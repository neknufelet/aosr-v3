"""#505 聲源指向性模型本體；本刀不接入求解或報表。"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import j1

from aosr.config.directivity_defaults import AllowedRange, DirectivityDefaults
from aosr.config.speaker_directivity import DIRECTIVITY_REAR_GAIN_MIN
from aosr.geometry.shoebox import Point
from aosr.physics.room_paths import RoomPath


class SourceModel(StrEnum):
    OMNIDIRECTIONAL = "omnidirectional"
    TWO_PARAMETER = "analytic_axisymmetric_two_parameter_v1"
    V2_COMPAT = "v2_compat_baffled_piston"


MODEL_VERSION = "analytic_axisymmetric_two_parameter_v1"


@dataclass(frozen=True)
class TwoParameterValues:
    """單頻固定參數；供邊界驗算，也可表示不隨頻率變化的專案設定。"""

    beta: float
    power_floor_db: float
    allowed_range: AllowedRange


def unit_vector(v: Sequence[float]) -> tuple[float, float, float]:
    """軸向與路徑出發方向共用的唯一三維單位化函式。"""
    if len(v) != 3 or not all(math.isfinite(float(x)) for x in v):
        raise ValueError("方向向量必須是三個有限數")
    norm = math.hypot(*v)
    if norm == 0.0:
        raise ValueError("零向量沒有方向")
    return (v[0] / norm, v[1] / norm, v[2] / norm)


def one_minus_cos(u: Sequence[float], a: Sequence[float]) -> float:
    """兩個單位向量差的平方一半；保住完全同向時的精確零。"""
    if len(u) != 3 or len(a) != 3:
        raise ValueError("方向向量必須是三維")
    value = sum((left - right) ** 2 for left, right in zip(u, a, strict=True)) / 2.0
    return min(2.0, max(0.0, value))


def _axis(frequencies_hz: Sequence[float]) -> np.ndarray:
    axis = np.asarray(frequencies_hz, dtype=float)
    if axis.ndim != 1 or not np.all(np.isfinite(axis)) or np.any(axis <= 0):
        raise ValueError("頻率軸必須是一維有限正數")
    return axis


def _values(
    axis: np.ndarray, params: DirectivityDefaults | TwoParameterValues
) -> tuple[np.ndarray, np.ndarray]:
    bounds = params.allowed_range
    if isinstance(params, TwoParameterValues):
        beta = np.full_like(axis, params.beta)
        floor_db = np.full_like(axis, params.power_floor_db)
    else:
        curve = params.two_parameter
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            beta = curve.beta_limit / (1.0 + (curve.beta_corner_hz / axis) ** curve.beta_exponent)
            floor_db = curve.power_floor_limit_db / (
                1.0 + (curve.power_floor_corner_hz / axis) ** curve.power_floor_exponent
            )
    for name, values, lower, upper in (
        ("beta", beta, bounds.beta_min, bounds.beta_max),
        ("power_floor_db", floor_db, bounds.power_floor_min_db, bounds.power_floor_max_db),
    ):
        invalid = np.flatnonzero(~np.isfinite(values) | (values < lower) | (values > upper))
        if invalid.size:
            index = int(invalid[0])
            raise ValueError(f"{name} 在 {axis[index]} Hz 的值 {values[index]} 超出 [{lower}, {upper}]")
    return beta, np.power(10.0, floor_db / 10.0)


def two_parameter_pressure_factor(
    x: float, frequencies_hz: Sequence[float], params: DirectivityDefaults | TwoParameterValues
) -> np.ndarray:
    """一次計算整條頻率軸的聲壓倍率；正前方逐位為一。

    穩定式 ``sqrt(1 + (1−ρ)·expm1(−2βx))`` 讓正前方逐位是 1；代價是功率下限很低時正後方
    有 1 與 −1 相消的捨入（允許範圍的角 β＝10、−80 dB 約 2e-9 相對），遠小於任何聲學意義。
    """
    if not math.isfinite(x) or not 0.0 <= x <= 2.0:
        raise ValueError(f"x 必須在 [0, 2]，收到 {x}")
    axis = _axis(frequencies_hz)
    beta, floor = _values(axis, params)
    return np.sqrt(1.0 + (1.0 - floor) * np.expm1(-2.0 * beta * x))


def two_parameter_power_ratio(
    frequencies_hz: Sequence[float], params: DirectivityDefaults | TwoParameterValues
) -> np.ndarray:
    """兩參數模型的球面平均功率公式解；beta 為零時取極限一。"""
    axis = _axis(frequencies_hz)
    beta, floor = _values(axis, params)
    # 先算 (1−e^{−4β})／(4β) 這個比值再乘 (1−ρ)：β 落在浮點最小那一段（約 1e-316 以下）時，
    # 先乘 (1−ρ) 會被那一段粗的間隔捨掉，公式解變成 1＋ρ、超出契約（對抗席實測）。
    shape = np.ones_like(beta)
    nonzero = beta != 0.0
    shape[nonzero] = (-np.expm1(-4.0 * beta[nonzero])) / (4.0 * beta[nonzero])
    ratio: np.ndarray = (1.0 - floor) * shape + floor
    return ratio


def v2_compat_pressure_factor(
    x: float, frequencies_hz: Sequence[float], baffle_width_m: float,
    piston_radius_m: float, sound_speed_m_s: float,
) -> np.ndarray:
    """上一代活塞形狀對照；保留正前方一，不做總功率正規化。"""
    if not math.isfinite(x) or not 0.0 <= x <= 2.0:
        raise ValueError("x 必須在 [0, 2]")
    if any(not math.isfinite(v) or v <= 0 for v in (baffle_width_m, piston_radius_m, sound_speed_m_s)):
        raise ValueError("面板寬、活塞半徑、聲速必須是有限正數")
    axis = _axis(frequencies_hz)
    argument = (2.0 * math.pi * axis * piston_radius_m / sound_speed_m_s) * math.sqrt(x * (2.0 - x))
    piston = np.ones_like(argument)
    nonzero = argument != 0.0
    piston[nonzero] = 2.0 * j1(argument[nonzero]) / argument[nonzero]
    corner = sound_speed_m_s / (math.pi * baffle_width_m)
    rear = DIRECTIVITY_REAR_GAIN_MIN + (1.0 - DIRECTIVITY_REAR_GAIN_MIN) / (1.0 + (axis / corner) ** 2)
    return piston * (1.0 - (1.0 - rear) * x / 2.0)


def v2_compat_power_ratio(
    frequencies_hz: Sequence[float], baffle_width_m: float,
    piston_radius_m: float, sound_speed_m_s: float,
) -> np.ndarray:
    """上一代相容對照唯一的功率算法：64 點高斯–勒讓德球面積分。"""
    nodes, weights = leggauss(64)
    axis = _axis(frequencies_hz)
    power = np.zeros_like(axis)
    for node, weight in zip(nodes, weights, strict=True):
        factor = v2_compat_pressure_factor(
            1.0 - float(node), tuple(axis), baffle_width_m, piston_radius_m, sound_speed_m_s
        )
        power += (float(weight) / 2.0) * factor**2
    return power


def speaker_axis(speaker: Point, aim: Point) -> tuple[float, float, float]:
    """從聲源指向對準點；重合點由 unit_vector 拒收。"""
    return unit_vector(tuple(b - a for a, b in zip(speaker.as_tuple(), aim.as_tuple(), strict=True)))


def departure_direction(path: RoomPath, receiver: Point) -> tuple[float, float, float]:
    """identity 奇數反射軸翻轉鏡像源→接收點；有別於路徑表的到達方向。

    bounces 由接收點往回列，最後一個是聲源離開後最先碰到的牆；
    出發方向與到達方向兩者都要保留，各有不同用途。
    """
    displacement = tuple(b - a for a, b in zip(path.image, receiver.as_tuple(), strict=True))
    return unit_vector(tuple(path.identity[2 * i + 1] * displacement[i] for i in range(3)))


def _pressure_function(
    model: SourceModel, axis: Sequence[float], params: DirectivityDefaults | TwoParameterValues | None,
    baffle_width_m: float | None, piston_radius_m: float | None, sound_speed_m_s: float | None,
) -> Callable[[float], np.ndarray]:
    """依聲源模型挑出「x → 整條頻率軸的 D」；該給的參數缺一個就拒收，不補預設。"""
    if model == SourceModel.TWO_PARAMETER:
        if params is None:
            raise ValueError("兩參數模型必須給 params")
        chosen = params
        return lambda x: two_parameter_pressure_factor(x, axis, chosen)
    if model == SourceModel.V2_COMPAT:
        if baffle_width_m is None or piston_radius_m is None or sound_speed_m_s is None:
            raise ValueError("上一代相容對照必須明給面板寬、活塞半徑與聲速")
        width, radius, speed = baffle_width_m, piston_radius_m, sound_speed_m_s
        return lambda x: v2_compat_pressure_factor(x, axis, width, radius, speed)
    raise ValueError(f"未知聲源模型：{model}")


def apply_pressure_factor(
    paths: Sequence[RoomPath], receiver: Point, axis: Sequence[float], model: SourceModel,
    *, source: Point | None = None, aim: Point | None = None,
    params: DirectivityDefaults | TwoParameterValues | None = None,
    baffle_width_m: float | None = None, piston_radius_m: float | None = None,
    sound_speed_m_s: float | None = None,
) -> list[RoomPath]:
    """逐路徑乘 D 後換掉複數聲壓；全向回原物件，尚未接入求解。

    喇叭軸線在這裡用 ``speaker_axis(source, aim)`` 算，跟出發方向共用同一支單位化函式，
    所以對準點＝接收點時直達那條的 x 逐位是 0、聲壓逐位不變（不收外面算好的軸向：
    再單位化一次會差一格，相容對照的活塞項就不是逐位 1）。``source`` 必須是算這批路徑的
    那個聲源——直達那條的鏡像座標就是它，對不上就拒收。
    上一代相容對照的面板寬、活塞半徑與聲速由呼叫端明給（兩組上一代預設尺寸在
    ``aosr.config.speaker_directivity.SPEAKER_PRESETS``，由呼叫端挑），這裡不藏預設型號。
    """
    if model == SourceModel.OMNIDIRECTIONAL:
        return list(paths)
    if source is None or aim is None:
        raise ValueError("非全向模型必須給聲源位置與對準點")
    direction = speaker_axis(source, aim)
    factor_at = _pressure_function(model, axis, params, baffle_width_m, piston_radius_m, sound_speed_m_s)
    result = []
    for path in paths:
        if path.order == 0 and path.image != source.as_tuple():
            raise ValueError("直達路徑的聲源座標跟給的 source 不同：路徑不是從這支喇叭算的")
        factors = factor_at(one_minus_cos(departure_direction(path, receiver), direction))
        if len(path.path_pressure) != len(factors):
            raise ValueError("路徑聲壓與頻率軸長度不符")
        pressure = tuple(p * float(d) for p, d in zip(path.path_pressure, factors, strict=True))
        result.append(replace(path, path_pressure=pressure))
    return result
