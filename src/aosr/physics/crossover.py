"""Schroeder 頻率與有限元素／幾何路的純函式接合。

規格錨點是 ``docs/decisions/stage-nine-three-lane-stitch-and-report-with-interference.md``。上一代只作
出處，不作逐點相容答案：上一代（記號 ``v3-donor``）的 Schroeder 係數與公式見
``lib.scoring.modal`` 模組；面積加權 Eyring 公式見
``lib.physics.ray_kernel_tracing`` 模組第 221–237 行；對數頻率上的 ``t²(3−2t)`` 見
``lib.scoring.crossover`` 模組第 42–46 行。

這支只 import 標準庫、``config`` 與 ``geometry``（同層 ``physics`` 沒有反向依賴），
不讀檔、不改環境、不 print。所有輸出只由輸入決定。
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aosr.config.three_lane_crossover import (
    CROSSOVER_LOWER_FLOOR_HZ,
    EYRING_COEFFICIENT_S_PER_M,
    SCHROEDER_COEFFICIENT_SI,
    SCHROEDER_T60_BANDS_HZ,
)
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.geometry.shoebox import Room, Wall

FaceAbsorption = Mapping[Wall, Mapping[float, float]]


@dataclass(frozen=True)
class CrossoverWeights:
    """每個輸入頻率的功率互補權重，以及是否因上限而硬切。"""

    w_fem: tuple[float, ...]
    w_geo: tuple[float, ...]
    capped_by_upper_limit: bool


def _room_measures(room: Room) -> tuple[float, dict[Wall, float]]:
    """驗證鞋盒三軸並回體積與六面牆面積。"""
    dimensions = (float(room.Lx), float(room.Ly), float(room.Lz))
    if not all(math.isfinite(length) and length > 0.0 for length in dimensions):
        raise ValueError("鞋盒房三軸都必須是有限正數")
    lx, ly, lz = dimensions
    areas = {
        Wall.FLOOR: lx * ly,
        Wall.CEILING: lx * ly,
        Wall.X0: ly * lz,
        Wall.XL: ly * lz,
        Wall.Y0: lx * lz,
        Wall.YL: lx * lz,
    }
    return lx * ly * lz, areas


def _validated_bands(absorption_by_wall: FaceAbsorption) -> tuple[float, ...]:
    """驗證六面牆帶寬一致，回排序後頻率。"""
    expected_walls = set(Wall.all())
    actual_walls = set(absorption_by_wall)
    if actual_walls != expected_walls:
        missing = sorted(wall.wall_name() for wall in expected_walls - actual_walls)
        extra = sorted(str(wall) for wall in actual_walls - expected_walls)
        raise ValueError(f"吸音率牆面不完整：missing={missing!r}, extra={extra!r}")
    first_wall = Wall.all()[0]
    bands = set(absorption_by_wall[first_wall])
    if not bands:
        raise ValueError("吸音率沒有任何頻帶")
    for wall in Wall.all()[1:]:
        if set(absorption_by_wall[wall]) != bands:
            raise ValueError(f"{wall.wall_name()} 的吸音率頻帶跟其他牆不同")
    frequencies = tuple(sorted(float(frequency) for frequency in bands))
    if not all(math.isfinite(frequency) and frequency > 0.0 for frequency in frequencies):
        raise ValueError("吸音率頻帶必須是有限正頻率")
    missing_midbands = tuple(
        frequency
        for frequency in SCHROEDER_T60_BANDS_HZ
        if frequency not in frequencies
    )
    if missing_midbands:
        missing_text = ", ".join(f"{frequency:g}" for frequency in missing_midbands)
        raise ValueError(
            f"吸音率輸入缺 Schroeder 必需頻帶：{missing_text} Hz"
        )
    return frequencies


def _absorption(value: float, wall: Wall, frequency_hz: float) -> float:
    """把一格吸音率驗成 Eyring 可用的 ``0 <= alpha < 1``。"""
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{wall.wall_name()} 在 {frequency_hz:g} Hz 的吸音率不是數字"
        ) from exc
    if not math.isfinite(alpha) or not 0.0 <= alpha < 1.0:
        raise ValueError(
            f"{wall.wall_name()} 在 {frequency_hz:g} Hz 的吸音率必須有限且落在 [0,1)"
        )
    return alpha


def eyring_t60_by_band(
    room: Room,
    absorption_by_wall: FaceAbsorption,
) -> dict[float, float]:
    """由鞋盒六面每頻帶吸音率算面積加權 Eyring T60（秒）。

    ``abar = sum(S_k alpha_k)/S``，再算
    ``T60 = 0.161 V / [-S ln(1-abar)]``。係數只從 config 具名常數讀；吸音率
    到 1 或以上直接報錯，不作上一代的夾擠。全室加權平均為零時公式沒有有限值，也報錯。
    """
    volume, areas = _room_measures(room)
    frequencies = _validated_bands(absorption_by_wall)
    surface_area = sum(areas[wall] for wall in Wall.all())
    result: dict[float, float] = {}
    for frequency_hz in frequencies:
        weighted = sum(
            areas[wall]
            * _absorption(
                absorption_by_wall[wall][frequency_hz], wall, frequency_hz
            )
            for wall in Wall.all()
        )
        average = weighted / surface_area
        if average <= 0.0:
            raise ValueError(f"{frequency_hz:g} Hz 的面積加權平均吸音率必須大於 0")
        denominator = -surface_area * math.log(1.0 - average)
        result[frequency_hz] = EYRING_COEFFICIENT_S_PER_M * volume / denominator
    return result


def schroeder_frequency_hz(
    room: Room,
    t60_by_band_s: Mapping[float, float],
) -> float:
    """由 500/1000 Hz Eyring T60 算 ``f_s=2000 sqrt(mean(T60)/V)``。

    兩帶任缺其一就報錯，不自動換帶；算術平均與係數皆照第九段決策紙。
    """
    volume, _ = _room_measures(room)
    t60_values: list[float] = []
    for frequency_hz in SCHROEDER_T60_BANDS_HZ:
        if frequency_hz not in t60_by_band_s:
            raise ValueError(f"Schroeder 頻率缺 {frequency_hz:g} Hz 的 Eyring T60")
        t60 = float(t60_by_band_s[frequency_hz])
        if not math.isfinite(t60) or t60 <= 0.0:
            raise ValueError(f"{frequency_hz:g} Hz 的 Eyring T60 必須是有限正數")
        t60_values.append(t60)
    mean_t60 = (t60_values[0] + t60_values[1]) / 2.0
    return SCHROEDER_COEFFICIENT_SI * math.sqrt(mean_t60 / volume)


def _soft_geometric_weight(frequency_hz: float, lower_hz: float) -> float:
    """在 ``[lower_hz, cap]`` 的 log2 座標上算 C1 平滑階梯。"""
    if frequency_hz <= lower_hz:
        return 0.0
    if frequency_hz >= FEM_GEOMETRIC_CROSSOVER_CAP_HZ:
        return 1.0
    t = math.log2(frequency_hz / lower_hz) / math.log2(
        FEM_GEOMETRIC_CROSSOVER_CAP_HZ / lower_hz
    )
    return t * t * (3.0 - 2.0 * t)


def crossover_weights(
    frequencies_hz: Sequence[float],
    f_s_hz: float,
) -> CrossoverWeights:
    """算每點 ``(w_fem, w_geo)``，並回報是否被有限元素上限硬切。

    ``f_s < cap`` 時，軟交接是 ``[max(f_s, floor), cap]``；下端以下只有
    有限元素，cap 以上只有幾何。``f_s >= cap`` 時，``<= cap`` 全有限元素、
    ``> cap`` 全幾何，並把 ``capped_by_upper_limit`` 設為真。有限元素權重永遠在
    幾何權重算完後用 ``1.0 - w_geo`` 定義。
    """
    f_s = float(f_s_hz)
    if not math.isfinite(f_s) or f_s <= 0.0:
        raise ValueError("f_s 必須是有限正數")
    frequencies = tuple(float(frequency) for frequency in frequencies_hz)
    if not all(math.isfinite(frequency) and frequency > 0.0 for frequency in frequencies):
        raise ValueError("輸入頻率必須全是有限正數")
    capped = f_s >= FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    if capped:
        w_geo = tuple(
            0.0 if frequency <= FEM_GEOMETRIC_CROSSOVER_CAP_HZ else 1.0
            for frequency in frequencies
        )
    else:
        lower_hz = max(f_s, CROSSOVER_LOWER_FLOOR_HZ)
        w_geo = tuple(
            _soft_geometric_weight(frequency, lower_hz)
            for frequency in frequencies
        )
    w_fem = tuple(1.0 - weight for weight in w_geo)
    return CrossoverWeights(w_fem, w_geo, capped)
