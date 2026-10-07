"""家具有限矩形面的鏡面能量修正（決策紙第 11 條）。

兩條邊用全長；投影長度經面積歸一後，按真實入射、反射距離算兩個截止頻率。
入射平面平行一邊時回到 Rindel 2005 簡報與 Zeng 等人 2006 的原文式；
任意方位角的投影與歸一是決策採用的推導，只有低頻端的積分核對。
回傳 K₁K₂ 是能量比；同調相加的聲壓須乘其平方根。聲速由呼叫端提供。

這是公式一致的近似，不是實際家具的精度驗證：尺寸遠小於距離的前提對
桌面與沙發常不成立；截止以上截在 1，不追精確式的起伏；靠邊反射會高估，
邊緣效應與完整繞射沒有建模。聲壓只乘實數，不改相位、不產生繞射路徑。
靠牆或接靠背的非自由邊仍照自由板算，可能低估；只收矩形面。
減掉的能量不當材料吸音，也不重新分配。牆面路徑不使用此修正。
"""
from __future__ import annotations

import math
from numbers import Real

from aosr.geometry.furniture import FurnitureFace, Vec3


def _finite_real(value: float, label: str) -> float:
    """數學輸入必須是有限實數；不把布林或複數當數值。"""
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{label}必須是有限實數")
    return float(value)


def _positive(value: float, label: str) -> float:
    value = _finite_real(value, label)
    if value <= 0.0:
        raise ValueError(f"{label}必須是正數")
    return value


def _unit_direction(direction: Vec3) -> Vec3:
    """把有限非零方向化為單位向量；不以容差或夾值修補退化方向。"""
    if len(direction) != 3:
        raise ValueError("入射方向須有三個分量")
    x, y, z = (_finite_real(value, "入射方向") for value in direction)
    length = _positive(math.sqrt(x * x + y * y + z * z), "入射方向長度")
    return x / length, y / length, z / length


def _dot(first: Vec3, second: Vec3) -> float:
    return sum(a * b for a, b in zip(first, second, strict=True))


def effective_edge_lengths(face: FurnitureFace, incident_direction: Vec3) -> tuple[float, float]:
    """回兩條 q_i（公尺）：p_i=L_i√(1−(ê_i·r̂)²)，g=√(L₁L₂cosθ/(p₁p₂))。

    方向按聲源→反射點，內部正規化；不因正負法向改能量。掠射退化時公式
    沒有定義，明確拒收，不塞 epsilon（小數救援值）或夾住根號內的數。
    """
    unit = _unit_direction(incident_direction)
    cosine = _positive(abs(_dot(face.normal, unit)), "入射角餘弦")
    first, second = face.edge_lengths_m
    edge_first, edge_second = face.edge_units
    p_first = _positive(first * math.sqrt(1.0 - _dot(edge_first, unit) ** 2), "投影邊長")
    p_second = _positive(second * math.sqrt(1.0 - _dot(edge_second, unit) ** 2), "投影邊長")
    gain = math.sqrt(first * second * cosine / (p_first * p_second))
    return _positive(gain * p_first, "有效邊長"), _positive(gain * p_second, "有效邊長")


def cutoff_frequencies(face: FurnitureFace, incident_direction: Vec3, *,
                       d_inc: float, d_refl: float, c: float) -> tuple[float, float]:
    """回兩方向 f_i=c/(q_i²(1/d_inc+1/d_refl))，順序與面的邊一致。

    不引入特徵距離 a*，避免原文分歧的係數二倍位置；兩段都是實際一次反射距離。
    """
    incident = _positive(d_inc, "入射距離")
    reflected = _positive(d_refl, "反射距離")
    speed = _positive(c, "聲速")
    q_first, q_second = effective_edge_lengths(face, incident_direction)
    inverse_distance = 1.0 / incident + 1.0 / reflected
    return (_positive(speed / (q_first ** 2 * inverse_distance), "截止頻率"),
            _positive(speed / (q_second ** 2 * inverse_distance), "截止頻率"))


def finite_size_energy(frequencies_hz: tuple[float, ...], face: FurnitureFace,
                       incident_direction: Vec3, *, d_inc: float, d_refl: float,
                       c: float) -> tuple[float, ...]:
    """逐頻回鏡面能量 K₁K₂；唯一夾值是決策紙指定的 min(1, f/f_i)。"""
    first, second = cutoff_frequencies(face, incident_direction, d_inc=d_inc, d_refl=d_refl, c=c)
    frequencies = tuple(_positive(frequency, "頻率") for frequency in frequencies_hz)
    return tuple(min(1.0, frequency / first) * min(1.0, frequency / second) for frequency in frequencies)
