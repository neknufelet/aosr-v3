"""家具一次反射、有限尺寸聲壓、牆面折線與直達遮擋的獨立計算零件。

Furniture 是凍結的 (furniture_id, FurnitureBox)；同一批 id 須唯一。
座標使用 geometry 的 Vec3（三軸公尺 tuple），多聲源／接收點使用 id→Vec3。
家具按 id、外露面按 geometry 的上／下／+x／−x／+y／−y 固定順序列舉。
margin_m、c、rho_c 都由呼叫端傳入，不讀設定、不內建容差或物理常數。

每條家具路徑只有一次反射，沒有牆面反射階數 K；兩段逐件查遮擋，包含
反射家具本身。家具一律不透聲。有限尺寸能量只作用於家具路徑，聲壓乘
√(K₁K₂)；材料使用呼叫端提供的逐頻實數阻抗。限制見 finite_reflector。
不接能量加總、晚期混響、整房散射、時間窗、指向性、路徑表、報表、輸入關或搜尋。
"""
from __future__ import annotations

import cmath
import math
from collections.abc import Mapping
from dataclasses import dataclass

from aosr.geometry.furniture import (
    FaceDirection,
    FurnitureBox,
    FurnitureFace,
    Vec3,
    mirror_point,
    segment_blocked_by_box,
    single_bounce_point,
)
from aosr.physics.amplitude import reflection_coefficient
from aosr.physics.finite_reflector import _finite_real, _positive, finite_size_energy
from aosr.physics.room_paths import RoomPath


@dataclass(frozen=True)
class Furniture:
    """物件識別與不可變盒子綁定；不修改 geometry 的既有資料型別。"""

    furniture_id: str
    box: FurnitureBox

    def __post_init__(self) -> None:
        if not isinstance(self.furniture_id, str) or not self.furniture_id.strip():
            raise ValueError("家具 id 必須是非空文字")
        if not isinstance(self.box, FurnitureBox):
            raise ValueError("家具須提供 FurnitureBox")


@dataclass(frozen=True)
class FurniturePath:
    """有效家具一次反射；image 為聲源對面的無限平面之真鏡像。"""

    furniture_id: str
    face: FurnitureFace
    hit: Vec3
    image: Vec3
    d_inc: float
    d_refl: float
    dist_m: float
    delay_s: float
    cos_theta: float
    departure_direction: Vec3

    @property
    def face_direction(self) -> FaceDirection:
        """面的房間座標方向；順序沿用 geometry，不增加反射階數欄位。"""
        return self.face.direction


def _ordered_furniture(furniture: tuple[Furniture, ...]) -> tuple[Furniture, ...]:
    ordered = tuple(sorted(furniture, key=lambda item: item.furniture_id))
    ids = tuple(item.furniture_id for item in ordered)
    if len(set(ids)) != len(ids):
        raise ValueError("同一批家具 id 不准重複")
    return ordered


def _checked_margin(margin_m: float) -> float:
    margin = _finite_real(margin_m, "接觸界線")
    if margin < 0.0:
        raise ValueError("接觸界線須為非負數")
    return margin


def _segment_blockers(start: Vec3, end: Vec3, furniture: tuple[Furniture, ...],
                      margin_m: float) -> tuple[str, ...]:
    """每件都查；端點在深內部的幾何輸入錯不被先遇到的遮擋短路藏起來。"""
    return tuple(item.furniture_id for item in furniture
                 if segment_blocked_by_box(start, end, item.box, margin_m=margin_m))


def _one_furniture_path(source: Vec3, receiver: Vec3, item: Furniture, face: FurnitureFace,
                        furniture: tuple[Furniture, ...], c: float, margin_m: float) -> FurniturePath | None:
    hit = single_bounce_point(source, receiver, face, margin_m=margin_m)
    if hit is None:
        return None
    # 兩段都逐件查，含自己；面上端點僅免接觸，沒有免除整顆盒子。
    incident_blockers = _segment_blockers(source, hit, furniture, margin_m)
    reflected_blockers = _segment_blockers(hit, receiver, furniture, margin_m)
    if incident_blockers or reflected_blockers:
        return None
    d_inc = math.dist(source, hit)
    d_refl = math.dist(hit, receiver)
    direction: Vec3 = ((hit[0] - source[0]) / d_inc, (hit[1] - source[1]) / d_inc,
                       (hit[2] - source[2]) / d_inc)
    cosine = abs(sum(normal * ray for normal, ray in zip(face.normal, direction, strict=True)))
    dist_m = d_inc + d_refl
    return FurniturePath(item.furniture_id, face, hit, mirror_point(source, face, margin_m=margin_m),
                         d_inc, d_refl, dist_m, dist_m / c, cosine, direction)


def single_bounce_furniture_paths(source: Vec3, receiver: Vec3, furniture: tuple[Furniture, ...], *,
                                  c: float, margin_m: float) -> tuple[FurniturePath, ...]:
    """所有有效家具一次反射；不接牆面列舉，不受牆面階數 K 限制。

    幾何零件負責同在外側、有限面邊界與接觸帶；被任何一件家具擋住任一段即移除。
    """
    speed, margin = _positive(c, "聲速"), _checked_margin(margin_m)
    ordered = _ordered_furniture(furniture)
    paths: list[FurniturePath] = []
    for item in ordered:
        for face in item.box.exposed_faces:
            path = _one_furniture_path(source, receiver, item, face, ordered, speed, margin)
            if path is not None:
                paths.append(path)
    return tuple(paths)


def furniture_reflection_coefficients(impedance_per_freq: tuple[float, ...], *,
                                      cos_theta: float, rho_c: float) -> tuple[complex, ...]:
    """呼叫端提供每件家具的逐頻實數 Z；沿用 amplitude 的局部反應反射係數。"""
    medium = _positive(rho_c, "空氣特性阻抗")
    cosine = _finite_real(cos_theta, "入射角餘弦")
    if not 0.0 <= cosine <= 1.0:
        raise ValueError("入射角餘弦須在零到一之間")
    return tuple(reflection_coefficient(complex(_positive(z, "家具阻抗"), 0.0), cosine, medium)
                 for z in impedance_per_freq)


def furniture_path_pressure(frequencies_hz: tuple[float, ...], *, dist_m: float, c: float,
                            refl_per_freq: tuple[complex, ...],
                            energy_factors: tuple[float, ...]) -> tuple[complex, ...]:
    """逐頻聲壓 (1/dist)R√K exp(−i2πf dist/c)；K 是能量而非聲壓比。

    與 amplitude.path_pressure 使用同一運算順序，K=1 時逐位相等；這裡不接 Materials，
    也不呼叫該函式。相位只取兩段總距離；有限尺寸修正本身不帶相位。
    """
    distance, speed = _positive(dist_m, "路徑距離"), _positive(c, "聲速")
    if len(frequencies_hz) != len(refl_per_freq) or len(frequencies_hz) != len(energy_factors):
        raise ValueError("頻率、反射係數與有限尺寸能量的列長必須一致")
    out: list[complex] = []
    for frequency, coefficient, factor in zip(frequencies_hz, refl_per_freq, energy_factors, strict=True):
        f = _positive(frequency, "頻率")
        energy = _finite_real(factor, "有限尺寸能量比")
        if not 0.0 <= energy <= 1.0:
            raise ValueError("有限尺寸能量比須在零到一之間")
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("反射係數須為有限數")
        omega = 2.0 * math.pi * f
        tau = distance / speed
        phase = cmath.exp(complex(0.0, -omega * tau))
        out.append((1.0 / distance) * coefficient * math.sqrt(energy) * phase)
    return tuple(out)


def furniture_path_amplitude(path: FurniturePath, frequencies_hz: tuple[float, ...],
                             impedance_per_freq: tuple[float, ...], *, rho_c: float,
                             c: float) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    """組合單條家具路徑的 (逐頻 R, 逐頻聲壓)，不乘整房散射或指向性。"""
    coefficients = furniture_reflection_coefficients(impedance_per_freq, cos_theta=path.cos_theta, rho_c=rho_c)
    energy = finite_size_energy(frequencies_hz, path.face, path.departure_direction,
                                d_inc=path.d_inc, d_refl=path.d_refl, c=c)
    pressure = furniture_path_pressure(frequencies_hz, dist_m=path.dist_m, c=c,
                                       refl_per_freq=coefficients, energy_factors=energy)
    return coefficients, pressure


def filter_room_paths(paths: tuple[RoomPath, ...], source: Vec3, receiver: Vec3,
                      furniture: tuple[Furniture, ...], *,
                      margin_m: float) -> tuple[tuple[RoomPath, ...], tuple[tuple[int, int, int, int, int, int], ...]]:
    """回 (原路徑中留下的項目, 被遮擋的 identity 清單)，各沿原輸入順序。

    RoomPath.bounces 從接收點往聲源排列，實際折線為 source→反序 bounces→receiver。
    沒有反彈的直達也查；不改留下項目的材料、聲壓或身分。家具為空時回同一個 paths tuple。
    """
    if not furniture:
        return paths, ()
    margin = _checked_margin(margin_m)
    ordered = _ordered_furniture(furniture)
    kept: list[RoomPath] = []
    blocked: list[tuple[int, int, int, int, int, int]] = []
    for path in paths:
        vertices = (source, *(bounce.point for bounce in reversed(path.bounces)), receiver)
        blockers = tuple(_segment_blockers(start, end, ordered, margin)
                         for start, end in zip(vertices[:-1], vertices[1:], strict=True))
        if any(blockers):
            blocked.append(path.identity)
        else:
            kept.append(path)
    return tuple(kept), tuple(blocked)


def direct_path_blockers(sources: Mapping[str, Vec3], receivers: Mapping[str, Vec3],
                         furniture: tuple[Furniture, ...], *,
                         margin_m: float) -> dict[tuple[str, str], tuple[str, ...]]:
    """每一對 (聲源 id, 接收點 id) 回所有擋住它的家具 id，空 tuple 就是暢通。

    主位、周圍點同樣逐對回傳，不在這裡決定搜尋剔除或方案拒收政策。
    單一聲源也用一項 mapping；鍵按聲源 id、接收點 id 排序，家具 id 亦固定排序。
    """
    margin, ordered = _checked_margin(margin_m), _ordered_furniture(furniture)
    return {(source_id, receiver_id): _segment_blockers(sources[source_id], receivers[receiver_id], ordered, margin)
            for source_id in sorted(sources) for receiver_id in sorted(receivers)}
