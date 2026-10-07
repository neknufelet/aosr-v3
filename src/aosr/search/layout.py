"""固定單位搜尋空間換成對稱擺位；只在限制檢查通過後才建方案。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from aosr.geometry.shoebox import Point, Wall
from aosr.reporting.scheme import Scheme, project_facing, project_midpoint, speaker_pair_ids
from aosr.scoring.receiver_set import ReceiverSet
from aosr.search.layout_settings import LayoutSettings, Span


SEARCH_QUANTITIES = ("front_distance", "spacing", "listening_distance")
UNIT_SPACE: Mapping[str, tuple[float, float]] = MappingProxyType(dict.fromkeys(SEARCH_QUANTITIES, (0.0, 1.0)))
CARDINAL_FACINGS: Final[tuple[tuple[float, float], ...]] = (
    (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0),
)
# 起點往內推的次數上限：一般房間尺寸實測最多驗 6 次；只有巨大房間配近 180° 夾角這種不可能的配置會超過。
MAX_START_ADJUSTMENTS: Final = 16


@dataclass(frozen=True)
class LayoutParams:
    """第二節第 4 條的三個有限正距離；退化連線不進此型別。"""

    front_distance_m: float
    spacing_m: float
    listening_distance_m: float

    def __post_init__(self) -> None:
        for value in (self.front_distance_m, self.spacing_m, self.listening_distance_m):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("layout distances must be finite and positive")


@dataclass(frozen=True)
class Placement:
    """尚未判合法的點擺位；receivers 保留專案順序，facing 是聆聽者朝前牆的單位向量。"""

    left: Point
    right: Point
    primary: Point
    receivers: tuple[tuple[str, Point], ...]
    facing: tuple[float, float]


def _spans(settings: LayoutSettings) -> tuple[Span, Span, Span]:
    return settings.front_distance_m, settings.spacing_m, settings.listening_distance_m


def params_from_unit(unit: Mapping[str, float], settings: LayoutSettings) -> LayoutParams:
    """各量獨立用 low + u * (high-low) 換算；上下限不隨別的量變動。"""
    if unit.keys() != UNIT_SPACE.keys():
        raise ValueError("unit parameters must name exactly the search quantities")
    values = []
    for name, span in zip(SEARCH_QUANTITIES, _spans(settings), strict=True):
        value = unit[name]
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"unit parameter {name} must be in [0, 1]")
        # 夾回範圍內：u=1 時 low + (high-low) 可能比 high 多出最末一位。
        values.append(min(span.high, max(span.low, span.low + value * (span.high - span.low))))
    return LayoutParams(*values)


def unit_from_params(params: LayoutParams, settings: LayoutSettings) -> Mapping[str, float]:
    """反算固定比例供起點入列；不裁切範圍外的參數。"""
    values = (params.front_distance_m, params.spacing_m, params.listening_distance_m)
    unit: dict[str, float] = {}
    for name, span, value in zip(SEARCH_QUANTITIES, _spans(settings), values, strict=True):
        if not span.low <= value <= span.high:
            raise ValueError(f"parameter {name} is outside its search range")
        unit[name] = (value - span.low) / (span.high - span.low)
    return MappingProxyType(unit)


def _rotate_offset(offset: tuple[float, float, float], turns: int) -> tuple[float, float, float]:
    """只交換、變號，不引入三角函數的近似；零轉時原位移逐位保留。"""
    x, y, z = offset
    return ((x, y, z), (-y, x, z), (-x, -y, z), (y, -x, z))[turns]


def _receivers(project: Scheme, primary: Point, facing: tuple[float, float]) -> tuple[tuple[str, Point], ...]:
    original_facing = project_facing(project)
    turns = (CARDINAL_FACINGS.index(facing) - CARDINAL_FACINGS.index(original_facing)) % 4
    original_primary = project.receiver_set.primary.position_m
    receivers = []
    for receiver in project.receiver_set.points:
        offset = tuple(a - b for a, b in zip(receiver.position_m, original_primary, strict=True))
        rotated = _rotate_offset((offset[0], offset[1], offset[2]), turns)
        receivers.append((receiver.receiver_id, Point(primary.x + rotated[0], primary.y + rotated[1],
                                                     primary.z + rotated[2])))
    return tuple(receivers)


def place(project: Scheme, settings: LayoutSettings, params: LayoutParams) -> Placement:
    """第二節第 2～5、9 條：對稱聲學中心、中軸主位、原座位佈局隨面向一起轉。"""
    room = project.scene.room_m
    wall = Wall.from_name(settings.front_wall)
    axis, plane = wall.axis(), wall.plane(room)
    inward = 1.0 if wall.kind() == "zero" else -1.0
    along = plane + inward * params.front_distance_m
    across = room.length(1 - axis) / 2.0 + settings.axis_offset_m
    mx, my = (along, across) if axis == 0 else (across, along)
    facing = (-inward, 0.0) if axis == 0 else (0.0, -inward)
    lx, ly = -facing[1], facing[0]  # 面向逆時針轉 90°＝聆聽者的左手方向。
    half = params.spacing_m / 2.0
    left = Point(mx + lx * half, my + ly * half, settings.speaker_height_m)
    right = Point(mx - lx * half, my - ly * half, settings.speaker_height_m)
    primary = Point(mx - facing[0] * params.listening_distance_m,
                    my - facing[1] * params.listening_distance_m, settings.ear_height_m)
    return Placement(left, right, primary, _receivers(project, primary, facing), facing)


def to_scheme(project: Scheme, placement: Placement, scheme_id: str) -> Scheme:
    """合法後才呼叫：保留場景、用途、聲源、聲道，以及每席的代號／角色／權重／方向。"""
    left_id, right_id = speaker_pair_ids(project)
    positions = dict(placement.receivers)
    if positions.keys() != {receiver.receiver_id for receiver in project.receiver_set.points}:
        raise ValueError("placement must preserve project receiver identities")
    points = tuple(receiver.model_dump() | {"position_m": positions[receiver.receiver_id].as_tuple()}
                   for receiver in project.receiver_set.points)
    return Scheme.model_validate(project.model_dump() | {
        "scheme_id": scheme_id,
        "speakers": {left_id: placement.left, right_id: placement.right},
        "receiver_set": ReceiverSet.model_validate({"points": points}),
    })


def _start_spacing_interval(settings: LayoutSettings, listening_factor: float) -> tuple[float, float] | None:
    """固定起點夾角，將三條距離範圍換成間距區間取交集，供選離原間距最近的值。"""
    low = max(settings.spacing_m.low, settings.listening_distance_m.low / listening_factor)
    high = min(settings.spacing_m.high, settings.listening_distance_m.high / listening_factor)
    limits = settings.listening_range_m
    if limits is not None:
        height = abs(settings.speaker_height_m - settings.ear_height_m)
        if limits.high <= height:
            return None
        radius_factor = math.hypot(0.5, listening_factor)
        low = max(low, math.sqrt(max(0.0, limits.low ** 2 - height ** 2)) / radius_factor)
        high = min(high, math.sqrt(limits.high ** 2 - height ** 2) / radius_factor)
    if low > high:
        return None
    return low, high


def _listening_factor(angle: float) -> float:
    """60° 保留原本 sqrt(3)/2 的換算，其他角度按正切算。"""
    return math.sqrt(3.0) / 2.0 if angle == 60.0 else 1.0 / (2.0 * math.tan(math.radians(angle / 2.0)))


def _push_endpoint(value: float, low: float, high: float, steps: int) -> float | None:
    """端點從一個最小可表示浮點單位往區間內推；單點交集無處可推，留給實際路線判定。"""
    if low == high or low < value < high or steps == 0:
        return value
    toward = high if value == low else low
    shifted = value + (math.nextafter(value, toward) - value) * steps
    return shifted if low <= shifted <= high else None


def _start_rounding_clear(project: Scheme, settings: LayoutSettings, params: LayoutParams) -> bool:
    """起點自己保證的三條：單位來回仍在搜尋範圍內（取樣器不收範圍外的點）、夾角與耳距不出界。

    入列與試算走同一條捨入鏈；牆面間隙、禁區、座位出房等幾何條件不參與，交給搜尋記原因。
    """
    from aosr.search.constraints import Reason, check

    try:
        actual = params_from_unit(unit_from_params(params, settings), settings)
    except ValueError:
        return False
    own = {Reason.BASE_ANGLE_OUT_OF_RANGE, Reason.LISTENING_DISTANCE_OUT_OF_RANGE}
    return not any(item.reason in own for item in check(project, settings, place(project, settings, actual)))


def _start_candidate(
    settings: LayoutSettings, front: float, original: float, factors: tuple[float, float, float], steps: int,
) -> LayoutParams | None:
    """factors＝(起點比例, 比例下限, 比例上限)；夾角端點的比例與間距各往區間內推 steps 個最小浮點單位。"""
    adjusted = _push_endpoint(*factors, steps)
    if adjusted is None:
        return None
    interval = _start_spacing_interval(settings, adjusted)
    if interval is None:
        return None
    low, high = interval
    spacing = _push_endpoint(min(high, max(low, original)), low, high, steps)
    if spacing is None:
        return None
    return LayoutParams(front, spacing, spacing * adjusted)


def _validated_start(
    project: Scheme, settings: LayoutSettings, front: float, original: float, angle: float,
) -> LayoutParams | None:
    """反推與擺位的兩條捨入鏈不同；三條自保條件驗不過才往內推，次數上限把推移限在捨入尺度。

    三條過了就交回，幾何不合法（例如撞牆面間隙、禁區）由搜尋記成不合法並寫原因；推到上限仍不過回 None。
    """
    factor = _listening_factor(angle)
    limits = settings.base_angle_deg
    factors = (factor, factor, factor)
    if limits is not None:
        factors = (factor, _listening_factor(limits.high), _listening_factor(limits.low))
    for attempt in range(MAX_START_ADJUSTMENTS):
        candidate = _start_candidate(settings, front, original, factors, 0 if attempt == 0 else 2 ** (attempt - 1))
        if candidate is None:
            return None
        if _start_rounding_clear(project, settings, candidate):
            return candidate
    return None


def standard_start(project: Scheme, settings: LayoutSettings) -> LayoutParams | None:
    """第二節第 6 條主對話做法：60° 正三角形僅為起點偏好，範圍外回 None。

    原離前牆＝兩聲學中心中點到**專案方案自己那面前牆**（主位面向的那面）的垂直距離——換牆搜尋時
    也沿用這個距離，起點是「專案原本的擺法換成正三角形」，不是專案喇叭到新那面牆的距離；
    原間距＝兩聲學中心的水平距離；新聆聽距離＝原間距 * sqrt(3) / 2。
    高度仍照專案設定，不以偏好改高度；箱體等硬限制交給搜尋同一條檢查路線判合法性。
    上述為原先的起點做法；以下調整同樣是主對話判斷，設計紙第二節第 6 條只說約 60° 當起點：
    60° 不在專案夾角範圍時選最近的端點；在固定夾角下，聆聽距離＝間距 / (2 tan(夾角 / 2))。
    將間距、水平聆聽距離、含高度差的三維耳距三條範圍取交集，原間距可行就保留，否則夾到
    可行區間內最近的間距。反推與擺位判定的捨入鏈不同，所以起點須經單位參數來回、擺位與
    硬限制檢查；搜尋範圍、夾角、耳距三條因捨入驗不過時，將間距與夾角端點的比例從一個最小
    浮點單位開始向內推，幅度逐次加倍，次數上限把推移限在捨入尺度。三條過了就交回；牆面間隙、
    禁區等幾何不合法不參與推的決定，由搜尋記成第 0 題不合法並寫原因。
    交集為空、原離前牆超出搜尋範圍、或推到上限三條仍不過時回 None。
    """
    facing = project_facing(project)
    axis = 0 if facing[0] != 0.0 else 1
    coordinate = project_midpoint(project)[axis]
    front = coordinate if facing[axis] < 0.0 else project.scene.room_m.length(axis) - coordinate
    left_id, right_id = speaker_pair_ids(project)
    left, right = project.speakers[left_id], project.speakers[right_id]
    angle = 60.0
    if settings.base_angle_deg is not None:
        angle = min(settings.base_angle_deg.high, max(settings.base_angle_deg.low, angle))
    if not settings.front_distance_m.low <= front <= settings.front_distance_m.high:
        return None
    return _validated_start(project, settings, front, math.hypot(left.x - right.x, left.y - right.y), angle)
