"""固定單位搜尋空間換成對稱擺位；只在限制檢查通過後才建方案。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from aosr.geometry.shoebox import Point, Wall
from aosr.reporting.scheme import Scheme
from aosr.scoring.receiver_set import ReceiverSet
from aosr.search.layout_settings import LayoutSettings, Span


SEARCH_QUANTITIES = ("front_distance", "spacing", "listening_distance")
UNIT_SPACE: Mapping[str, tuple[float, float]] = MappingProxyType(dict.fromkeys(SEARCH_QUANTITIES, (0.0, 1.0)))
CARDINAL_FACINGS: Final[tuple[tuple[float, float], ...]] = (
    (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0), (0.0, 1.0),
)


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


def _speaker_ids(project: Scheme) -> tuple[str, str]:
    roles = {channel.role: channel.speaker_id for channel in project.channel_group.channels}
    if "left" not in roles or "right" not in roles:
        raise ValueError("project requires left and right channel roles")
    return roles["left"], roles["right"]


def _project_midpoint(project: Scheme) -> tuple[float, float]:
    left_id, right_id = _speaker_ids(project)
    left, right = project.speakers[left_id], project.speakers[right_id]
    return (left.x + right.x) / 2.0, (left.y + right.y) / 2.0


def _project_facing(project: Scheme) -> tuple[float, float]:
    """專案方案的聆聽者面向：兩喇叭中點相對主位、水平分量絕對值較大的那一軸（主對話判斷）。

    候選擺法一律由搜尋設定重建、本來就對稱；專案方案只拿來定周圍座位的相對佈局要轉幾個直角
    與 60° 起點的原距離，所以不要求專案座位剛好在中軸上（客戶現況可能不對稱）。兩軸一樣大
    （含兩者都是零）就定不出前牆，拒收。不設容差門檻：比的是大小，不是相等。
    """
    mx, my = _project_midpoint(project)
    px, py, _ = project.receiver_set.primary.position_m
    dx, dy = mx - px, my - py
    if abs(dx) > abs(dy):
        return (1.0 if dx > 0.0 else -1.0), 0.0
    if abs(dy) > abs(dx):
        return 0.0, (1.0 if dy > 0.0 else -1.0)
    raise ValueError("專案方案的兩喇叭中點相對主位沒有主要方向（x、y 一樣大），定不出前牆")


def _rotate_offset(offset: tuple[float, float, float], turns: int) -> tuple[float, float, float]:
    """只交換、變號，不引入三角函數的近似；零轉時原位移逐位保留。"""
    x, y, z = offset
    return ((x, y, z), (-y, x, z), (-x, -y, z), (y, -x, z))[turns]


def _receivers(project: Scheme, primary: Point, facing: tuple[float, float]) -> tuple[tuple[str, Point], ...]:
    original_facing = _project_facing(project)
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
    left_id, right_id = _speaker_ids(project)
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


def standard_start(project: Scheme, settings: LayoutSettings) -> LayoutParams | None:
    """第二節第 6 條主對話做法：60° 正三角形僅為起點偏好，範圍外回 None。

    原離前牆＝兩聲學中心中點到**專案方案自己那面前牆**（主位面向的那面）的垂直距離——換牆搜尋時
    也沿用這個距離，起點是「專案原本的擺法換成正三角形」，不是專案喇叭到新那面牆的距離；
    原間距＝兩聲學中心的水平距離；新聆聽距離＝原間距 * sqrt(3) / 2。
    高度仍照專案設定，不以偏好改高度，也不在這裡判箱體合法性。
    """
    facing = _project_facing(project)
    axis = 0 if facing[0] != 0.0 else 1
    coordinate = _project_midpoint(project)[axis]
    front = coordinate if facing[axis] < 0.0 else project.scene.room_m.length(axis) - coordinate
    left_id, right_id = _speaker_ids(project)
    left, right = project.speakers[left_id], project.speakers[right_id]
    spacing = math.hypot(left.x - right.x, left.y - right.y)
    listening = spacing * math.sqrt(3.0) / 2.0
    values = (front, spacing, listening)
    if not all(span.low <= value <= span.high for span, value in zip(_spans(settings), values, strict=True)):
        return None
    return LayoutParams(*values)
