"""求解前的工程硬限制；只回原因代碼與以公尺量的實際穿透／缺口。"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from aosr.geometry.shoebox import Point, Room, distance
from aosr.reporting.scheme import Scheme
from aosr.search.layout import Placement
from aosr.search.layout_settings import Box, Cabinet, LayoutSettings
from aosr.search.sampler import Illegal


class Reason(StrEnum):
    """設計紙第二節第 1、5 條的求解前不合法原因。"""

    CABINET_OUTSIDE_ROOM = "cabinet_outside_room"
    WALL_GAP = "wall_gap"
    CABINETS_OVERLAP = "cabinets_overlap"
    CABINET_IN_KEEP_OUT = "cabinet_in_keep_out"
    SEAT_IN_KEEP_OUT = "seat_in_keep_out"
    SEAT_OUTSIDE_ROOM = "seat_outside_room"
    OUTSIDE_SPEAKER_AREA = "outside_speaker_area"
    LISTENING_DISTANCE_OUT_OF_RANGE = "listening_distance_out_of_range"


@dataclass(frozen=True)
class Violation:
    """某一種硬限制的最大違反量；不准用零、非數字或假分數冒充。"""

    reason: Reason
    amount_m: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.amount_m) or self.amount_m <= 0.0:
            raise ValueError("violation amount must be finite and positive")


@dataclass(frozen=True)
class _Prism:
    """水平凸多邊形加高度閉區間；箱體只轉水平角、不俯仰。"""

    polygon: tuple[tuple[float, float], ...]
    low_z: float
    high_z: float

    def corners(self) -> tuple[Point, ...]:
        return tuple(Point(x, y, z) for x, y in self.polygon for z in (self.low_z, self.high_z))


def _cabinet(center: Point, primary: Point, cabinet: Cabinet) -> _Prism:
    dx, dy = primary.x - center.x, primary.y - center.y
    norm = math.hypot(dx, dy)
    if norm == 0.0:
        raise ValueError("speaker and primary must define a horizontal facing")
    fx, fy = dx / norm, dy / norm
    sx, sy = -fy, fx
    half_width = cabinet.width_m / 2.0
    front = cabinet.acoustic_center_behind_front_m
    back = front - cabinet.depth_m
    polygon = tuple((center.x + f * fx + w * sx, center.y + f * fy + w * sy)
                    for f, w in ((back, -half_width), (front, -half_width),
                                 (front, half_width), (back, half_width)))
    above_bottom = cabinet.acoustic_center_above_bottom_m
    if above_bottom is None:
        above_bottom = cabinet.height_m / 2.0
    low_z = center.z - above_bottom
    return _Prism(polygon, low_z, low_z + cabinet.height_m)


def _box_prism(box: Box) -> _Prism:
    return _Prism(((box.x.low, box.y.low), (box.x.high, box.y.low),
                   (box.x.high, box.y.high), (box.x.low, box.y.high)), box.z.low, box.z.high)


def _axes(polygon: tuple[tuple[float, float], ...]) -> Iterable[tuple[float, float]]:
    """每条邊的單位法向量，投影後的重疊量才是公尺。"""
    for start, end in zip(polygon, (*polygon[1:], polygon[0]), strict=True):
        dx, dy = end[0] - start[0], end[1] - start[1]
        norm = math.hypot(dx, dy)
        yield -dy / norm, dx / norm


def _penetration(first: _Prism, second: _Prism) -> float:
    """高度有開區間重疊時，取全部分離軸的最小水平退出距離。

    投影 [a,b] 與 [c,d] 的退出距離是 min(b-c,d-a)，不是交集長度；
    這也涵蓋一個多邊形完全包住另一個的情形。接觸邊界算合法。
    高度只做重疊門檻；箱體移出的量按施工單在水平分離軸上量。
    """
    if min(first.high_z, second.high_z) <= max(first.low_z, second.low_z):
        return 0.0
    amounts = []
    for ax, ay in (*_axes(first.polygon), *_axes(second.polygon)):
        a = tuple(x * ax + y * ay for x, y in first.polygon)
        b = tuple(x * ax + y * ay for x, y in second.polygon)
        overlap = min(max(a) - min(b), max(b) - min(a))
        if overlap <= 0.0:
            return 0.0
        amounts.append(overlap)
    return min(amounts)


def _outside(point: Point, room: Room, gap: float = 0.0, *, axes: tuple[int, ...] = (0, 1, 2)) -> float:
    """指定軸上最大的缺口；gap=0 就是角／點越界量。"""
    values = point.as_tuple()
    return max(0.0, *(max(gap - values[axis], values[axis] - (room.length(axis) - gap)) for axis in axes))


# 喇叭必要間隙只對四面牆（x、y 兩軸）：地板與天花板不算——落地喇叭箱底本來就貼地板，箱高與喇叭高
# 都是專案固定輸入，把地板算進去會讓每個候選都不合法（主對話判斷，2026-10-02 審查抓到）。
WALL_AXES = (0, 1)


def _seat_penetration(point: Point, box: Box) -> float:
    """座位在禁區裡（三軸都在裡面）時，水平方向退出去的最短距離；耳高不搜，不算往上下退。"""
    if not box.z.low < point.z < box.z.high:
        return 0.0
    distances = tuple(min(value - span.low, span.high - value)
                      for value, span in ((point.x, box.x), (point.y, box.y)))
    return max(0.0, min(distances))


def _distance_to_box(point: Point, box: Box) -> float:
    components = tuple(max(span.low - value, value - span.high, 0.0)
                       for value, span in zip(point.as_tuple(), (box.x, box.y, box.z), strict=True))
    return math.sqrt(sum(value * value for value in components))


def _record(amounts: dict[Reason, float], reason: Reason, amount: float) -> None:
    if amount > 0.0:
        amounts[reason] = max(amounts.get(reason, 0.0), amount)


def _check_cabinet(
    room: Room, settings: LayoutSettings, cabinet: _Prism, amounts: dict[Reason, float],
) -> None:
    corners = cabinet.corners()
    _record(amounts, Reason.CABINET_OUTSIDE_ROOM, max(_outside(point, room) for point in corners))
    # 零表示專案未宣告額外間隙；出牆仍由上面的箱體越界記錄，不重複歸到零間隙。
    if settings.wall_gap_m > 0.0:
        _record(amounts, Reason.WALL_GAP,
                max(_outside(point, room, settings.wall_gap_m, axes=WALL_AXES) for point in corners))
    for box in settings.keep_out:
        _record(amounts, Reason.CABINET_IN_KEEP_OUT, _penetration(cabinet, _box_prism(box)))


def check(project: Scheme, settings: LayoutSettings, placement: Placement) -> tuple[Violation, ...]:
    """空 tuple＝合法；每種原因只回最大違反量，不建方案、不求解、不評分。"""
    amounts: dict[Reason, float] = {}
    room = project.scene.room_m
    speakers = (placement.left, placement.right)
    cabinets = tuple(_cabinet(center, placement.primary, settings.cabinet) for center in speakers)
    for cabinet in cabinets:
        _check_cabinet(room, settings, cabinet, amounts)
    _record(amounts, Reason.CABINETS_OVERLAP, _penetration(cabinets[0], cabinets[1]))
    for _, seat in placement.receivers:
        _record(amounts, Reason.SEAT_OUTSIDE_ROOM, _outside(seat, room))
        for box in settings.keep_out:
            _record(amounts, Reason.SEAT_IN_KEEP_OUT, _seat_penetration(seat, box))
    for center in speakers:
        if settings.speaker_areas is not None:
            _record(amounts, Reason.OUTSIDE_SPEAKER_AREA,
                    min(_distance_to_box(center, box) for box in settings.speaker_areas))
        limits = settings.listening_range_m
        if limits is not None:
            value = distance(center, placement.primary)
            _record(amounts, Reason.LISTENING_DISTANCE_OUT_OF_RANGE,
                    max(limits.low - value, value - limits.high, 0.0))
    return tuple(Violation(reason, amounts[reason]) for reason in sorted(amounts))


def to_illegal(violations: Iterable[Violation]) -> Illegal:
    """各原因按字母排並以 + 連接，最大違反量逐項相加，交回取樣器。"""
    items = tuple(violations)
    if not items:
        raise ValueError("legal placements cannot be reported as illegal")
    return Illegal("+".join(sorted(item.reason.value for item in items)), sum(item.amount_m for item in items))
