"""從單一方案列擺位標準的幾何實際值；不開檔、不求解、不篩選或排名。"""
from __future__ import annotations

import math
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from aosr.config.placement_standards import Measurement, PlacementStandard, PlacementStandards
from aosr.geometry.shoebox import Point, Room, Wall, distance
from aosr.reporting.scheme import Scheme
from aosr.search.layout import _speaker_ids


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
NOTICE: Final = "這張表只列標準有沒有守：沒守的不淘汰，守了也不代表聲學較好"
Unit = Literal["m", "deg", "1"]


class Verdict(StrEnum):
    """條文判定值；只列值與條件可接受都獨立於守／沒守。"""

    MET = "met"
    NOT_MET = "not_met"
    ACCEPTABLE_IN_SUITABLE_ROOMS = "acceptable_in_suitable_rooms"
    VALUE_ONLY = "value_only"

    @property
    def label(self) -> str:
        """給人看的中文判定。"""
        return {
            Verdict.MET: "守", Verdict.NOT_MET: "沒守",
            Verdict.ACCEPTABLE_IN_SUITABLE_ROOMS: "房間設計合適時可接受",
            Verdict.VALUE_ONLY: "標準沒給容許範圍，只列實際值",
        }[self]


class ActualValue(BaseModel):
    """一項有限的幾何值；最近面保留 Wall（房間牆面）既有名稱。"""

    model_config = FROZEN

    label: str
    value: float
    unit: Unit
    closest_surface: str | None = None
    judged_as: float | None = None
    """跟某個門檻只差浮點尾差時照那個門檻值判（#614）；None＝照原值判。value 一律是原值。"""
    judged_upper_factor: float | None = None
    """照按基寬縮放的上限判時，保留資料檔的係數供加註；其他門檻留空。"""


class ChecklistRow(BaseModel):
    """一條原文及其幾何結果；多支喇叭的值按左右角色列在同一條下。"""

    model_config = FROZEN

    id: str
    standard: str
    clause: str
    pdf_page: int
    quote: str
    description: str
    actual: tuple[ActualValue, ...] = Field(min_length=1)
    verdict: Verdict


class StandardsChecklist(BaseModel):
    """凍結的逐條結果與固定閱讀聲明；沒有分數或淘汰欄。"""

    model_config = FROZEN

    rows: tuple[ChecklistRow, ...] = Field(min_length=1)
    notice: Literal["這張表只列標準有沒有守：沒守的不淘汰，守了也不代表聲學較好"] = NOTICE


def _nearest(point: Point, room: Room, walls: tuple[Wall, ...], label: str) -> ActualValue:
    """垂直距離取最短；等距時採 Wall（牆面）的固定順序。"""
    distances = tuple((abs(point.as_tuple()[wall.axis()] - wall.plane(room)), wall)
                      for wall in walls)
    gap, wall = min(distances, key=lambda item: item[0])
    return ActualValue(label=label, value=gap, unit="m", closest_surface=wall.wall_name())


def _listener_walls(front_wall: str) -> tuple[Wall, ...]:
    """由前牆的軸與面別推側牆、後牆；不重寫牆名座標表。"""
    front = Wall.from_name(front_wall)
    vertical = tuple(wall for wall in Wall.all() if wall not in (Wall.FLOOR, Wall.CEILING))
    if front not in vertical:
        raise ValueError("前牆必須是四面垂直牆之一")
    return tuple(wall for wall in vertical
                 if wall.axis() != front.axis() or wall.kind() != front.kind())


def _horizontal_geometry(left: Point, right: Point, primary: Point) -> tuple[float, float]:
    """水平向量的 atan2（反正切）夾角與到喇叭無限延伸連線的垂距。"""
    lx, ly = left.x - primary.x, left.y - primary.y
    rx, ry = right.x - primary.x, right.y - primary.y
    bx, by = right.x - left.x, right.y - left.y
    horizontal_base = math.hypot(bx, by)
    if horizontal_base == 0.0 or math.hypot(lx, ly) == 0.0 or math.hypot(rx, ry) == 0.0:
        raise ValueError("水平喇叭連線或主位視線退化，無法列夾角與垂距")
    angle = math.degrees(math.atan2(abs(lx * ry - ly * rx), lx * rx + ly * ry))
    h = abs(bx * (primary.y - left.y) - by * (primary.x - left.x)) / horizontal_base
    return angle, h


def _measurements(scheme: Scheme, front_wall: str) -> dict[Measurement, tuple[ActualValue, ...]]:
    """只從聲學中心、主位耳點、座位清單與房間尺寸量幾何。"""
    listener_walls = _listener_walls(front_wall)
    left_id, right_id = _speaker_ids(scheme)
    left, right = scheme.speakers[left_id], scheme.speakers[right_id]
    primary = Point(*scheme.receiver_set.primary.position_m)
    room = scheme.scene.room_m
    speakers = (("左喇叭", left), ("右喇叭", right))
    walls = tuple(wall for wall in Wall.all() if wall not in (Wall.FLOOR, Wall.CEILING))
    base = distance(left, right)
    angle, h = _horizontal_geometry(left, right, primary)
    radius = max(distance(primary, Point(*receiver.position_m)) for receiver in scheme.receiver_set.points)
    angle_value = ActualValue(label="水平夾角", value=angle, unit="deg")
    radius_value = ActualValue(label="離主位最遠距離", value=radius, unit="m")
    return {
        "height_difference": tuple(ActualValue(label=label, value=point.z - primary.z, unit="m")
                                   for label, point in speakers),
        "speaker_surfaces": tuple(_nearest(point, room, Wall.all(), label) for label, point in speakers),
        "base_width": (ActualValue(label="聲學中心基寬", value=base, unit="m"),),
        "listening_distance": tuple(ActualValue(label=label, value=distance(point, primary), unit="m")
                                    for label, point in speakers),
        "listening_angle": (angle_value,),
        "area_radius": (radius_value,),
        "speaker_height": tuple(ActualValue(label=label, value=point.z, unit="m")
                                for label, point in speakers),
        "ear_height": (ActualValue(label="主位耳高", value=primary.z, unit="m"),),
        "inclination": tuple(ActualValue(label=label, unit="deg", value=math.degrees(math.atan2(
            abs(primary.z - point.z), math.hypot(primary.x - point.x, primary.y - point.y))))
            for label, point in speakers),
        "speaker_walls": tuple(_nearest(point, room, walls, label) for label, point in speakers),
        "listener_walls": (_nearest(primary, room, listener_walls, "主位"),),
        "angle_distance": (angle_value, ActualValue(label="水平垂距 h", value=h, unit="m"),
                           ActualValue(label="h／b", value=h / base, unit="1")),
    }


def _required(value: float | None) -> float:
    """資料模型已驗過所需門檻；若呼叫端繞過驗證仍明確拒絕。"""
    if value is None:
        raise ValueError("判法缺必要門檻")
    return value


def _on_limit(value: float, limits: tuple[float, ...], boundary_rel: float) -> float:
    """離某個門檻在相對 boundary_rel 之內就當剛好在那個門檻上：只吸收座標加減的浮點尾差（#614）。"""
    for limit in limits:
        if abs(value - limit) <= abs(limit) * boundary_rel:
            return limit
    return value


def _limits(entry: PlacementStandard, base: float) -> tuple[float, ...]:
    """這一條判定會拿來比的門檻值；只列值的沒有。"""
    limits = entry.thresholds
    values = (limits.target, limits.lower, limits.upper, limits.acceptable_upper,
              None if limits.upper_factor is None else limits.upper_factor * base)
    return () if entry.comparison == "value_only" else tuple(value for value in values if value is not None)


def _judge(entry: PlacementStandard, actual: tuple[ActualValue, ...], base: float,
           boundary_rel: float) -> Verdict:
    """只依資料的比較列舉與門檻；左右都符合才回守。比之前先把貼著門檻的值放回門檻上。"""
    kind, limits = entry.comparison, entry.thresholds
    if kind == "value_only":
        return Verdict.VALUE_ONLY
    if kind == "equal":
        target = _required(limits.target)
        met = all(_on_limit(item.value, (target,), boundary_rel) == target for item in actual)
    elif kind == "minimum":
        lower = _required(limits.lower)
        met = all(_on_limit(item.value, (lower,), boundary_rel) >= lower for item in actual)
    elif kind == "maximum":
        upper = _required(limits.upper)
        met = all(_on_limit(item.value, (upper,), boundary_rel) <= upper for item in actual)
    elif kind == "open_range":
        lower, upper = _required(limits.lower), _required(limits.upper)
        met = all(lower < _on_limit(item.value, (lower, upper), boundary_rel) < upper for item in actual)
    elif kind == "scaled_range":
        lower, upper = _required(limits.lower), _required(limits.upper_factor) * base
        met = all(lower <= _on_limit(item.value, (lower, upper), boundary_rel) <= upper for item in actual)
    else:
        lower, upper = _required(limits.lower), _required(limits.upper)
        values = tuple(_on_limit(item.value, (lower, upper), boundary_rel) for item in actual)
        met = all(lower <= value <= upper for value in values)
        if kind == "preferred_range" and not met:
            acceptable = _required(limits.acceptable_upper)
            if all(upper < _on_limit(value, (acceptable,), boundary_rel) <= acceptable for value in values):
                return Verdict.ACCEPTABLE_IN_SUITABLE_ROOMS
    return Verdict.MET if met else Verdict.NOT_MET


def check_placement_standards(scheme: Scheme, front_wall: str, standards: PlacementStandards, *,
                              boundary_rel: float) -> StandardsChecklist:
    """純函式：逐條列實際值與判定，不改方案或任何結果檔。

    boundary_rel 由呼叫端從精度契約登記簿 placement_standard_boundary 讀進來（#614）；
    列出的實際值照原值，不跟著放回門檻。
    """
    if not math.isfinite(boundary_rel) or boundary_rel < 0.0:
        raise ValueError("門檻邊界的相對範圍必須是有限的非負數")
    measurements = _measurements(scheme, front_wall)
    base = measurements["base_width"][0].value
    rows = []
    for entry in standards.entries:
        actual = measurements[entry.measurement]
        verdict = _judge(entry, actual, base, boundary_rel)
        limits = _limits(entry, base)
        actual = tuple(_annotate_actual(item, entry, base, limits, boundary_rel) for item in actual)
        description = entry.description
        if verdict == Verdict.NOT_MET and entry.failure_note is not None:
            description = f"{description} {entry.failure_note}"
        rows.append(ChecklistRow(
            id=entry.id, standard=entry.standard, clause=entry.clause, pdf_page=entry.pdf_page,
            quote=entry.quote, description=description, actual=actual, verdict=verdict,
        ))
    return StandardsChecklist(rows=tuple(rows))


def _annotate_actual(actual: ActualValue, entry: PlacementStandard, base: float,
                     limits: tuple[float, ...], boundary_rel: float) -> ActualValue:
    """只加註有浮點尾差的實際值；縮放上限保留係數，下限仍照數字列。"""
    on = _on_limit(actual.value, limits, boundary_rel)
    if on == actual.value:
        return actual
    factor = entry.thresholds.upper_factor
    scaled = factor if factor is not None and on == factor * base else None
    return actual.model_copy(update={"judged_as": on, "judged_upper_factor": scaled})


def _render_actual(actual: ActualValue) -> str:
    """實際值照原值印；被當成剛好在門檻上判的另註明照哪個門檻值判，免得原值跟判定看起來矛盾。"""
    units = {"m": "m（公尺）", "deg": "deg（度）", "1": "1（無單位）"}
    surface = "" if actual.closest_surface is None else f"；最近面 {actual.closest_surface}"
    limit = (str(actual.judged_as) if actual.judged_upper_factor is None
             else f"{actual.judged_upper_factor}×B（＝{actual.judged_as}）")
    judged = ("" if actual.judged_as is None
              else f"（跟門檻 {limit} 只差浮點尾差，照 {limit} 判）")
    return f"{actual.label}={actual.value} {units[actual.unit]}{judged}{surface}"


def render_checklist_text(checklist: StandardsChecklist) -> str:
    """中文純文字逐條呈現；原文另起一行，不推論聲學品質。"""
    lines: list[str] = [checklist.notice]
    for row in checklist.rows:
        actual = "；".join(_render_actual(item) for item in row.actual)
        lines.append(f"{row.id} | {row.standard} {row.clause}（PDF 頁 {row.pdf_page}）"
                     f" | {row.verdict.label} | {actual} | {row.description}")
        lines.append(f"原文：{row.quote}")
    return "\n".join(lines)
