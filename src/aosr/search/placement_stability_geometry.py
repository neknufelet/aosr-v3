"""具名座標移位與逐段判合法；沒有求解、取樣或檔案輸出。"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Final, Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.geometry.furniture import FurnitureKind, Vec3, contact_margin_m
from aosr.geometry.shoebox import Point, Wall
from aosr.reporting import validation
from aosr.reporting.scheme import ListenerPlacement, Scheme, project_facing, speaker_pair_ids
from aosr.reporting.validation import SchemeProblem
from aosr.search import constraints, furniture_prefilter
from aosr.search.constraints import Reason, Violation
from aosr.search.layout import Placement, to_scheme
from aosr.search.layout_settings import Cabinet, LayoutSettings
from aosr.search.settings import SearchSettings

PLACEMENT_SHIFT_M: Final = 0.02
SPEC_REASONS = frozenset((Reason.BASE_ANGLE_OUT_OF_RANGE, Reason.LISTENING_DISTANCE_OUT_OF_RANGE))


@dataclass(frozen=True)
class Shift:
    """scheme 延後建構，讓越界點先由硬限制給出真實違反量，再走 to_scheme。"""

    name: str
    template: Scheme
    settings: SearchSettings
    placement: Placement
    scheme_id: str

    @property
    def scheme(self) -> Scheme:
        return to_scheme(self.template, self.placement, self.scheme_id)


@dataclass(frozen=True)
class Legality:
    outcome: Literal["unplaceable", "placement_requirement_failed", "ready"]
    violations: tuple[Violation, ...] = ()
    problems: tuple[SchemeProblem, ...] = ()
    out_of_spec: tuple[Violation, ...] = ()
    outside_search: tuple[str, ...] = ()
    search_range_not_checked: bool = False


def _translate(point: Point, delta: Vec3) -> Point:
    return Point(point.x + delta[0], point.y + delta[1], point.z + delta[2])


def _scale(vector: Vec3, factor: float) -> Vec3:
    return vector[0] * factor, vector[1] * factor, vector[2] * factor


def _displace(base: Placement, speaker: Vec3, spread: Vec3, seat: Vec3) -> Placement:
    left = _translate(_translate(base.left, speaker), spread)
    right = _translate(_translate(base.right, speaker), _scale(spread, -1.0))
    return replace(base, left=left, right=right, primary=_translate(base.primary, seat),
                   receivers=tuple((key, _translate(point, seat)) for key, point in base.receivers))


def _center_shift(template: Scheme, settings: SearchSettings, placement: Placement,
                  delta: float) -> tuple[Scheme, SearchSettings, Placement]:
    setup = template.speaker_setup
    if setup is None or setup.mount == "stand":
        return template, settings, placement
    center = setup.cabinet.acoustic_center_above_bottom_m + delta
    cabinet = setup.cabinet.model_copy(update={"acoustic_center_above_bottom_m": center})
    template = template.model_copy(update={"speaker_setup": setup.model_copy(update={"cabinet": cabinet})})
    # 不在這裡驗：聲學中心下移可能變負，要留給建方案那一段判成擺不出來（方案的箱體同樣 model_copy 不驗）。
    layout = settings.layout.model_copy(update={"cabinet": Cabinet.model_construct(**cabinet.model_dump())})
    settings = settings.model_copy(update={"layout": layout})
    top = 0.0
    if setup.mount == "desk":
        table = next(item for item in template.furniture or ()
                     if item.kind in (FurnitureKind.DESK, FurnitureKind.COFFEE_TABLE))
        if not isinstance(table.placement, ListenerPlacement):
            raise ValueError("承托桌必須跟著主位")
        top = table.placement.bottom_height_m + table.height_m
    height = top + center
    return template, settings, replace(placement, left=Point(placement.left.x, placement.left.y, height),
                                        right=Point(placement.right.x, placement.right.y, height))


def generate_shifts(project: Scheme, settings: SearchSettings, refined_scheme: Scheme) -> tuple[Shift, ...]:
    """用細算方案座標、場景與主位跟隨關係；搜尋範圍不改。"""
    if {p.receiver_id for p in project.receiver_set.points} != {p.receiver_id for p in refined_scheme.receiver_set.points}:
        raise ValueError("細算方案必須保留專案座位代號")
    facing = project_facing(refined_scheme)
    left_id, right_id = speaker_pair_ids(refined_scheme)
    base = Placement(refined_scheme.speakers[left_id], refined_scheme.speakers[right_id],
        Point(*refined_scheme.receiver_set.primary.position_m),
        tuple((p.receiver_id, Point(*p.position_m)) for p in refined_scheme.receiver_set.points), facing)
    d = PLACEMENT_SHIFT_M
    front, left, up = (facing[0] * d, facing[1] * d, 0.0), (-facing[1] * d, facing[0] * d, 0.0), (0.0, 0.0, d)
    side = (base.left.x - base.right.x) * left[0] + (base.left.y - base.right.y) * left[1]
    if side == 0.0:
        raise ValueError("兩喇叭在聽者左右軸上重合，定不出往外方向")
    outward = left if side > 0.0 else _scale(left, -1.0)
    zero = (0.0, 0.0, 0.0)
    axes = (("speakers_forward", front, zero, zero), ("speakers_outward", zero, outward, zero),
            ("seat_forward", zero, zero, front), ("seat_left", zero, zero, left),
            ("ear_up", zero, zero, up), ("acoustic_center_up", up, zero, zero))
    opposites = ("speakers_backward", "speakers_inward", "seat_backward", "seat_right", "ear_down", "acoustic_center_down")
    shifts = []
    for (name, speaker, spread, seat), opposite in zip(axes, opposites, strict=True):
        for key, sign in ((name, 1.0), (opposite, -1.0)):
            vectors = tuple(_scale(vector, sign) for vector in (speaker, spread, seat))
            placement = _displace(base, vectors[0], vectors[1], vectors[2])
            template, copied = refined_scheme, settings.model_copy(deep=True)
            if name == "acoustic_center_up":
                template, copied, placement = _center_shift(template, copied, placement, sign * d)
            identifier = str(uuid5(NAMESPACE_URL, f"aosr:placement-stability:{refined_scheme.scheme_id}:{key}"))
            shifts.append(Shift(key, template, copied, placement, identifier))
    return tuple(shifts)


def _search_range_matches(layout: LayoutSettings, placement: Placement) -> bool:
    """沿搜尋 place 同一面牆的軸與 0／L 面判方向；換牆前的原方案不套這把範圍尺。"""
    wall = Wall.from_name(layout.front_wall)
    return placement.facing[wall.axis()] == (-1.0 if wall.kind() == "zero" else 1.0)


def _outside_search(project: Scheme, layout: LayoutSettings, placement: Placement, *, contact_rel: float) -> tuple[str, ...]:
    """範圍邊界留家具紙第 14 條那把接觸界線：基準點剛好在邊界上時，重算的浮點尾差不標超出。"""
    if not _search_range_matches(layout, placement):
        return ()
    wall = Wall.from_name(layout.front_wall)
    axis = wall.axis()
    midpoint = tuple((a + b) / 2.0 for a, b in zip(placement.left.as_tuple(), placement.right.as_tuple(), strict=True))
    front = abs(midpoint[axis] - wall.plane(project.scene.room_m))
    spacing = math.hypot(placement.left.x - placement.right.x, placement.left.y - placement.right.y)
    listening = abs((placement.primary.x - midpoint[0]) * placement.facing[0]
                    + (placement.primary.y - midpoint[1]) * placement.facing[1])
    quantities = [("front_distance", front, layout.front_distance_m), ("spacing", spacing, layout.spacing_m)]
    if not layout.seat_locked and layout.listening_distance_m is not None:
        quantities.append(("listening_distance", listening, layout.listening_distance_m))
    room = project.scene.room_m
    margin = contact_margin_m((room.Lx, room.Ly, room.Lz), contact_rel=contact_rel)
    return tuple(name for name, value, span in quantities if not span.low - margin <= value <= span.high + margin)


def check_shift(project: Scheme, shift: Shift, *, contact_rel: float,
                capabilities: CapabilityTable, directivity: DirectivityDefaults) -> Legality:
    """物理項與家具擺放錯先停；直達被擋仍核候選限制，最後走現成方案驗證。"""
    violations = constraints.check(project, shift.settings.layout, shift.placement)
    physical = tuple(v for v in violations if v.reason not in SPEC_REASONS)
    if physical:
        return Legality("unplaceable", physical)
    marked = Legality("ready", out_of_spec=violations,
                      search_range_not_checked=not _search_range_matches(shift.settings.layout, shift.placement),
                      outside_search=_outside_search(project, shift.settings.layout, shift.placement,
                                                     contact_rel=contact_rel))
    try:
        moved = shift.scheme
    except ValidationError as error:
        problems = tuple(SchemeProblem(".".join(map(str, e["loc"])) or "scheme", str(e["msg"])) for e in error.errors())
        return replace(marked, outcome="unplaceable", problems=problems)
    prefilter = furniture_prefilter.check(moved, contact_rel=contact_rel)
    if any(v.reason == Reason.FURNITURE_PLACEMENT_INVALID for v in prefilter):
        return replace(marked, outcome="unplaceable", violations=prefilter)
    candidate = furniture_prefilter.check_candidate(moved, shift.settings.layout, contact_rel=contact_rel)
    if prefilter or candidate:
        blocked = any(v.reason == Reason.DIRECT_PATH_BLOCKED for v in prefilter)
        problems = tuple(p for p in validation.furniture_problems(moved) if p.path.startswith("pairs.")) if blocked else ()
        return replace(marked, outcome="unplaceable" if candidate else "placement_requirement_failed",
                       violations=(*prefilter, *candidate), problems=problems)
    problems = validation.validate_scheme(moved, capabilities=capabilities, directivity=directivity)
    if problems:
        return replace(marked, outcome="unplaceable", problems=problems)
    return marked
