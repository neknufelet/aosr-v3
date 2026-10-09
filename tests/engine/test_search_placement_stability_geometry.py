"""十二點逐座標手算；家具、箱體與合法性順序都有實體反例。"""
from dataclasses import replace
import math
from typing import Literal

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point
from aosr.reporting.scheme import Scheme, absolute_furniture
from aosr.reporting.validation import furniture_problems, validate_scheme
from aosr.search import constraints, furniture_prefilter
from aosr.search.constraints import Reason
from aosr.search.layout_settings import Box, Span
from aosr.search.placement_stability_geometry import PLACEMENT_SHIFT_M, check_shift, generate_shifts
from tests.engine._placement_stability_cases import scheme, settings
from tests.engine._precision_contracts import contract_value

CONTACT_REL = contract_value("furniture_geometry_contact")
CAPABILITIES = load_capabilities(config_path("capabilities.toml"))
DIRECTIVITY = load_directivity_defaults(config_path("directivity_defaults.toml"))


@pytest.mark.parametrize("facing", ["west", "north"])
@pytest.mark.parametrize("locked", [False, True])
def test_twelve_coordinates_hand_calculated_in_two_facings_and_lock_modes(
    facing: Literal["west", "north"], locked: bool,
) -> None:
    base = scheme(facing)
    # 原專案仍用搜尋軸，移位要拿細算方案的驗證軸，不可從原專案抄回去。
    project = base.model_copy(update={"scene": base.scene.model_copy(update={"low_frequency_axis": "search_octave_24"})})
    config = settings(base, locked=locked)
    shifts = {s.name: s for s in generate_shifts(project, config, base)}
    # 順序是左喇叭、右喇叭、主位的世界座標移動方向；每格由手算旋轉得到。
    west = {
        "speakers_forward": ((-1, 0, 0), (-1, 0, 0), (0, 0, 0)),
        "speakers_backward": ((1, 0, 0), (1, 0, 0), (0, 0, 0)),
        "speakers_outward": ((0, -1, 0), (0, 1, 0), (0, 0, 0)),
        "speakers_inward": ((0, 1, 0), (0, -1, 0), (0, 0, 0)),
        "seat_forward": ((0, 0, 0), (0, 0, 0), (-1, 0, 0)),
        "seat_backward": ((0, 0, 0), (0, 0, 0), (1, 0, 0)),
        "seat_left": ((0, 0, 0), (0, 0, 0), (0, -1, 0)),
        "seat_right": ((0, 0, 0), (0, 0, 0), (0, 1, 0)),
        "ear_up": ((0, 0, 0), (0, 0, 0), (0, 0, 1)),
        "ear_down": ((0, 0, 0), (0, 0, 0), (0, 0, -1)),
        "acoustic_center_up": ((0, 0, 1), (0, 0, 1), (0, 0, 0)),
        "acoustic_center_down": ((0, 0, -1), (0, 0, -1), (0, 0, 0)),
    }
    north = west | {
        "speakers_forward": ((0, 1, 0), (0, 1, 0), (0, 0, 0)),
        "speakers_backward": ((0, -1, 0), (0, -1, 0), (0, 0, 0)),
        "speakers_outward": ((-1, 0, 0), (1, 0, 0), (0, 0, 0)),
        "speakers_inward": ((1, 0, 0), (-1, 0, 0), (0, 0, 0)),
        "seat_forward": ((0, 0, 0), (0, 0, 0), (0, 1, 0)),
        "seat_backward": ((0, 0, 0), (0, 0, 0), (0, -1, 0)),
        "seat_left": ((0, 0, 0), (0, 0, 0), (-1, 0, 0)),
        "seat_right": ((0, 0, 0), (0, 0, 0), (1, 0, 0)),
    }
    expected = west if facing == "west" else north
    assert shifts.keys() == expected.keys()
    original_furniture = {f.furniture_id: f.bottom_center_m for f in absolute_furniture(base) or ()}
    for name, (left_delta, right_delta, seat_delta) in expected.items():
        shift = shifts[name]
        moved = shift.scheme
        for key, delta in (("left", left_delta), ("right", right_delta)):
            want = tuple(value + step * PLACEMENT_SHIFT_M for value, step in zip(base.speakers[key].as_tuple(), delta, strict=True))
            assert moved.speakers[key].as_tuple() == pytest.approx(want, abs=8 * CONTACT_REL, rel=0)
        for before, after in zip(base.receiver_set.points, moved.receiver_set.points, strict=True):
            want = tuple(value + step * PLACEMENT_SHIFT_M for value, step in zip(before.position_m, seat_delta, strict=True))
            assert after.position_m == pytest.approx(want, abs=8 * CONTACT_REL, rel=0)
            assert after.model_dump(exclude={"position_m"}) == before.model_dump(exclude={"position_m"})
        furniture = {f.furniture_id: f.bottom_center_m for f in absolute_furniture(moved) or ()}
        table = original_furniture["table"]
        assert furniture["table"] == pytest.approx((table[0] + seat_delta[0] * PLACEMENT_SHIFT_M,
            table[1] + seat_delta[1] * PLACEMENT_SHIFT_M, 0.70), abs=8 * CONTACT_REL, rel=0)
        assert furniture["cloud"] == (3.0, 3.0, 2.5)
        assert moved.scene.low_frequency_axis == base.scene.low_frequency_axis
        assert shift.settings == config
        assert shift.settings is not config
    repeated = {s.name: s.scheme_id for s in generate_shifts(base, config, base)}
    assert repeated == {name: shift.scheme_id for name, shift in shifts.items()}
    assert base.speakers["left"].z == 1.2


@pytest.mark.parametrize("mount,center,top", [("stand", 0.205, 0.0), ("desk", 0.205, 0.73), ("floor", 0.8, 0.0)])
@pytest.mark.parametrize("name,sign", [("acoustic_center_up", 1), ("acoustic_center_down", -1)])
def test_acoustic_center_three_mounts_change_cabinet_and_settings_together(
    mount: str, center: float, top: float, name: str, sign: int,
) -> None:
    base = scheme(mount=mount, furniture=mount == "desk")
    config = settings(base)
    shift = next(s for s in generate_shifts(base, config, base) if s.name == name)
    moved = shift.scheme
    assert moved.speaker_setup is not None
    changed = center if mount == "stand" else center + sign * PLACEMENT_SHIFT_M
    assert moved.speaker_setup.cabinet.acoustic_center_above_bottom_m == changed
    assert shift.settings.layout.cabinet.acoustic_center_above_bottom_m == changed
    assert shift.settings.layout.cabinet.model_dump() == moved.speaker_setup.cabinet.model_dump()
    expected_z = 1.2 + sign * PLACEMENT_SHIFT_M if mount == "stand" else top + changed
    for speaker in moved.speakers.values():
        assert speaker.z == expected_z
    assert furniture_problems(moved) == ()
    assert check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY).outcome == "ready"
    assert config.layout.cabinet.acoustic_center_above_bottom_m == center


def test_desk_moves_under_stationary_speakers_and_cloud_stays_put() -> None:
    base = scheme(mount="desk")
    shift = next(s for s in generate_shifts(base, settings(base, locked=True), base) if s.name == "seat_left")
    moved = shift.scheme
    assert moved.speakers == base.speakers
    positions = {f.furniture_id: f.bottom_center_m for f in absolute_furniture(moved) or ()}
    assert positions["table"] == (2.0, 3.0 - PLACEMENT_SHIFT_M, 0.70)
    assert positions["cloud"] == (3.0, 3.0, 2.5)


@pytest.mark.parametrize("reason,amount", [
    (Reason.CABINET_OUTSIDE_ROOM, 0.38), (Reason.WALL_GAP, 0.03),
    (Reason.CABINETS_OVERLAP, 0.21), (Reason.CABINET_IN_KEEP_OUT, 0.155),
    (Reason.SEAT_IN_KEEP_OUT, 0.1), (Reason.SEAT_OUTSIDE_ROOM, 0.2),
    (Reason.OUTSIDE_SPEAKER_AREA, 0.1),
])
def test_physical_constraints_stop_before_construction_and_override_spec_markers(reason: Reason, amount: float) -> None:
    base = scheme(furniture=False)
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "ear_up")
    placement = replace(shift.placement, left=Point(1.0, 3.0, 1.2), right=Point(1.0, 4.0, 1.2))
    changes: dict[str, object] = {"listening_range_m": Span(low=0.1, high=0.2)}
    if reason == Reason.CABINET_OUTSIDE_ROOM:
        placement = replace(placement, left=Point(-0.1, 3.0, 1.2))
    elif reason == Reason.WALL_GAP:
        placement = replace(placement, left=Point(0.3, 3.0, 1.2))
        changes["wall_gap_m"] = 0.05
    elif reason == Reason.CABINETS_OVERLAP:
        placement = replace(placement, right=placement.left)
    elif reason == Reason.CABINET_IN_KEEP_OUT:
        changes["keep_out"] = (Box(x=Span(low=0.8, high=1.1), y=Span(low=2.95, high=3.05), z=Span(low=1.0, high=1.3)),)
    elif reason == Reason.SEAT_IN_KEEP_OUT:
        changes["keep_out"] = (Box(x=Span(low=2.9, high=3.1), y=Span(low=2.9, high=3.1), z=Span(low=1.0, high=1.4)),)
    elif reason == Reason.SEAT_OUTSIDE_ROOM:
        primary = Point(8.2, 3.0, 1.2)
        placement = replace(placement, primary=primary, receivers=tuple(
            (key, primary if key == "main" else point) for key, point in placement.receivers))
    else:
        changes["speaker_areas"] = (Box(x=Span(low=0.9, high=1.1), y=Span(low=2.9, high=4.1), z=Span(low=0.1, high=1.1)),)
    config = shift.settings.model_copy(update={"layout": shift.settings.layout.model_copy(update=changes)})
    # 左箱面向 +x：x=[.72,1]、y=[2.895,3.105]。出房 .38、間隙 .03、重疊 .21、禁區退出 .155。
    # 座位禁區退出 .1、x=8.2 出房 .2；聲學中心 z=1.2 比可用區上緣 1.1 高 .1。
    before = constraints.check(base, config.layout, placement)
    assert {v.reason for v in before} == {reason, Reason.LISTENING_DISTANCE_OUT_OF_RANGE}
    result = check_shift(base, replace(shift, settings=config, placement=placement), contact_rel=CONTACT_REL,
                         capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "unplaceable"
    assert {v.reason for v in result.violations} == {reason}
    assert tuple(v.amount_m for v in result.violations) == pytest.approx((amount,))
    assert result.problems == ()
    assert result.out_of_spec == ()


def test_spec_and_search_range_are_markers_and_locked_distance_is_not_searched() -> None:
    base = scheme(furniture=False)
    original = settings(base)
    layout = original.layout.model_copy(update={"base_angle_deg": Span(low=10, high=20),
        "listening_range_m": Span(low=0.2, high=0.5), "front_distance_m": Span(low=0.2, high=0.5),
        "spacing_m": Span(low=0.2, high=0.5), "listening_distance_m": Span(low=0.2, high=0.5)})
    shift = next(s for s in generate_shifts(base, original.model_copy(update={"layout": layout}), base) if s.name == "ear_up")
    result = check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "ready"
    assert {v.reason for v in result.out_of_spec} == {Reason.BASE_ANGLE_OUT_OF_RANGE, Reason.LISTENING_DISTANCE_OUT_OF_RANGE}
    assert set(result.outside_search) == {"front_distance", "spacing", "listening_distance"}
    locked = replace(shift, settings=original.model_copy(update={"layout": layout.model_copy(update={"seat_locked": True})}))
    result = check_shift(base, locked, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert set(result.outside_search) == {"front_distance", "spacing"}


def test_invalid_furniture_boxes_stop_before_candidate_that_requires_boxes() -> None:
    base = scheme()
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "seat_left")
    template = base.model_dump()
    template["furniture"] = [dict(furniture_id="huge", kind="desk", material="wood", width_m=20.0,
        depth_m=0.2, height_m=0.1, placement=dict(forward_m=0, left_m=0, bottom_height_m=0.4, yaw_deg=0))]
    changed = replace(shift, template=Scheme.model_validate(template))
    assert {v.reason for v in furniture_prefilter.check(changed.scheme, contact_rel=CONTACT_REL)} == {
        Reason.FURNITURE_PLACEMENT_INVALID}
    # 不是兩段都回違反：盒子建不出來，候選段會丟錯，必須在預篩停下。
    with pytest.raises(ValueError):
        furniture_prefilter.check_candidate(changed.scheme, changed.settings.layout, contact_rel=CONTACT_REL)
    result = check_shift(base, changed, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "unplaceable"
    assert {v.reason for v in result.violations} == {Reason.FURNITURE_PLACEMENT_INVALID}


def test_direct_block_is_separate_and_problem_paths_include_surrounding_seats() -> None:
    base = scheme(furniture=False)
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "seat_left")
    # 薄桌板穿過到 down 的直達；喇叭與耳朵都不在盒內。
    document = base.model_dump()
    document["furniture"] = [dict(furniture_id="block", kind="desk", material="wood", width_m=3.0,
        depth_m=0.1, height_m=0.04, placement=dict(forward_m=1.0, left_m=0, bottom_height_m=1.13, yaw_deg=0))]
    changed = replace(shift, template=Scheme.model_validate(document))
    result = check_shift(base, changed, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "placement_requirement_failed"
    assert {v.reason for v in result.violations} == {Reason.DIRECT_PATH_BLOCKED}
    assert {p.path for p in result.problems} == {"pairs.left.down", "pairs.right.down"}


def test_candidate_restriction_precedes_scheme_height_validation() -> None:
    base = scheme(mount="desk")
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "speakers_outward")
    placement = replace(shift.placement, left=Point(1.0, 0.5, 1.3))
    result = check_shift(base, replace(shift, placement=placement), contact_rel=CONTACT_REL,
                         capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "unplaceable"
    assert {v.reason for v in result.violations} == {Reason.CABINET_OFF_TABLE}
    assert result.problems == ()


def test_scheme_height_validation_is_last_and_keeps_problem_paths() -> None:
    base = scheme(mount="desk")
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "ear_up")
    placement = replace(shift.placement, left=Point(1.0, 2.0, 1.1))
    result = check_shift(base, replace(shift, placement=placement), contact_rel=CONTACT_REL,
                         capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "unplaceable"
    assert result.violations == ()
    assert {p.path for p in result.problems} == {"speakers.left.z"}


def test_scheme_construction_error_precedes_furniture_prefilter() -> None:
    base = scheme(mount="desk")
    document = base.model_dump()
    setup = base.speaker_setup
    assert setup is not None
    document["speaker_setup"] = setup.model_dump() | {"cabinet": setup.cabinet.model_dump() |
                                                     {"acoustic_center_above_bottom_m": 0.01}}
    document["speakers"] = {key: Point(point.x, point.y, 0.74) for key, point in base.speakers.items()}
    base = Scheme.model_validate(document)
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "acoustic_center_down")
    result = check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "unplaceable"
    assert result.violations == ()
    assert {p.path for p in result.problems} == {"speaker_setup.cabinet.acoustic_center_above_bottom_m"}


@pytest.mark.parametrize(("past_bound", "flagged"), ((0.5, False), (2.0, True)), ids=("within-contact", "beyond-contact"))
def test_search_range_edge_absorbs_contact_margin_only(past_bound: float, flagged: bool) -> None:
    # 房間最長邊 8 m，接觸界線＝8×登記簿相對值；喇叭 y=2、4，間距手算 2.0 m，耳高移位不改間距。
    # 範圍上限放在 2.0 減「past_bound 倍界線」：半倍是基準點重算的浮點尾差，不准標超出；兩倍才標。
    base = scheme(furniture=False)
    margin = 8.0 * CONTACT_REL
    original = settings(base)
    layout = original.layout.model_copy(update={"spacing_m": Span(low=0.2, high=2.0 - past_bound * margin)})
    shift = next(s for s in generate_shifts(base, original.model_copy(update={"layout": layout}), base)
                 if s.name == "ear_up")
    result = check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "ready"
    assert ("spacing" in result.outside_search) is flagged


@pytest.mark.parametrize(("name", "sign"), (("acoustic_center_up", 1), ("acoustic_center_down", -1)))
def test_floor_center_ignores_a_desk_in_the_room(name: str, sign: int) -> None:
    # 落地擺法：喇叭高度＝聲學中心離箱底（0.8 m）±2 公分；房裡有書桌（頂面 0.73 m）也不准拿桌面頂墊高。
    base = scheme(mount="floor")
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == name)
    expected = 0.8 + sign * PLACEMENT_SHIFT_M
    assert all(speaker.z == expected for speaker in shift.scheme.speakers.values())


def test_clean_sofa_moves_back_into_rear_wall() -> None:
    document = scheme(furniture=False).model_dump()
    # 面西，沙發中心 x=7.49、深 1，背面 x=7.99；往後移後出房 1 公分。
    document["furniture"] = [dict(furniture_id="sofa", kind="sofa", material="fabric", width_m=1.0,
        depth_m=1.0, height_m=0.65, placement=dict(forward_m=-4.49, left_m=0, bottom_height_m=0, yaw_deg=0))]
    base = Scheme.model_validate(document)
    assert furniture_prefilter.check(base, contact_rel=CONTACT_REL) == ()
    assert furniture_prefilter.check_candidate(base, settings(base).layout, contact_rel=CONTACT_REL) == ()
    assert validate_scheme(base, capabilities=CAPABILITIES, directivity=DIRECTIVITY) == ()
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "seat_backward")
    result = check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "unplaceable"
    assert {v.reason for v in result.violations} == {Reason.FURNITURE_PLACEMENT_INVALID}
    assert result.violations[0].amount_m == pytest.approx(PLACEMENT_SHIFT_M - 0.01)


def test_clean_fixed_panel_blocks_only_after_seat_moves_left() -> None:
    document = scheme(furniture=False).model_dump()
    # x=2 處：left→down 的基準 y=2.5，左移後 y=2.49；薄板 y=[2.486,2.494]，z=[1.14,1.16]。
    document["furniture"] = [dict(furniture_id="panel", kind="ceiling_cloud", material="wood", width_m=0.008,
        depth_m=0.008, height_m=0.02, placement=dict(bottom_center_m=(2.0, 2.49, 1.14), yaw_deg=0))]
    base = Scheme.model_validate(document)
    assert furniture_prefilter.check(base, contact_rel=CONTACT_REL) == ()
    assert furniture_prefilter.check_candidate(base, settings(base).layout, contact_rel=CONTACT_REL) == ()
    assert validate_scheme(base, capabilities=CAPABILITIES, directivity=DIRECTIVITY) == ()
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "seat_left")
    result = check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "placement_requirement_failed"
    assert {v.reason for v in result.violations} == {Reason.DIRECT_PATH_BLOCKED}
    assert {p.path for p in result.problems} == {"pairs.left.down"}


@pytest.mark.parametrize("quantity", ["listening_range_m", "listening_distance_m"])
@pytest.mark.parametrize("bound", ["low", "high"])
def test_primary_moves_across_model_and_search_listening_distance_limits(quantity: str, bound: str) -> None:
    base = scheme(furniture=False)
    original = settings(base)
    # 型號距離 sqrt(2²+1²)，搜尋量為軸向 2 m；界線放兩個移位距離之間。
    value = math.sqrt(5.0) if quantity == "listening_range_m" else 2.0
    span = Span(low=value - PLACEMENT_SHIFT_M / 2, high=3.0) if bound == "low" else Span(
        low=1.0, high=value + PLACEMENT_SHIFT_M / 2)
    config = original.model_copy(update={"layout": original.layout.model_copy(update={quantity: span})})
    for shift in generate_shifts(base, config, base):
        if shift.name not in ("seat_forward", "seat_backward"):
            continue
        result = check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
        assert result.outcome == "ready"
        flagged = shift.name == ("seat_forward" if bound == "low" else "seat_backward")
        if quantity == "listening_range_m":
            assert bool(result.out_of_spec) is flagged
            assert {v.reason for v in result.out_of_spec} == ({Reason.LISTENING_DISTANCE_OUT_OF_RANGE} if flagged else set())
        else:
            assert ("listening_distance" in result.outside_search) is flagged


@pytest.mark.parametrize("facing", ["west", "north"])
@pytest.mark.parametrize("quantity,value", [("front_distance", 1.0), ("spacing", 2.0), ("listening_distance", 2.0)])
@pytest.mark.parametrize("bound", ["low", "high"])
@pytest.mark.parametrize("offset,flagged", [(0.5, False), (2.0, True)])
def test_search_three_quantities_both_bounds_two_walls_and_off_axis_listener(
    facing: Literal["west", "north"], quantity: str, value: float, bound: str, offset: float, flagged: bool,
) -> None:
    base = scheme(facing, furniture=False)
    # 北牆 y=8，喇叭中點 y=5，離前牆 3 m。主位偏軸 1 m，軸向距離仍 2 m，不是 sqrt(5)。
    if quantity == "front_distance" and facing == "north":
        value = 3.0
    delta = (0.0, -1.0, 0.0) if facing == "west" else (-1.0, 0.0, 0.0)
    base = base.model_copy(update={"receiver_set": base.receiver_set.model_copy(update={"points": tuple(
        p.model_copy(update={"position_m": tuple(v + step for v, step in zip(p.position_m, delta, strict=True))})
        for p in base.receiver_set.points)})})
    margin = 8.0 * CONTACT_REL
    edge = value + offset * margin if bound == "low" else value - offset * margin
    span = Span(low=edge, high=value + 1.0) if bound == "low" else Span(low=0.2, high=edge)
    config = settings(base)
    config = config.model_copy(update={"layout": config.layout.model_copy(update={f"{quantity}_m": span})})
    shift = next(s for s in generate_shifts(base, config, base) if s.name == "ear_up")
    result = check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "ready"
    assert set(result.outside_search) == ({quantity} if flagged else set())


@pytest.mark.parametrize("front_wall", ["yL", "xL"])
def test_changed_search_wall_skips_range_and_uses_finalist_facing(front_wall: str) -> None:
    project, refined = scheme("north", furniture=False), scheme("west", furniture=False)
    # 兩軸都有喇叭間距：拿錯面向要以錯座標被抓到，不讓左右軸退化的丟錯遮住這一題。
    refined = refined.model_copy(update={"speakers": refined.speakers | {"right": Point(1.1, 4.0, 1.2)}})
    original = settings(project)
    layout = original.layout.model_copy(update={"front_wall": front_wall, "front_distance_m": Span(low=0.2, high=0.5),
        "spacing_m": Span(low=0.2, high=0.5), "listening_distance_m": Span(low=0.2, high=0.5)})
    shifts = generate_shifts(project, original.model_copy(update={"layout": layout}), refined)
    forward = next(s for s in shifts if s.name == "speakers_forward")
    assert forward.placement.left.as_tuple() == (1.0 - PLACEMENT_SHIFT_M, 2.0, 1.2)
    result = check_shift(project, forward, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "ready"
    assert result.outside_search == ()
    assert result.search_range_not_checked
    same = next(s for s in generate_shifts(refined, settings(refined), refined) if s.name == "ear_up")
    assert not check_shift(refined, same, contact_rel=CONTACT_REL,
                           capabilities=CAPABILITIES, directivity=DIRECTIVITY).search_range_not_checked


def test_speaker_outward_is_geometric_even_when_left_channel_is_on_listener_right() -> None:
    base = scheme(furniture=False)
    base = base.model_copy(update={"speakers": {"left": base.speakers["right"], "right": base.speakers["left"]}})
    shifts = {s.name: s.scheme for s in generate_shifts(base, settings(base), base)}
    assert shifts["speakers_outward"].speakers["left"].y == 4.0 + PLACEMENT_SHIFT_M
    assert shifts["speakers_outward"].speakers["right"].y == 2.0 - PLACEMENT_SHIFT_M
    assert shifts["speakers_inward"].speakers["left"].y == 4.0 - PLACEMENT_SHIFT_M
    assert shifts["speakers_inward"].speakers["right"].y == 2.0 + PLACEMENT_SHIFT_M


def test_speakers_coincident_on_listener_left_right_axis_are_rejected() -> None:
    base = scheme(furniture=False)
    base = base.model_copy(update={"speakers": {"left": Point(1.0, 3.0, 1.2), "right": Point(1.2, 3.0, 1.2)}})
    with pytest.raises(ValueError):
        generate_shifts(base, settings(base), base)


def test_shift_identifiers_are_unique_and_repeatable() -> None:
    base = scheme(furniture=False)
    config = settings(base)
    ids = [s.scheme_id for s in generate_shifts(base, config, base)]
    assert len(set(ids)) == len(ids)
    assert ids == [s.scheme_id for s in generate_shifts(base, config, base)]


def test_direct_block_and_occupied_stand_are_both_recorded_as_unplaceable() -> None:
    document = scheme(furniture=False).model_dump()
    document["furniture"] = [
        dict(furniture_id="block", kind="desk", material="wood", width_m=3.0, depth_m=0.1, height_m=0.04,
             placement=dict(forward_m=1.0, left_m=0, bottom_height_m=1.13, yaw_deg=0)),
        dict(furniture_id="under-stand", kind="ceiling_cloud", material="wood", width_m=0.2, depth_m=0.2, height_m=0.1,
             placement=dict(bottom_center_m=(0.9, 2.0, 0.5), yaw_deg=0)),
    ]
    base = Scheme.model_validate(document)
    shift = next(s for s in generate_shifts(base, settings(base), base) if s.name == "ear_up")
    moved = shift.scheme
    assert {v.reason for v in furniture_prefilter.check(moved, contact_rel=CONTACT_REL)} == {Reason.DIRECT_PATH_BLOCKED}
    assert {v.reason for v in furniture_prefilter.check_candidate(moved, shift.settings.layout,
                                                                contact_rel=CONTACT_REL)} == {Reason.STAND_SPACE_OCCUPIED}
    result = check_shift(base, shift, contact_rel=CONTACT_REL, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    assert result.outcome == "unplaceable"
    assert {v.reason for v in result.violations} == {Reason.DIRECT_PATH_BLOCKED, Reason.STAND_SPACE_OCCUPIED}
    assert {p.path for p in result.problems} == {"pairs.left.down", "pairs.right.down"}
