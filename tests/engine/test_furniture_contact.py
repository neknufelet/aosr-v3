"""家具接觸界線、遮擋與 L50 退化語料；預期結果由盒區間手算。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine.test_furniture import box

if TYPE_CHECKING:
    from aosr.geometry.furniture import Vec3


@pytest.mark.parametrize(("start", "end", "blocked"), [
    ((1.0, 4.0, 1.5), (5.0, 4.0, 1.5), True),
    ((3.0, 1.0, 1.5), (3.0, 7.0, 1.5), True),
    ((3.0, 4.0, 0.0), (3.0, 4.0, 3.0), True),
    ((1.0, 4.0, 2.01), (5.0, 4.0, 2.01), False),
    ((1.0, 4.0, 2.0), (5.0, 4.0, 2.0), False),
    ((1.0, 2.0, 2.0), (5.0, 2.0, 2.0), False),
    ((1.0, 3.0, 1.5), (3.0, 1.0, 1.5), False),
    ((1.0, 3.0, 1.0), (3.0, 1.0, 3.0), False),
    ((1.0, 6.01, 1.5), (5.0, 6.01, 1.5), False),
    ((1.0, 4.0, 0.5), (5.0, 4.0, 0.5), False),
    ((1.0, 1.0, 1.5), (1.0, 7.0, 1.5), False),
    ((1.0, 4.0, 1.5), (2.0, 4.0, 1.5), False),
])
def test_segment_crossings_tangencies_and_parallel_outside(start: Vec3, end: Vec3, blocked: bool) -> None:
    from aosr.geometry.furniture import segment_blocked_by_box

    for first, last in ((start, end), (end, start)):
        assert segment_blocked_by_box(first, last, box(), margin_m=0.0) is blocked


@pytest.mark.parametrize(("axis", "edge", "inward"), [
    (0, 2.0, 1.0), (0, 4.0, -1.0), (1, 2.0, 1.0), (1, 6.0, -1.0), (2, 1.0, 1.0), (2, 2.0, -1.0),
])
def test_contact_margin_mutant_beyond_tolerance_is_red(axis: int, edge: float, inward: float) -> None:
    from aosr.geometry.furniture import contact_margin_m, segment_blocked_by_box

    margin = contact_margin_m((4.0, 8.0, 3.0), contact_rel=contract_value("furniture_geometry_contact"))
    # 每個面的內側：深 2T 有開區間可穿過，0.5T 仍屬接觸。取樣自正式登記值，沒有第二把尺。
    moving_axis = (axis + 1) % 3
    for factor, expected in ((0.5, False), (1.0 - MUTANT_MARGIN, False), (1.0, False),
                             (1.0 + MUTANT_MARGIN, True), (2.0, True)):
        start = [3.0, 4.0, 1.5]
        end = start.copy()
        start[axis] = end[axis] = edge + inward * factor * margin
        start[moving_axis], end[moving_axis] = -1.0, 8.0
        first = (start[0], start[1], start[2])
        last = (end[0], end[1], end[2])
        assert segment_blocked_by_box(first, last, box(), margin_m=margin) is expected


@pytest.mark.parametrize(("start", "end", "blocked"), [
    ((3.0, 4.0, 2.0), (3.0, 4.0, 3.0), False),
    ((2.0, 4.0, 1.5), (1.0, 4.0, 1.5), False),
    ((2.0, 4.0, 1.5), (4.0, 4.0, 1.5), True),
    ((2.0, 4.0, 1.5), (5.0, 4.0, 1.5), True),
])
def test_face_endpoint_contact_does_not_hide_interior_crossing(start: Vec3, end: Vec3, blocked: bool) -> None:
    from aosr.geometry.furniture import segment_blocked_by_box

    assert segment_blocked_by_box(start, end, box(), margin_m=0.0) is blocked
    assert segment_blocked_by_box(end, start, box(), margin_m=0.0) is blocked


@pytest.mark.parametrize("interior_first", [False, True])
def test_endpoint_in_solid_is_an_input_error(interior_first: bool) -> None:
    from aosr.geometry.furniture import segment_blocked_by_box

    inside, outside = (3.0, 4.0, 1.5), (1.0, 4.0, 1.5)
    start, end = (inside, outside) if interior_first else (outside, inside)
    with pytest.raises(ValueError, match="端點.*家具.*內部"):
        segment_blocked_by_box(start, end, box(), margin_m=0.0)


def test_endpoint_in_contact_band_is_allowed() -> None:
    from aosr.geometry.furniture import segment_blocked_by_box

    margin = contract_value("furniture_geometry_contact") * 8.0
    assert segment_blocked_by_box((2.0 + 0.5 * margin, 4.0, 1.5), (1.0, 4.0, 1.5), box(), margin_m=margin) is False
    with pytest.raises(ValueError, match="端點"):
        segment_blocked_by_box((2.0 + 2.0 * margin, 4.0, 1.5), (1.0, 4.0, 1.5), box(), margin_m=margin)


@pytest.mark.parametrize("point", [(1.0, 4.0, 1.5), (2.0, 4.0, 1.5), (2.0, 2.0, 1.0)])
def test_zero_length_outside_or_touching_is_clear(point: Vec3) -> None:
    from aosr.geometry.furniture import segment_blocked_by_box

    assert segment_blocked_by_box(point, point, box(), margin_m=0.0) is False


def test_zero_length_inside_is_an_input_error() -> None:
    from aosr.geometry.furniture import segment_blocked_by_box

    with pytest.raises(ValueError, match="端點"):
        segment_blocked_by_box((3.0, 4.0, 1.5), (3.0, 4.0, 1.5), box(), margin_m=0.0)


@pytest.mark.parametrize(("start", "end", "blocked"), [
    ((1.0, 3.0, 2.0), (5.0, 5.0, 2.0), False),  # 共面，整段貼頂面。
    ((2.0, 2.0, 1.0), (2.0, 2.0, 2.0), False),  # 同時共用兩個面的交線。
    ((1.0, 4.0, 1.5), (10.0, 4.0, 1.5), True),  # 中點 x=5.5 在外，仍穿過 x=2..4。
    ((1.0, 4.0, 2.0), (5.0, 4.0, 1.0), True),  # 斜穿內部，不能只看端點。
])
def test_l50_degenerate_segment_corpus(start: Vec3, end: Vec3, blocked: bool) -> None:
    from aosr.geometry.furniture import segment_blocked_by_box

    assert segment_blocked_by_box(start, end, box(), margin_m=0.0) is blocked


@pytest.mark.parametrize(("bottom", "yaw", "within"), [
    ((3.0, 4.0, 1.0), 0.0, True),
    ((1.0, 2.0, 0.0), 0.0, True),
    ((0.9, 2.0, 0.0), 0.0, False),
    ((5.1, 2.0, 0.0), 0.0, False),
    ((1.0, 1.9, 0.0), 0.0, False),
    ((1.0, 6.1, 0.0), 0.0, False),
    ((1.0, 2.0, 2.1), 0.0, False),
    ((1.0, 2.0, 0.0), 90.0, False),
])
def test_box_within_room_checks_all_extents(bottom: Vec3, yaw: float, within: bool) -> None:
    from aosr.geometry.furniture import box_within_room

    kind = "sofa" if bottom[2] == 0.0 else "desk"
    assert box_within_room(box(kind=kind, bottom=bottom, yaw=yaw), (6.0, 8.0, 3.0), margin_m=0.0) is within


def test_room_boundary_absorbs_only_contact_band() -> None:
    from aosr.geometry.furniture import box_within_room

    margin = contract_value("furniture_geometry_contact") * 8.0
    for factor, within in ((0.5, True), (1.0, True), (2.0, False)):
        left = box(kind="sofa", bottom=(1.0 - factor * margin, 2.0, 0.0), margin=margin)
        right = box(kind="sofa", bottom=(5.0 + factor * margin, 2.0, 0.0), margin=margin)
        assert box_within_room(left, (6.0, 8.0, 3.0), margin_m=margin) is within
        assert box_within_room(right, (6.0, 8.0, 3.0), margin_m=margin) is within


@pytest.mark.parametrize(("bottom", "overlap"), [
    ((3.0, 4.0, 1.0), True), ((4.0, 4.0, 1.0), True),
    ((5.0, 4.0, 1.0), False), ((5.0, 8.0, 2.0), False),
    ((3.0, 8.0, 1.0), False), ((3.0, 4.0, 2.0), False),
    ((3.0, 4.0, 3.0), False),
])
def test_box_overlap_requires_volume_not_shared_faces(bottom: Vec3, overlap: bool) -> None:
    from aosr.geometry.furniture import boxes_overlap

    other = box(bottom=bottom)
    assert boxes_overlap(box(), other, margin_m=0.0) is overlap
    assert boxes_overlap(other, box(), margin_m=0.0) is overlap


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_overlap_depth_must_exceed_contact_band(axis: int) -> None:
    from aosr.geometry.furniture import boxes_overlap

    margin = contract_value("furniture_geometry_contact") * 8.0
    touching = [5.0, 8.0, 2.0]
    for factor, overlap in ((0.5, False), (1.0, False), (2.0, True)):
        bottom = [3.0, 4.0, 1.0]
        bottom[axis] = touching[axis] - factor * margin
        other = box(bottom=(bottom[0], bottom[1], bottom[2]))
        assert boxes_overlap(box(), other, margin_m=margin) is overlap


@pytest.mark.parametrize(("point", "inside"), [
    ((3.0, 4.0, 1.5), True), ((2.0, 4.0, 1.5), False),
    ((4.0, 4.0, 1.5), False), ((3.0, 2.0, 1.5), False),
    ((3.0, 6.0, 1.5), False), ((3.0, 4.0, 1.0), False),
    ((3.0, 4.0, 2.0), False), ((9.0, 4.0, 1.5), False),
])
def test_point_inside_excludes_boundary(point: Vec3, inside: bool) -> None:
    from aosr.geometry.furniture import point_inside_box

    assert point_inside_box(point, box(), margin_m=0.0) is inside


def test_point_inside_uses_contact_band() -> None:
    from aosr.geometry.furniture import point_inside_box

    margin = contract_value("furniture_geometry_contact") * 8.0
    for factor, inside in ((0.5, False), (1.0, False), (2.0, True)):
        assert point_inside_box((2.0 + factor * margin, 4.0, 1.5), box(), margin_m=margin) is inside


def test_box_thinner_than_contact_band_has_no_deep_interior() -> None:
    from aosr.geometry.furniture import point_inside_box, segment_blocked_by_box

    margin = contract_value("furniture_geometry_contact") * 8.0
    thin = box(height=margin, bottom=(3.0, 4.0, 1.0))
    assert point_inside_box((3.0, 4.0, 1.0 + margin / 2.0), thin, margin_m=margin) is False
    assert segment_blocked_by_box((3.0, 4.0, 0.0), (3.0, 4.0, 2.0), thin, margin_m=margin) is False


def test_contact_margin_scales_by_longest_room_edge() -> None:
    from aosr.geometry.furniture import contact_margin_m

    relative = contract_value("furniture_geometry_contact")
    assert contact_margin_m((4.0, 8.0, 3.0), contact_rel=relative) == 8.0 * relative
    assert contact_margin_m((16.0, 4.0, 3.0), contact_rel=relative) == 16.0 * relative
    assert contact_margin_m((4.0, 8.0, 3.0), contact_rel=0.0) == 0.0


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_margin_is_rejected_everywhere(bad: float) -> None:
    from aosr.geometry.furniture import (
        box_within_room, boxes_overlap, contact_margin_m, mirror_point, point_inside_box,
        segment_blocked_by_box, single_bounce_point,
    )
    from tests.engine.test_furniture import face

    value, top, point = box(), face("top"), (3.0, 4.0, 3.0)
    operations = (
        lambda: box(margin=bad),
        lambda: contact_margin_m((4.0, 8.0, 3.0), contact_rel=bad),
        lambda: mirror_point(point, top, margin_m=bad),
        lambda: single_bounce_point(point, point, top, margin_m=bad),
        lambda: point_inside_box(point, value, margin_m=bad),
        lambda: segment_blocked_by_box(point, point, value, margin_m=bad),
        lambda: box_within_room(value, (4.0, 8.0, 3.0), margin_m=bad),
        lambda: boxes_overlap(value, value, margin_m=bad),
    )
    for operation in operations:
        with pytest.raises(ValueError, match="界線"):
            operation()


@pytest.mark.parametrize("room", [(0.0, 8.0, 3.0), (-1.0, 8.0, 3.0), (4.0, math.nan, 3.0), (4.0, 8.0, math.inf)])
def test_invalid_room_size_is_rejected(room: Vec3) -> None:
    from aosr.geometry.furniture import box_within_room, contact_margin_m

    with pytest.raises(ValueError, match="房間"):
        contact_margin_m(room, contact_rel=0.0)
    with pytest.raises(ValueError, match="房間"):
        box_within_room(box(), room, margin_m=0.0)


def test_registry_points_to_live_contact_mutant() -> None:
    from aosr.config.precision_contracts import load_precision_contracts

    registry = Path(__file__).resolve().parents[2] / "blueprint" / "precision_contracts.toml"
    entry = load_precision_contracts(registry)["furniture_geometry_contact"]
    assert entry.unit == "relative"
    assert entry.decision_paper == "stage-two-furniture-v1-boxes-single-bounce-approximate-ranking.md"
    assert entry.mutant_test == "tests/engine/test_furniture_contact.py::test_contact_margin_mutant_beyond_tolerance_is_red"
