"""家具盒子與有限面反射的手算考卷；不借上一代程式當答案。"""
from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING, cast

import pytest

from tests.engine._precision_contracts import contract_value

if TYPE_CHECKING:
    from aosr.geometry.furniture import FurnitureBox, FurnitureFace, Vec3


def box(*, kind: str = "desk", width: float = 2.0, depth: float = 4.0,
        height: float = 1.0, bottom: Vec3 = (3.0, 4.0, 1.0), yaw: float = 0.0,
        margin: float = 0.0) -> FurnitureBox:
    from aosr.geometry.furniture import FurnitureBox

    return FurnitureBox(kind=kind, width_m=width, depth_m=depth, height_m=height,
                        bottom_center_m=bottom, yaw_deg=yaw, margin_m=margin)


def face(direction: str) -> FurnitureFace:
    return next(item for item in box().faces if item.direction == direction)


def test_bounds_use_bottom_center_and_height() -> None:
    value = box()
    assert value.minimum_m == (2.0, 2.0, 1.0)
    assert value.maximum_m == (4.0, 6.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        setattr(value, "width_m", 9.0)


@pytest.mark.parametrize(("direction", "center", "normal", "edges", "lengths"), [
    ("top", (3.0, 4.0, 2.0), (0.0, 0.0, 1.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (2.0, 4.0)),
    ("bottom", (3.0, 4.0, 1.0), (0.0, 0.0, -1.0), ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), (2.0, 4.0)),
    ("+x", (4.0, 4.0, 1.5), (1.0, 0.0, 0.0), ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (4.0, 1.0)),
    ("-x", (2.0, 4.0, 1.5), (-1.0, 0.0, 0.0), ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (4.0, 1.0)),
    ("+y", (3.0, 6.0, 1.5), (0.0, 1.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (2.0, 1.0)),
    ("-y", (3.0, 2.0, 1.5), (0.0, -1.0, 0.0), ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (2.0, 1.0)),
])
def test_face_frames_are_in_room_coordinates(direction: str, center: Vec3, normal: Vec3,
                                            edges: tuple[Vec3, Vec3], lengths: tuple[float, float]) -> None:
    item = face(direction)
    assert item.center_m == center
    assert item.normal == normal
    assert item.edge_units == edges
    assert item.edge_lengths_m == lengths


@pytest.mark.parametrize("kind", ["sofa", "chair", "coffee_table", "desk", "ceiling_cloud"])
def test_exposed_faces_follow_floor_or_suspended_kind(kind: str) -> None:
    grounded = kind in {"sofa", "chair"}
    value = box(kind=kind, bottom=(3.0, 4.0, 0.0 if grounded else 1.0))
    names = {item.direction for item in value.exposed_faces}
    expected = {"top", "+x", "-x", "+y", "-y"}
    if not grounded:
        expected.add("bottom")
    assert names == expected
    assert {item.direction for item in value.faces} == expected | {"bottom"}


@pytest.mark.parametrize(("yaw", "minimum", "maximum", "lengths"), [
    (0.0, (2.0, 2.0, 1.0), (4.0, 6.0, 2.0), (2.0, 4.0)),
    (90.0, (1.0, 3.0, 1.0), (5.0, 5.0, 2.0), (4.0, 2.0)),
    (180.0, (2.0, 2.0, 1.0), (4.0, 6.0, 2.0), (2.0, 4.0)),
    (270.0, (1.0, 3.0, 1.0), (5.0, 5.0, 2.0), (4.0, 2.0)),
])
def test_cardinal_rotation_swaps_width_depth(yaw: float, minimum: Vec3, maximum: Vec3,
                                           lengths: tuple[float, float]) -> None:
    value = box(yaw=yaw)
    assert value.minimum_m == minimum
    assert value.maximum_m == maximum
    assert next(item for item in value.faces if item.direction == "top").edge_lengths_m == lengths
    assert next(item for item in value.faces if item.direction == "+x").normal == (1.0, 0.0, 0.0)


@pytest.mark.parametrize("bad", [45.0, 360.0, -90.0, math.nan, math.inf, "90", True, None])
def test_rotation_rejects_non_cardinal_and_non_numeric(bad: object) -> None:
    with pytest.raises(ValueError, match="第一版只收正放的盒子"):
        box(yaw=cast(float, bad))


@pytest.mark.parametrize("dimension", ["width", "depth", "height"])
@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, "2", True])
def test_dimensions_must_be_finite_positive(dimension: str, bad: object) -> None:
    dimensions = dict(width=2.0, depth=4.0, height=1.0)
    dimensions[dimension] = cast(float, bad)
    with pytest.raises(ValueError, match="尺寸"):
        box(width=dimensions["width"], depth=dimensions["depth"], height=dimensions["height"])


@pytest.mark.parametrize("bad", [(math.nan, 4.0, 1.0), (3.0, math.inf, 1.0), (3.0, 4.0, math.nan)])
def test_bottom_center_must_be_finite(bad: Vec3) -> None:
    with pytest.raises(ValueError, match="底面中心"):
        box(bottom=bad)


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="家具種類"):
        box(kind="bookshelf")


@pytest.mark.parametrize("kind", ["sofa", "chair"])
def test_floor_kind_accepts_only_contact_band(kind: str) -> None:
    margin = contract_value("furniture_geometry_contact") * 8.0
    for z in (-margin, 0.0, margin):
        assert box(kind=kind, bottom=(3.0, 4.0, z), margin=margin).bottom_center_m == (3.0, 4.0, z)
    for z in (-2.0 * margin, 2.0 * margin):
        with pytest.raises(ValueError, match="貼地"):
            box(kind=kind, bottom=(3.0, 4.0, z), margin=margin)


@pytest.mark.parametrize("kind", ["coffee_table", "desk", "ceiling_cloud"])
def test_suspended_kind_requires_bottom_above_contact_band(kind: str) -> None:
    margin = contract_value("furniture_geometry_contact") * 8.0
    for z in (-margin, 0.0, 0.5 * margin, margin):
        with pytest.raises(ValueError, match="懸空"):
            box(kind=kind, bottom=(3.0, 4.0, z), margin=margin)
    assert box(kind=kind, bottom=(3.0, 4.0, 2.0 * margin), margin=margin).bottom_center_m[2] > margin


@pytest.mark.parametrize(("direction", "point", "expected"), [
    ("top", (3.0, 4.0, 3.0), (3.0, 4.0, 1.0)),
    ("bottom", (3.0, 4.0, 0.0), (3.0, 4.0, 2.0)),
    ("+x", (6.0, 4.0, 1.5), (2.0, 4.0, 1.5)),
    ("-x", (0.0, 4.0, 1.5), (4.0, 4.0, 1.5)),
    ("+y", (3.0, 9.0, 1.5), (3.0, 3.0, 1.5)),
    ("-y", (3.0, 0.0, 1.5), (3.0, 4.0, 1.5)),
])
def test_mirror_each_face_has_hand_computed_answer(direction: str, point: Vec3, expected: Vec3) -> None:
    from aosr.geometry.furniture import mirror_point

    assert mirror_point(point, face(direction), margin_m=0.0) == expected


def test_tabletop_example_reflects_at_horizontal_midpoint() -> None:
    from aosr.geometry.furniture import single_bounce_point

    # 第 11 條：0.8×1.6 m 桌板頂面 z=0.75；兩點同高 0.45、沿 y 相距 1.2。
    table = box(width=0.8, depth=1.6, height=0.05, bottom=(2.0, 2.0, 0.7))
    top = next(item for item in table.faces if item.direction == "top")
    hit = single_bounce_point((2.0, 1.4, 1.2), (2.0, 2.6, 1.2), top, margin_m=0.0)
    assert hit == (2.0, 2.0, 0.75)


@pytest.mark.parametrize(("direction", "source", "receiver", "expected"), [
    ("top", (2.5, 3.0, 3.0), (3.5, 5.0, 3.0), (3.0, 4.0, 2.0)),
    ("bottom", (2.5, 3.0, 0.0), (3.5, 5.0, 0.0), (3.0, 4.0, 1.0)),
    ("+x", (5.0, 3.0, 1.25), (5.0, 5.0, 1.75), (4.0, 4.0, 1.5)),
    ("-x", (1.0, 3.0, 1.25), (1.0, 5.0, 1.75), (2.0, 4.0, 1.5)),
    ("+y", (2.5, 7.0, 1.25), (3.5, 7.0, 1.75), (3.0, 6.0, 1.5)),
    ("-y", (2.5, 1.0, 1.25), (3.5, 1.0, 1.75), (3.0, 2.0, 1.5)),
])
def test_single_bounce_all_outward_directions(direction: str, source: Vec3, receiver: Vec3,
                                             expected: Vec3) -> None:
    from aosr.geometry.furniture import single_bounce_point

    assert single_bounce_point(source, receiver, face(direction), margin_m=0.0) == expected


def test_single_bounce_unequal_heights_uses_distance_ratio() -> None:
    from aosr.geometry.furniture import single_bounce_point

    # 頂面 z=2，像源 z=1 到接收 z=5 的四分之一處：y=3+(5-3)/4=3.5。
    assert single_bounce_point((3.0, 3.0, 3.0), (3.0, 5.0, 5.0), face("top"), margin_m=0.0) == (3.0, 3.5, 2.0)


@pytest.mark.parametrize(("factor", "accepted"), [(-2.0, True), (0.0, True), (0.5, True), (1.0, True), (2.0, False)])
@pytest.mark.parametrize(("axis", "edge"), [(0, 2.0), (0, 4.0), (1, 2.0), (1, 6.0)])
def test_reflection_edge_contact_band(axis: int, edge: float, factor: float, accepted: bool) -> None:
    from aosr.geometry.furniture import single_bounce_point

    margin = contract_value("furniture_geometry_contact") * 8.0
    outward = -1.0 if edge == 2.0 else 1.0
    xyz = [3.0, 4.0, 3.0]
    xyz[axis] = edge + outward * factor * margin
    point = cast("Vec3", tuple(xyz))
    hit = single_bounce_point(point, point, face("top"), margin_m=margin)
    if accepted:
        assert hit == (xyz[0], xyz[1], 2.0)
    else:
        assert hit is None


def test_reflection_far_outside_face_has_no_path() -> None:
    from aosr.geometry.furniture import single_bounce_point

    assert single_bounce_point((8.0, 3.0, 3.0), (8.0, 5.0, 3.0), face("top"), margin_m=0.0) is None


@pytest.mark.parametrize(("source_z", "receiver_z"), [(1.5, 3.0), (3.0, 1.5), (1.5, 1.5), (2.0, 3.0), (3.0, 2.0)])
def test_backside_or_on_plane_has_no_reflection(source_z: float, receiver_z: float) -> None:
    from aosr.geometry.furniture import single_bounce_point

    assert single_bounce_point((3.0, 3.0, source_z), (3.0, 5.0, receiver_z), face("top"), margin_m=0.0) is None


@pytest.mark.parametrize("near_source", [False, True])
@pytest.mark.parametrize(("factor", "accepted"), [(0.5, False), (1.0, False), (2.0, True)])
def test_reflection_requires_both_points_beyond_contact_band(near_source: bool, factor: float, accepted: bool) -> None:
    from aosr.geometry.furniture import single_bounce_point

    margin = contract_value("furniture_geometry_contact") * 8.0
    near, far = (3.0, 4.0, 2.0 + factor * margin), (3.0, 4.0, 3.0)
    source, receiver = (near, far) if near_source else (far, near)
    hit = single_bounce_point(source, receiver, face("top"), margin_m=margin)
    assert (hit is not None) is accepted


def test_thin_table_does_not_fill_space_below_it() -> None:
    from aosr.geometry.furniture import segment_blocked_by_box

    table = box(width=0.8, depth=1.6, height=0.05, bottom=(2.0, 2.0, 0.7))
    assert segment_blocked_by_box((1.0, 2.0, 0.5), (3.0, 2.0, 0.5), table, margin_m=0.0) is False


def test_repeated_input_is_bitwise_identical() -> None:
    from aosr.geometry.furniture import mirror_point, single_bounce_point

    source, receiver = (3.125, 3.375, 3.0), (3.375, 4.625, 3.5)
    top = face("top")
    first = single_bounce_point(source, receiver, top, margin_m=0.0)
    assert first is not None
    expected = tuple(value.hex() for value in first)
    mirror_hex = tuple(value.hex() for value in mirror_point(source, top, margin_m=0.0))
    for _ in range(4):
        result = single_bounce_point(source, receiver, face("top"), margin_m=0.0)
        assert result is not None
        assert tuple(value.hex() for value in result) == expected
        assert tuple(value.hex() for value in mirror_point(source, face("top"), margin_m=0.0)) == mirror_hex


def test_furniture_is_outside_physics_and_modal_import_closures() -> None:
    from aosr.reporting.physics_identity import physics_import_closure
    from aosr.reporting.modal_diagnosis import modal_import_closure

    assert "aosr.geometry.furniture" not in physics_import_closure().modules
    assert "aosr.geometry.furniture" not in modal_import_closure().modules
