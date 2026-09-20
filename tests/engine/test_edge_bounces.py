"""票 #305：反彈點落在兩面牆交線時，幾何、階數與材料都要連續。"""
from __future__ import annotations

import pytest

from aosr.geometry import shoebox
from aosr.geometry.shoebox import (
    Point,
    Room,
    expand_bounces,
    image_from_identity,
    order_of,
)
from aosr.physics.amplitude import (
    CANONICAL_WALLS,
    Materials,
    WallGrid,
    incidence_cos,
    reflection_coefficient,
)
from aosr.physics.room_paths import RoomPath, _one_path, image_source_paths
from tests.engine._precision_contracts import contract_value


_ROOM = Room(Lx=2.0, Ly=2.0, Lz=2.0)
_RECEIVER = Point(x=1.5, y=1.5, z=1.0)
_EDGE_IDENTITY = (0, -1, 0, -1, 0, 1)
_RHO_C = 400.0
_FREQUENCY_HZ = 1000.0
_CONTINUITY_REL = contract_value("edge_bounce_continuity")


def _materials(*, patched: bool) -> Materials:
    impedances = {
        "floor": complex(7.0 * _RHO_C, 0.0),
        "ceiling": complex(8.0 * _RHO_C, 0.0),
        "x0": complex(2.0 * _RHO_C, 0.0),
        "xL": complex(9.0 * _RHO_C, 0.0),
        "y0": complex(4.0 * _RHO_C, 0.0),
        "yL": complex(10.0 * _RHO_C, 0.0),
    }
    walls: dict[str, tuple[complex, ...]] = {
        wall: (impedances[wall],) for wall in CANONICAL_WALLS
    }
    wall_grids: dict[str, WallGrid] | None = None
    if patched:
        wall_grids = {
            wall: (1, 2, ((impedances[wall],), (impedances[wall],)))
            for wall in CANONICAL_WALLS
        }
    return Materials(
        rho_c=_RHO_C,
        frequencies_hz=(_FREQUENCY_HZ,),
        walls=walls,
        wall_grids=wall_grids,
    )


def _edge_path(source: Point, *, patched: bool = False) -> RoomPath:
    return next(
        path
        for path in image_source_paths(
            _ROOM,
            source,
            _RECEIVER,
            343.0,
            max_order=2,
            materials=_materials(patched=patched),
        )
        if path.identity == _EDGE_IDENTITY
    )


def test_edge_energy_joins_the_nearby_path_continuously() -> None:
    """分格材料逐跳乘係數時，10 cm、1 cm、1 mm 到交線的能量連續收斂。"""
    offsets_m = (0.1, 0.01, 0.001, 0.0)
    energies = tuple(
        abs(
            _edge_path(
                Point(0.5, 0.5 + offset, 1.0),
                patched=True,
            ).path_pressure[0]
        )
        ** 2
        for offset in offsets_m
    )
    steps = tuple(abs(right - left) for left, right in zip(energies, energies[1:]))

    assert all(later < earlier for earlier, later in zip(steps, steps[1:])), steps
    endpoint_relative_difference = abs(energies[-1] - energies[-2]) / energies[-1]
    assert endpoint_relative_difference <= _CONTINUITY_REL


def test_one_edge_point_consumes_both_identity_reflections() -> None:
    """交線上的同一點同時帶 x0、y0，牆數加總仍等於 identity 的二階。"""
    path = _edge_path(Point(0.5, 0.5, 1.0))
    edge = next(
        bounce for bounce in path.bounces if set(bounce.walls) == {"x0", "y0"}
    )

    assert edge.point == (0.0, 0.0, 1.0)
    assert sum(len(bounce.walls) for bounce in path.bounces) == path.order


def test_three_wall_corner_uses_the_same_accounting_rule() -> None:
    """三面交會是同一規則的延伸：同一點帶 floor、x0、y0 三面。"""
    identity = (0, -1, 0, -1, 0, -1)
    bounces = expand_bounces(
        _ROOM,
        identity,
        Point(0.5, 0.5, 0.5),
        Point(1.5, 1.5, 1.5),
    )
    corner = next(
        bounce
        for bounce in bounces
        if set(bounce.walls) == {"floor", "x0", "y0"}
    )

    assert corner.point == (0.0, 0.0, 0.0)
    assert sum(len(bounce.walls) for bounce in bounces) == order_of(identity)


def test_patch_reflection_product_multiplies_both_edge_walls() -> None:
    """分格材料也要逐牆查值；交線的一點不能只乘其中一面。"""
    source = Point(0.5, 0.5, 1.0)
    path = _edge_path(source, patched=True)
    materials = _materials(patched=True)
    image = image_from_identity(_ROOM, _EDGE_IDENTITY, source).as_tuple()
    x_cos = incidence_cos(0, path.dist_m, _RECEIVER.as_tuple(), image)
    y_cos = incidence_cos(1, path.dist_m, _RECEIVER.as_tuple(), image)
    expected = reflection_coefficient(
        materials.impedance_at("x0", 0, 0, 0), x_cos, materials.rho_c
    ) * reflection_coefficient(
        materials.impedance_at("y0", 0, 0, 0), y_cos, materials.rho_c
    )

    assert path.reflection_product == (expected,)
    assert tuple(wall for wall, _row, _col in path.bounce_cells) == ("x0", "y0")


def test_edge_point_only_multiplies_walls_still_owed_by_identity() -> None:
    """點雖落在 x0/y0 交線，identity 只欠 x0 時不得把 y0 也乘進去。"""
    source = Point(0.5, 0.0, 1.0)
    receiver = Point(1.5, 0.0, 1.0)
    identity = (0, -1, 0, 1, 0, 1)
    materials = _materials(patched=True)
    path = _one_path(
        _ROOM,
        source,
        receiver,
        343.0,
        identity,
        materials,
    )
    image = image_from_identity(_ROOM, identity, source).as_tuple()
    expected = reflection_coefficient(
        materials.impedance_at("x0", 0, 0, 0),
        incidence_cos(0, path.dist_m, receiver.as_tuple(), image),
        materials.rho_c,
    )

    assert path.bounces[0].point == (0.0, 0.0, 1.0)
    assert path.bounces[0].walls == ("x0",)
    assert tuple(wall for wall, _row, _col in path.bounce_cells) == ("x0",)
    assert path.reflection_product == (expected,)


def test_symmetric_third_order_room_now_computes() -> None:
    """最常見的左右對稱佈局不再讓整份三階路徑失敗。"""
    paths = image_source_paths(
        Room(Lx=6.0, Ly=4.0, Lz=3.0),
        Point(1.0, 2.0, 1.5),
        Point(4.0, 2.0, 1.5),
        343.0,
        max_order=3,
    )

    assert paths
    assert all(
        sum(len(bounce.walls) for bounce in path.bounces) == path.order
        for path in paths
    )


def test_segment_that_crosses_no_wall_still_raises() -> None:
    """接收點與鏡像都在 x0 外側時沒有牆交點，仍然明確拒絕。"""
    with pytest.raises(ValueError, match="線段沒穿過任何一面牆"):
        expand_bounces(
            _ROOM,
            (0, -1, 0, 1, 0, 1),
            Point(0.5, 1.0, 1.0),
            Point(-1.0, 1.0, 1.0),
        )


def test_crossing_pinned_outside_wall_reports_wall_and_point() -> None:
    """有穿過 x0 平面但交點落在房外時，不能誤報成完全沒有穿牆。"""
    with pytest.raises(ValueError, match="反彈點落在那面牆的範圍外") as caught:
        expand_bounces(
            _ROOM,
            (0, -1, 0, 1, 0, 1),
            Point(0.5, 1.0, 1.0),
            Point(1.5, 9.0, 1.0),
        )

    message = str(caught.value)
    assert "牆=x0" in message
    assert "反彈點=(0.0, 3.0, 1.0)" in message
    assert "線段沒穿過任何一面牆" not in message


def test_one_point_cannot_consume_more_reflections_than_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """候選閘門若多放一面牆進來，顯式階數帳仍須在加入反彈點前拒絕。"""
    identity = (0, -1, 0, 1, 0, 1)
    overdrawn_signature = dict(shoebox.wall_count_signature(identity))
    overdrawn_signature["y0"] = 1
    monkeypatch.setattr(
        shoebox,
        "wall_count_signature",
        lambda _identity: overdrawn_signature,
    )

    with pytest.raises(ValueError, match="反射階數超過 identity 的階數"):
        expand_bounces(
            _ROOM,
            identity,
            Point(0.5, 0.5, 1.0),
            Point(1.5, -1.5, 1.0),
        )
