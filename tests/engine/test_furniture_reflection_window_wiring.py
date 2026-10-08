"""桌面手算場景：牆面證明維持全集，補算列只移除被擋的牆面路徑。"""
from __future__ import annotations

from dataclasses import asdict

import pytest

from aosr.geometry.furniture import FurnitureKind
from aosr.geometry.shoebox import Point, Room
from aosr.physics import report_io
from aosr.physics.furniture_paths import filter_room_paths
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.furniture_scene import furniture_lane_inputs, furniture_scene
from aosr.physics.reflection_window import ReflectionWindow, build_reflection_window
from aosr.physics.report_path_table import PathTableData, build_path_table
from aosr.physics.room_paths import SUPPORTED_MAX_ORDER, image_source_paths
from tests.engine import _furniture_energy_cases as case
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine.test_reflection_window import _row_key


FREQUENCIES = (125.0, 250.0)
SCATTERING = (0.2, 0.3)


def inputs(*, furnished: bool = True, order: int = 1) -> report_io.ReportInput:
    document = {
        "room_m": asdict(case.ROOM), "source_m": asdict(case.SOURCE),
        "receiver_m": asdict(case.RECEIVER), "source_model": {"kind": "omnidirectional"},
        "sound_speed_m_s": case.SPEED, "density_kg_m3": case.DENSITY,
        "impedance_pa_s_per_m_by_wall": {wall.wall_name(): value for wall, value in case.WALLS.items()},
        "reflection_order_k": order,
        "furniture": [case.desk().model_dump(mode="json")] if furnished else None,
    }
    return report_io.load_input_document(document, CAPABILITIES, DIRECTIVITY)


def window(data: report_io.ReportInput, seconds: float) -> ReflectionWindow:
    return build_reflection_window(data, frequencies_hz=FREQUENCIES,
        scattering_coefficient=SCATTERING, window_s=seconds, contact_rel=case.CONTACT_REL)


def table(data: report_io.ReportInput, order: int) -> PathTableData:
    solved = report_io.solver_inputs(data)
    furniture = furniture_lane_inputs(data.furniture, (case.ROOM.Lx, case.ROOM.Ly, case.ROOM.Lz),
                                     FREQUENCIES, case.RHO_C, contact_rel=case.CONTACT_REL)
    return build_path_table(room=solved.room, source=solved.source, receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s, rho_c_pa_s_per_m=case.RHO_C,
        impedance_by_wall=solved.impedance_by_wall, frequencies_hz=FREQUENCIES,
        scattering_coefficient=SCATTERING, reflection_order_k=order,
        source_model=solved.source_model, furniture=furniture, furniture_rows=False)


def test_desk_removes_only_the_two_blocked_extra_wall_paths() -> None:
    empty, furnished = window(inputs(furnished=False), 0.015), window(inputs(), 0.015)
    blocked = {("floor", "ceiling"), ("ceiling", "floor")}
    assert blocked <= {row.wall_sequence for row in empty.rows}
    assert tuple(_row_key(row) for row in furnished.rows) == tuple(
        _row_key(row) for row in empty.rows if row.wall_sequence not in blocked)
    assert furnished.computed_order_k == empty.computed_order_k == 2
    assert furnished.coverage == "approximate" and empty.coverage == "complete"
    assert furnished.furniture_ids == ("desk",)
    assert furnished.next_uncomputed_earliest_relative_s == pytest.approx(0.0178142, abs=0.0000001)


def test_wall_proof_never_uses_the_filtered_earliest_arrival() -> None:
    empty = window(inputs(furnished=False, order=3), 0.0235)
    furnished = window(inputs(order=3), 0.0235)
    assert furnished.computed_order_k == empty.computed_order_k == 4
    assert furnished.next_uncomputed_earliest_relative_s == empty.next_uncomputed_earliest_relative_s
    assert furnished.next_uncomputed_earliest_relative_s == pytest.approx(0.0281190, abs=0.0000001)
    assert empty.rows and not furnished.rows


def test_window_ids_follow_normalized_input_not_declaration_order() -> None:
    # 兩件、宣告順序跟代號序相反；書架擺在角落、不擋直達。單件家具看不出順序寫反。
    shelf = case.desk("a-shelf").model_copy(update={"bottom_center_m": (5.5, 3.5, 0.5), "width_m": 0.4, "depth_m": 0.4})
    document = inputs().model_dump(mode="json")
    document["furniture"] = [case.desk("z-desk").model_dump(mode="json"), shelf.model_dump(mode="json")]
    data = report_io.load_input_document(document, CAPABILITIES, DIRECTIVITY)
    assert window(data, 0.015).furniture_ids == ("a-shelf", "z-desk")


def test_unprovable_furniture_window_keeps_ids() -> None:
    result = window(inputs(order=3), 1.0)
    assert result.coverage == "not_provable"
    assert result.computed_order_k == SUPPORTED_MAX_ORDER
    assert result.furniture_ids == ("desk",)


@pytest.mark.parametrize(("order", "seconds"), [(1, 0.001), (1, 0.015), (3, 0.0235), (3, 0.04), (3, 1.0)])
def test_furniture_does_not_change_wall_proof_or_numeric_validation(order: int, seconds: float) -> None:
    empty, furnished = window(inputs(furnished=False, order=order), seconds), window(inputs(order=order), seconds)
    assert furnished.coverage != "complete"
    assert furnished.furniture_ids == ("desk",)
    assert furnished.computed_order_k == empty.computed_order_k
    assert furnished.next_uncomputed_earliest_relative_s == empty.next_uncomputed_earliest_relative_s
    assert furnished.direct_delay_s == empty.direct_delay_s
    assert furnished.validation == empty.validation


@pytest.mark.parametrize(("order", "seconds"), [(1, 0.015), (3, 0.0235)])
def test_furniture_window_plus_k_table_equals_full_wall_only_table(order: int, seconds: float) -> None:
    data = inputs(order=order)
    result, ordinary, full = window(data, seconds), table(data, order), table(data, SUPPORTED_MAX_ORDER)
    direct = next(row.delay_s for row in full.rows if row.order == 0)
    actual = tuple(_row_key(row) for row in ordinary.rows if row.delay_s - direct <= seconds)
    actual += tuple(_row_key(row) for row in result.rows)
    expected = tuple(_row_key(row) for row in full.rows if row.delay_s - direct <= seconds)
    assert actual == expected
    assert all(row.furniture_id is None and row.wall_sequence != ("furniture",) for row in result.rows)


def test_furniture_requires_explicit_keyword_contact() -> None:
    with pytest.raises(ValueError, match="有家具時必須提供 contact_rel（家具幾何接觸界線）"):
        build_reflection_window(inputs(), frequencies_hz=FREQUENCIES,
            scattering_coefficient=SCATTERING, window_s=0.015)


@pytest.mark.parametrize(("depth_m", "blocked"), [(2e-12, False), (0.02, True)])
def test_direct_check_without_extra_table_uses_the_contact_boundary(depth_m: float, blocked: bool) -> None:
    # 決策紙第 14 條：界線以內擦過桌面頂面（z=0.6）不算擋；computed == K 那一支也要用同一把界線。
    document = inputs(order=3).model_dump(mode="json")
    document["source_m"]["z"] = document["receiver_m"]["z"] = 0.6 - depth_m
    data = report_io.load_input_document(document, CAPABILITIES, DIRECTIVITY)
    assert window(data.model_copy(update={"furniture": None}), 0.001).computed_order_k == 3
    if blocked:
        with pytest.raises(ValueError, match="直達路徑被家具 desk 擋住，不符合擺位要求"):
            window(data, 0.001)
    else:
        assert window(data, 0.001).coverage == "approximate"


def test_blocked_direct_is_rejected_even_without_an_extra_table(monkeypatch: pytest.MonkeyPatch) -> None:
    data = inputs(order=3)
    assert window(data.model_copy(update={"furniture": None}), 0.001).computed_order_k == 3
    desk = case.desk().model_copy(update={"bottom_center_m": (3.0, 2.0, 1.0)})
    data = data.model_copy(update={"furniture": (desk,)})

    def forbidden_table(**kwargs: object) -> None:
        pytest.fail("computed == K 不准建補算表")

    monkeypatch.setattr("aosr.physics.reflection_window.build_path_table", forbidden_table)
    with pytest.raises(ValueError, match="直達路徑被家具 desk 擋住，不符合擺位要求"):
        window(data, 0.001)


def _plate(furniture_id: str, x: float, y: float) -> AbsoluteFurniture:
    return AbsoluteFurniture(furniture_id=furniture_id, kind=FurnitureKind.CEILING_CLOUD, material="wood",
        width_m=0.5, depth_m=0.5, height_m=0.05, bottom_center_m=(x, y, 1.5), yaw_deg=0.0)


def test_filtered_wall_proof_would_miss_a_surviving_third_order_path() -> None:
    """真正的反例（第六步物理審查找到）：聲源、接收點正上方各一片小板，擋掉第 2 階的地板與天花；
    第 3 階有一條倖存、比第 2 階倖存最早還早。用過濾後的最早當證明會停在 1 階、漏掉它。"""
    room, source, receiver = (20.0, 20.0, 2.5), (9.0, 10.0, 1.25), (11.0, 10.0, 1.25)
    items = (_plate("over-receiver", *receiver[:2]), _plate("over-source", *source[:2]))
    document = inputs(furnished=False).model_dump(mode="json")
    document.update(room_m=dict(zip(("Lx", "Ly", "Lz"), room)), source_m=dict(zip("xyz", source)),
                    receiver_m=dict(zip("xyz", receiver)), furniture=[item.model_dump(mode="json") for item in items])
    data = report_io.load_input_document(document, CAPABILITIES, DIRECTIVITY)
    boxes, margin_m = furniture_scene(items, room, contact_rel=case.CONTACT_REL)
    paths = image_source_paths(Room(*room), Point(*source), Point(*receiver), data.sound_speed_m_s, max_order=3)
    kept, _ = filter_room_paths(tuple(paths), source, receiver, boxes, margin_m=margin_m)
    direct = next(path.delay_s for path in paths if path.order == 0)
    second = min(path.delay_s for path in kept if path.order == 2) - direct
    third = min((path for path in kept if path.order == 3), key=lambda path: path.delay_s)
    assert third.delay_s - direct < second, "幾何要讓倖存的第 3 階比倖存的第 2 階早"
    result = window(data, (second + third.delay_s - direct) / 2)
    assert result.computed_order_k > 2 and result.coverage == "approximate"
    assert tuple(wall for bounce in third.bounces for wall in bounce.walls) in {row.wall_sequence for row in result.rows}


def test_screen_next_order_ignores_furniture_blocking() -> None:
    # 篩查第 K+1 階只算牆面、不扣遮擋；這個場景的第 4 階最早那條會被桌子擋，扣了就分得出來。
    empty = build_reflection_screen(inputs(furnished=False, order=3), FREQUENCIES)
    screened = build_reflection_screen(inputs(order=3), FREQUENCIES)
    assert screened.next_order_earliest_delay_s == empty.next_order_earliest_delay_s
    assert screened.pairs == empty.pairs
    paths = tuple(image_source_paths(case.ROOM, case.SOURCE, case.RECEIVER, case.SPEED, max_order=4))
    lane = case.lane_inputs((case.desk(),))
    kept, _ = filter_room_paths(paths, case.SOURCE.as_tuple(), case.RECEIVER.as_tuple(), lane.furniture,
                                margin_m=lane.margin_m)
    assert min(path.delay_s for path in kept if path.order == 4) != min(path.delay_s for path in paths if path.order == 4)


def test_blocked_direct_is_rejected_when_the_report_starts_at_the_supported_limit() -> None:
    desk = case.desk().model_copy(update={"bottom_center_m": (3.0, 2.0, 1.0)})
    data = inputs(order=SUPPORTED_MAX_ORDER).model_copy(update={"furniture": (desk,)})
    with pytest.raises(ValueError, match="直達路徑被家具 desk 擋住，不符合擺位要求"):
        window(data, 0.001)

