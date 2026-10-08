"""三路與批次家具接線：共用阻抗、有限元素前拒收、輸出家具綁定。"""
from __future__ import annotations

from dataclasses import asdict, fields, replace

import pytest

from aosr.config.frequency_axis import GEOMETRIC_BAND_FREQUENCIES_HZ, LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.physics import furniture_scene, geometric_lane, report_io, report_output, three_lane_report as report
from aosr.physics.room_paths import RoomPath
from aosr.physics.furniture_paths import Furniture, FurniturePath, filter_room_paths, single_bounce_furniture_paths
from aosr.geometry.furniture import FaceDirection, Vec3
from aosr.geometry.shoebox import Wall
from aosr.physics import report_path_table, three_lane_report_batch as batch
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.report_io import PathTableSection
from aosr.materials.furniture_materials import FurnitureImpedanceOnAxis
from tests.engine import _furniture_energy_cases as case, _furniture_third_order_case as third, _source_model_control as stand_ins
from tests.engine._report_cache import _key_value
from tests.engine.test_report_scene import _inputs


@pytest.fixture
def fast_room(monkeypatch: pytest.MonkeyPatch) -> None:
    # 有限元素與晚期慢求解用已知替身；家具路徑與兩軸接合都真跑。
    monkeypatch.setattr(report, "_solve_fem_energy", stand_ins.fake_fem_energy)
    monkeypatch.setattr(report, "_solve_report_late_decay", stand_ins.fast_late_decay)
    monkeypatch.setattr(batch, "solve_geometric_late_energy", stand_ins.fake_late_energy)


def _prepared(furniture: tuple[AbsoluteFurniture, ...] | None = None) -> batch._Shared:
    return batch._prepare(source_model=case.OMNI, room=case.ROOM, sound_speed_m_s=case.SPEED,
        density_kg_m3=case.DENSITY, impedance_by_wall=case.WALLS, scattering_by_wall=None,
        capability=None, reflection_order_k=1, low_frequency_axis=LowFrequencyAxis.SEARCH,
        furniture=furniture, contact_rel=case.CONTACT_REL if furniture else None)


def test_shared_room_fields_bitwise_equal_except_furniture() -> None:
    plain, furnished = _prepared(), _prepared((case.desk(),))
    for field in fields(plain):
        if field.name != "furniture":
            # Python 浮點 repr 是往返不失位的表示，亦分開正負零；遞迴展開每格。
            assert repr(_key_value(getattr(plain, field.name))) == repr(_key_value(getattr(furnished, field.name))), field.name
    assert plain.furniture is None and furnished.furniture is not None


@pytest.mark.usefixtures("fast_room")
def test_fine_and_dense_enumerate_their_own_room_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    axes: list[int] = []
    events: list[str] = []
    original = filter_room_paths
    original_furniture = single_bounce_furniture_paths

    def observed(paths: tuple[RoomPath, ...], source: Vec3, receiver: Vec3,
                 furniture: tuple[Furniture, ...], *, margin_m: float
                 ) -> tuple[tuple[RoomPath, ...], tuple[tuple[int, int, int, int, int, int], ...]]:
        axes.append(len(paths[0].path_pressure))
        events.append("牆面")
        return original(paths, source, receiver, furniture, margin_m=margin_m)

    def observed_furniture(source: Vec3, receiver: Vec3, furniture: tuple[Furniture, ...], *,
                           c: float, margin_m: float) -> tuple[FurniturePath, ...]:
        events.append("家具")
        return original_furniture(source, receiver, furniture, c=c, margin_m=margin_m)

    monkeypatch.setattr(geometric_lane, "filter_room_paths", observed)
    monkeypatch.setattr(geometric_lane, "single_bounce_furniture_paths", observed_furniture)
    shared = _prepared((case.desk(),))
    result = batch._pair_report(shared, case.SOURCE, case.RECEIVER, (0.1,) * len(shared.fem_frequencies_hz))
    assert axes == [len(shared.report_frequencies_hz), len(GEOMETRIC_BAND_FREQUENCIES_HZ)]
    assert events == ["牆面", "家具", "牆面", "家具"]
    assert result.furniture == (case.desk(),)


@pytest.mark.usefixtures("fast_room")
@pytest.mark.parametrize("medium_scale", (1.0, 2.0))
def test_batch_prepares_impedances_once_per_axis_and_reports_keep_absolute_furniture(
    monkeypatch: pytest.MonkeyPatch, medium_scale: float,
) -> None:
    calls: list[tuple[tuple[float, ...], float]] = []
    original = furniture_scene.furniture_impedances

    def observed(items: tuple[AbsoluteFurniture, ...], frequencies: tuple[float, ...],
                 rho_c: float) -> dict[str, FurnitureImpedanceOnAxis]:
        calls.append((frequencies, rho_c))
        return original(items, frequencies, rho_c)

    monkeypatch.setattr(furniture_scene, "furniture_impedances", observed)
    sources = {"left": case.SOURCE, "right": replace(case.SOURCE, y=1.8)}
    receivers = {"main": case.RECEIVER, "surround": replace(case.RECEIVER, y=2.2)}
    fs = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[0]
    actual = report.solve_three_lane_reports(source_model=case.analytic(), room=case.ROOM,
        sources=sources, receivers=receivers, sound_speed_m_s=case.SPEED, density_kg_m3=case.DENSITY * medium_scale,
        impedance_by_wall=case.WALLS, reflection_order_k=1,
        furniture=(case.desk(),), contact_rel=case.CONTACT_REL,
        fem_energies={(s, r): (0.1,) * len(fs) for s in sources for r in receivers})
    assert calls == [(low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[1], case.RHO_C * medium_scale),
                     (GEOMETRIC_BAND_FREQUENCIES_HZ, case.RHO_C * medium_scale)]
    for result in actual.values():
        assert result.furniture == result.geometric_lane.furniture == (case.desk(),)
    calls.clear()
    one = report.solve_three_lane_report(source_model=case.analytic(), room=case.ROOM,
        source=case.SOURCE, receiver=case.RECEIVER, sound_speed_m_s=case.SPEED, density_kg_m3=case.DENSITY,
        impedance_by_wall=case.WALLS, reflection_order_k=1,
        furniture=(case.desk(),), contact_rel=case.CONTACT_REL)
    assert one.furniture == (case.desk(),)


@pytest.mark.usefixtures("fast_room")
@pytest.mark.parametrize("batch_fem", (False, True))
def test_all_blocked_pairs_are_named_before_fem(monkeypatch: pytest.MonkeyPatch, batch_fem: bool) -> None:
    def forbidden(**_: object) -> None:
        pytest.fail("直達被擋時有限元素不得開始")

    monkeypatch.setattr(report, "_solve_fem_energy", forbidden)
    monkeypatch.setattr(report, "_solve_fem_energies", forbidden)
    blocker = case.desk("barrier").model_copy(update={"height_m": 1.0})
    with pytest.raises(ValueError, match="直達") as caught:
        batch.solve_reports(source_model=case.OMNI, room=case.ROOM,
            sources={"left": case.SOURCE, "right": replace(case.SOURCE, y=1.9)},
            receivers={"main": case.RECEIVER, "surround": replace(case.RECEIVER, y=2.1)},
            sound_speed_m_s=case.SPEED, density_kg_m3=case.DENSITY, impedance_by_wall=case.WALLS,
            scattering_by_wall=None, capability=None, reflection_order_k=1,
            low_frequency_axis=LowFrequencyAxis.SEARCH, batch_fem=batch_fem,
            furniture=(blocker,), contact_rel=case.CONTACT_REL)
    for name in ("left", "right", "main", "surround", "barrier"):
        assert name in str(caught.value)


@pytest.mark.parametrize("batch_entry", (False, True))
def test_furniture_requires_explicit_contact_rel(batch_entry: bool) -> None:
    with pytest.raises(ValueError, match="contact_rel"):
        if batch_entry:
            report.solve_three_lane_reports(source_model=case.OMNI, room=case.ROOM,
                sources={"left": case.SOURCE}, receivers={"main": case.RECEIVER},
                sound_speed_m_s=case.SPEED, density_kg_m3=case.DENSITY,
                impedance_by_wall=case.WALLS, furniture=(case.desk(),))
        else:
            report.solve_three_lane_report(source_model=case.OMNI, room=case.ROOM,
                source=case.SOURCE, receiver=case.RECEIVER, sound_speed_m_s=case.SPEED,
                density_kg_m3=case.DENSITY, impedance_by_wall=case.WALLS, furniture=(case.desk(),))


@pytest.mark.usefixtures("fast_room")
@pytest.mark.parametrize("mismatch", ("inputs-only", "report-only", "same-id-other-size"))
def test_output_refuses_different_furniture_before_solver_input_gate(mismatch: str) -> None:
    inputs = _inputs()
    actual = report.solve_three_lane_report(**report_io.solver_inputs(inputs)._asdict())
    wrong_inputs = inputs.model_copy(update={"furniture": (case.desk(),)})
    if mismatch != "inputs-only":
        actual = replace(actual, furniture=(case.desk(),))
        wrong_inputs = inputs
    if mismatch == "same-id-other-size":
        # 兩邊都有家具、代號與件數相同、只差尺寸：只比代號或只比有沒有家具都會放過。
        wrong_inputs = inputs.model_copy(update={"furniture": (case.desk().model_copy(update={"width_m": 0.9}),)})
    with pytest.raises(ValueError, match="inputs.*家具.*report"):
        report_output.output_from_report(actual, inputs=wrong_inputs, with_points=False,
            path_table_inputs=report_io.solver_inputs(inputs))


@pytest.mark.usefixtures("fast_room")
def test_output_accepts_report_solved_with_the_same_furniture() -> None:
    # 正向對照：核對只擋「不同」，有家具時使用真的求解輸入與路徑表。
    inputs = _inputs()
    furnished = inputs.model_copy(update={"furniture": (case.desk(),)})
    plain_report = report.solve_three_lane_report(**report_io.solver_inputs(inputs)._asdict())
    furnished_report = report.solve_three_lane_report(**report_io.solver_inputs(inputs)._asdict()
        | {"furniture": furnished.furniture, "contact_rel": case.CONTACT_REL})
    plain = report_output.output_from_report(plain_report, inputs=inputs, with_points=False)
    output = report_output.output_from_report(furnished_report, inputs=furnished, with_points=False,
        path_table_inputs=report_io.solver_inputs(furnished), contact_rel=case.CONTACT_REL)
    assert furnished_report.furniture == furnished.furniture
    assert output.scene.scene_fingerprint != plain.scene.scene_fingerprint


def _third_solved(scattering: float) -> report_io.SolverInputs:
    return report_io.solver_inputs(_inputs())._replace(room=third.ROOM, source=third.SOURCE,
        receiver=third.RECEIVER, sound_speed_m_s=third.SPEED, density_kg_m3=third.DENSITY,
        impedance_by_wall={wall: third.WALLS[wall.wall_name()] for wall in Wall.all()},
        scattering_by_wall={wall: scattering for wall in Wall.all()}, reflection_order_k=2,
        source_model=third.MODEL, furniture=(third.GLASS_DESK,))


@pytest.mark.usefixtures("fast_room")
@pytest.mark.parametrize("explicit_contact", (False, True))
def test_section_builds_its_furniture_inputs_from_the_scheme_medium_like_the_energy_lane(explicit_contact: bool) -> None:
    """路徑表段自己組家具輸入（ρc、房間、頻率軸）；散射 1 消掉牆面鏡面，家具列×直達＝能量路反射欄。"""
    solved = _third_solved(1.0)
    actual = report.solve_three_lane_report(**solved._asdict(), contact_rel=case.CONTACT_REL)
    section = report_path_table.build_path_table_section(actual, solved,
        contact_rel=case.CONTACT_REL if explicit_contact else None)
    rows = [row for row in section.rows if row.furniture_id is not None]
    assert [(row.furniture_id, row.furniture_face) for row in rows] == [("glassdesk", FaceDirection.TOP)]
    lane = actual.geometric_lane
    assert tuple(relative * direct for relative, direct in zip(rows[0].relative_direct_energy, lane.direct_energy,
        strict=True)) == pytest.approx(lane.reflected_energy, rel=1e-12)


@pytest.mark.usefixtures("fast_room")
def test_section_wall_rows_use_the_scheme_medium() -> None:
    """散射小於 1 時牆面列也在：整張段落表要等於直接用方案 ρc（密度×聲速）建的表。"""
    solved = _third_solved(0.3)
    actual = report.solve_three_lane_report(**solved._asdict(), contact_rel=case.CONTACT_REL)
    section = report_path_table.build_path_table_section(actual, solved, contact_rel=case.CONTACT_REL)
    lane = actual.geometric_lane
    expected = report_path_table.build_path_table(room=third.ROOM, source=third.SOURCE, receiver=third.RECEIVER,
        sound_speed_m_s=third.SPEED, rho_c_pa_s_per_m=third.RHO_C, impedance_by_wall=solved.impedance_by_wall,
        frequencies_hz=lane.frequencies_hz, scattering_coefficient=lane.scattering, reflection_order_k=2,
        source_model=third.MODEL, furniture=furniture_scene.furniture_lane_inputs((third.GLASS_DESK,),
            (third.ROOM.Lx, third.ROOM.Ly, third.ROOM.Lz), lane.frequencies_hz, third.RHO_C,
            contact_rel=case.CONTACT_REL))
    assert any(row.furniture_id is None and row.order > 0 for row in section.rows)
    assert section == PathTableSection.model_validate(asdict(expected))


@pytest.mark.usefixtures("fast_room")
def test_output_hands_its_contact_to_the_path_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """真求解輸入帶家具，輸出組裝把呼叫端給的界線原樣交給路徑表。"""
    inputs = _inputs()
    furnished = inputs.model_copy(update={"furniture": (case.desk(),)})
    solved = report_io.solver_inputs(furnished)
    actual = report.solve_three_lane_report(**solved._asdict(), contact_rel=case.CONTACT_REL)
    marker = case.CONTACT_REL * 3.0  # 跟登記簿不同：拿到的若是重讀的值就分得出來
    seen: list[float | None] = []
    real = furniture_scene.furniture_lane_inputs

    def spy(items: tuple[AbsoluteFurniture, ...] | None, room_size_m: Vec3, frequencies_hz: tuple[float, ...],
            rho_c: float, *, contact_rel: float | None) -> furniture_scene.FurnitureLaneInputs | None:
        seen.append(contact_rel)
        return real(items, room_size_m, frequencies_hz, rho_c, contact_rel=contact_rel)

    monkeypatch.setattr(report_path_table, "furniture_lane_inputs", spy)
    output = report_output.output_from_report(actual, inputs=furnished, with_points=False,
        path_table_inputs=solved, contact_rel=marker)
    assert seen == [marker]
    assert output.path_table is not None and output.path_table.furniture_ids == ("desk",)
