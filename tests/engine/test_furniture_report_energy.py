"""三路與批次家具接線：共用阻抗、有限元素前拒收、輸出家具綁定。"""
from __future__ import annotations

from dataclasses import fields, replace

import pytest

from aosr.config.frequency_axis import GEOMETRIC_BAND_FREQUENCIES_HZ, LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.physics import furniture_scene, geometric_lane, report_io, report_output, three_lane_report as report
from aosr.physics.room_paths import RoomPath
from aosr.physics.furniture_paths import Furniture, FurniturePath, filter_room_paths, single_bounce_furniture_paths
from aosr.geometry.furniture import Vec3
from aosr.physics import three_lane_report_batch as batch
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.materials.furniture_materials import FurnitureImpedanceOnAxis
from tests.engine import _furniture_energy_cases as case, _source_model_control as stand_ins
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
@pytest.mark.parametrize("report_has_furniture", (False, True))
def test_output_refuses_different_furniture_before_solver_input_gate(report_has_furniture: bool) -> None:
    inputs = _inputs()
    actual = report.solve_three_lane_report(**report_io.solver_inputs(inputs)._asdict())
    wrong_inputs = inputs.model_copy(update={"furniture": (case.desk(),)})
    if report_has_furniture:
        actual = replace(actual, furniture=(case.desk(),))
        wrong_inputs = inputs
    with pytest.raises(ValueError, match="inputs.*家具.*report"):
        report_output.output_from_report(actual, inputs=wrong_inputs, with_points=False,
            path_table_inputs=report_io.solver_inputs(inputs))
