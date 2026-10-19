"""材料核心當次註記的收集、聯集、輸出界線與白話。"""
from __future__ import annotations

import pytest
from pathlib import Path
from aosr.reporting.result import SchemeResult
from tests.engine.test_scheme_repair import result as result
from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import ConstructionNote
from aosr.physics import wall_surfaces
from tests.engine.test_report_output_contract import _output


def note(frequencies: tuple[float, ...], *, layer: int = 1) -> ConstructionNote:
    return ConstructionNote(code="komatsu_frequency_extrapolation", layer_index=layer,
                            frequencies_hz=frequencies, material_class="fibrous", message="頻率外推照算")


def test_merge_preserves_layers_codes_and_unions_actual_frequencies() -> None:
    collected: dict[Wall, tuple[ConstructionNote, ...]] = {}
    wall_surfaces.merge_construction_notes(collected, Wall.X0, (note((80., 20.)),))
    wall_surfaces.merge_construction_notes(collected, Wall.X0, (note((20., 40.)), note((80.,), layer=2)))
    by_layer = {item.layer_index: item for item in collected[Wall.X0]}
    assert by_layer[1].frequencies_hz == (20., 40., 80.)
    assert by_layer[2] == note((80.,), layer=2)
    assert by_layer[1].message == "頻率外推照算"


def test_note_output_is_optional_and_has_facts_and_closed_wall_names() -> None:
    assert "construction_notes_by_wall" not in _output().model_dump(mode="json")
    output = _output(construction_notes_by_wall={"x0": (note((20.,)),)})
    assert type(output).model_validate_json(output.model_dump_json()) == output
    facts = type(output).model_fields["construction_notes_by_wall"].json_schema_extra
    assert isinstance(facts, dict)
    assert {"quantity", "unit", "reference", "validity"} <= facts.keys()
    whole = type(output).model_json_schema()
    schema = whole["properties"]["construction_notes_by_wall"]
    assert set(schema["propertyNames"]["enum"]) == set(Wall.wall_names())
    assert whole["$defs"]["ConstructionNote"]["properties"]["frequencies_hz"]["uniqueItems"] is True
    for broken in ({"ceilling": (note((20.,)),)}, {"x0": (note((40., 20.)),)},
                   {"x0": (note((20.,)), note((40.,)))}, {"x0": ()}, {}):
        with pytest.raises(ValueError):
            _output(construction_notes_by_wall=broken)


def test_each_wall_response_keeps_the_notes_returned_with_its_values(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    from aosr.materials import construction
    from aosr.materials.construction_models import Assembly, BoardLayer
    from tests.engine._material_flow_cases import fixed_materials

    assembly = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),), backing="rigid")
    materials = fixed_materials({wall: 1646.4 for wall in Wall.all()}) | {Wall.X0: assembly}
    normal_note, path_note, late_note = note((20.,)), note((40.,)), note((80.,))
    monkeypatch.setattr(construction, "surface_impedance", lambda *args, **kwargs: (np.array([1000.+2.j]), (normal_note,)))
    monkeypatch.setattr(construction, "surface_impedance_at_angles", lambda *args, **kwargs: (np.full((1, 8), 1000.+2.j), (path_note,)))
    monkeypatch.setattr(construction, "random_incidence_absorption", lambda *args, **kwargs: (np.array([.5]), (late_note,)))
    actual: dict[Wall, tuple[ConstructionNote, ...]] = {}
    wall_surfaces.wall_normal_impedances(materials, (20.,), density_kg_m3=1.2, sound_speed_m_s=343., notes_by_wall=actual)
    wall_surfaces.wall_path_impedances(assembly, (40.,), (.5,), wall_name="x0", density_kg_m3=1.2,
                                      sound_speed_m_s=343., notes_by_wall=actual)
    response = wall_surfaces.wall_surface_response(materials, (80.,), density_kg_m3=1.2, sound_speed_m_s=343.)
    assert response.notes_by_wall[Wall.X0][0] is late_note
    for wall, items in response.notes_by_wall.items():
        wall_surfaces.merge_construction_notes(actual, wall, items)
    assert actual[Wall.X0][0].frequencies_hz == (20., 40., 80.)


def test_saved_pair_notes_read_back_without_a_result_version_change(tmp_path: Path, result: SchemeResult) -> None:
    from aosr.reporting.evaluation import ResultStanding
    from aosr.gui.result_view import _construction_notes
    from tests.engine.test_result_reload_compatibility import load_document

    pair = result.pairs[0]
    with_notes = pair.model_copy(update={"report": pair.report.model_copy(update={
        "construction_notes_by_wall": {"x0": (note((20., 40.)),)}})})
    saved = result.model_copy(update={"pairs": (with_notes, *result.pairs[1:])})
    loaded = load_document(tmp_path, saved.model_dump(mode="json"), result.physics_identity)
    assert loaded.result.schema_version == result.schema_version
    assert loaded.standing is ResultStanding.CURRENT
    assert loaded.result.pairs[0].report.construction_notes_by_wall == with_notes.report.construction_notes_by_wall
    assert _construction_notes(loaded.result) == ("x 起點牆第 1 層：20–40 Hz 超出材料模型原研究的量測範圍，仍以外推照算",)


def test_formal_report_merges_three_lane_returns_without_recalculation(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from aosr.physics import report_io, report_output, three_lane_report as report, three_lane_report_batch as batch
    from aosr.physics import three_lane_report_geometry as geometry
    from tests.engine.test_report_io_contract import _fake_fem_energy, _input_document, _table
    from tests.engine._directivity import DIRECTIVITY

    real_fem, real_late, real_geometry = batch._prepare_fem_impedances, batch._prepare_late_family, geometry._solve_both_geometric_report_lanes
    def fem(*args: object, **kwargs: object) -> object:
        collector = kwargs["notes_by_wall"]
        assert isinstance(collector, dict)
        collector[Wall.X0] = (note((20.,)),)
        return real_fem(*args, **kwargs)  # type: ignore[arg-type]  # expires=2026-12-08 reason=包裝原入口，保留當次所有參數
    def late(**kwargs: object) -> object:
        surfaces, t60, fs, energy, decay = real_late(**kwargs)  # type: ignore[arg-type]  # expires=2026-12-08 reason=同上
        return replace(surfaces, notes_by_wall={Wall.X0: (note((40.,)),)}), t60, fs, energy, decay
    def geometric(**kwargs: object) -> object:
        collector = kwargs["notes_by_wall"]
        assert isinstance(collector, dict)
        wall_surfaces.merge_construction_notes(collector, Wall.X0, (note((80.,)),))
        return real_geometry(**kwargs)  # type: ignore[arg-type]  # expires=2026-12-08 reason=同上
    monkeypatch.setattr(batch, "_prepare_fem_impedances", fem)
    monkeypatch.setattr(batch, "_prepare_late_family", late)
    monkeypatch.setattr(geometry, "_solve_both_geometric_report_lanes", geometric)
    monkeypatch.setattr(report, "_solve_fem_energy", _fake_fem_energy)
    inputs = report_io.load_input_document(_input_document(), _table(), DIRECTIVITY)
    solved = report_io.solver_inputs(inputs)
    raw = report.solve_three_lane_report(source_model=solved.source_model, room=solved.room,
        source=solved.source, receiver=solved.receiver, sound_speed_m_s=solved.sound_speed_m_s,
        density_kg_m3=solved.density_kg_m3, impedance_by_wall=solved.impedance_by_wall,
        materials_by_wall=solved.materials_by_wall)
    output = report_output.output_from_report(raw, inputs=inputs, with_points=False)
    assert output.construction_notes_by_wall == {"x0": (note((20., 40., 80.)),)}


@pytest.mark.parametrize("low_flow", (False, True))
def test_two_sided_extrapolation_is_written_as_two_spans(low_flow: bool) -> None:
    """流阻夠低、頻軸夠寬時低高兩端都外推：寫成兩段，中間照常算的頻率不連進去。"""
    import jax.numpy as jnp
    from aosr.config.construction_environment import construction_environment
    from aosr.materials import construction
    from aosr.materials.construction_models import Assembly, PorousLayer

    _air, ranges = construction_environment(density_kg_m3=1.2, sound_speed_m_s=343.)
    limits = ranges.komatsu
    sigma = 4000. if low_flow else (limits.sample_flow_resistivity_min_pa_s_m2 + limits.sample_flow_resistivity_max_pa_s_m2) / 2
    lower, upper = (bound * sigma for bound in limits.f_over_sigma_bounds_m3_kg)
    axis = (lower / 2, lower * 2, upper / 2, upper * 2)
    layer = PorousLayer(thickness_m=.05, flow_resistivity_pa_s_m2=sigma, material_class="fibrous")
    notes = construction._komatsu_notes(Assembly(layers=(layer,), backing="rigid"),
                                                   jnp.asarray(axis), ranges)
    extrapolation = next(note for note in notes if note.code == "komatsu_frequency_extrapolation")
    low, high = extrapolation.frequencies_hz  # 考卷行程沒開雙精度，頻點是單精度：比相對誤差
    assert (low, high) == pytest.approx((axis[0], axis[-1]), rel=1e-6)
    assert wall_surfaces.extrapolated_frequency_text(extrapolation.frequencies_hz) == f"{low:g} Hz、{high:g} Hz"
    wide = (axis[0] / 2, axis[0], axis[-1], axis[-1] * 2)
    assert wall_surfaces.extrapolated_frequency_text(wide) == f"{wide[0]:g}–{wide[1]:g} Hz、{wide[2]:g}–{wide[3]:g} Hz"


def test_each_pair_keeps_only_its_own_reflection_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    """鏡像法每對各算各的：一對拿到的註記不准漏進另一對。"""
    from aosr.config.frequency_axis import low_frequency_axis_frequencies
    from aosr.geometry.shoebox import Point
    from aosr.physics import report_io, three_lane_report_batch as batch, three_lane_report_geometry as geometry
    from tests.engine.test_report_io_contract import _input_document, _table
    from tests.engine._directivity import DIRECTIVITY

    solved = report_io.solver_inputs(report_io.load_input_document(_input_document(), _table(), DIRECTIVITY))
    near = solved.receiver
    far = Point(solved.room.Lx - near.x, near.y, near.z)
    frequency = {near: 30., far: 60.}
    real_geometry = geometry._solve_both_geometric_report_lanes
    def geometric(**kwargs: object) -> object:
        collector, receiver = kwargs["notes_by_wall"], kwargs["receiver"]
        assert isinstance(collector, dict) and isinstance(receiver, Point)
        wall_surfaces.merge_construction_notes(collector, Wall.X0, (note((frequency[receiver],)),))
        return real_geometry(**kwargs)  # type: ignore[arg-type]  # expires=2026-12-08 reason=包裝原入口，保留當次所有參數
    monkeypatch.setattr(geometry, "_solve_both_geometric_report_lanes", geometric)
    fem_axis, _report_axis = low_frequency_axis_frequencies(solved.low_frequency_axis)
    reports = batch.solve_reports(source_model=solved.source_model, room=solved.room,
        sources={"s": solved.source}, receivers={"near": near, "far": far},
        sound_speed_m_s=solved.sound_speed_m_s, density_kg_m3=solved.density_kg_m3,
        impedance_by_wall=solved.impedance_by_wall, materials_by_wall=solved.materials_by_wall,
        scattering_by_wall=solved.scattering_by_wall, capability=None,
        reflection_order_k=solved.reflection_order_k, low_frequency_axis=solved.low_frequency_axis,
        batch_fem=True, fem_energies={("s", name): (1.,) * len(fem_axis) for name in ("near", "far")})
    for name, receiver in (("near", near), ("far", far)):
        assert reports[("s", name)].construction_notes_by_wall == {"x0": (note((frequency[receiver],)),)}

def test_scattering_rows_collect_their_own_notes(monkeypatch: pytest.MonkeyPatch) -> None:
    """散射權重那一支（垂直入射）自己算的註記也要收，不靠路徑那支剛好算到同一條軸。"""
    import numpy as np
    from aosr.materials import construction
    from aosr.materials.construction_models import Assembly, BoardLayer
    from aosr.physics.wall_reflections import wall_reflection_sources
    from tests.engine._material_flow_cases import fixed_materials

    assembly = Assembly(layers=(BoardLayer(surface_mass_kg_m2=.15, resistance_pa_s_m=80.),), backing="rigid")
    materials = fixed_materials({wall: 1646.4 for wall in Wall.all()}) | {Wall.X0: assembly}
    scattering_note = note((20.,))
    monkeypatch.setattr(construction, "surface_impedance",
                        lambda *args, **kwargs: (np.array([1000.+2.j]), (scattering_note,)))
    collected: dict[Wall, tuple[ConstructionNote, ...]] = {}
    sources = wall_reflection_sources(materials, (20.,), density_kg_m3=1.2, sound_speed_m_s=343.,
                                      notes_by_wall=collected)
    assert sources.normal_rows()["x0"] == (1000.+2.j,)
    assert collected == {Wall.X0: (scattering_note,)}
