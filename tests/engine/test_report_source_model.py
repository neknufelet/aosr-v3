"""#505 第二刀：聲源模型的輸入、輸出與全向逐位控制組。"""

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.directivity_defaults import TwoParameterCurve
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point
from aosr.physics import (
    geometric_lane,
    reflection_window,
    report_io,
    report_output,
    report_path_table,
    three_lane_report,
    three_lane_report_batch,
)
from aosr.physics.report_source import (
    ANALYTIC_STATUS,
    AnalyticAxisymmetricInput,
    SourceModelKind,
    SourceModelSection,
    SourceModelSpec,
)
from aosr.physics.source_directivity import SourceModel
from tests.engine import _source_model_control as control


def _table() -> CapabilityTable:
    return load_capabilities(config_path("capabilities.toml"))


def _document(source_model: object = None) -> dict[str, object]:
    scene = dict(control.SCENE_WITHOUT_SOURCE_MODEL)
    scene["source_model"] = source_model if source_model is not None else {"kind": "omnidirectional"}
    return scene


def _parameters() -> TwoParameterCurve:
    return TwoParameterCurve(
        beta_limit=3.2, beta_corner_hz=4000.0, beta_exponent=0.67,
        power_floor_limit_db=-45.0, power_floor_corner_hz=2550.0,
        power_floor_exponent=1.12,
    )


def _analytic_input() -> dict[str, object]:
    return {
        "kind": SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value,
        "parameters": _parameters().model_dump(),
        "aim_m": {"x": 4.35, "y": 2.45, "z": 1.05},
    }


def _analytic_spec() -> SourceModelSpec:
    return SourceModelSpec(
        SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1,
        parameters=_parameters(), aim=Point(4.35, 2.45, 1.05),
    )


def test_report_kinds_reuse_physics_model_values() -> None:
    assert SourceModelKind.OMNIDIRECTIONAL.value == SourceModel.OMNIDIRECTIONAL.value
    assert (
        SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value
        == SourceModel.TWO_PARAMETER.value
    )


def test_omnidirectional_spec_and_section_are_explicit() -> None:
    spec = SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL)
    assert spec.parameters is None and spec.aim is None
    section = SourceModelSection.from_spec(spec, source=None)
    assert section.parameters is None
    assert section.aim_m is None
    assert section.axis_unit_vector is None
    assert section.model_version is None
    with pytest.raises(ValueError):
        SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL, aim=Point(1.0, 2.0, 3.0))
    with pytest.raises(ValidationError):
        SourceModelSection.model_validate({**section.model_dump(), "model_version": "wrong"})


@pytest.mark.parametrize("bad", [
    {"kind": "unknown"},
    {"kind": "omnidirectional", "parameters": {}},
    {"kind": "omnidirectional", "extra": 1},
])
def test_source_model_rejects_unknown_or_extra_fields(bad: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="source_model"):
        report_io.load_input_document(_document(bad), _table())


def test_source_model_is_required() -> None:
    document = _document()
    del document["source_model"]
    with pytest.raises(ValueError, match="source_model"):
        report_io.load_input_document(document, _table())


def test_analytic_input_rejection_reads_capability_note() -> None:
    table = _table()
    entry = table.for_entry("source_directivity")
    original = next(item for item in entry.capability if item.materials == SourceModel.TWO_PARAMETER.value)
    marker = "暫存能力表指定的拒收理由"
    modified = entry.model_copy(update={"capability": tuple(
        item.model_copy(update={"note": marker}) if item is original else item
        for item in entry.capability
    )})
    changed = table.model_copy(update={"entry": tuple(
        modified if item is entry else item for item in table.entry
    )})
    with pytest.raises(ValueError, match=marker):
        report_io.load_input_document(_document(_analytic_input()), changed)
    assert marker not in original.note


def test_source_model_section_accepts_both_shapes_and_rejects_mixed_shapes() -> None:
    spec = _analytic_spec()
    section = SourceModelSection.from_spec(spec, Point(1.15, 0.95, 1.2))
    assert section.kind == spec.kind
    assert section.parameters == spec.parameters
    assert section.aim_m == spec.aim
    assert section.axis_unit_vector is not None
    assert section.verification_status == ANALYTIC_STATUS
    with pytest.raises(ValidationError):
        SourceModelSection.model_validate({**section.model_dump(), "axis_unit_vector": None})
    with pytest.raises(ValidationError):
        SourceModelSection.model_validate({**section.model_dump(), "verification_status": "wrong"})
    with pytest.raises(ValueError):
        SourceModelSpec(spec.kind)
    with pytest.raises(ValueError):
        SourceModelSpec(spec.kind, parameters=_parameters())
    with pytest.raises(ValueError):
        SourceModelSpec(spec.kind, aim=Point(4.35, 2.45, 1.05))


@pytest.mark.parametrize("invalid", [
    {"kind": SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value},
    {**_analytic_input(), "aim_m": {"x": 1.0, "y": 2.0, "z": 3.0, "extra": 4.0}},
    {**_analytic_input(), "parameters": {"beta_limit": 3.2}},
])
def test_analytic_input_needs_all_parameters_and_exact_aim_shape(invalid: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="source_model"):
        report_io.load_input_document(_document(invalid), _table())


def test_omnidirectional_path_table_rejects_departure_angle() -> None:
    from aosr.physics.report_path_output import PathDirectionAngles, PathRow, PathTableSection

    row = PathRow(
        order=0, wall_sequence=(), delay_s=0.01, distance_m=3.43,
        direction_vector=(1.0, 0.0, 0.0),
        direction_angles=PathDirectionAngles(azimuth_deg=0.0, elevation_deg=0.0),
        departure_off_axis_deg=30.0, relative_direct_energy=(1.0,),
    )
    with pytest.raises(ValidationError, match="departure_off_axis_deg"):
        PathTableSection(
            reflection_order_k=3, frequencies_hz=(100.0,), scattering_coefficient=(0.0,),
            source_model_kind=SourceModelKind.OMNIDIRECTIONAL, rows=(row,),
        )
    analytic = PathTableSection(
        reflection_order_k=3, frequencies_hz=(100.0,), scattering_coefficient=(0.0,),
        source_model_kind=SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1,
        rows=(row,),
    )
    assert analytic.rows[0].departure_off_axis_deg == 30.0
    with pytest.raises(ValidationError, match="departure_off_axis_deg"):
        PathTableSection.model_validate({
            **analytic.model_dump(),
            "rows": ({**row.model_dump(), "departure_off_axis_deg": None},),
        })


@pytest.mark.parametrize("function", [
    geometric_lane.solve_geometric_early_lane,
    geometric_lane.solve_geometric_lane,
    three_lane_report.solve_three_lane_report,
    three_lane_report.solve_three_lane_reports,
    three_lane_report._solve_both_geometric_report_lanes,
    three_lane_report_batch.solve_reports,
    report_path_table.build_path_table,
])
def test_each_raw_physics_entrance_rejects_analytic_model(
    function: Callable[..., object],
) -> None:
    kwargs: dict[str, object] = {
        name: None for name, parameter in inspect.signature(function).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    kwargs["source_model"] = _analytic_spec()
    with pytest.raises(ValueError, match="analytic_axisymmetric_two_parameter_v1.*第四刀才接上"):
        function(**kwargs)


def test_input_owned_physics_entrances_reject_analytic_model() -> None:
    inputs = report_io.load_input_document(_document(), _table())
    analytic = AnalyticAxisymmetricInput.model_validate(_analytic_input())
    changed = inputs.model_copy(update={"source_model": analytic})
    with pytest.raises(ValueError, match="第四刀才接上"):
        reflection_window.build_reflection_window(
            changed, frequencies_hz=(100.0,), scattering_coefficient=(0.0,), window_s=0.01,
        )
    solved = report_io.solver_inputs(inputs)._replace(source_model=_analytic_spec())
    report = SimpleNamespace(geometric_lane=SimpleNamespace(frequencies_hz=(), scattering=()))
    with pytest.raises(ValueError, match="第四刀才接上"):
        report_path_table.build_path_table_section(report, solved)


def test_output_rejects_source_model_mismatch() -> None:
    inputs = report_io.load_input_document(_document(), _table())
    report = object.__new__(three_lane_report.ThreeLaneReport)
    object.__setattr__(report, "reflection_order_k", inputs.reflection_order_k)
    object.__setattr__(report, "low_frequency_axis", inputs.low_frequency_axis)
    object.__setattr__(report, "source_model", _analytic_spec())
    with pytest.raises(ValueError, match="source_model"):
        report_output.output_from_report(report, inputs=inputs, with_points=False)


def test_dense_and_fine_axes_reject_different_source_models() -> None:
    omni = SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL)
    fine = geometric_lane.GeometricLaneResult(
        frequencies_hz=(), direct_energy=(), reflected_energy=(), interference_energy=(),
        late_energy=(), scattering=(), geometric_energy=(), reflection_order_k=3,
        source_model=omni,
    )
    dense = geometric_lane.GeometricEarlyResult(
        frequencies_hz=(), direct_energy=(), reflected_energy=(), interference_energy=(),
        scattering=(), reflection_order_k=3, source_model=_analytic_spec(),
    )
    with pytest.raises(ValueError, match="source_model"):
        geometric_lane.average_geometric_lane_to_bands_with_dense_early(
            fine, dense, reflection_order_k=3,
        )


def _compare_answer_fields(actual: dict[str, object], expected: dict[str, object]) -> int:
    checked = 0
    for name, value in expected.items():
        cell = actual[name]
        if isinstance(value, str) and value.startswith(("0x", "-0x")):
            assert isinstance(cell, float), name
            assert cell.hex() == value, name
            checked += 1
        else:
            assert cell == value, name
    return checked


def test_omnidirectional_report_matches_frozen_float_hex_control(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """控制組只取舊聲學欄；新增模型欄與按設計改變的場景指紋不重錄。"""
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", control.fake_fem_energy)
    monkeypatch.setattr(three_lane_report, "_solve_report_late_decay", control.fast_late_decay)
    inputs = report_io.load_input_document(_document(), _table())
    solved = report_io.solver_inputs(inputs)
    report = three_lane_report.solve_three_lane_report(**solved._asdict())
    assert report.source_model == solved.source_model
    assert report.geometric_lane.source_model == solved.source_model
    output = report_output.output_from_report(
        report, inputs=inputs, with_points=True, path_table_inputs=solved,
    )
    assert output.scene.source_model == SourceModelSection.from_spec(solved.source_model, None)
    assert output.path_table is not None
    assert output.path_table.source_model_kind is SourceModelKind.OMNIDIRECTIONAL
    answers = control.ANSWERS
    points = output.points
    path_table = output.path_table
    assert points is not None and path_table is not None
    assert len(points) == answers["point_count"]
    checked = _compare_answer_fields(
        output.top.model_dump(), cast(dict[str, object], answers["top"]),
    )
    for index, fields in cast(dict[int, dict[str, object]], answers["points"]).items():
        checked += _compare_answer_fields(points[index].model_dump(), fields)
    for actual, fields in zip(output.bands, cast(list[dict[str, object]], answers["bands"]), strict=True):
        checked += _compare_answer_fields(actual.model_dump(), fields)
    for index, fields in cast(dict[int, dict[str, object]], answers["path_rows"]).items():
        row = path_table.rows[index]
        assert row.order == fields["order"]
        for frequency_index, expected in cast(dict[int, str], fields["relative_direct_energy"]).items():
            assert frequency_index in control.PATH_FREQ_INDEXES
            assert row.relative_direct_energy[frequency_index].hex() == expected
            checked += 1
    assert checked > 0
