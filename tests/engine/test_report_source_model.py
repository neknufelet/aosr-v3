"""#505 第二刀：聲源模型的輸入、輸出與全向逐位控制組。"""

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
import numpy as np
from pydantic import TypeAdapter, ValidationError

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
    SourceCurveParameters,
    SourceModelInput,
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
    assert section.parameters is not None and spec.parameters is not None
    assert section.parameters.model_dump() == spec.parameters.model_dump()
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


@pytest.mark.parametrize(("invalid", "where"), [
    ({"kind": SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1.value}, "parameters"),
    ({**_analytic_input(), "aim_m": {"x": 1.0, "y": 2.0, "z": 3.0, "extra": 4.0}}, "aim_m"),
    ({**_analytic_input(), "parameters": {"beta_limit": 3.2}}, "parameters"),
])
def test_analytic_input_needs_all_parameters_and_exact_aim_shape(invalid: dict[str, object], where: str) -> None:
    """整份輸入裡的壞形狀在形狀那一關就擋、訊息指名那一格；不是靠能力表那一關（那一關訊息帶「第四刀」）。"""
    with pytest.raises(ValueError, match="source_model") as caught:
        report_io.load_input_document(_document(invalid), _table())
    assert where in str(caught.value)
    assert "第四刀" not in str(caught.value)


@pytest.mark.parametrize(("change", "where"), [
    ({"aim_m": {"x": 1.0, "y": 2.0, "z": 3.0, "extra": 4.0}}, "aim_m"),
    ({"aim_m": {"x": 1.0, "y": 2.0}}, "aim_m"),
    ({"aim_m": [1.0, 2.0, 3.0]}, "aim_m"),
    ({"aim_m": {"x": True, "y": 2.0, "z": 3.0}}, "aim_m"),
    ({"aim_m": {"x": float("nan"), "y": 2.0, "z": 3.0}}, "aim_m"),
    ({"aim_m": {"x": 1.0, "y": float("inf"), "z": 3.0}}, "aim_m"),
    ({"parameters": {"beta_limit": 3.2}}, "parameters"),
    ({"parameters": {**_parameters().model_dump(), "beta_limit": True}}, "parameters"),
])
def test_analytic_shape_is_checked_on_its_own_not_only_by_the_capability_gate(
    change: dict[str, object], where: str,
) -> None:
    """這一刀整份報表輸入一律拒收解析近似，所以形狀要單獨驗：合法的收下、每一種壞形狀各自拒收、訊息指名那一格。"""
    adapter: TypeAdapter[object] = TypeAdapter(SourceModelInput)
    accepted = adapter.validate_python(_analytic_input())
    assert isinstance(accepted, AnalyticAxisymmetricInput)
    assert accepted.aim_m == Point(4.35, 2.45, 1.05)
    with pytest.raises(ValidationError, match=where):
        adapter.validate_python({**_analytic_input(), **change})


def test_analytic_rejection_reads_its_own_row_not_any_unsupported_row() -> None:
    """拒收只看解析近似那一列：那一列改成試驗中時，不准借用別列（實測）的 unsupported 理由。"""
    table = _table()
    entry = table.for_entry("source_directivity")
    marker = "實測那一列的暫存理由"
    own_marker = "解析近似那一列改成試驗中後的暫存理由"
    rows = []
    for item in entry.capability:
        if item.materials == SourceModel.TWO_PARAMETER.value:
            rows.append(item.model_copy(update={"status": "experimental", "note": own_marker}))
        else:
            rows.append(item.model_copy(update={"note": marker}))
    assert any(item.status == "unsupported" for item in rows)
    changed = table.model_copy(update={"entry": tuple(
        entry.model_copy(update={"capability": tuple(rows)}) if item is entry else item
        for item in table.entry
    )})
    with pytest.raises(ValueError, match="第四刀才接上計算") as caught:
        report_io.load_input_document(_document(_analytic_input()), changed)
    assert marker not in str(caught.value)
    assert own_marker not in str(caught.value)


def test_spec_refuses_an_unregistered_kind_string() -> None:
    """種類要是登記過的列舉成員；同字串的 str 也不收（StrEnum 跟字串比較會相等，所以要看型別）。"""
    with pytest.raises(ValueError, match="已登記"):
        SourceModelSpec(cast(SourceModelKind, SourceModelKind.OMNIDIRECTIONAL.value))


def test_analytic_section_needs_the_source_and_points_its_axis_at_the_aim() -> None:
    """解析近似的輸出區段：沒有聲源位置不能算軸線；有的話軸線就是單位化（對準點 − 聲源），對 numpy 另算。"""
    spec = _analytic_spec()
    with pytest.raises(ValueError, match="聲源位置"):
        SourceModelSection.from_spec(spec, None)
    source = Point(1.15, 0.95, 1.2)
    section = SourceModelSection.from_spec(spec, source)
    delta = np.subtract((4.35, 2.45, 1.05), (1.15, 0.95, 1.2))
    assert section.axis_unit_vector is not None
    np.testing.assert_allclose(section.axis_unit_vector, delta / np.linalg.norm(delta), rtol=0, atol=1e-15)


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
    declared = inspect.signature(function).parameters["source_model"]
    assert declared.default is inspect.Parameter.empty
    assert declared.kind is inspect.Parameter.KEYWORD_ONLY
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
    scene_model = output.scene.source_model
    assert scene_model.kind is SourceModelKind.OMNIDIRECTIONAL
    assert (scene_model.model_version, scene_model.parameters, scene_model.aim_m, scene_model.axis_unit_vector) == (
        None, None, None, None)
    assert "全向點聲源" in scene_model.verification_status
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


def test_status_sentences_carry_their_meaning_in_words() -> None:
    """狀態句的內容用字面詞比，不拿常數比常數：解析近似那句三個詞都要在，全向那句說明是全向點聲源。"""
    analytic = SourceModelSection.from_spec(_analytic_spec(), Point(1.15, 0.95, 1.2)).verification_status
    for phrase in ("水平面擬合", "上下方向沿用同一條曲線", "尚未獨立驗證"):
        assert phrase in analytic
    omni = SourceModelSection.from_spec(SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL), None).verification_status
    assert "全向點聲源" in omni
    assert "驗證" not in omni


_DETAILS = ("model_version", "parameters", "aim_m", "axis_unit_vector")


def _sections() -> tuple[dict[str, object], dict[str, object]]:
    omni = SourceModelSection.from_spec(SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL), None).model_dump()
    analytic = SourceModelSection.from_spec(_analytic_spec(), Point(1.15, 0.95, 1.2)).model_dump()
    return omni, analytic


@pytest.mark.parametrize("field", _DETAILS)
def test_omnidirectional_section_refuses_each_detail(field: str) -> None:
    omni, analytic = _sections()
    assert analytic[field] is not None
    with pytest.raises(ValidationError, match="全向"):
        SourceModelSection.model_validate({**omni, field: analytic[field]})


@pytest.mark.parametrize("field", _DETAILS)
def test_analytic_section_needs_each_detail(field: str) -> None:
    _omni, analytic = _sections()
    with pytest.raises(ValidationError, match="解析近似"):
        SourceModelSection.model_validate({**analytic, field: None})


def test_section_status_and_version_must_match_the_kind() -> None:
    omni, analytic = _sections()
    with pytest.raises(ValidationError, match="全向"):
        SourceModelSection.model_validate({**omni, "verification_status": analytic["verification_status"]})
    with pytest.raises(ValidationError, match="解析近似"):
        SourceModelSection.model_validate({**analytic, "verification_status": omni["verification_status"]})
    with pytest.raises(ValidationError, match="解析近似"):
        SourceModelSection.model_validate({**analytic, "model_version": "v0"})
    with pytest.raises(ValueError, match="全向"):
        SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL, parameters=_parameters())


@pytest.mark.parametrize("change", [
    {"beta_corner_hz": 0.0}, {"power_floor_exponent": 0.0}, {"beta_limit": True}, {"unlisted": 1.0},
])
def test_curve_parameters_follow_the_registry_rules(change: dict[str, object]) -> None:
    """報表那一側的六個標量逐欄帶四件事，值的規則跟登記簿同一套（交給 TwoParameterCurve 驗）。"""
    good = _parameters().model_dump()
    assert SourceCurveParameters.model_validate(good).to_curve() == _parameters()
    with pytest.raises(ValidationError):
        SourceCurveParameters.model_validate({**good, **change})

