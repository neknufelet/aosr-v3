"""家具近似只由路徑表表頭發出，並沿既有評分聯集傳遞。"""
from __future__ import annotations

import pytest
from dataclasses import replace

from aosr.config.paths import config_path
from aosr.scoring import reflections as reflection_evaluator
from aosr.physics.report_io import ReportOutput, PathTableSection
from aosr.scoring.contract import (
    CategoryEvaluation, ChannelMatchingPayload, EvaluationState, Flag,
    ListeningAreaStabilityPayload, QualityCategory, ReasonCode, TimbrePayload,
)
from aosr.scoring.listening_area_channels import evaluate_listening_area_channels
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from aosr.scoring.timbre import timbre_input_from_report
from tests.engine import test_reflections as reflections
from tests.engine import test_timbre_channels as timbres
from tests.engine import test_listening_area as area
from tests.engine import test_listening_area_channels as areas
from tests.engine import test_channel_matching as matching


def furniture_report(report: ReportOutput, furniture_id: str = "desk") -> ReportOutput:
    """手組已含家具模型的表頭；不拆產品輸入關、不改物理計算。"""
    assert report.path_table is not None
    document = report.path_table.model_dump(mode="python")
    document.update(furniture_ids=(furniture_id,),
                    furniture_model="single_bounce_finite_size_v1", blocked_wall_paths=(),
                    furniture_materials=({"furniture_id": furniture_id, "material": "wood",
                                          "unknown_bands_hz": (63.0, 8000.0)},))
    table = PathTableSection.model_validate(document)
    return report.model_copy(update={"path_table": table})


@pytest.mark.parametrize("kind", ["furniture", "empty", "no_table"])
def test_timbre_report_flags_use_only_path_table_header(kind: str) -> None:
    record = reflections._pair()[0]
    report = furniture_report(record.report) if kind == "furniture" else record.report
    if kind == "no_table":
        report = report.model_copy(update={"path_table": None})
    data = timbre_input_from_report(report, candidate_id="candidate-a", speaker_id="left",
        receiver_id="main", source_reference="fixture", provenance=timbres._single("left").provenance)
    assert tuple(flag.value for flag in data.report_flags) == (
        ("furniture_model_approximate",) if kind == "furniture" else ())


def test_unknown_furniture_model_is_rejected() -> None:
    from aosr.scoring.furniture_model_state import furniture_model_state
    report = reflections._pair()[0].report
    assert report.path_table is not None
    unknown = report.path_table.model_copy(update={"furniture_model": "unknown-model"})
    with pytest.raises(ValueError, match="未知家具模型"):
        furniture_model_state(unknown)


@pytest.mark.parametrize("measured", [True, False])
def test_reflection_flag_reads_header_only_for_measured_channels(measured: bool) -> None:
    records = reflections._pair()
    payload = reflections._evaluate(records).payload if measured else None
    assert payload is None or isinstance(payload, ReflectionsAndEchoPayload)
    changed = tuple(replace(item, report=furniture_report(item.report)) for item in records)
    settings = reflection_evaluator._settings(config_path("quality_targets.toml"),
        "dedicated_two_channel_listening_room")
    flags = reflection_evaluator._flags(changed, changed[0], settings, payload)
    assert (Flag.FURNITURE_MODEL_APPROXIMATE in flags) is measured


def test_existing_unions_propagate_approximation_without_changing_support() -> None:
    flag = Flag.FURNITURE_MODEL_APPROXIMATE
    singles = {role: timbres._single(role).model_copy(update={"flags": (flag,)})
               for role in ("left", "right")}
    assert flag in timbres._aggregate(timbres._group(), singles).flags
    receivers = area._receiver_set()
    records = area._results(receivers)
    changed = tuple(item.model_copy(update={"timbre_evaluation":
        item.timbre_evaluation.model_copy(update={"flags": (flag,)})}) for item in records)
    baseline, approximate = area._evaluate(receivers, records), area._evaluate(receivers, changed)
    assert flag in approximate.flags
    assert baseline.settings_fingerprint == approximate.settings_fingerprint
    assert isinstance(baseline.payload, ListeningAreaStabilityPayload)
    assert isinstance(approximate.payload, ListeningAreaStabilityPayload)
    assert baseline.payload.frequency_support == approximate.payload.frequency_support
    channels = {role: areas._channel(role).model_copy(update={"flags": (flag,)})
                for role in ("left", "right")}
    aggregate = evaluate_listening_area_channels(areas._group(), areas._receivers(), channels,
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a")
    assert flag in aggregate.flags
    receivers, group = matching._receivers(), matching._group()
    points = tuple(matching._point(receivers, group, point.receiver_id) for point in receivers.points)
    changed_points = tuple(point.model_copy(update={"responses": tuple(
        response.model_copy(update={"timbre_evaluation": response.timbre_evaluation.model_copy(
            update={"flags": (flag,)})}) for response in point.responses)}) for point in points)
    baseline = matching._evaluate(receivers, group, points)
    approximate = matching._evaluate(receivers, group, changed_points)
    assert flag in approximate.flags
    assert baseline.settings_fingerprint == approximate.settings_fingerprint
    assert isinstance(baseline.payload, ChannelMatchingPayload)
    assert isinstance(approximate.payload, ChannelMatchingPayload)
    assert baseline.payload.broadband_support == approximate.payload.broadband_support
    for single in singles.values():
        assert isinstance(single.payload, TimbrePayload)
        assert all(flag not in feature.flags for feature in single.payload.features)


@pytest.mark.parametrize("category", [QualityCategory.REVERBERATION, QualityCategory.LOW_FREQUENCY_DECAY])
def test_non_geometry_categories_reject_furniture_approximation(category: QualityCategory) -> None:
    document = timbres._single("left").model_dump(mode="python")
    document.update(category=category, source_model_fingerprint=None,
        state=EvaluationState.UNAVAILABLE, payload=None, raw_quantities=(),
        reason_codes=(ReasonCode.SOLVER_UNAVAILABLE,), flags=("furniture_model_approximate",))
    with pytest.raises(ValueError, match="家具近似"):
        CategoryEvaluation.model_validate(document)
