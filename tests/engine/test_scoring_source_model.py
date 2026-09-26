"""#505 第三刀：聲源模型身分、旗標、評估與同表條件的獨立考卷。"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

import pytest
from pydantic import ValidationError

from aosr.config.directivity_defaults import TwoParameterCurve
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point
from aosr.physics.report_io import PathTableSection
from aosr.physics.report_source import SourceModelKind, SourceModelSection, SourceModelSpec
from aosr.physics.source_directivity import MODEL_VERSION
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CandidateEvaluation, CategoryEvaluation, ChannelMatchingPayload,
    EvaluationState, Flag, ModelValidationStatus, QualityCategory, ReasonCode,
)
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections
from aosr.scoring.reflections_contract import REFLECTIONS_AND_ECHO_EVALUATOR_VERSION, ReflectionsAndEchoPayload
from aosr.scoring.source_model_identity import (
    REFLECTION_DIRECTIVITY_FLAGS, SOURCE_MODEL_EXCLUDED_FIELDS,
    SOURCE_MODEL_FINGERPRINT_FIELDS, TIMBRE_DIRECTIVITY_FLAGS,
    reflection_directivity_flags, source_model_fingerprint, timbre_directivity_flags,
)
from aosr.scoring.timbre import TIMBRE_EVALUATOR_VERSION, evaluate_timbre, timbre_input_from_report
from aosr.scoring.timbre_channels import TIMBRE_CHANNELS_EVALUATOR_VERSION
from aosr.scoring.listening_area import LISTENING_AREA_EVALUATOR_VERSION
from aosr.scoring.channel_matching import CHANNEL_MATCHING_EVALUATOR_VERSION, ChannelPointInput
from aosr.scoring.reverberation import REVERBERATION_EVALUATOR_VERSION
from tests.engine import _ranking_fixtures as ranking_fixtures
from tests.engine import _reflection_fixtures as reflection_fixtures
from tests.engine._source_model import OMNI_SOURCE_MODEL, OMNI_SOURCE_MODEL_FINGERPRINT


_ANALYTIC_JSON = (
    '{"kind":"analytic_axisymmetric_two_parameter_v1",'
    '"model_version":"analytic_axisymmetric_two_parameter_v1",'
    '"parameters":{"beta_corner_hz":4100.0,"beta_exponent":0.7,'
    '"beta_limit":3.4,"power_floor_corner_hz":2400.0,'
    '"power_floor_exponent":1.2,"power_floor_limit_db":-43.0}}'
)
_OMNI_JSON = '{"kind":"omnidirectional","model_version":null,"parameters":null}'


def _analytic(aim: Point = Point(4.0, 2.0, 1.2)) -> SourceModelSection:
    curve = TwoParameterCurve(
        beta_limit=3.4, beta_corner_hz=4100.0, beta_exponent=0.7,
        power_floor_limit_db=-43.0, power_floor_corner_hz=2400.0,
        power_floor_exponent=1.2,
    )
    return SourceModelSection.from_spec(
        SourceModelSpec(SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1,
                        curve, aim), Point(1.0, 1.0, 1.2)
    )


def _analytic_record(record: ReflectionInput, section: SourceModelSection | None = None) -> ReflectionInput:
    """只手造報表身分，不重算場景指紋；評估器本來就只讀報表指紋。"""
    section = section or _analytic()
    scene = record.report.scene.model_copy(update={"source_model": section})
    table = record.report.path_table
    assert table is not None and record.window is not None
    document = table.model_dump(mode="python")
    document["source_model_kind"] = section.kind
    for row in document["rows"]:
        row["departure_off_axis_deg"] = 35.0
    changed_table = PathTableSection.model_validate(document)
    report = record.report.model_copy(update={"scene": scene, "path_table": changed_table})
    window = record.window.model_copy(update={"source_model_kind": section.kind})
    return replace(record, report=report, window=window)


def _reflection(records: Sequence[ReflectionInput]) -> CategoryEvaluation:
    return evaluate_reflections(
        reflection_fixtures._group(), records,
        primary_receiver_id="main", candidate_id="candidate-a",
        purpose="dedicated_two_channel_listening_room",
        quality_targets_path=config_path("quality_targets.toml"),
    )


def test_category_evaluation_requires_source_model_fingerprint() -> None:
    assert "source_model_fingerprint" in CategoryEvaluation.model_fields
    assert CategoryEvaluation.model_fields["source_model_fingerprint"].is_required()


def test_source_model_fingerprints_match_hand_written_json() -> None:
    assert source_model_fingerprint(OMNI_SOURCE_MODEL) == hashlib.sha256(_OMNI_JSON.encode()).hexdigest()
    assert source_model_fingerprint(_analytic()) == hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()


def test_source_model_identity_excludes_aim_axis_and_status() -> None:
    first = _analytic()
    moved = _analytic(Point(4.0, 3.0, 1.2))
    assert (first.aim_m, first.axis_unit_vector) != (moved.aim_m, moved.axis_unit_vector)
    assert source_model_fingerprint(first) == source_model_fingerprint(moved)
    assert SOURCE_MODEL_FINGERPRINT_FIELDS | SOURCE_MODEL_EXCLUDED_FIELDS == set(SourceModelSection.model_fields)
    assert not SOURCE_MODEL_FINGERPRINT_FIELDS & SOURCE_MODEL_EXCLUDED_FIELDS


@pytest.mark.parametrize("parameter", tuple(TwoParameterCurve.model_fields))
def test_each_curve_parameter_changes_source_model_fingerprint(parameter: str) -> None:
    first = _analytic()
    assert first.parameters is not None
    current = getattr(first.parameters, parameter)
    changed = first.model_copy(update={"parameters": first.parameters.model_copy(update={parameter: current + 0.1})})
    assert source_model_fingerprint(changed) != source_model_fingerprint(first)


def test_model_version_changes_source_model_fingerprint() -> None:
    first = _analytic()
    assert first.model_version == MODEL_VERSION
    changed = first.model_copy(update={"model_version": "next-model-version"})
    assert source_model_fingerprint(changed) != source_model_fingerprint(first)


def test_directivity_flag_tables_cover_every_kind_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert set(REFLECTION_DIRECTIVITY_FLAGS) == set(SourceModelKind)
    assert set(TIMBRE_DIRECTIVITY_FLAGS) == set(SourceModelKind)
    assert reflection_directivity_flags(SourceModelKind.OMNIDIRECTIONAL) == (Flag.NO_DIRECTIVITY,)
    assert timbre_directivity_flags(SourceModelKind.OMNIDIRECTIONAL) == ()
    assert reflection_directivity_flags(_analytic().kind) == (Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED,)
    assert timbre_directivity_flags(_analytic().kind) == (Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED,)
    # 拿掉的是非全向那一格：查不到時若偷偷退回全向那一格，這裡才分得出來。
    missing = SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1
    monkeypatch.delitem(REFLECTION_DIRECTIVITY_FLAGS, missing)
    monkeypatch.delitem(TIMBRE_DIRECTIVITY_FLAGS, missing)
    with pytest.raises(ValueError, match=missing.value):
        reflection_directivity_flags(missing)
    with pytest.raises(ValueError, match=missing.value):
        timbre_directivity_flags(missing)


def test_reflection_analytic_and_omni_report_identity_and_flags() -> None:
    omni_pair = reflection_fixtures._pair()
    analytic_pair = tuple(_analytic_record(item) for item in omni_pair)
    omni = _reflection(omni_pair)
    analytic = _reflection(analytic_pair)
    assert omni.state is EvaluationState.MEASURED
    assert analytic.state is EvaluationState.MEASURED
    assert isinstance(omni.payload, ReflectionsAndEchoPayload)
    assert isinstance(analytic.payload, ReflectionsAndEchoPayload)
    assert omni.payload.source_model_kind is SourceModelKind.OMNIDIRECTIONAL
    assert analytic.payload.source_model_kind is SourceModelKind.ANALYTIC_AXISYMMETRIC_TWO_PARAMETER_V1
    assert omni.source_model_fingerprint == OMNI_SOURCE_MODEL_FINGERPRINT
    assert analytic.source_model_fingerprint == hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()
    assert Flag.NO_DIRECTIVITY in omni.flags
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in analytic.flags
    assert Flag.NO_DIRECTIVITY not in analytic.flags
    assert analytic.settings_fingerprint == omni.settings_fingerprint


def test_reflection_unavailable_keeps_model_identity_and_flag() -> None:
    pair = tuple(_analytic_record(item) for item in reflection_fixtures._pair())
    wrong = replace(pair[1], speaker_id="wrong-speaker")
    result = _reflection((pair[0], wrong))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.source_model_fingerprint == hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in result.flags
    assert Flag.NO_DIRECTIVITY not in result.flags


def test_reflection_rejects_mixed_models_even_with_shared_scene_fingerprint() -> None:
    pair = reflection_fixtures._pair()
    result = _reflection((pair[0], _analytic_record(pair[1])))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.SOURCE_MODEL_MISMATCH,)
    # 不可估時的身分與旗標取自跟場景指紋同一份（主位左聲道，全向），不取另一份。
    assert result.source_model_fingerprint == OMNI_SOURCE_MODEL_FINGERPRINT
    assert Flag.NO_DIRECTIVITY in result.flags
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED not in result.flags


def test_reflection_rejects_same_kind_with_different_parameters() -> None:
    parameters = _analytic().parameters
    assert parameters is not None
    other = _analytic().model_copy(update={"parameters": parameters.model_copy(update={"beta_limit": 3.5})})
    pair = reflection_fixtures._pair()
    result = _reflection((_analytic_record(pair[0]), _analytic_record(pair[1], other)))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.SOURCE_MODEL_MISMATCH,)


def _around(role: str, source_y: float) -> ReflectionInput:
    return reflection_fixtures._record(role, source_y, "around", receiver_y=2.1)


def _other_scene(record: ReflectionInput) -> ReflectionInput:
    scene = record.report.scene.model_copy(update={"scene_fingerprint": "b" * 64})
    return replace(record, report=record.report.model_copy(update={"scene": scene}))


def test_reflection_surrounding_point_from_another_scene_is_recorded_on_that_point_only() -> None:
    """模型不同的真實報表，場景指紋一定也不同：周圍點那一支只記場景不一（#480），整類照算。"""
    pair = reflection_fixtures._pair()
    stray = _other_scene(_analytic_record(_around("left", 1.3)))
    result = _reflection((*pair, stray, _around("right", 2.5)))
    assert result.state is EvaluationState.MEASURED
    assert result.source_model_fingerprint == OMNI_SOURCE_MODEL_FINGERPRINT
    assert isinstance(result.payload, ReflectionsAndEchoPayload)
    reasons = {(item.role, item.receiver_id): item.reason_codes for item in result.payload.channels}
    assert reasons[("left", "around")] == (ReasonCode.SCENE_FINGERPRINT_MISMATCH,)
    assert reasons[("right", "around")] == ()


def test_reflection_model_check_covers_surrounding_points_and_survives_a_stray_point() -> None:
    pair = reflection_fixtures._pair()
    surrounding = _reflection((*pair, _analytic_record(_around("left", 1.3)), _around("right", 2.5)))
    assert surrounding.state is EvaluationState.UNAVAILABLE
    assert surrounding.reason_codes == (ReasonCode.SOURCE_MODEL_MISMATCH,)
    # 另一支周圍點場景不一，只把那一支記下；主位混用兩種聲源模型照樣要拒收。
    stray = (_other_scene(_around("left", 1.3)), _around("right", 2.5))
    mixed = _reflection((pair[0], _analytic_record(pair[1]), *stray))
    assert mixed.state is EvaluationState.UNAVAILABLE
    assert mixed.reason_codes == (ReasonCode.SOURCE_MODEL_MISMATCH,)


@pytest.mark.parametrize("part", ("table", "window", "scene"))
def test_reflection_rejects_internal_model_kind_mismatch(part: str) -> None:
    pair = reflection_fixtures._pair()
    first = _analytic_record(pair[0])
    if part == "scene":
        # 只有場景說是解析近似，路徑表與補算窗仍是全向：兩者彼此相同，所以只比這兩格抓不到。
        scene = pair[0].report.scene.model_copy(update={"source_model": _analytic()})
        first = replace(pair[0], report=pair[0].report.model_copy(update={"scene": scene}))
    elif part == "table":
        table = first.report.path_table
        assert table is not None
        report = first.report.model_copy(update={
            "path_table": table.model_copy(update={"source_model_kind": SourceModelKind.OMNIDIRECTIONAL})
        })
        first = replace(first, report=report)
    else:
        assert first.window is not None
        first = replace(first, window=first.window.model_copy(update={
            "source_model_kind": SourceModelKind.OMNIDIRECTIONAL
        }))
    result = _reflection((first, _analytic_record(pair[1])))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.SOURCE_MODEL_MISMATCH,)


@pytest.mark.parametrize(("actual", "expected"), (
    (TIMBRE_EVALUATOR_VERSION, "aosr.scoring.timbre.v8"),
    (TIMBRE_CHANNELS_EVALUATOR_VERSION, "aosr.scoring.timbre_channels.v2"),
    (LISTENING_AREA_EVALUATOR_VERSION, "aosr.scoring.listening_area.v6"),
    (CHANNEL_MATCHING_EVALUATOR_VERSION, "aosr.scoring.channel_matching.v8"),
    (REFLECTIONS_AND_ECHO_EVALUATOR_VERSION, "aosr.scoring.reflections.v3"),
    (REVERBERATION_EVALUATOR_VERSION, "aosr.scoring.reverberation.v4"),
))
def test_evaluator_version_advances_once(actual: str, expected: str) -> None:
    assert actual == expected


def _analytic_evaluation(
    evaluation: CategoryEvaluation, section: SourceModelSection | None = None,
) -> CategoryEvaluation:
    """同一場景下手造解析近似評估，保留所有聲學數字。"""
    return evaluation.model_copy(update={
        "source_model_fingerprint": source_model_fingerprint(section or _analytic()),
        "flags": (*evaluation.flags, Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED),
    })


def test_timbre_analytic_flags_follow_report_flags_before_validation_flags() -> None:
    from tests.engine import test_scoring_timbre as fixtures

    data = fixtures._flat_input().model_copy(update={
        "source_model": _analytic(), "report_flags": (Flag.CROSSOVER_BAND,),
    })
    result = fixtures._evaluate(data)
    assert result.state is EvaluationState.MEASURED
    assert result.source_model_fingerprint == hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()
    assert result.flags == (
        Flag.CROSSOVER_BAND, Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED,
        Flag.BASELINE_SETTINGS,
    )
    experimental = fixtures._evaluate(data.model_copy(update={
        "model_validation_status": ModelValidationStatus.EXPERIMENTAL,
    }))
    assert experimental.flags == (
        Flag.CROSSOVER_BAND, Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED, Flag.UNVALIDATED,
        Flag.BASELINE_SETTINGS,
    )
    omni = fixtures._evaluate(fixtures._flat_input())
    assert Flag.NO_DIRECTIVITY not in omni.flags
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED not in omni.flags
    assert result.settings_fingerprint == omni.settings_fingerprint


def test_timbre_unavailable_retains_analytic_identity_and_flag() -> None:
    from tests.engine import test_scoring_timbre as fixtures

    data = fixtures._flat_input().model_copy(update={
        "source_model": _analytic(), "total_energy": (0.0,) * len(fixtures._flat_input().total_energy),
    })
    result = fixtures._evaluate(data)
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.source_model_fingerprint == hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in result.flags


def test_timbre_report_collector_copies_whole_source_model_section() -> None:
    from aosr.scoring.contract import InputProvenance

    report = _analytic_record(reflection_fixtures._pair()[0]).report
    provenance = InputProvenance(report_id="manual", engine_commit="fixture", speaker_id="left", receiver_id="main")
    collected = timbre_input_from_report(
        report, candidate_id="candidate-a", speaker_id="left", receiver_id="main",
        source_reference="fixture-source", provenance=provenance,
    )
    assert collected.source_model == report.scene.source_model
    assert collected.report_flags == ()
    result = evaluate_timbre(
        collected, purpose="dedicated_two_channel_listening_room",
        quality_targets_path=config_path("quality_targets.toml"),
    )
    assert result.source_model_fingerprint == hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in result.flags


def test_channel_aggregate_copies_analytic_identity_and_rejects_mixture() -> None:
    from tests.engine import test_timbre_channels as fixtures

    group = fixtures._group()
    left = _analytic_evaluation(fixtures._single("left"))
    right = _analytic_evaluation(fixtures._single("right"))
    matching = fixtures._aggregate(group, {"left": left, "right": right})
    assert matching.state is EvaluationState.MEASURED
    assert matching.source_model_fingerprint == left.source_model_fingerprint
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in matching.flags
    mixed = fixtures._aggregate(group, {"left": left, "right": fixtures._single("right")})
    assert mixed.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.SOURCE_MODEL_MISMATCH in mixed.reason_codes
    assert mixed.source_model_fingerprint is None
    missing = fixtures._aggregate(group, {"left": left})
    assert missing.state is EvaluationState.UNAVAILABLE
    assert missing.source_model_fingerprint == left.source_model_fingerprint
    # 不可估、指紋空的上游不算另一種聲源模型：不報混用，照帶另一支的值。
    document = fixtures._single("right").model_dump(mode="python")
    document.update(state=EvaluationState.UNAVAILABLE, payload=None, raw_quantities=(),
                    reason_codes=(ReasonCode.INSUFFICIENT_COVERAGE,), source_model_fingerprint=None)
    blank = fixtures._aggregate(group, {"left": left, "right": CategoryEvaluation.model_validate(document)})
    assert blank.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.SOURCE_MODEL_MISMATCH not in blank.reason_codes
    assert blank.source_model_fingerprint == left.source_model_fingerprint
    omni = fixtures._aggregate(group, {"left": fixtures._single("left"), "right": fixtures._single("right")})
    assert matching.settings_fingerprint == omni.settings_fingerprint


def test_listening_area_copies_analytic_identity_and_rejects_mixture() -> None:
    from tests.engine import test_listening_area as fixtures

    receivers = fixtures._receiver_set()
    base = fixtures._results(receivers)
    analytic = tuple(item.model_copy(update={
        "timbre_evaluation": _analytic_evaluation(item.timbre_evaluation)
    }) for item in base)
    matching = fixtures._evaluate(receivers, analytic)
    assert matching.state is EvaluationState.MEASURED
    assert matching.source_model_fingerprint == analytic[0].timbre_evaluation.source_model_fingerprint
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in matching.flags
    mixed = fixtures._evaluate(receivers, (base[0], *analytic[1:]))
    assert mixed.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.SOURCE_MODEL_MISMATCH in mixed.reason_codes
    assert mixed.source_model_fingerprint is None
    missing = fixtures._evaluate(receivers, analytic[:-1])
    assert missing.state is EvaluationState.UNAVAILABLE
    assert missing.source_model_fingerprint == analytic[0].timbre_evaluation.source_model_fingerprint
    assert matching.settings_fingerprint == fixtures._evaluate(receivers, base).settings_fingerprint


def test_listening_area_model_check_ignores_other_seats() -> None:
    """聆聽區的聲源模型核對跟版本核對用同一批：其他座位不採用，也不拿來比。"""
    from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet
    from tests.engine import test_listening_area as fixtures

    receivers = ReceiverSet(points=(*fixtures._receiver_set().points, ReceiverPoint(
        receiver_id="sofa", position_m=(2.5, 2.0, 1.2), role=ReceiverRole.OTHER_SEAT, importance=1.0,
    )))
    results = fixtures._results(receivers)
    sofa = results[0].model_copy(update={
        "receiver_id": "sofa", "timbre_evaluation": _analytic_evaluation(results[0].timbre_evaluation),
    })
    evaluation = fixtures._evaluate(receivers, (*results, sofa))
    assert evaluation.state is EvaluationState.MEASURED
    assert evaluation.source_model_fingerprint == OMNI_SOURCE_MODEL_FINGERPRINT


def _matching_analytic_points(
    points: tuple[ChannelPointInput, ...],
) -> tuple[ChannelPointInput, ...]:

    changed = []
    for point in points:
        assert isinstance(point, ChannelPointInput)
        responses = tuple(response.model_copy(update={
            "timbre_evaluation": _analytic_evaluation(response.timbre_evaluation)
        }) for response in point.responses)
        changed.append(point.model_copy(update={"responses": responses}))
    return tuple(changed)


def test_channel_matching_copies_analytic_identity_and_rejects_mixture() -> None:
    from tests.engine import test_channel_matching as fixtures

    receivers, group = fixtures._receivers(), fixtures._group()
    base = fixtures._channel_points(receivers, group)
    analytic = _matching_analytic_points(base)
    matching = fixtures._evaluate(receivers, group, analytic)
    assert matching.state is EvaluationState.MEASURED
    assert matching.source_model_fingerprint == hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()
    assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in matching.flags
    assert matching.payload is not None
    mixed = fixtures._evaluate(receivers, group, (base[0], *analytic[1:]))
    assert mixed.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.SOURCE_MODEL_MISMATCH in mixed.reason_codes
    assert mixed.source_model_fingerprint is None
    missing = fixtures._evaluate(receivers, group, analytic[:-1])
    assert missing.state is EvaluationState.UNAVAILABLE
    assert missing.source_model_fingerprint == hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()
    # 施工單：聲源模型不准折進設定指紋（它走比較身分那一格）。
    assert matching.settings_fingerprint == fixtures._evaluate(receivers, group, base).settings_fingerprint


def test_channel_matching_reflection_mismatch_is_local_to_diagnosis() -> None:
    from aosr.scoring.channel_matching_reflections_contract import ReflectionAsymmetry
    from tests.engine import test_channel_matching_reflections_wiring as fixtures

    upstream = fixtures._integrated_upstream()
    inputs = fixtures._integrated_inputs(upstream)
    changed = upstream.model_copy(update={
        "source_model_fingerprint": hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest()
    })
    result = fixtures._integrated_evaluate(changed, inputs)
    assert result.state is EvaluationState.MEASURED
    assert isinstance(result.payload, ChannelMatchingPayload)
    section: ReflectionAsymmetry = result.payload.reflection_asymmetry
    assert section.reason_codes == (ReasonCode.SOURCE_MODEL_MISMATCH,)
    assert result.reason_codes == ()
    # 場景與聲源模型都對不上時，先報場景（真實報表的場景指紋已含聲源模型）。
    from aosr.scoring.channel_matching_reflections import reflection_asymmetry
    from tests.engine import test_channel_matching_reflections as diagnosis

    both = reflection_asymmetry(
        upstream, candidate_id=upstream.candidate_id, scene_fingerprint="b" * 64,
        source_model_fingerprint=hashlib.sha256(_ANALYTIC_JSON.encode()).hexdigest(),
        channels=diagnosis._CHANNELS, comparisons=diagnosis._PAIR, receiver_ids=("main",),
        primary_receiver_id="main", placement=upstream.placement,
    )
    assert both.reason_codes == (ReasonCode.SCENE_FINGERPRINT_MISMATCH,)


def test_contract_model_identity_rules_and_candidate_consistency() -> None:
    from tests.engine import test_scoring_contract as fixtures

    for state in ("measured", "costed"):
        missing = {**fixtures._evaluation(state=state), "source_model_fingerprint": None}
        with pytest.raises(ValidationError, match="source_model_fingerprint"):
            CategoryEvaluation.model_validate(missing)
    unavailable = fixtures._unavailable("timbre_balance", {"speaker_positions_m": (), "receiver_positions_m": ()})
    assert CategoryEvaluation.model_validate(unavailable.model_copy(update={
        "source_model_fingerprint": None
    }).model_dump()).source_model_fingerprint is None
    for category in ("reverberation", "low_frequency_decay"):
        independent = fixtures._unavailable(category, {"speaker_positions_m": (), "receiver_positions_m": ()})
        with pytest.raises(ValidationError, match="source_model_fingerprint"):
            CategoryEvaluation.model_validate(independent.model_copy(update={
                "source_model_fingerprint": OMNI_SOURCE_MODEL_FINGERPRINT
            }).model_dump())
    first = unavailable
    second = fixtures._unavailable("spatial_impression", {"speaker_positions_m": (), "receiver_positions_m": ()})
    alternate = second.model_copy(update={"source_model_fingerprint": "b" * 64})
    def candidate(rows: tuple[CategoryEvaluation, ...]) -> CandidateEvaluation:
        return CandidateEvaluation(
            schema_version=CONTRACT_SCHEMA_VERSION, candidate_id="candidate-a",
            scene_fingerprint=first.scene_fingerprint, evaluations=rows,
        )
    with pytest.raises(ValueError, match="source_model_fingerprint"):
        candidate((first, alternate))
    assert candidate((first.model_copy(update={"source_model_fingerprint": None}), second))


def test_ranking_splits_models_but_keeps_aim_variants_together() -> None:
    from aosr.scoring.ranking_models import CandidateStatus

    omni = ranking_fixtures._candidate(ranking_fixtures._timbre("candidate-o"))
    first_model = _analytic()
    second_aim = _analytic(Point(4.0, 3.0, 1.2))
    assert source_model_fingerprint(first_model) == source_model_fingerprint(second_aim)
    analytic_a = ranking_fixtures._candidate(_analytic_evaluation(
        ranking_fixtures._timbre("candidate-a"), first_model))
    analytic_b = ranking_fixtures._candidate(_analytic_evaluation(
        ranking_fixtures._timbre("candidate-b"), second_aim))
    result = ranking_fixtures._rank(omni, analytic_a, analytic_b)
    assert result.status_of("candidate-o") is CandidateStatus.NOT_COMPARABLE
    assert result.status_of("candidate-a") is CandidateStatus.RANKABLE
    assert result.status_of("candidate-b") is CandidateStatus.RANKABLE
    identity = result.header.main_table_identity[0]
    assert identity.source_model_fingerprint == analytic_a.evaluations[0].source_model_fingerprint
    for row in result.rankable:
        evaluation = row.categories[0].evaluation
        assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in evaluation.flags
        assert Flag.NO_DIRECTIVITY not in evaluation.flags
        assert Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED in row.flags
        assert Flag.NO_DIRECTIVITY not in row.flags
    assert result.not_comparable.rows[0].identity[0].source_model_fingerprint == OMNI_SOURCE_MODEL_FINGERPRINT


def test_ranking_splits_same_kind_with_different_parameters() -> None:
    from aosr.scoring.ranking_models import CandidateStatus

    first = ranking_fixtures._candidate(_analytic_evaluation(ranking_fixtures._timbre("candidate-a")))
    original = first.evaluations[0]
    alternative = _analytic().parameters
    assert alternative is not None
    changed = _analytic().model_copy(update={
        "parameters": alternative.model_copy(update={"beta_limit": 3.5})
    })
    different = ranking_fixtures._timbre("candidate-b").model_copy(update={
        "source_model_fingerprint": source_model_fingerprint(changed),
        "flags": (*original.flags, Flag.ANALYTIC_DIRECTIVITY_UNVALIDATED),
    })
    second = ranking_fixtures._candidate(different)
    result = ranking_fixtures._rank(first, second)
    assert {result.status_of("candidate-a"), result.status_of("candidate-b")} == {
        CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE,
    }
    assert result.header.main_table_identity[0].source_model_fingerprint != result.not_comparable.rows[0].identity[0].source_model_fingerprint


def test_comparison_identity_requires_source_model_fingerprint() -> None:
    from aosr.scoring.ranking_models import ComparisonIdentity

    assert ComparisonIdentity.model_fields["source_model_fingerprint"].is_required()


def test_model_independent_measured_categories_reject_a_fingerprint() -> None:
    from tests.engine import test_scoring_contract as fixtures

    document = fixtures._evaluation(state="measured")
    document["category"] = "low_frequency_decay"
    document["payload"] = {"category": "low_frequency_decay"}
    with pytest.raises(ValidationError, match="source_model_fingerprint"):
        CategoryEvaluation.model_validate(document)


def test_reflections_prefers_real_scene_mismatch_over_model_mismatch() -> None:
    pair = reflection_fixtures._pair()
    analytic = _analytic_record(pair[1])
    scene = analytic.report.scene.model_copy(update={"scene_fingerprint": "b" * 64})
    analytic = replace(analytic, report=analytic.report.model_copy(update={"scene": scene}))
    result = _reflection((pair[0], analytic))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.SCENE_FINGERPRINT_MISMATCH,)


def test_aggregates_without_upstream_do_not_invent_model_identity() -> None:
    from tests.engine import test_channel_matching as matching
    from tests.engine import test_listening_area as listening
    from tests.engine import test_timbre_channels as channels

    grouped = channels._aggregate(channels._group(), {})
    receivers = listening._receiver_set()
    area = listening._evaluate(receivers, ())
    matching_receivers, group = matching._receivers(), matching._group()
    channel = matching._evaluate(matching_receivers, group, ())
    for evaluation in (grouped, area, channel):
        assert evaluation.state is EvaluationState.UNAVAILABLE
        assert evaluation.source_model_fingerprint is None
