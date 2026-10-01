"""聆聽區逐聲道彙總與暫定警戒的接線考卷（#518、#519）。"""

from __future__ import annotations

from datetime import date
import json
import math
import pytest

from aosr.config.quality_targets import SettingEntry, TargetEntry
from aosr.scoring.channel_group import ChannelGroup
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CandidateEvaluation, CategoryEvaluation,
    EvaluationState, ListeningAreaChannelsPayload, ListeningAreaStabilityPayload,
    ReasonCode,
)
from aosr.scoring.listening_area_channels import evaluate_listening_area_channels
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.listening_area_cost import (
    comparison_support, cost_listening_area_evaluation, listening_area_floor_reasons,
)
from aosr.scoring.ranking import (
    CandidateStatus, NotEvaluatedReason, RankingContext, RankingResult, rank_candidates,
)
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.review_alert import ListeningAreaReviewAlert
from aosr.scoring.verification_selection import _LINES, _raw_value, select_for_verification
from tests.engine.test_listening_area_cost import _listening_only_registry, _measured, _registry
from tests.engine.test_timbre_listening_area_support import _AXIS, _timbre, _receivers as _curve_receivers


def _group() -> ChannelGroup:
    return ChannelGroup.model_validate({
        "channels": [{"role": role, "speaker_id": role} for role in ("left", "right")],
        "comparisons": [{"left_role": "left", "right_role": "right"}],
        "feature_match_tolerance_hz": 1.0,
    })


def _receivers() -> ReceiverSet:
    return ReceiverSet.model_validate({"points": [
        {"receiver_id": "main", "position_m": (0, 0, 0), "role": "primary",
         "importance": 1.0, "direction_relative_to_primary": None},
        {"receiver_id": "front", "position_m": (0, 0.1, 0), "role": "surrounding",
         "importance": 1.0, "direction_relative_to_primary": "front"},
    ]})


def _channel(role: str, *, worst: float = 0.0, level_worst: float = 0.0,
             peer_tilt_worst: float | None = None) -> CategoryEvaluation:
    base = _measured(tilt_mean=0.2 if role == "left" else 0.8,
                     tilt_worst=worst, level_worst=level_worst,
                     peer_means=(0.0, 0.0, 0.0) if peer_tilt_worst is not None else None,
                     peer_worsts=(peer_tilt_worst, 0.0, 0.0)
                     if peer_tilt_worst is not None else None)
    document = base.model_dump(mode="python")
    document["provenance"]["speaker_id"] = role
    document["payload"].update(
        speaker_id=role, receiver_set_fingerprint=_receivers().fingerprint,
    )
    if peer_tilt_worst is not None:
        document["payload"]["tilt_stability"]["surrounding_to_surrounding"][
            "worst_deviation"]["reference"]["receiver_id"] = "front"
    return CategoryEvaluation.model_validate(document)


def _aggregate(*, right_worst: float = 0.0) -> CategoryEvaluation:
    return evaluate_listening_area_channels(
        _group(), _receivers(),
        {"right": _channel("right", worst=right_worst), "left": _channel("left")},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )


def _rank_area(
    measured: CategoryEvaluation, group: ChannelGroup | None = None,
) -> RankingResult:
    chosen = group or _group()
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id="candidate-a",
        scene_fingerprint="a" * 64, evaluations=(measured,),
    )
    context = RankingContext(
        purpose="dedicated_two_channel_listening_room",
        receiver_set_fingerprint=_receivers().fingerprint,
        channel_group_fingerprint=chosen.fingerprint,
        run_date=date(2026, 9, 28), engine_version="fixture",
    )
    return rank_candidates([candidate], _listening_only_registry(), context)


def test_single_is_a_valid_channel_role_in_ranking_alerts() -> None:
    group = ChannelGroup.model_validate({
        "channels": [{"role": role, "speaker_id": role} for role in ("left", "single")],
        "comparisons": [{"left_role": "left", "right_role": "single"}],
        "feature_match_tolerance_hz": 1.0,
    })
    measured = evaluate_listening_area_channels(
        group, _receivers(),
        {"left": _channel("left"), "single": _channel("single", worst=2.0)},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    ranked = _rank_area(measured, group)
    assert ranked.status_of("candidate-a") is CandidateStatus.RANKABLE
    (row,) = ranked.rankable
    alerts = [item for item in row.review_alerts if isinstance(item, ListeningAreaReviewAlert)]
    assert [(item.role, item.speaker_id, item.deviation) for item in alerts] == [
        ("single", "single", 2.0),
    ]


@pytest.mark.parametrize(
    ("change", "reason"),
    (
        ("source_model", ReasonCode.SOURCE_MODEL_MISMATCH),
        ("candidate_outer", ReasonCode.CANDIDATE_ID_MISMATCH),
        ("candidate_payload", ReasonCode.CANDIDATE_ID_MISMATCH),
        ("scene", ReasonCode.SCENE_FINGERPRINT_MISMATCH),
        ("evaluator_version", ReasonCode.EVALUATOR_VERSION_MISMATCH),
        ("receiver_id", ReasonCode.RECEIVER_ID_MISMATCH),
        ("placement", ReasonCode.PLACEMENT_MISMATCH),
        ("speaker_provenance", ReasonCode.CHANNEL_ROLE_MISMATCH),
        ("speaker_payload", ReasonCode.CHANNEL_ROLE_MISMATCH),
    ),
    ids=("source-model", "candidate-outer", "candidate-payload", "scene",
         "evaluator-version", "primary-receiver", "placement",
         "speaker-provenance", "speaker-payload"),
)
def test_channel_identity_mismatch_makes_whole_area_unavailable(
    change: str, reason: ReasonCode,
) -> None:
    """逐項移除彙總身分核對時，對應原因碼考卷會紅。"""
    right = _channel("right")
    if change == "source_model":
        right = right.model_copy(update={"source_model_fingerprint": "b" * 64})
    elif change == "candidate_outer":
        right = right.model_copy(update={"candidate_id": "other-candidate"})
    elif change == "candidate_payload":
        assert isinstance(right.payload, ListeningAreaStabilityPayload)
        right = right.model_copy(update={"payload": right.payload.model_copy(
            update={"candidate_id": "other-candidate"})})
    elif change == "scene":
        right = right.model_copy(update={"scene_fingerprint": "b" * 64})
    elif change == "evaluator_version":
        right = right.model_copy(update={"evaluator_version": "listening-area-v2"})
    elif change == "receiver_id":
        right = right.model_copy(update={"provenance": right.provenance.model_copy(
            update={"receiver_id": "front"})})
    elif change == "placement":
        right = right.model_copy(update={"placement": right.placement.model_copy(
            update={"receiver_positions_m": (("main-seat", (9.0, 8.0, 7.0)),)})})
    elif change == "speaker_provenance":
        right = right.model_copy(update={"provenance": right.provenance.model_copy(
            update={"speaker_id": "wrong-speaker"})})
    else:
        assert isinstance(right.payload, ListeningAreaStabilityPayload)
        right = right.model_copy(update={"payload": right.payload.model_copy(
            update={"speaker_id": "wrong-speaker"})})
    result = evaluate_listening_area_channels(
        _group(), _receivers(), {"left": _channel("left"), "right": right},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    assert result.state is EvaluationState.UNAVAILABLE
    assert reason in result.reason_codes


def test_listening_area_group_fingerprint_mismatch_is_not_evaluated() -> None:
    result = _rank_area(_aggregate(), ChannelGroup.model_validate({
        "channels": [{"role": "left", "speaker_id": "left"}],
        "comparisons": [], "feature_match_tolerance_hz": 1.0,
    }))
    assert result.status_of("candidate-a") is CandidateStatus.NOT_EVALUATED
    assert NotEvaluatedReason.CHANNEL_GROUP_FINGERPRINT_MISMATCH in {
        item.reason for row in result.not_evaluated for item in row.missing
    }


def test_both_channels_over_same_line_keep_own_identity_and_deviation() -> None:
    measured = evaluate_listening_area_channels(
        _group(), _receivers(),
        {"left": _channel("left", worst=2.0),
         "right": _channel("right", worst=2.5)},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    ranked = _rank_area(measured)
    (row,) = ranked.rankable
    alerts = [item for item in row.review_alerts if isinstance(item, ListeningAreaReviewAlert)]
    assert [(item.role, item.speaker_id, item.metric, item.group, item.deviation)
            for item in alerts] == [
        ("left", "left", "tilt", "primary_to_surrounding", 2.0),
        ("right", "right", "tilt", "primary_to_surrounding", 2.5),
    ]


def test_peer_alert_uses_its_own_worst_pair_and_deviation() -> None:
    measured = evaluate_listening_area_channels(
        _group(), _receivers(),
        {"left": _channel("left"),
         "right": _channel("right", worst=2.0, peer_tilt_worst=3.0)},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    (row,) = _rank_area(measured).rankable
    peer = next(item for item in row.review_alerts
                if isinstance(item, ListeningAreaReviewAlert)
                and item.group == "surrounding_to_surrounding")
    assert (peer.role, peer.speaker_id, peer.receiver_id, peer.reference_id,
            peer.deviation) == ("right", "right", "back", "front", 3.0)


def test_alert_order_uses_role_metric_and_group_before_deviation() -> None:
    measured = evaluate_listening_area_channels(
        _group(), _receivers(),
        {"right": _channel("right", worst=2.0, peer_tilt_worst=3.0),
         "left": _channel("left", level_worst=5.0)},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    (row,) = _rank_area(measured).rankable
    assert [(item.role, item.metric, item.group, item.deviation)
            for item in row.review_alerts if isinstance(item, ListeningAreaReviewAlert)] == [
        ("left", "overall_level", "primary_to_surrounding", 5.0),
        ("right", "tilt", "primary_to_surrounding", 2.0),
        ("right", "tilt", "surrounding_to_surrounding", 3.0),
    ]


def test_aggregate_retains_both_channels_and_is_order_independent() -> None:
    result = _aggregate()
    assert result.state is EvaluationState.MEASURED
    assert result.settings_fingerprint == "listening-area-settings-a"
    assert isinstance(result.payload, ListeningAreaChannelsPayload)
    assert tuple(item.role for item in result.payload.channels) == ("left", "right")
    assert result == evaluate_listening_area_channels(
        _group(), _receivers(),
        {"left": _channel("left"), "right": _channel("right")},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )


def test_reversing_channel_declaration_keeps_every_output_field() -> None:
    group = _group()
    reversed_group = group.model_copy(update={"channels": tuple(reversed(group.channels))})
    assert reversed_group.fingerprint == group.fingerprint
    result = _aggregate()
    reordered = evaluate_listening_area_channels(
        reversed_group, _receivers(),
        {"left": _channel("left"), "right": _channel("right")},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    assert reordered == result


def test_only_right_breaches_alert_but_candidate_ranks_at_channel_mean() -> None:
    purpose = _registry().purpose("dedicated_two_channel_listening_room")
    target = purpose.entry("listening_area_stability.tilt_worst_deviation")
    assert isinstance(target, TargetEntry)
    limit = target.value
    assert isinstance(limit, float)
    measured = _aggregate(right_worst=limit + 0.5)
    costed = cost_listening_area_evaluation(measured, purpose, "cost-fixture")
    left = cost_listening_area_evaluation(_channel("left"), purpose, "cost-fixture")
    right = cost_listening_area_evaluation(
        _channel("right", worst=limit + 0.5), purpose, "cost-fixture")
    assert costed.category_cost is not None
    assert left.category_cost is not None and right.category_cost is not None
    assert costed.category_cost.value == (
        left.category_cost.value + right.category_cost.value
    ) / 2
    assert "right.tilt_worst_deviation.primary_to_surrounding" in costed.category_cost.components
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id="candidate-a",
        scene_fingerprint="a" * 64, evaluations=(measured,),
    )
    context = RankingContext(
        purpose="dedicated_two_channel_listening_room",
        receiver_set_fingerprint=_receivers().fingerprint,
        channel_group_fingerprint=_group().fingerprint,
        run_date=date(2026, 9, 28), engine_version="fixture",
    )
    ranked = rank_candidates([candidate], _listening_only_registry(), context)
    assert ranked.status_of("candidate-a") is CandidateStatus.RANKABLE
    (row,) = ranked.rankable
    alerts = [item for item in row.review_alerts if isinstance(item, ListeningAreaReviewAlert)]
    assert {(item.role, item.metric, item.group) for item in alerts} == {
        ("right", "tilt", "primary_to_surrounding")}
    alert = next(item for item in alerts if item.role == "right")
    assert (alert.role, alert.speaker_id, alert.receiver_id, alert.reference_id) == (
        "right", "right", "front", "main",
    )
    assert alert.deviation == limit + 0.5
    assert alert.limit == limit
    assert "尚未正式校準" in alert.note


def _from_curves(role: str, receivers: ReceiverSet,
                 main_energy: tuple[float, ...], front_energy: tuple[float, ...]
                 ) -> CategoryEvaluation:
    points = tuple(ReceiverPointResult(
        receiver_id=receiver, receiver_set_fingerprint=receivers.fingerprint,
        timbre_evaluation=_timbre("candidate-a", receiver, role, _AXIS, energy),
        frequencies_hz=_AXIS, total_energy=energy,
    ) for receiver, energy in (("main", main_energy), ("front", front_energy)))
    result = evaluate_listening_area(
        receivers, points, candidate_id="candidate-a", speaker_id=role,
        timbre_settings_fingerprint=points[0].timbre_evaluation.settings_fingerprint,
        scene_fingerprint="a" * 64, feature_match_tolerance_hz=10.0,
        broadband_range_hz=(20.0, 8000.0),
    )
    assert result.state is EvaluationState.MEASURED
    return result


def test_distinct_input_curves_alert_only_right_without_elimination() -> None:
    receivers = _curve_receivers()
    flat = tuple(1e7 for _ in _AXIS)
    tilted = tuple(1e7 * 10 ** (2.5 * math.log2(f / 1000.0) / 10.0)
                   for f in _AXIS)
    left = _from_curves("left", receivers, flat, flat)
    right = _from_curves("right", receivers, flat, tilted)
    group = _group()
    measured = evaluate_listening_area_channels(
        group, receivers, {"left": left, "right": right},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint=left.settings_fingerprint,
    )
    registry = _listening_only_registry()
    context = RankingContext(
        purpose="dedicated_two_channel_listening_room",
        receiver_set_fingerprint=receivers.fingerprint,
        channel_group_fingerprint=group.fingerprint,
        run_date=date(2026, 9, 28), engine_version="fixture",
    )
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id="candidate-a",
        scene_fingerprint="a" * 64, evaluations=(measured,),
    )
    result = rank_candidates([candidate], registry, context)
    assert result.status_of("candidate-a") is CandidateStatus.RANKABLE
    (row,) = result.rankable
    alerts = [item for item in row.review_alerts if isinstance(item, ListeningAreaReviewAlert)]
    assert all(item.role == "right" and item.speaker_id == "right" for item in alerts)
    tilt = next(item for item in alerts if item.metric == "tilt")
    assert (tilt.receiver_id, tilt.reference_id) == ("front", "main")
    assert isinstance(right.payload, ListeningAreaStabilityPayload)
    assert tilt.deviation == right.payload.tilt_stability.primary_to_surrounding.worst_deviation.value
    assert tilt.deviation > tilt.limit
    purpose = registry.purpose("dedicated_two_channel_listening_room")
    left_cost = cost_listening_area_evaluation(left, purpose, "cost")
    right_cost = cost_listening_area_evaluation(right, purpose, "cost")
    assert left_cost.category_cost is not None and right_cost.category_cost is not None
    assert row.categories[0].category_cost == (
        left_cost.category_cost.value + right_cost.category_cost.value) / 2


def test_aggregate_comparison_support_keeps_each_channel_and_version() -> None:
    support = json.loads(comparison_support(_aggregate()))
    assert support["channel_group_fingerprint"] == _group().fingerprint
    assert [item["role"] for item in support["channels"]] == ["left", "right"]
    assert [item["speaker_id"] for item in support["channels"]] == ["left", "right"]
    assert all(item["frequency_support"]["points"] for item in support["channels"])
    assert support["listening_area_evaluator_version"] == "listening-area-v1"


def test_verification_reads_right_channel_worst_value() -> None:
    evaluation = _aggregate(right_worst=2.0)
    assert _raw_value(evaluation, _LINES[0]) == 2.0


def test_right_channel_near_line_reaches_verification_selection() -> None:
    registry = _listening_only_registry()
    purpose = registry.purpose("dedicated_two_channel_listening_room")
    target = purpose.entry("listening_area_stability.tilt_worst_deviation")
    shift = purpose.entry(
        "verification.observed_axis_shift.listening_area_stability."
        "tilt_worst_deviation.primary_to_surrounding"
    )
    multiplier = purpose.entry("verification.range_multiplier")
    assert isinstance(target, TargetEntry)
    assert isinstance(shift, SettingEntry) and isinstance(multiplier, SettingEntry)
    assert isinstance(target.value, float)
    assert isinstance(shift.value, float)
    assert isinstance(multiplier.value, int | float)
    worst = target.value - shift.value * float(multiplier.value) / 2
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id="candidate-a",
        scene_fingerprint="a" * 64, evaluations=(_aggregate(right_worst=worst),),
    )
    context = RankingContext(
        purpose=purpose.name, receiver_set_fingerprint=_receivers().fingerprint,
        channel_group_fingerprint=_group().fingerprint,
        run_date=date(2026, 9, 28), engine_version="fixture",
    )
    ranked = rank_candidates([candidate], registry, context)
    selected = select_for_verification(ranked, registry, None, None, 7)
    assert any(reason.kind == "near_line" and reason.line_key ==
               "listening_area_stability.tilt_worst_deviation.primary_to_surrounding"
               for item in selected.candidates for reason in item.reasons)


def test_aggregate_component_lines_have_role_and_mean_weight() -> None:
    measured = _aggregate()
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id="candidate-a",
        scene_fingerprint="a" * 64, evaluations=(measured,),
    )
    context = RankingContext(
        purpose="dedicated_two_channel_listening_room",
        receiver_set_fingerprint=_receivers().fingerprint,
        channel_group_fingerprint=_group().fingerprint,
        run_date=date(2026, 9, 28), engine_version="fixture",
    )
    result = rank_candidates([candidate], _listening_only_registry(), context)
    (row,) = result.rankable
    (line,) = row.categories
    components = {item.name: item for item in line.components}
    assert components["left.tilt_weighted_mean_deviation"].role == "principal"
    assert components["right.tilt_weighted_mean_deviation"].role == "principal"
    from aosr.scoring.listening_area_cost import _listening_area_principal_weights
    weight = _listening_area_principal_weights(
        _registry().purpose("dedicated_two_channel_listening_room")
    )["tilt_weighted_mean_deviation"]
    assert components["left.tilt_weighted_mean_deviation"].weight == weight / 2
    assert components["right.tilt_weighted_mean_deviation"].weight == weight / 2
    assert components["right.tilt_worst_deviation.primary_to_surrounding"].role == "protection"
    identity = next(item for item in result.header.main_table_identity
                    if item.category.value == "listening_area_stability")
    assert json.loads(identity.assessed_support)["channel_group_fingerprint"] == (
        _group().fingerprint)


def test_breached_aggregate_has_no_floor_reason() -> None:
    purpose = _registry().purpose("dedicated_two_channel_listening_room")
    costed = cost_listening_area_evaluation(_aggregate(right_worst=2.0), purpose, "cost")
    assert listening_area_floor_reasons(costed, purpose) == ()


def test_one_unavailable_channel_makes_whole_category_unavailable() -> None:
    right = _channel("right")
    document = right.model_dump(mode="python")
    document.update(state=EvaluationState.UNAVAILABLE, payload=None,
                    raw_quantities=(), reason_codes=(ReasonCode.MISSING_POINTS,))
    unavailable = CategoryEvaluation.model_validate(document)
    result = evaluate_listening_area_channels(
        _group(), _receivers(), {"left": _channel("left"), "right": unavailable},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    assert result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.CHANNEL_RESULT_UNAVAILABLE in result.reason_codes
    assert ReasonCode.REQUIRED_CHANNEL_POINT_UNAVAILABLE in result.reason_codes
    assert ReasonCode.MISSING_POINTS in result.reason_codes


def test_payload_receiver_set_mismatch_is_named() -> None:
    right = _channel("right")
    document = right.model_dump(mode="python")
    document["payload"]["receiver_set_fingerprint"] = "wrong-set"
    result = evaluate_listening_area_channels(
        _group(), _receivers(),
        {"left": _channel("left"), "right": CategoryEvaluation.model_validate(document)},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    assert result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.RECEIVER_SET_FINGERPRINT_MISMATCH in result.reason_codes


def test_payload_settings_mismatch_is_named() -> None:
    right = _channel("right")
    document = right.model_dump(mode="python")
    document["payload"]["settings_fingerprint"] = "wrong-settings"
    result = evaluate_listening_area_channels(
        _group(), _receivers(),
        {"left": _channel("left"), "right": CategoryEvaluation.model_validate(document)},
        candidate_id="candidate-a", scene_fingerprint="a" * 64,
        listening_area_settings_fingerprint="listening-area-settings-a",
    )
    assert result.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.LISTENING_AREA_SETTINGS_FINGERPRINT_MISMATCH in result.reason_codes
