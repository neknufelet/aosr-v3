"""聲道匹配從逐點音色結果收真實擺位的考卷（票 #417）。"""
from __future__ import annotations

from typing import Final

from aosr.config.paths import config_path
from aosr.scoring.channel_matching import (
    ChannelComparison,
    ChannelDefinition,
    ChannelGroup,
    ChannelPointInput,
    ChannelResponse,
    evaluate_channel_matching,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    EvaluationState,
    InputProvenance,
    ModelValidationStatus,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbrePayload,
)
from aosr.scoring.placement import point_placement
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet


_CANDIDATE: Final[str] = "candidate-placement"
_SCENE: Final[str] = "a" * 64
_TIMBRE_SETTINGS: Final[str] = "timbre-settings"
_LISTENING_SETTINGS: Final[str] = "listening-settings"
_PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
_SPEAKERS = {
    "left": (0.2, 0.3, 1.1),
    "right": (0.2, 0.7, 1.1),
}
_RECEIVERS = {
    "main": (4.7, 0.5, 1.2),
    "front": (4.5, 0.5, 1.2),
}


def _receiver_set() -> ReceiverSet:
    return ReceiverSet(
        points=(
            ReceiverPoint(
                receiver_id="main",
                position_m=_RECEIVERS["main"],
                role=ReceiverRole.PRIMARY,
                importance=1.0,
            ),
            ReceiverPoint(
                receiver_id="front",
                position_m=_RECEIVERS["front"],
                role=ReceiverRole.SURROUNDING,
                importance=1.0,
                direction_relative_to_primary="front",
            ),
        )
    )


def _group() -> ChannelGroup:
    return ChannelGroup(
        channels=tuple(
            ChannelDefinition(role=role, speaker_id=f"speaker-{role}")
            for role in ("left", "right")
        ),
        comparisons=(ChannelComparison(left_role="left", right_role="right"),),
        feature_match_tolerance_hz=10.0,
    )


def _payload() -> TimbrePayload:
    return TimbrePayload(
        category="timbre_balance",
        tilt_db_per_octave=0.0,
        tilt_fit_range_hz=(80.0, 4000.0),
        tilt_dependency_range_hz=(71.0, 4490.0),
        target_tilt_db_per_octave=0.0,
        target_deviation_rms_db=0.0,
        deviation_curve=((100.0, 0.0), (200.0, 0.0)),
        residual_rms_db=0.0,
        ripple_range_hz=(40.0, 4000.0),
        ripple_dependency_range_hz=(37.0, 4240.0),
        features=(),
        strongest_peak_index=None,
        deepest_dip_index=None,
        data_range_hz=(20.0, 8000.0),
        coverage_range_hz=(20.0, 8000.0),
        model_validation_status=ModelValidationStatus.VALIDATED,
        model_validation_frequency_range_hz=(20.0, 8000.0),
    )


def _timbre(receiver_id: str, role: str) -> CategoryEvaluation:
    speaker_id = f"speaker-{role}"
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE,
        placement=point_placement(
            speaker_id,
            _SPEAKERS[role],
            receiver_id,
            _RECEIVERS[receiver_id],
        ),
        category=QualityCategory.TIMBRE_BALANCE,
        state=EvaluationState.MEASURED,
        payload=_payload(),
        raw_quantities=(RawQuantity(name="tilt", value=0.0, unit="dB/oct"),),
        category_cost=None,
        flags=(),
        reason_codes=(),
        evaluator_version="timbre-fixture-v1",
        settings_fingerprint=_TIMBRE_SETTINGS,
        provenance=InputProvenance(
            report_id=f"report-{role}-{receiver_id}",
            engine_commit="engine-fixture",
            speaker_id=speaker_id,
            receiver_id=receiver_id,
        ),
    )


def _points(receivers: ReceiverSet, group: ChannelGroup) -> tuple[ChannelPointInput, ...]:
    return tuple(
        ChannelPointInput(
            receiver_id=receiver_id,
            receiver_set_fingerprint=receivers.fingerprint,
            timbre_settings_fingerprint=_TIMBRE_SETTINGS,
            listening_area_settings_fingerprint=_LISTENING_SETTINGS,
            channel_group_fingerprint=group.fingerprint,
            responses=tuple(
                ChannelResponse(
                    role=role,
                    timbre_evaluation=_timbre(receiver_id, role),
                    frequencies_hz=(100.0, 200.0),
                    total_energy=(1.0, 1.0),
                    direct_distance_m=2.0,
                )
                for role in ("left", "right")
            ),
        )
        for receiver_id in ("main", "front")
    )


def _evaluate(
    receivers: ReceiverSet,
    group: ChannelGroup,
    points: tuple[ChannelPointInput, ...],
) -> CategoryEvaluation:
    return evaluate_channel_matching(
        receivers,
        points,
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE,
        timbre_settings_fingerprint=_TIMBRE_SETTINGS,
        listening_area_settings_fingerprint=_LISTENING_SETTINGS,
        channel_group=group,
        purpose=_PURPOSE,
        quality_targets_path=config_path("quality_targets.toml"),
        sound_speed_m_s=343.0,
    )


def test_channel_batch_placement_mismatch_is_unavailable_and_order_independent() -> None:
    """同一真實喇叭代號跨點落在不同座標時，整類不可估且不受輸入順序影響。"""
    receivers = _receiver_set()
    group = _group()
    points = list(_points(receivers, group))
    front = points[1]
    responses = list(front.responses)
    left = responses[0]
    changed_placement = left.timbre_evaluation.placement.model_copy(
        update={"speaker_positions_m": (("speaker-left", (9.0, 8.0, 7.0)),)}
    )
    responses[0] = left.model_copy(
        update={
            "timbre_evaluation": left.timbre_evaluation.model_copy(
                update={"placement": changed_placement}
            )
        }
    )
    points[1] = front.model_copy(update={"responses": tuple(responses)})

    forward = _evaluate(receivers, group, tuple(points))
    reversed_input = _evaluate(receivers, group, tuple(reversed(points)))
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=_CANDIDATE,
        scene_fingerprint=_SCENE,
        evaluations=(forward,),
    )

    assert forward == reversed_input
    assert forward.state is EvaluationState.UNAVAILABLE
    assert forward.reason_codes == (ReasonCode.PLACEMENT_MISMATCH,)
    assert forward.placement.speaker_positions_m == ()
    assert forward.placement.receiver_positions_m == ()
    assert candidate.evaluations == (forward,)
