"""聲道匹配從逐點音色結果收真實擺位的考卷（票 #417）。"""
from __future__ import annotations

import math

from pathlib import Path
from typing import Final

import pytest

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
    ChannelMatchingPayload,
    EvaluationState,
    Flag,
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


def _with_timbre_update(point: ChannelPointInput, update: dict[str, object]) -> ChannelPointInput:
    """把這一點第一個聲道疊的音色評估改掉一格。"""
    responses = list(point.responses)
    first = responses[0]
    responses[0] = first.model_copy(
        update={"timbre_evaluation": first.timbre_evaluation.model_copy(update=update)}
    )
    return point.model_copy(update={"responses": tuple(responses)})


def test_two_different_mistakes_give_same_output_in_either_order() -> None:
    """兩點各犯一種錯：原因碼照列舉宣告的順序排，輸入反過來輸出逐格相同；擺位沒衝突就照樣帶著。"""
    receivers = _receiver_set()
    group = _group()
    points = list(_points(receivers, group))
    points[0] = _with_timbre_update(points[0], {"candidate_id": "wrong-candidate"})
    points[1] = _with_timbre_update(points[1], {"settings_fingerprint": "wrong-settings"})

    forward = _evaluate(receivers, group, tuple(points))
    reversed_input = _evaluate(receivers, group, tuple(reversed(points)))

    assert forward == reversed_input
    assert forward.state is EvaluationState.UNAVAILABLE
    assert {
        ReasonCode.CANDIDATE_ID_MISMATCH,
        ReasonCode.TIMBRE_SETTINGS_FINGERPRINT_MISMATCH,
    } <= set(forward.reason_codes)
    assert {name for name, _ in forward.placement.speaker_positions_m} == {
        response.timbre_evaluation.provenance.speaker_id
        for point in points
        for response in point.responses
    }
    assert {name for name, _ in forward.placement.receiver_positions_m} == {
        point.receiver_id for point in points
    }


def test_flags_from_different_points_do_not_follow_input_order() -> None:
    """兩點各帶一種旗標：彙總出來的旗標順序不准跟著輸入順序變。"""
    receivers = _receiver_set()
    group = _group()
    points = list(_points(receivers, group))
    for index, flag in ((0, Flag.UNVALIDATED), (1, Flag.DATA_COVERAGE_SHORT)):
        first = points[index].responses[0].timbre_evaluation
        points[index] = _with_timbre_update(points[index], {"flags": (*first.flags, flag)})

    forward = _evaluate(receivers, group, tuple(points))
    reversed_input = _evaluate(receivers, group, tuple(reversed(points)))

    assert forward == reversed_input
    assert {Flag.UNVALIDATED, Flag.DATA_COVERAGE_SHORT} <= set(forward.flags)


def test_group_without_comparisons_is_refused_by_channel_matching() -> None:
    """單聲道的聲道組（沒有比較對）是給聲道音色彙總用的：聲道匹配拿到它要明講拒收，不准量出一份空的結果。"""
    receivers = _receiver_set()
    group = _group()
    mono = ChannelGroup(
        channels=group.channels[:1],
        comparisons=(),
        feature_match_tolerance_hz=group.feature_match_tolerance_hz,
    )

    with pytest.raises(ValueError, match="比較對"):
        _evaluate(receivers, mono, tuple(_points(receivers, group)))


def test_negative_ripple_smoothing_width_is_refused_but_zero_is_raw(tmp_path: Path) -> None:
    """左右差異曲線用音色的起伏平滑寬度：0 是看原始曲線（票 #432）要收，負的要紅——只擋「≤0」會把 0 也擋掉。"""
    original = config_path("quality_targets.toml").read_text(encoding="utf-8")
    key = 'key = "timbre_balance.smoothing_width_octave_ripple"\nvalue = 0.0'
    assert key in original
    negative = tmp_path / "quality_targets.toml"
    negative.write_text(original.replace(key, key.replace("0.0", "-0.1"), 1), encoding="utf-8")
    receivers = _receiver_set()
    group = _group()
    points = tuple(_points(receivers, group))

    assert _evaluate(receivers, group, points).state is not None
    with pytest.raises(ValueError, match="不准是負的"):
        evaluate_channel_matching(
            receivers,
            points,
            candidate_id=_CANDIDATE,
            scene_fingerprint=_SCENE,
            timbre_settings_fingerprint=_TIMBRE_SETTINGS,
            listening_area_settings_fingerprint=_LISTENING_SETTINGS,
            channel_group=group,
            purpose=_PURPOSE,
            quality_targets_path=negative,
            sound_speed_m_s=343.0,
        )


def test_difference_curve_with_zero_width_is_the_pointwise_raw_difference() -> None:
    """左右差異曲線在寬度 0 時要逐點等於原始的左減右（dB）：暗中沿用舊平滑這一題就會紅。"""
    receivers = _receiver_set()
    group = _group()
    points = tuple(_points(receivers, group))
    evaluation = _evaluate(receivers, group, points)
    assert isinstance(evaluation.payload, ChannelMatchingPayload)
    point = next(m for m in evaluation.payload.point_results if m.receiver_id == points[0].receiver_id)
    responses = {item.role: item for item in points[0].responses}
    left, right = responses["left"], responses["right"]
    expected = [
        10.0 * math.log10((a / max(left.total_energy)) / (b / max(right.total_energy)))
        for a, b in zip(left.total_energy, right.total_energy, strict=True)
    ]

    assert [item.left_minus_right_db for item in point.frequency_difference_curve_db] == pytest.approx(expected)
