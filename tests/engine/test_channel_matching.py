"""聲道匹配第一層評估器考卷（票 #350 第一段）。"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final, Literal

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
    CategoryEvaluation,
    ChannelMatchingPayload,
    EvaluationState,
    Feature,
    InputProvenance,
    MetricState,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbrePayload,
)
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet


_CANDIDATE: Final[str] = "candidate-a"
_TIMBRE_SETTINGS: Final[str] = "timbre-settings-a"
_LISTENING_SETTINGS: Final[str] = "listening-settings-a"
_PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
_TARGETS = config_path("quality_targets.toml")


def _receivers() -> ReceiverSet:
    return ReceiverSet(
        points=(
            ReceiverPoint(
                receiver_id="main",
                position_m=(0.0, 0.0, 1.2),
                role=ReceiverRole.PRIMARY,
                importance=1.0,
            ),
            ReceiverPoint(
                receiver_id="front",
                position_m=(0.0, 0.5, 1.2),
                role=ReceiverRole.SURROUNDING,
                importance=1.0,
                direction_relative_to_primary="front",
            ),
        )
    )


def _group(*, include_center: bool = False) -> ChannelGroup:
    channels = [
        ChannelDefinition(role="left", speaker_id="speaker-left"),
        ChannelDefinition(role="right", speaker_id="speaker-right"),
    ]
    if include_center:
        channels.append(ChannelDefinition(role="center", speaker_id="speaker-center"))
    return ChannelGroup(
        channels=tuple(channels),
        comparisons=(ChannelComparison(left_role="left", right_role="right"),),
        feature_match_tolerance_hz=10.0,
    )


def _feature(kind: Literal["peak", "dip"], frequency_hz: float) -> Feature:
    return Feature(
        kind=kind,
        center_frequency_hz=frequency_hz,
        depth_db=4.0 if kind == "peak" else -4.0,
        width_octave=0.25,
        flags=(),
    )


def _timbre(
    receiver_id: str,
    role: str,
    *,
    tilt: float,
    ripple: float,
    features: Sequence[Feature] = (),
    candidate_id: str = _CANDIDATE,
    settings_fingerprint: str = _TIMBRE_SETTINGS,
) -> CategoryEvaluation:
    payload = TimbrePayload(
        category="timbre_balance",
        tilt_db_per_octave=tilt,
        tilt_fit_range_hz=(80.0, 4000.0),
        target_tilt_db_per_octave=0.0,
        target_deviation_rms_db=ripple,
        deviation_curve=((100.0, 0.0), (200.0, ripple)),
        residual_rms_db=ripple,
        ripple_range_hz=(40.0, 4000.0),
        features=tuple(features),
        strongest_peak_index=None,
        deepest_dip_index=None,
        data_range_hz=(20.0, 8000.0),
        coverage_range_hz=(20.0, 8000.0),
    )
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        category=QualityCategory.TIMBRE_BALANCE,
        state=EvaluationState.MEASURED,
        payload=payload,
        raw_quantities=(RawQuantity(name="tilt", value=tilt, unit="dB/oct"),),
        category_cost=None,
        flags=(),
        reason_codes=(),
        evaluator_version="timbre-fixture-v1",
        settings_fingerprint=settings_fingerprint,
        provenance=InputProvenance(
            report_id=f"report-{receiver_id}-{role}",
            engine_commit="engine-fixture",
            speaker_id=f"speaker-{role}",
            receiver_id=receiver_id,
        ),
    )


def _response(
    receiver_id: str,
    role: str,
    *,
    tilt: float,
    ripple: float,
    energy: tuple[float, ...],
    distance_m: float,
    features: Sequence[Feature] = (),
) -> ChannelResponse:
    return ChannelResponse(
        role=role,
        timbre_evaluation=_timbre(
            receiver_id,
            role,
            tilt=tilt,
            ripple=ripple,
            features=features,
        ),
        frequencies_hz=(100.0, 200.0),
        total_energy=energy,
        direct_distance_m=distance_m,
    )


def _point(
    receivers: ReceiverSet,
    group: ChannelGroup,
    receiver_id: str,
    *,
    left_tilt: float = 1.0,
    right_tilt: float = 0.0,
    left_ripple: float = 2.0,
    right_ripple: float = 1.0,
    left_energy: tuple[float, ...] = (2.0, 2.0),
    right_energy: tuple[float, ...] = (1.0, 1.0),
    left_distance_m: float = 2.0,
    right_distance_m: float = 2.0,
) -> ChannelPointInput:
    responses = [
        _response(
            receiver_id,
            "left",
            tilt=left_tilt,
            ripple=left_ripple,
            energy=left_energy,
            distance_m=left_distance_m,
            features=(_feature("dip", 100.0),),
        ),
        _response(
            receiver_id,
            "right",
            tilt=right_tilt,
            ripple=right_ripple,
            energy=right_energy,
            distance_m=right_distance_m,
        ),
    ]
    if any(channel.role == "center" for channel in group.channels):
        responses.append(
            _response(
                receiver_id,
                "center",
                tilt=8.0,
                ripple=8.0,
                energy=(8.0, 8.0),
                distance_m=8.0,
            )
        )
    return ChannelPointInput(
        receiver_id=receiver_id,
        receiver_set_fingerprint=receivers.fingerprint,
        timbre_settings_fingerprint=_TIMBRE_SETTINGS,
        listening_area_settings_fingerprint=_LISTENING_SETTINGS,
        channel_group_fingerprint=group.fingerprint,
        responses=tuple(responses),
    )


def _evaluate(
    receivers: ReceiverSet,
    group: ChannelGroup,
    points: Sequence[ChannelPointInput],
) -> CategoryEvaluation:
    return evaluate_channel_matching(
        receivers,
        points,
        candidate_id=_CANDIDATE,
        timbre_settings_fingerprint=_TIMBRE_SETTINGS,
        listening_area_settings_fingerprint=_LISTENING_SETTINGS,
        channel_group=group,
        purpose=_PURPOSE,
        quality_targets_path=_TARGETS,
        sound_speed_m_s=343.0,
    )


def _payload(evaluation: CategoryEvaluation) -> ChannelMatchingPayload:
    assert evaluation.state == EvaluationState.MEASURED
    assert isinstance(evaluation.payload, ChannelMatchingPayload)
    return evaluation.payload


@pytest.mark.parametrize(
    ("identity", "reason"),
    (
        ("candidate", ReasonCode.CANDIDATE_ID_MISMATCH),
        ("receiver_set", ReasonCode.RECEIVER_SET_FINGERPRINT_MISMATCH),
        ("timbre_settings", ReasonCode.TIMBRE_SETTINGS_FINGERPRINT_MISMATCH),
        ("listening_settings", ReasonCode.LISTENING_AREA_SETTINGS_FINGERPRINT_MISMATCH),
        ("channel_group", ReasonCode.CHANNEL_GROUP_FINGERPRINT_MISMATCH),
    ),
)
def test_any_of_five_identities_mismatching_is_unavailable_with_specific_reason(
    identity: str, reason: ReasonCode
) -> None:
    """五個身分任一錯位都不能比較，且原因碼不得縮成籠統的 other_error。"""
    receivers = _receivers()
    group = _group()
    point = _point(receivers, group, "main")
    if identity == "candidate":
        first = point.responses[0]
        changed = first.model_copy(
            update={
                "timbre_evaluation": first.timbre_evaluation.model_copy(
                    update={"candidate_id": "candidate-b"}
                )
            }
        )
        point = point.model_copy(update={"responses": (changed, *point.responses[1:])})
    elif identity == "receiver_set":
        point = point.model_copy(update={"receiver_set_fingerprint": "other-set"})
    elif identity == "timbre_settings":
        point = point.model_copy(update={"timbre_settings_fingerprint": "other-timbre"})
    elif identity == "listening_settings":
        point = point.model_copy(
            update={"listening_area_settings_fingerprint": "other-listening"}
        )
    else:
        point = point.model_copy(update={"channel_group_fingerprint": "other-group"})

    evaluation = _evaluate(
        receivers, group, (point, _point(receivers, group, "front"))
    )

    assert evaluation.state == EvaluationState.UNAVAILABLE
    assert evaluation.payload is None
    assert reason in evaluation.reason_codes


def test_unavailable_side_keeps_the_point_unavailable_and_preserves_both_sources() -> None:
    """同點任一聲道不可估時，不能借另一點補值；兩邊原始評估仍要留在逐點來源。"""
    receivers = _receivers()
    group = _group()
    main = _point(receivers, group, "main")
    right = main.responses[1]
    unavailable = right.timbre_evaluation.model_copy(
        update={
            "state": EvaluationState.UNAVAILABLE,
            "payload": None,
            "raw_quantities": (),
            "reason_codes": (ReasonCode.INSUFFICIENT_COVERAGE,),
        }
    )
    main = main.model_copy(
        update={
            "responses": (
                main.responses[0],
                right.model_copy(update={"timbre_evaluation": unavailable}),
            )
        }
    )
    front = _point(receivers, group, "front", left_tilt=4.0, right_tilt=1.0)

    payload = _payload(_evaluate(receivers, group, (main, front)))
    main_result = next(item for item in payload.point_results if item.receiver_id == "main")
    aggregate = payload.aggregates[0].tilt_difference
    source = next(item for item in payload.point_sources if item.receiver_id == "main")

    assert main_result.state == MetricState.UNAVAILABLE
    assert main_result.tilt_difference_db_per_octave is None
    assert ReasonCode.CHANNEL_RESULT_UNAVAILABLE in main_result.reason_codes
    assert aggregate is not None
    assert aggregate.weighted_mean_absolute_difference == pytest.approx(3.0)
    assert {item.role for item in source.channels} == {"left", "right"}
    assert {item.state for item in source.channels} == {
        EvaluationState.MEASURED,
        EvaluationState.UNAVAILABLE,
    }


def test_each_receiver_is_compared_before_opposite_differences_are_aggregated() -> None:
    """主位 +12、周圍 -4 必須先各自取絕對值；先跨點平均會錯得 4。"""
    receivers = _receivers()
    group = _group()
    points = (
        _point(receivers, group, "main", left_tilt=12.0, right_tilt=0.0),
        _point(receivers, group, "front", left_tilt=0.0, right_tilt=4.0),
    )

    payload = _payload(_evaluate(receivers, group, points))
    aggregate = payload.aggregates[0].tilt_difference

    assert aggregate is not None
    assert aggregate.weighted_mean_absolute_difference == pytest.approx(8.0)
    assert aggregate.worst_absolute_difference == pytest.approx(12.0)
    assert {item.tilt_difference_db_per_octave for item in payload.point_results} == {
        -4.0,
        12.0,
    }


def test_tilt_and_ripple_compare_each_channels_smoothed_timbre_summary() -> None:
    """原始曲線直接相減會得另一答案；聲道匹配只能相減第一層各自平滑後的摘要。"""
    receivers = _receivers()
    group = _group()
    main = _point(
        receivers,
        group,
        "main",
        left_tilt=2.5,
        right_tilt=-0.5,
        left_ripple=4.0,
        right_ripple=1.0,
        left_energy=(1.0, 100.0),
        right_energy=(100.0, 1.0),
    )

    payload = _payload(
        _evaluate(receivers, group, (main, _point(receivers, group, "front")))
    )
    result = next(item for item in payload.point_results if item.receiver_id == "main")

    # 兩點原始能量若未經第一層各自平滑與擬合，左右一八度斜率差會是 40 dB/oct。
    assert result.tilt_difference_db_per_octave == pytest.approx(3.0)
    assert result.tilt_difference_db_per_octave != pytest.approx(40.0)
    assert result.ripple_rms_difference_db == pytest.approx(3.0)


def test_direct_time_difference_is_left_minus_right_and_positive_when_right_arrives_first() -> None:
    """把距離差方向反寫、或忘記轉毫秒，都會讓這題紅。"""
    receivers = _receivers()
    group = _group()
    point = _point(
        receivers,
        group,
        "main",
        left_distance_m=3.43,
        right_distance_m=0.0,
    )

    payload = _payload(
        _evaluate(receivers, group, (point, _point(receivers, group, "front")))
    )
    result = next(item for item in payload.point_results if item.receiver_id == "main")

    difference_ms = result.direct_time_difference_ms
    assert difference_ms is not None
    assert difference_ms == pytest.approx(10.0)
    # 正值＝右聲道先到（票 #350 第 2 格拍的方向）。
    assert difference_ms > 0.0


def test_broadband_level_sums_linear_energy_before_converting_the_ratio_to_db() -> None:
    """逐頻率先轉 dB 再平均會得到另一個答案，不能冒充寬頻能量差。"""
    receivers = _receivers()
    group = _group()
    point = _point(
        receivers,
        group,
        "main",
        left_energy=(1.0, 9.0),
        right_energy=(4.0, 4.0),
    )

    payload = _payload(
        _evaluate(receivers, group, (point, _point(receivers, group, "front")))
    )
    result = next(item for item in payload.point_results if item.receiver_id == "main")

    assert result.broadband_level_difference_db == pytest.approx(
        10.0 * math.log10(10.0 / 8.0)
    )


def test_extra_channel_is_preserved_but_not_automatically_added_to_comparisons() -> None:
    """加中置只能擴充清單；未明列的中置比較不得自動出現在結果裡。"""
    receivers = _receivers()
    group = _group(include_center=True)
    point = _point(receivers, group, "main")

    payload = _payload(
        _evaluate(receivers, group, (point, _point(receivers, group, "front")))
    )

    assert {item.role for item in payload.channels} == {"left", "right", "center"}
    assert {(item.left_role, item.right_role) for item in payload.comparisons} == {
        ("left", "right")
    }
    assert {item.role for item in payload.point_sources[0].channels} == {
        "left",
        "right",
        "center",
    }
