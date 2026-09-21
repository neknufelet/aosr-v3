"""聲道匹配第一層評估器考卷（票 #350 第一段）。"""
from __future__ import annotations

import math
from functools import partial
from collections.abc import Sequence
from datetime import date
from typing import Final, Literal

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring.category_registry import EliminationReason
from aosr.scoring.channel_matching import (
    ChannelComparison,
    ChannelDefinition,
    ChannelGroup,
    ChannelPointInput,
    ChannelResponse,
    evaluate_channel_matching,
)
from aosr.scoring.channel_matching_cost import (
    channel_matching_floor_reasons,
    comparison_support,
    cost_channel_matching_evaluation,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    ChannelMatchingPayload,
    EvaluationState,
    Feature,
    Flag,
    InputProvenance,
    ModelValidationStatus,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    TimbrePayload,
)
from aosr.scoring.ranking import CandidateStatus, RankingContext, rank_candidates
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
    evaluator_version: str = "timbre-fixture-v1",
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
        model_validation_status=ModelValidationStatus.VALIDATED,
        model_validation_frequency_range_hz=(20.0, 8000.0),
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
        evaluator_version=evaluator_version,
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
    frequencies_hz: tuple[float, ...] = (100.0, 200.0),
    features: Sequence[Feature] = (),
    candidate_id: str = _CANDIDATE,
    settings_fingerprint: str = _TIMBRE_SETTINGS,
    evaluator_version: str = "timbre-fixture-v1",
) -> ChannelResponse:
    return ChannelResponse(
        role=role,
        timbre_evaluation=_timbre(
            receiver_id,
            role,
            tilt=tilt,
            ripple=ripple,
            features=features,
            candidate_id=candidate_id,
            settings_fingerprint=settings_fingerprint,
            evaluator_version=evaluator_version,
        ),
        frequencies_hz=frequencies_hz,
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
    left_frequencies_hz: tuple[float, ...] = (100.0, 200.0),
    right_frequencies_hz: tuple[float, ...] = (100.0, 200.0),
    candidate_id: str = _CANDIDATE,
    timbre_settings_fingerprint: str = _TIMBRE_SETTINGS,
    listening_area_settings_fingerprint: str = _LISTENING_SETTINGS,
    evaluator_version: str = "timbre-fixture-v1",
) -> ChannelPointInput:
    response = partial(
        _response,
        receiver_id,
        candidate_id=candidate_id,
        settings_fingerprint=timbre_settings_fingerprint,
        evaluator_version=evaluator_version,
    )
    responses = [
        response(
            "left",
            tilt=left_tilt,
            ripple=left_ripple,
            energy=left_energy,
            distance_m=left_distance_m,
            frequencies_hz=left_frequencies_hz,
            features=(_feature("dip", 100.0),),
        ),
        response(
            "right",
            tilt=right_tilt,
            ripple=right_ripple,
            energy=right_energy,
            distance_m=right_distance_m,
            frequencies_hz=right_frequencies_hz,
        ),
    ]
    if any(channel.role == "center" for channel in group.channels):
        responses.append(
            response(
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
        timbre_settings_fingerprint=timbre_settings_fingerprint,
        listening_area_settings_fingerprint=listening_area_settings_fingerprint,
        channel_group_fingerprint=group.fingerprint,
        responses=tuple(responses),
    )


def _evaluate(
    receivers: ReceiverSet,
    group: ChannelGroup,
    points: Sequence[ChannelPointInput],
    *,
    timbre_settings_fingerprint: str = _TIMBRE_SETTINGS,
    listening_area_settings_fingerprint: str = _LISTENING_SETTINGS,
    candidate_id: str = _CANDIDATE,
) -> CategoryEvaluation:
    return evaluate_channel_matching(
        receivers,
        points,
        candidate_id=candidate_id,
        timbre_settings_fingerprint=timbre_settings_fingerprint,
        listening_area_settings_fingerprint=listening_area_settings_fingerprint,
        channel_group=group,
        purpose=_PURPOSE,
        quality_targets_path=_TARGETS,
        sound_speed_m_s=343.0,
    )


def _payload(evaluation: CategoryEvaluation) -> ChannelMatchingPayload:
    assert evaluation.state == EvaluationState.MEASURED
    assert isinstance(evaluation.payload, ChannelMatchingPayload)
    return evaluation.payload


def _translated_receivers(receivers: ReceiverSet) -> ReceiverSet:
    translation = (0.1, 0.2, 0.3)
    return ReceiverSet(
        points=tuple(
            point.model_copy(
                update={
                    "position_m": tuple(
                        coordinate + offset
                        for coordinate, offset in zip(
                            point.position_m, translation, strict=True
                        )
                    )
                }
            )
            for point in receivers.points
        )
    )


def _channel_points(
    receivers: ReceiverSet,
    group: ChannelGroup,
    *,
    timbre_settings_fingerprint: str = _TIMBRE_SETTINGS,
    evaluator_version: str = "timbre-fixture-v1",
) -> tuple[ChannelPointInput, ...]:
    return tuple(
        _point(
            receivers,
            group,
            receiver_id,
            timbre_settings_fingerprint=timbre_settings_fingerprint,
            evaluator_version=evaluator_version,
        )
        for receiver_id in ("main", "front")
    )


def _channel_only_registry() -> QualityTargets:
    document = load_quality_targets(_TARGETS).model_dump(mode="json", by_alias=True)
    for row in document["purpose"][0]["qualification"]:
        if row["key"] == "ranking.mandatory_categories":
            row["value"] = ["channel_matching"]
        elif row["key"] == "ranking.optional_categories":
            row["value"] = [
                name for name in row["value"] if name != "channel_matching"
            ]
    return QualityTargets.model_validate(document)


def _candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=evaluation.candidate_id,
        provenance=evaluation.provenance,
        evaluations=(evaluation,),
    )


def _ranking_context(receivers: ReceiverSet, group: ChannelGroup) -> RankingContext:
    return RankingContext(
        purpose=_PURPOSE,
        receiver_set_fingerprint=receivers.fingerprint,
        channel_group_fingerprint=group.fingerprint,
        run_date=date(2026, 9, 21),
        engine_version="engine-fixture",
    )


@pytest.mark.parametrize(
    ("changed_settings", "changed_version"),
    (
        ("timbre-settings-b", "timbre-fixture-v1"),
        (_TIMBRE_SETTINGS, "timbre-fixture-v2"),
    ),
)
def test_channel_matching_identity_tracks_upstream_measurement_method(
    changed_settings: str, changed_version: str
) -> None:
    """整批音色設定或版本不同時，聲道匹配的對外身分必須不同。"""
    receivers = _receivers()
    group = _group()
    original = _evaluate(receivers, group, _channel_points(receivers, group))
    changed = _evaluate(
        receivers,
        group,
        _channel_points(
            receivers,
            group,
            timbre_settings_fingerprint=changed_settings,
            evaluator_version=changed_version,
        ),
        timbre_settings_fingerprint=changed_settings,
    )

    assert original.state is EvaluationState.MEASURED
    assert changed.state is EvaluationState.MEASURED
    assert original.settings_fingerprint != changed.settings_fingerprint


def test_unvalidated_flag_from_any_channel_reaches_the_comparison() -> None:
    """任何一支聲道的音色是用沒驗過的物理模型量的，左右比較的結果也要帶著那個標記。"""
    receivers = _receivers()
    group = _group()
    main, front = _channel_points(receivers, group)
    response = front.responses[0]
    flagged = response.model_copy(
        update={
            "timbre_evaluation": response.timbre_evaluation.model_copy(
                update={"flags": (Flag.UNVALIDATED,)}
            )
        }
    )
    front = front.model_copy(update={"responses": (flagged, *front.responses[1:])})

    evaluation = _evaluate(receivers, group, (main, front))

    assert evaluation.state is EvaluationState.MEASURED
    assert Flag.UNVALIDATED in evaluation.flags


def test_mixed_upstream_evaluator_versions_are_not_compared() -> None:
    """批內上游音色評估器版本不一致時不可比，原因指名是版本。"""
    receivers = _receivers()
    group = _group()
    main = _point(receivers, group, "main", evaluator_version="timbre-fixture-v1")
    front = _point(receivers, group, "front", evaluator_version="timbre-fixture-v2")

    evaluation = _evaluate(receivers, group, (main, front))

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.EVALUATOR_VERSION_MISMATCH,)


def test_channel_matching_identity_uses_relative_receiver_layout() -> None:
    """帶浮點尾數的平移保持同表；只改周圍點重要性則必須分表。"""
    receivers = _receivers()
    translated = _translated_receivers(receivers)
    front = receivers.points[1]
    changed = ReceiverSet(
        points=(
            receivers.points[0],
            front.model_copy(update={"importance": front.importance + 0.5}),
        )
    )
    original = _evaluate(receivers, _group(), _channel_points(receivers, _group()))
    moved = _evaluate(translated, _group(), _channel_points(translated, _group()))
    reweighted = _evaluate(changed, _group(), _channel_points(changed, _group()))

    assert receivers.fingerprint != translated.fingerprint
    assert receivers.layout_fingerprint == translated.layout_fingerprint
    assert original.settings_fingerprint == moved.settings_fingerprint
    assert original.settings_fingerprint != reweighted.settings_fingerprint


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


def test_unavailable_side_makes_the_whole_category_unavailable() -> None:
    """同點任一聲道不可估時整類拒算；輸出不再帶來源，但輸入兩邊仍原樣保留。"""
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

    evaluation = _evaluate(receivers, group, (main, front))

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.payload is None
    assert ReasonCode.CHANNEL_RESULT_UNAVAILABLE in evaluation.reason_codes
    assert ReasonCode.INSUFFICIENT_COVERAGE in evaluation.reason_codes
    assert {item.role for item in main.responses} == {"left", "right"}
    assert {item.timbre_evaluation.state for item in main.responses} == {
        EvaluationState.MEASURED,
        EvaluationState.UNAVAILABLE,
    }


def _with_unavailable_right_channel(point: ChannelPointInput) -> ChannelPointInput:
    """把這一點右聲道的音色評估換成不可估（資料不足），左聲道原樣。"""
    right = point.responses[1]
    unavailable = right.timbre_evaluation.model_copy(
        update={
            "state": EvaluationState.UNAVAILABLE,
            "payload": None,
            "raw_quantities": (),
            "reason_codes": (ReasonCode.INSUFFICIENT_COVERAGE,),
        }
    )
    responses = (
        point.responses[0],
        right.model_copy(update={"timbre_evaluation": unavailable}),
    )
    return point.model_copy(update={"responses": responses})


def test_missing_bad_surrounding_point_cannot_erase_level_floor_or_win_ranking() -> None:
    """3 dB 控制組可排名；8 dB 會淘汰，讓該點一側不可估只能整類拒算。"""
    receivers = _receivers()
    group = _group()

    def evaluated(candidate_id: str, surrounding_db: float) -> CategoryEvaluation:
        ratio = 10.0 ** (surrounding_db / 10.0)
        points = (
            _point(
                receivers,
                group,
                "main",
                left_energy=(1.0, 1.0),
                right_energy=(1.0, 1.0),
                candidate_id=candidate_id,
            ),
            _point(
                receivers,
                group,
                "front",
                left_energy=(ratio, ratio),
                right_energy=(1.0, 1.0),
                candidate_id=candidate_id,
            ),
        )
        return _evaluate(receivers, group, points, candidate_id=candidate_id)

    safe = evaluated("level-3db", 3.0)
    breached = evaluated("level-8db", 8.0)
    purpose = load_quality_targets(_TARGETS).purpose(_PURPOSE)
    breached_costed = cost_channel_matching_evaluation(
        breached, purpose, load_quality_targets(_TARGETS).fingerprint
    )
    front = _point(
        receivers,
        group,
        "front",
        left_energy=(10.0 ** 0.8,) * 2,
        right_energy=(1.0, 1.0),
        candidate_id="level-missing",
    )
    front = _with_unavailable_right_channel(front)
    missing = _evaluate(
        receivers,
        group,
        (_point(receivers, group, "main", candidate_id="level-missing"), front),
        candidate_id="level-missing",
    )
    ranked = rank_candidates(
        [_candidate(safe), _candidate(missing)],
        _channel_only_registry(),
        _ranking_context(receivers, group),
    )

    assert safe.state is EvaluationState.MEASURED
    assert breached.state is EvaluationState.MEASURED
    assert channel_matching_floor_reasons(breached_costed, purpose) == (
        EliminationReason.CHANNEL_MATCHING_LEVEL_WORST_BEYOND_LIMIT.value,
    )
    assert missing.state is EvaluationState.UNAVAILABLE
    assert missing.category_cost is None
    assert ranked.status_of("level-3db") is CandidateStatus.RANKABLE
    assert ranked.status_of("level-missing") is CandidateStatus.NOT_EVALUATED


def test_symmetric_high_frequency_crop_stays_measured_but_separates_tables() -> None:
    """左右同步砍掉 4.5 kHz 以上仍可估，但實際寬頻支撐不同所以不可同表。"""
    receivers = _receivers()
    group = _group()
    full_axis = (100.0, 200.0, 1000.0, 4000.0, 6000.0)
    short_axis = full_axis[:-1]

    def evaluated(candidate_id: str, axis: tuple[float, ...]) -> CategoryEvaluation:
        left = (1.0,) * (len(axis) - 1) + ((3.0,) if axis == full_axis else (1.0,))
        points = tuple(
            _point(
                receivers,
                group,
                receiver_id,
                left_frequencies_hz=axis,
                right_frequencies_hz=axis,
                left_energy=left,
                right_energy=(1.0,) * len(axis),
                candidate_id=candidate_id,
            )
            for receiver_id in ("main", "front")
        )
        return _evaluate(receivers, group, points, candidate_id=candidate_id)

    full = evaluated("full-axis", full_axis)
    cropped = evaluated("cropped-axis", short_axis)
    result = rank_candidates(
        [_candidate(full), _candidate(cropped)],
        _channel_only_registry(),
        _ranking_context(receivers, group),
    )

    assert full.state is EvaluationState.MEASURED
    assert cropped.state is EvaluationState.MEASURED
    assert comparison_support(full) != comparison_support(cropped)
    assert {
        result.status_of("full-axis"),
        result.status_of("cropped-axis"),
    } == {CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE}


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


def test_broadband_support_must_match_across_required_receivers() -> None:
    """各點即使左右各自同軸，實際用進寬頻音量的頻率支撐不同仍不可同類估分。"""
    receivers = _receivers()
    group = _group()
    full_axis = (100.0, 200.0, 1000.0, 4000.0, 6000.0)
    short_axis = full_axis[:-1]
    evaluation = _evaluate(
        receivers,
        group,
        (
            _point(
                receivers,
                group,
                "main",
                left_frequencies_hz=full_axis,
                right_frequencies_hz=full_axis,
                left_energy=(1.0,) * len(full_axis),
                right_energy=(1.0,) * len(full_axis),
            ),
            _point(
                receivers,
                group,
                "front",
                left_frequencies_hz=short_axis,
                right_frequencies_hz=short_axis,
                left_energy=(1.0,) * len(short_axis),
                right_energy=(1.0,) * len(short_axis),
            ),
        ),
    )

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert evaluation.reason_codes == (ReasonCode.FREQUENCY_AXIS_MISMATCH,)


def test_matching_broadband_support_is_recorded_without_requiring_range_ceiling() -> None:
    """所有點與聲道同樣只到 4 kHz 仍可估，payload 必須記下實際最低、最高與點數。"""
    receivers = _receivers()
    group = _group()
    frequencies = (100.0, 200.0, 1000.0, 4000.0)
    points = tuple(
        _point(
            receivers,
            group,
            receiver_id,
            left_frequencies_hz=frequencies,
            right_frequencies_hz=frequencies,
            left_energy=(1.0,) * len(frequencies),
            right_energy=(1.0,) * len(frequencies),
        )
        for receiver_id in ("main", "front")
    )

    payload = _payload(_evaluate(receivers, group, points))

    assert payload.broadband_support.lowest_frequency_hz == 100.0
    assert payload.broadband_support.highest_frequency_hz == 4000.0
    assert payload.broadband_support.frequency_count == len(frequencies)


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
