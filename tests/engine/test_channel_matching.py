"""聲道匹配第一層評估器考卷（票 #350 第一段）。"""
from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from typing import Final, Literal, cast

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
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
    cost_channel_matching_evaluation,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CategoryEvaluation,
    ChannelMatchingPayload,
    EvaluationState,
    Feature,
    Flag,
    InputProvenance,
    MetricState,
    ModelValidationStatus,
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
    features: Sequence[Feature] = (),
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
            settings_fingerprint=settings_fingerprint,
            evaluator_version=evaluator_version,
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
    timbre_settings_fingerprint: str = _TIMBRE_SETTINGS,
    listening_area_settings_fingerprint: str = _LISTENING_SETTINGS,
    evaluator_version: str = "timbre-fixture-v1",
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
            settings_fingerprint=timbre_settings_fingerprint,
            evaluator_version=evaluator_version,
        ),
        _response(
            receiver_id,
            "right",
            tilt=right_tilt,
            ripple=right_ripple,
            energy=right_energy,
            distance_m=right_distance_m,
            settings_fingerprint=timbre_settings_fingerprint,
            evaluator_version=evaluator_version,
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
                settings_fingerprint=timbre_settings_fingerprint,
                evaluator_version=evaluator_version,
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
) -> CategoryEvaluation:
    return evaluate_channel_matching(
        receivers,
        points,
        candidate_id=_CANDIDATE,
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


def _two_pair_evaluation() -> CategoryEvaluation:
    receivers = _receivers()
    base = _group(include_center=True)
    group = ChannelGroup(
        channels=base.channels,
        comparisons=(
            *base.comparisons,
            ChannelComparison(left_role="right", right_role="center"),
        ),
        feature_match_tolerance_hz=base.feature_match_tolerance_hz,
    )
    return _evaluate(receivers, group, _channel_points(receivers, group))


def _costed(evaluation: CategoryEvaluation) -> CategoryEvaluation:
    registry = load_quality_targets(_TARGETS)
    return cost_channel_matching_evaluation(
        evaluation,
        registry.purpose(_PURPOSE),
        registry.fingerprint,
    )


def _broadband_mean(aggregate: dict[str, object]) -> float:
    summary = cast(dict[str, object], aggregate["broadband_level_difference"])
    return cast(float, summary["weighted_mean_absolute_difference"])


def test_contract_rejects_duplicate_aggregate_for_the_better_comparison() -> None:
    """重讀時重複較好比較對，不得讓逐列平均把它的權重偷偷加倍。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    aggregates = cast(tuple[dict[str, object], ...], document["aggregates"])
    better = min(aggregates, key=_broadband_mean)
    left = cast(str, better["left_role"])
    right = cast(str, better["right_role"])
    document["aggregates"] = (*aggregates, deepcopy(better))

    with pytest.raises(ValidationError, match=rf"{left}/{right}.*彙總列"):
        ChannelMatchingPayload.model_validate(document)


def test_contract_rejects_duplicate_receiver_and_comparison_result() -> None:
    """同一接收點與比較對只能有一列，否則重讀後的彙總身分不唯一。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    results = cast(tuple[dict[str, object], ...], document["point_results"])
    repeated = results[0]
    receiver = cast(str, repeated["receiver_id"])
    left = cast(str, repeated["left_role"])
    right = cast(str, repeated["right_role"])
    document["point_results"] = (*results, deepcopy(repeated))

    with pytest.raises(
        ValidationError,
        match=rf"{receiver}.*{left}/{right}.*逐點結果",
    ):
        ChannelMatchingPayload.model_validate(document)


@pytest.mark.parametrize(
    "receiver_list",
    ("assessed_receiver_ids", "unavailable_receiver_ids"),
)
def test_contract_rejects_duplicate_receiver_inside_each_aggregate_list(
    receiver_list: str,
) -> None:
    """已評與不可估清單各自都不能用重複接收點偽造列數。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    aggregates = list(cast(tuple[dict[str, object], ...], document["aggregates"]))
    changed = deepcopy(aggregates[0])
    assessed = cast(tuple[str, ...], changed["assessed_receiver_ids"])
    unavailable = cast(tuple[str, ...], changed["unavailable_receiver_ids"])
    receiver = assessed[0]
    if receiver_list == "assessed_receiver_ids":
        changed[receiver_list] = (*assessed, receiver)
    else:
        changed["assessed_receiver_ids"] = tuple(
            item for item in assessed if item != receiver
        )
        changed[receiver_list] = (*unavailable, receiver, receiver)
    aggregates[0] = changed
    document["aggregates"] = tuple(aggregates)
    left = cast(str, changed["left_role"])
    right = cast(str, changed["right_role"])

    with pytest.raises(
        ValidationError,
        match=rf"{left}/{right}.*{receiver_list}.*{receiver}",
    ):
        ChannelMatchingPayload.model_validate(document)


def test_contract_rejects_missing_comparison_result_for_a_receiver() -> None:
    """接收點仍存在於另一比較對時，不能漏掉其中一對的逐點列。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    results = cast(tuple[dict[str, object], ...], document["point_results"])
    missing = results[0]
    receiver = cast(str, missing["receiver_id"])
    left = cast(str, missing["left_role"])
    right = cast(str, missing["right_role"])
    document["point_results"] = results[1:]
    aggregates = list(cast(tuple[dict[str, object], ...], document["aggregates"]))
    for aggregate in aggregates:
        if (
            aggregate["left_role"] == left
            and aggregate["right_role"] == right
        ):
            assessed = cast(tuple[str, ...], aggregate["assessed_receiver_ids"])
            aggregate["assessed_receiver_ids"] = tuple(
                item for item in assessed if item != receiver
            )
    document["aggregates"] = tuple(aggregates)

    with pytest.raises(
        ValidationError,
        match=rf"{receiver}.*{left}/{right}.*缺逐點結果",
    ):
        ChannelMatchingPayload.model_validate(document)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("overlap", "同時列為已評與不可估"),
        ("missing", "漏掉逐點結果"),
        ("phantom", "沒有逐點結果"),
        ("misfiled", "歸類跟逐點結果不符"),
    ),
)
def test_contract_rejects_aggregate_receiver_membership_errors(
    mutation: str, message: str
) -> None:
    """彙總的接收點分類若重疊或漏列，不能再代表那一對的逐點結果。"""
    document = _payload(_two_pair_evaluation()).model_dump(mode="python")
    aggregates = list(cast(tuple[dict[str, object], ...], document["aggregates"]))
    changed = deepcopy(aggregates[0])
    assessed = cast(tuple[str, ...], changed["assessed_receiver_ids"])
    receiver = assessed[0]
    left = cast(str, changed["left_role"])
    right = cast(str, changed["right_role"])
    unavailable = cast(tuple[str, ...], changed["unavailable_receiver_ids"])
    if mutation == "overlap":
        changed["unavailable_receiver_ids"] = (*unavailable, receiver)
    elif mutation == "phantom":
        receiver = "receiver-without-results"
        changed["unavailable_receiver_ids"] = (*unavailable, receiver)
    elif mutation == "misfiled":
        changed["assessed_receiver_ids"] = tuple(
            item for item in assessed if item != receiver
        )
        changed["unavailable_receiver_ids"] = (*unavailable, receiver)
    else:
        changed["assessed_receiver_ids"] = tuple(
            item for item in assessed if item != receiver
        )
    aggregates[0] = changed
    document["aggregates"] = tuple(aggregates)

    with pytest.raises(
        ValidationError,
        match=rf"{left}/{right}.*{receiver}.*{message}",
    ):
        ChannelMatchingPayload.model_validate(document)


def test_contract_and_cost_are_independent_of_result_row_order() -> None:
    """彙總與逐點列換序後，重讀、類代價及淘汰原因都必須相同。"""
    evaluation = _two_pair_evaluation()
    document = evaluation.model_dump(mode="python")
    payload = cast(dict[str, object], document["payload"])
    aggregates = cast(tuple[dict[str, object], ...], payload["aggregates"])
    results = cast(tuple[dict[str, object], ...], payload["point_results"])
    payload["aggregates"] = tuple(reversed(aggregates))
    payload["point_results"] = tuple(reversed(results))
    reordered = CategoryEvaluation.model_validate(document)

    registry = load_quality_targets(_TARGETS)
    purpose = registry.purpose(_PURPOSE)
    original_costed = _costed(evaluation)
    reordered_costed = _costed(reordered)

    assert original_costed.category_cost is not None
    assert reordered_costed.category_cost is not None
    assert reordered_costed.category_cost.value == pytest.approx(
        original_costed.category_cost.value
    )
    assert channel_matching_floor_reasons(
        reordered_costed, purpose
    ) == channel_matching_floor_reasons(original_costed, purpose)


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
