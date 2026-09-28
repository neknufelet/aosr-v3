"""聆聽區穩定性的容許帶代價與暫定複核警戒。"""

from __future__ import annotations

import json
from typing import Final, Literal, cast

from aosr.config.quality_targets import EntryStatus, QualityPurpose, TargetEntry, Unit
from aosr.scoring.contract import (
    CategoryCost,
    CategoryEvaluation,
    DeviationAggregate,
    EvaluationState,
    Flag,
    ListeningAreaChannelsPayload,
    ListeningAreaStabilityPayload,
    QualityCategory,
    StabilityComparison,
)
from aosr.scoring.cost_shapes import (
    ComponentRole,
    shape_cost as _shape_cost,
    target as _target,
    weight_table as _weight_table,
    scalar as _scalar,
)
from aosr.scoring.review_alert import ListeningAreaReviewAlert, ReviewAlert


_LISTENING_AREA_WEIGHTS_KEY: Final[str] = (
    "listening_area_stability.within_category_weights"
)
_LISTENING_AREA_ROLES: Final[dict[str, ComponentRole]] = {
    "tilt_weighted_mean_deviation": "principal",
    "ripple_rms_weighted_mean_deviation": "principal",
    "overall_level_weighted_mean_deviation": "principal",
    "tilt_worst_deviation": "protection",
    "ripple_rms_worst_deviation": "protection",
    "overall_level_worst_deviation": "protection",
}
_LISTENING_AREA_TARGET_KEYS: Final[dict[str, str]] = {
    name: f"listening_area_stability.{name}" for name in _LISTENING_AREA_ROLES
}
_LISTENING_AREA_TARGET_UNITS: Final[dict[str, Unit]] = {
    _LISTENING_AREA_TARGET_KEYS["tilt_weighted_mean_deviation"]: "dB/oct",
    _LISTENING_AREA_TARGET_KEYS["ripple_rms_weighted_mean_deviation"]: "dB",
    _LISTENING_AREA_TARGET_KEYS["overall_level_weighted_mean_deviation"]: "dB",
    _LISTENING_AREA_TARGET_KEYS["tilt_worst_deviation"]: "dB/oct",
    _LISTENING_AREA_TARGET_KEYS["ripple_rms_worst_deviation"]: "dB",
    _LISTENING_AREA_TARGET_KEYS["overall_level_worst_deviation"]: "dB",
}


def _comparison_aggregates(
    comparison: StabilityComparison,
) -> dict[str, DeviationAggregate]:
    """回傳具名且真的存在的比較組；周圍彼此留空時不補一筆零。"""
    aggregates = {"primary_to_surrounding": comparison.primary_to_surrounding}
    if comparison.surrounding_to_surrounding is None:
        return aggregates
    aggregates["surrounding_to_surrounding"] = comparison.surrounding_to_surrounding
    return aggregates


def _mean_deviation_group_costs(
    comparison: StabilityComparison, target: TargetEntry
) -> dict[str, float]:
    """兩組主要偏差各自套容許帶，保留組名供報表追查。"""
    return {
        name: _shape_cost(target, (aggregate.weighted_mean_deviation,))
        for name, aggregate in _comparison_aggregates(comparison).items()
    }


def _mean_deviation_cost(comparison: StabilityComparison, target: TargetEntry) -> float:
    """每組各自套容許帶後取較嚴重者，避免好的一組稀釋災難組。

    不取平均也讓缺少周圍彼此組時不會因分母從二變一而改變同一組的代價。
    """
    if target.cost_shape != "in_range_best":
        raise ValueError(f"{target.key} 是主要分項，cost_shape 必須是 in_range_best")
    return max(_mean_deviation_group_costs(comparison, target).values())


def _worst_deviation_group_costs(
    comparison: StabilityComparison, target: TargetEntry
) -> dict[str, float]:
    """兩組最差偏差各自套暫定線，保留組名供複核警戒使用。"""
    return {
        name: _shape_cost(target, (aggregate.worst_deviation.value,))
        for name, aggregate in _comparison_aggregates(comparison).items()
    }


def _worst_deviation_cost(
    comparison: StabilityComparison, target: TargetEntry
) -> float:
    """兩組各自套暫定線，取真的存在的組裡最嚴重的一筆。"""
    if target.cost_shape != "beyond_threshold_only":
        raise ValueError(
            f"{target.key} 是暫定警戒，cost_shape 必須是 beyond_threshold_only"
        )
    return max(_worst_deviation_group_costs(comparison, target).values())


def _listening_area_components(
    payload: ListeningAreaStabilityPayload, purpose: QualityPurpose
) -> dict[str, float]:
    """三種位置偏差的主分項與最差值保護；峰谷一致性只留在 payload。"""
    comparisons = {
        "tilt": payload.tilt_stability,
        "ripple_rms": payload.ripple_rms_stability,
        "overall_level": payload.overall_level_stability,
    }
    components: dict[str, float] = {}
    for name, comparison in comparisons.items():
        mean_name = f"{name}_weighted_mean_deviation"
        worst_name = f"{name}_worst_deviation"
        mean_key = _LISTENING_AREA_TARGET_KEYS[mean_name]
        worst_key = _LISTENING_AREA_TARGET_KEYS[worst_name]
        mean_target = _target(
            purpose, mean_key, _LISTENING_AREA_TARGET_UNITS[mean_key]
        )
        worst_target = _target(
            purpose, worst_key, _LISTENING_AREA_TARGET_UNITS[worst_key]
        )
        mean_group_costs = _mean_deviation_group_costs(comparison, mean_target)
        worst_group_costs = _worst_deviation_group_costs(comparison, worst_target)
        components[mean_name] = _mean_deviation_cost(comparison, mean_target)
        components.update(
            (f"{mean_name}.{group}", cost) for group, cost in mean_group_costs.items()
        )
        components[worst_name] = _worst_deviation_cost(comparison, worst_target)
        components.update(
            (f"{worst_name}.{group}", cost) for group, cost in worst_group_costs.items()
        )
    return components


def _listening_area_principal_weights(
    purpose: QualityPurpose,
) -> dict[str, float]:
    """聆聽區權重名稱必須剛好涵蓋三個主要分項。"""
    weights = {
        item.name: item.value
        for item in _weight_table(purpose, _LISTENING_AREA_WEIGHTS_KEY).item
    }
    principal = {
        name for name, role in _LISTENING_AREA_ROLES.items() if role == "principal"
    }
    if set(weights) != principal:
        raise ValueError(
            f"{_LISTENING_AREA_WEIGHTS_KEY} 的名稱必須剛好是 {sorted(principal)}"
        )
    return weights


def cost_listening_area_evaluation(
    evaluation: CategoryEvaluation,
    purpose: QualityPurpose,
    cost_settings_fingerprint: str,
) -> CategoryEvaluation:
    """把已量的聆聽區穩定性升成新的 ``costed`` 物件；輸入不動。"""
    if evaluation.state is not EvaluationState.MEASURED:
        raise ValueError("聆聽區代價只接 measured 評估")
    payload = evaluation.payload
    channels: tuple[tuple[str | None, ListeningAreaStabilityPayload], ...]
    if isinstance(payload, ListeningAreaStabilityPayload):
        channels = ((None, payload),)
    elif isinstance(payload, ListeningAreaChannelsPayload):
        channels = tuple((item.role, item.payload) for item in payload.channels)
    else:
        raise TypeError("聆聽區代價必須收到單支或逐聲道聆聽區 payload")
    weights = _listening_area_principal_weights(purpose)
    per_channel = tuple(
        (role, _listening_area_components(item, purpose)) for role, item in channels
    )
    channel_costs = tuple(
        sum(parts[name] * weight for name, weight in weights.items())
        for _, parts in per_channel
    )
    components = {
        name if role is None else f"{role}.{name}": value
        for role, parts in per_channel for name, value in parts.items()
    }
    category_cost = CategoryCost(
        value=sum(channel_costs) / len(channel_costs),
        components=components,
        cost_settings_fingerprint=cost_settings_fingerprint,
    )
    comparisons = tuple(
        comparison for _, item in channels
        for comparison in (item.tilt_stability, item.ripple_rms_stability,
                           item.overall_level_stability)
    )
    flags = evaluation.flags
    if any(item.surrounding_to_surrounding is None for item in comparisons):
        missing_flag = Flag.LISTENING_AREA_PEER_GROUP_MISSING
        if missing_flag not in flags:
            flags = (*flags, missing_flag)
    document = evaluation.model_dump(mode="python")
    document.update(
        state=EvaluationState.COSTED,
        category_cost=category_cost,
        flags=flags,
    )
    return CategoryEvaluation.model_validate(document)


def listening_area_registry_sources(
    purpose: QualityPurpose,
) -> tuple[tuple[str, EntryStatus], ...]:
    """回排名真正讀過的聆聽區登記簿列。"""
    records = [
        (key, _target(purpose, key, _LISTENING_AREA_TARGET_UNITS[key]).status)
        for key in _LISTENING_AREA_TARGET_KEYS.values()
    ]
    records.extend(
        (f"{_LISTENING_AREA_WEIGHTS_KEY}.{item.name}", item.status)
        for item in _weight_table(purpose, _LISTENING_AREA_WEIGHTS_KEY).item
    )
    return tuple(records)


def listening_area_floor_reasons(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> tuple[str, ...]:
    """票 #519：聆聽區三條線只掛複核警戒，不產生淘汰原因。"""
    del evaluation, purpose
    return ()


def _channel_alerts(
    payload: ListeningAreaStabilityPayload, purpose: QualityPurpose,
    role: str, speaker_id: str, components: dict[str, float],
) -> tuple[ReviewAlert, ...]:
    alerts: list[ReviewAlert] = []
    for metric, comparison in (
        ("tilt", payload.tilt_stability),
        ("ripple_rms", payload.ripple_rms_stability),
        ("overall_level", payload.overall_level_stability),
    ):
        key = _LISTENING_AREA_TARGET_KEYS[f"{metric}_worst_deviation"]
        target = _target(purpose, key, _LISTENING_AREA_TARGET_UNITS[key])
        limit = _scalar(target)
        metric_name = {"tilt": "傾斜", "ripple_rms": "起伏均方根",
                       "overall_level": "寬頻音量"}[metric]
        for group, aggregate in _comparison_aggregates(comparison).items():
            name = (f"{metric}_worst_deviation.{group}" if role == "single" else
                    f"{role}.{metric}_worst_deviation.{group}")
            if name not in components:
                raise ValueError(f"聆聽區現有的比較組缺最差值分項：{name}")
            if components[name] <= 0.0:
                continue
            worst = aggregate.worst_deviation
            group_name = {"primary_to_surrounding": "主位對周圍",
                          "surrounding_to_surrounding": "周圍彼此"}[group]
            alerts.append(ListeningAreaReviewAlert(
                category=QualityCategory.LISTENING_AREA_STABILITY,
                role=role, speaker_id=speaker_id,
                metric=cast(Literal["tilt", "ripple_rms", "overall_level"], metric),
                group=cast(Literal["primary_to_surrounding", "surrounding_to_surrounding"], group),
                receiver_id=worst.receiver.receiver_id,
                reference_id=worst.reference.receiver_id,
                deviation=worst.value, limit=limit,
                note=(f"{role} 聲道 {speaker_id} 的{metric_name}（{group_name}）："
                      f"{worst.receiver.receiver_id} 對 {worst.reference.receiver_id} "
                      f"差 {worst.value:g} {target.unit}，超過暫定線 {limit:g} {target.unit}；"
                      "超標、待複核，尚未正式校準"),
            ))
    return tuple(alerts)


def listening_area_review_alerts(
    evaluation: CategoryEvaluation, purpose: QualityPurpose,
) -> tuple[ReviewAlert, ...]:
    """逐聲道、逐量、逐組把超出暫定線的最差位置差掛為警戒。"""
    payload = evaluation.payload
    if not isinstance(payload, (ListeningAreaStabilityPayload, ListeningAreaChannelsPayload)):
        return ()
    cost = evaluation.category_cost
    if cost is None:
        raise ValueError("costed 聆聽區評估缺 category_cost")
    if isinstance(payload, ListeningAreaStabilityPayload):
        return _channel_alerts(payload, purpose, "single", payload.speaker_id,
                               cost.components)
    return tuple(alert for channel in payload.channels
                 for alert in _channel_alerts(channel.payload, purpose, channel.role,
                                              channel.speaker_id, cost.components))


cost_evaluation = cost_listening_area_evaluation
registry_sources = listening_area_registry_sources
floor_reasons = listening_area_floor_reasons
review_alerts = listening_area_review_alerts


def comparison_support(evaluation: CategoryEvaluation) -> str:
    """#489 聆聽區完整頻率支撐的正規 JSON；不同支撐先分表。"""
    payload = evaluation.payload
    if isinstance(payload, ListeningAreaChannelsPayload):
        document: dict[str, object] = {
            "channel_group_fingerprint": payload.channel_group_fingerprint,
            "channels": [
                {"role": item.role, "speaker_id": item.speaker_id,
                 "frequency_support": _frequency_support(item.payload)}
                for item in sorted(payload.channels, key=lambda channel: channel.role)
            ],
            "listening_area_evaluator_version": payload.listening_area_evaluator_version,
        }
    elif isinstance(payload, ListeningAreaStabilityPayload):
        document = _frequency_support(payload)
    else:
        return ""
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    )


def _frequency_support(payload: ListeningAreaStabilityPayload) -> dict[str, object]:
    document = payload.frequency_support.model_dump(mode="json")
    # 接收點清單可以重排，比較身分先照代號排好。
    document["points"] = sorted(document["points"], key=lambda point: str(point["receiver_id"]))
    return document
