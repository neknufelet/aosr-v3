"""聲道匹配第二層：逐點彙總的容許帶代價、最差值保護與時間開關。"""

from __future__ import annotations

import json
from typing import Final

from aosr.config.quality_targets import (
    EntryStatus,
    QualityPurpose,
    SettingEntry,
    TargetEntry,
    Unit,
)
from aosr.scoring.contract import (
    CategoryCost,
    CategoryEvaluation,
    ChannelComparisonAggregate,
    ChannelMatchingPayload,
    EvaluationState,
)
from aosr.scoring.cost_shapes import (
    shape_cost as _shape_cost,
    target as _target,
    weight_table as _weight_table,
)


_SWITCH_KEY: Final[str] = "channel_matching.direct_time_cost_enabled"
_WEIGHTS_KEY: Final[str] = "channel_matching.within_category_weights"
_METRICS: Final[tuple[str, ...]] = (
    "tilt_difference",
    "ripple_rms_difference",
    "broadband_level_difference",
    "direct_time_difference",
)
_MEAN_KEYS: Final[dict[str, str]] = {
    name: f"channel_matching.{name}_weighted_mean" for name in _METRICS
}
_WORST_KEYS: Final[dict[str, str]] = {
    name: f"channel_matching.{name}_worst" for name in _METRICS
}
_FLOOR_REASONS: Final[dict[str, str]] = {
    "tilt_difference": "channel_matching_tilt_worst_beyond_limit",
    "ripple_rms_difference": "channel_matching_ripple_worst_beyond_limit",
    "broadband_level_difference": "channel_matching_level_worst_beyond_limit",
    "direct_time_difference": "channel_matching_direct_time_worst_beyond_limit",
}
_METRIC_UNITS: Final[dict[str, Unit]] = {
    "tilt_difference": "dB/oct",
    "ripple_rms_difference": "dB",
    "broadband_level_difference": "dB",
    "direct_time_difference": "ms",
}
_SWITCH_UNIT: Final[Unit] = "1"


def _setting(
    purpose: QualityPurpose, key: str, expected_unit: Unit
) -> SettingEntry:
    entry = purpose.entry(key)
    if not isinstance(entry, SettingEntry):
        raise TypeError(f"{key} 不是量法設定")
    if entry.unit != expected_unit:
        raise ValueError(
            f"{key} 單位應為 {expected_unit}，登記簿寫 {entry.unit}"
        )
    return entry


def _switch(purpose: QualityPurpose) -> bool:
    value = _setting(purpose, _SWITCH_KEY, _SWITCH_UNIT).value
    if value not in (0, 1):
        raise ValueError(f"{_SWITCH_KEY} 必須是 0 或 1")
    return bool(value)


def _weights(purpose: QualityPurpose) -> dict[str, float]:
    weights = {
        item.name: item.value for item in _weight_table(purpose, _WEIGHTS_KEY).item
    }
    if set(weights) != set(_METRICS):
        raise ValueError(f"{_WEIGHTS_KEY} 的名稱必須剛好是 {sorted(_METRICS)}")
    return weights


def _targets(purpose: QualityPurpose, metric: str) -> tuple[TargetEntry, TargetEntry]:
    mean = _target(purpose, _MEAN_KEYS[metric], _METRIC_UNITS[metric])
    worst = _target(purpose, _WORST_KEYS[metric], _METRIC_UNITS[metric])
    if mean.cost_shape != "in_range_best":
        raise ValueError(f"{mean.key} 必須是 in_range_best")
    if worst.cost_shape != "beyond_threshold_only":
        raise ValueError(f"{worst.key} 必須是 beyond_threshold_only")
    return mean, worst


def _pair_name(aggregate: ChannelComparisonAggregate) -> str:
    return f"{aggregate.left_role}-{aggregate.right_role}"


def _metric_components(
    payload: ChannelMatchingPayload,
    purpose: QualityPurpose,
    metric: str,
) -> tuple[float, float, dict[str, float]] | None:
    mean_target, worst_target = _targets(purpose, metric)
    assessed = [
        (aggregate, getattr(aggregate, metric))
        for aggregate in payload.aggregates
        if getattr(aggregate, metric) is not None
    ]
    if not assessed:
        return None
    mean_costs = [
        (
            aggregate,
            _shape_cost(mean_target, (summary.weighted_mean_absolute_difference,)),
        )
        for aggregate, summary in assessed
        if summary is not None
    ]
    worst_costs = [
        (
            aggregate,
            summary,
            _shape_cost(worst_target, (summary.worst_absolute_difference,)),
        )
        for aggregate, summary in assessed
        if summary is not None
    ]
    mean_cost = sum(cost for _, cost in mean_costs) / len(mean_costs)
    worst_cost = max(cost for _, _, cost in worst_costs)
    details = {
        **{
            f"{metric}.weighted_mean.{_pair_name(aggregate)}": cost
            for aggregate, cost in mean_costs
        },
        **{
            (
                f"{metric}.worst.{_pair_name(aggregate)}."
                f"{summary.worst_receiver_id}"
            ): cost
            for aggregate, summary, cost in worst_costs
        },
    }
    return mean_cost, worst_cost, details


def _components(
    payload: ChannelMatchingPayload, purpose: QualityPurpose
) -> tuple[dict[str, float], dict[str, float]]:
    components: dict[str, float] = {}
    principal: dict[str, float] = {}
    active = _METRICS if payload.direct_time_cost_enabled else _METRICS[:-1]
    for metric in active:
        result = _metric_components(payload, purpose, metric)
        if result is None:
            continue
        mean_cost, worst_cost, details = result
        components[f"{metric}.weighted_mean"] = mean_cost
        components[f"{metric}.worst"] = worst_cost
        components.update(details)
        principal[metric] = mean_cost
    return components, principal


def cost_channel_matching_evaluation(
    evaluation: CategoryEvaluation,
    purpose: QualityPurpose,
    cost_settings_fingerprint: str,
) -> CategoryEvaluation:
    """把已量的逐點彙總換成類代價；診斷只留在 payload，不進代價。"""
    if evaluation.state is not EvaluationState.MEASURED:
        raise ValueError("聲道匹配代價只接 measured 評估")
    payload = evaluation.payload
    if not isinstance(payload, ChannelMatchingPayload):
        raise TypeError("聲道匹配代價必須收到 ChannelMatchingPayload")
    if payload.direct_time_cost_enabled != _switch(purpose):
        raise ValueError("評估器與代價設定的直達時間開關不同")
    components, principal = _components(payload, purpose)
    if not principal:
        return evaluation
    weights = _weights(purpose)
    denominator = sum(weights[name] for name in principal)
    # 選直接報錯：主分項明明存在卻沒有任何參與權重時，維持 measured 會把整類靜默消失。
    if denominator <= 0.0:
        raise ValueError("聲道匹配參與主代價的權重和必須大於零")
    value = sum(cost * weights[name] for name, cost in principal.items()) / denominator
    document = evaluation.model_dump(mode="python")
    document.update(
        state=EvaluationState.COSTED,
        category_cost=CategoryCost(
            value=value,
            components=components,
            cost_settings_fingerprint=cost_settings_fingerprint,
        ),
    )
    return CategoryEvaluation.model_validate(document)


def comparison_support(evaluation: CategoryEvaluation) -> str:
    """回寬頻音量實際使用的頻率支撐之可讀正規 JSON。"""
    payload = evaluation.payload
    if not isinstance(payload, ChannelMatchingPayload):
        return ""
    return json.dumps(
        payload.broadband_support.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )


def channel_matching_registry_sources(
    purpose: QualityPurpose,
) -> tuple[tuple[str, EntryStatus], ...]:
    """回排名真正讀過的聲道匹配登記簿列，供表頭判定 baseline。"""
    records = [(_SWITCH_KEY, _setting(purpose, _SWITCH_KEY, _SWITCH_UNIT).status)]
    records.extend(
        (key, _target(purpose, key, _METRIC_UNITS[metric]).status)
        for keys in (_MEAN_KEYS, _WORST_KEYS)
        for metric, key in keys.items()
    )
    records.extend(
        (f"{_WEIGHTS_KEY}.{item.name}", item.status)
        for item in _weight_table(purpose, _WEIGHTS_KEY).item
    )
    return tuple(records)


def channel_matching_floor_reasons(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> tuple[str, ...]:
    """回啟用量的最差值違反；該有 worst 明細卻全缺時報錯。

    已量的聲道匹配每一點每一對都是已量（票 #403），所以只有直達時間開關關著時不要求明細。
    """
    del purpose
    if not isinstance(evaluation.payload, ChannelMatchingPayload):
        return ()
    cost = evaluation.category_cost
    if cost is None:
        raise ValueError("costed 聲道匹配評估缺 category_cost")
    reasons: list[str] = []
    for metric, reason in _FLOOR_REASONS.items():
        if metric == "direct_time_difference" and not evaluation.payload.direct_time_cost_enabled:
            continue
        prefix = f"{metric}.worst."
        worst_costs = tuple(
            value
            for name, value in cost.components.items()
            if name.startswith(prefix)
        )
        if not worst_costs:
            raise ValueError(f"可估的 {metric} 缺 worst 分項明細")
        if any(value > 0.0 for value in worst_costs):
            reasons.append(reason)
    # 殘響目前仍走註冊表的空底線處理器；那個洞另開票，不在 #350 順手改。
    return tuple(reasons)


cost_evaluation = cost_channel_matching_evaluation
registry_sources = channel_matching_registry_sources
floor_reasons = channel_matching_floor_reasons
