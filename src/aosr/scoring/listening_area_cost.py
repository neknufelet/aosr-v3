"""聆聽區穩定性的容許帶代價與底線保護。"""
from __future__ import annotations

from typing import Final

from aosr.config.quality_targets import QualityPurpose, TargetEntry
from aosr.scoring.contract import (
    CategoryCost,
    CategoryEvaluation,
    DeviationAggregate,
    EvaluationState,
    Flag,
    ListeningAreaStabilityPayload,
    StabilityComparison,
)
from aosr.scoring.cost_shapes import (
    ComponentRole,
    shape_cost as _shape_cost,
    target as _target,
    weight_table as _weight_table,
)


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


def _comparison_aggregates(
    comparison: StabilityComparison,
) -> dict[str, DeviationAggregate]:
    """回傳具名且真的存在的比較組；周圍彼此留空時不補一筆零。"""
    aggregates = {"primary_to_surrounding": comparison.primary_to_surrounding}
    if comparison.surrounding_to_surrounding is None:
        return aggregates
    aggregates["surrounding_to_surrounding"] = (
        comparison.surrounding_to_surrounding
    )
    return aggregates


def _mean_deviation_group_costs(
    comparison: StabilityComparison, target: TargetEntry
) -> dict[str, float]:
    """兩組主要偏差各自套容許帶，保留組名供報表追查。"""
    return {
        name: _shape_cost(target, (aggregate.weighted_mean_deviation,))
        for name, aggregate in _comparison_aggregates(comparison).items()
    }


def _mean_deviation_cost(
    comparison: StabilityComparison, target: TargetEntry
) -> float:
    """每組各自套容許帶後取較嚴重者，避免好的一組稀釋災難組。

    不取平均也讓缺少周圍彼此組時不會因分母從二變一而改變同一組的代價。
    """
    if target.cost_shape != "in_range_best":
        raise ValueError(f"{target.key} 是主要分項，cost_shape 必須是 in_range_best")
    return max(_mean_deviation_group_costs(comparison, target).values())


def _worst_deviation_group_costs(
    comparison: StabilityComparison, target: TargetEntry
) -> dict[str, float]:
    """兩組最差偏差各自套底線，保留組名供淘汰理由使用。"""
    return {
        name: _shape_cost(target, (aggregate.worst_deviation.value,))
        for name, aggregate in _comparison_aggregates(comparison).items()
    }


def _worst_deviation_cost(
    comparison: StabilityComparison, target: TargetEntry
) -> float:
    """兩組各自套底線門檻，取真的存在的組裡最嚴重的一筆。"""
    if target.cost_shape != "beyond_threshold_only":
        raise ValueError(
            f"{target.key} 是底線保護，cost_shape 必須是 beyond_threshold_only"
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
        mean_target = _target(purpose, _LISTENING_AREA_TARGET_KEYS[mean_name])
        worst_target = _target(purpose, _LISTENING_AREA_TARGET_KEYS[worst_name])
        mean_group_costs = _mean_deviation_group_costs(comparison, mean_target)
        worst_group_costs = _worst_deviation_group_costs(comparison, worst_target)
        components[mean_name] = _mean_deviation_cost(comparison, mean_target)
        components.update(
            (f"{mean_name}.{group}", cost)
            for group, cost in mean_group_costs.items()
        )
        components[worst_name] = _worst_deviation_cost(comparison, worst_target)
        components.update(
            (f"{worst_name}.{group}", cost)
            for group, cost in worst_group_costs.items()
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
    if not isinstance(payload, ListeningAreaStabilityPayload):
        raise TypeError("聆聽區代價必須收到 ListeningAreaStabilityPayload")
    components = _listening_area_components(payload, purpose)
    weights = _listening_area_principal_weights(purpose)
    value = sum(components[name] * weight for name, weight in weights.items())
    category_cost = CategoryCost(
        value=value,
        components=components,
        cost_settings_fingerprint=cost_settings_fingerprint,
    )
    comparisons = (
        payload.tilt_stability,
        payload.ripple_rms_stability,
        payload.overall_level_stability,
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
