"""音色的主要分項、底線保護、報表欄位與登記簿來源。"""

from __future__ import annotations

from typing import Final

from aosr.config.quality_targets import EntryStatus, QualityPurpose
from aosr.scoring.contract import (
    CategoryCost,
    CategoryEvaluation,
    EvaluationState,
    Feature,
    Flag,
    TimbrePayload,
)
from aosr.scoring.cost_shapes import (
    ComponentRole,
    scalar as _scalar,
    shape_cost as _shape_cost,
    target as _target,
    weight_table as _weight_table,
)


_WEIGHTS_KEY: Final[str] = "timbre_balance.within_category_weights"
_TILT_KEY: Final[str] = "timbre_balance.target_tilt_db_per_octave"
_RESIDUAL_KEY: Final[str] = "timbre_balance.residual_rms_db"
_DEVIATION_KEY: Final[str] = "timbre_balance.target_deviation_rms_db"
_PEAK_KEY: Final[str] = "timbre_balance.peak_depth_db"
_DIP_KEY: Final[str] = "timbre_balance.dip_depth_db"

# 主要分項進類代價、對照只印、保護只擋（#345 第 5 格，直線目標）。
TIMBRE_ROLES: Final[dict[str, ComponentRole]] = {
    "tilt": "principal",
    "residual_rms": "principal",
    "target_deviation": "reference",
    "peaks_dips": "protection",
}
TIMBRE_TARGET_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "tilt": (_TILT_KEY,),
    "residual_rms": (_RESIDUAL_KEY,),
    "target_deviation": (_DEVIATION_KEY,),
    "peaks_dips": (_PEAK_KEY, _DIP_KEY),
}


def counted_depths(features: tuple[Feature, ...], kind: str) -> tuple[float, ...]:
    """峰或谷之中進底線保護的深度；太窄的保留在清單上、但不進代價。"""
    return tuple(
        feature.depth_db
        for feature in features
        if feature.kind == kind and Flag.FEATURE_TOO_NARROW not in feature.flags
    )


def protection_costs(
    payload: TimbrePayload, purpose: QualityPurpose
) -> dict[str, float]:
    """峰、谷各自超出界線的代價；大於零就是踩到底線保護。"""
    costs: dict[str, float] = {}
    for kind, key in (("peak", _PEAK_KEY), ("dip", _DIP_KEY)):
        target = _target(purpose, key)
        if target.cost_shape != "beyond_threshold_only":
            raise ValueError(
                f"{key} 是底線保護，cost_shape 必須是 beyond_threshold_only"
            )
        costs[kind] = _shape_cost(target, counted_depths(payload.features, kind))
    return costs


def _components(payload: TimbrePayload, purpose: QualityPurpose) -> dict[str, float]:
    """音色四樣輸出各自的代價（#345 第 5 格的配法）。"""
    protection = protection_costs(payload, purpose)
    return {
        "tilt": _shape_cost(_target(purpose, _TILT_KEY), (payload.tilt_db_per_octave,)),
        "residual_rms": _shape_cost(
            _target(purpose, _RESIDUAL_KEY), (payload.residual_rms_db,)
        ),
        "target_deviation": _shape_cost(
            _target(purpose, _DEVIATION_KEY), (payload.target_deviation_rms_db,)
        ),
        "peaks_dips": protection["peak"] + protection["dip"],
    }


def principal_weights(purpose: QualityPurpose) -> dict[str, float]:
    """類內權重表的名稱必須剛好是主要分項。"""
    weights = {
        item.name: item.value for item in _weight_table(purpose, _WEIGHTS_KEY).item
    }
    principal = {name for name, role in TIMBRE_ROLES.items() if role == "principal"}
    if set(weights) != principal:
        raise ValueError(f"{_WEIGHTS_KEY} 的名稱必須剛好是 {sorted(principal)}")
    return weights


def cost_timbre_evaluation(
    evaluation: CategoryEvaluation,
    purpose: QualityPurpose,
    cost_settings_fingerprint: str,
) -> CategoryEvaluation:
    """把一條已量的音色輸出升成新的 ``costed`` 物件；輸入不動。"""
    if evaluation.state is not EvaluationState.MEASURED:
        raise ValueError("音色代價只接 measured 評估")
    payload = evaluation.payload
    if not isinstance(payload, TimbrePayload):
        raise TypeError("音色代價必須收到 TimbrePayload")
    if payload.target_tilt_db_per_octave != _scalar(_target(purpose, _TILT_KEY)):
        raise ValueError("評估器用的目標傾斜與登記簿不同，代價沒有唯一答案")
    components = _components(payload, purpose)
    weights = principal_weights(purpose)
    category_cost = CategoryCost(
        value=sum(components[name] * weight for name, weight in weights.items()),
        components=components,
        cost_settings_fingerprint=cost_settings_fingerprint,
    )
    document = evaluation.model_dump(mode="python")
    document.update(state=EvaluationState.COSTED, category_cost=category_cost)
    return CategoryEvaluation.model_validate(document)


def timbre_registry_sources(
    purpose: QualityPurpose,
) -> tuple[tuple[str, EntryStatus], ...]:
    """回排名真正讀過的音色登記簿列。"""
    records = [
        (key, _target(purpose, key).status)
        for keys in TIMBRE_TARGET_KEYS.values()
        for key in keys
    ]
    records.extend(
        (f"{_WEIGHTS_KEY}.{item.name}", item.status)
        for item in _weight_table(purpose, _WEIGHTS_KEY).item
    )
    return tuple(records)


def timbre_floor_reasons(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> tuple[str, ...]:
    """音色峰谷底線的全部違反原因。"""
    payload = evaluation.payload
    if not isinstance(payload, TimbrePayload):
        return ()
    protection = protection_costs(payload, purpose)
    reasons: list[str] = []
    if protection["peak"] > 0.0:
        reasons.append("timbre_peak_beyond_limit")
    if protection["dip"] > 0.0:
        reasons.append("timbre_dip_beyond_limit")
    return tuple(reasons)


# category_registry 只靠這些共同名字載入各類；新增類時排名層不需要再加分支。
cost_evaluation = cost_timbre_evaluation
registry_sources = timbre_registry_sources
floor_reasons = timbre_floor_reasons
