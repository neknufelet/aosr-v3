"""音色的主要分項、底線保護、報表欄位與登記簿來源。"""

from __future__ import annotations

import json
from typing import Final

from aosr.config.quality_targets import EntryStatus, QualityPurpose, Unit
from aosr.scoring.contract import (
    CategoryCost,
    CategoryEvaluation,
    EvaluationState,
    Feature,
    Flag,
    TimbreChannelsPayload,
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
TIMBRE_TARGET_UNITS: Final[dict[str, Unit]] = {
    _TILT_KEY: "dB/oct",
    _RESIDUAL_KEY: "dB",
    _DEVIATION_KEY: "dB",
    _PEAK_KEY: "dB",
    _DIP_KEY: "dB",
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
        target = _target(purpose, key, TIMBRE_TARGET_UNITS[key])
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
        "tilt": _shape_cost(
            _target(purpose, _TILT_KEY, TIMBRE_TARGET_UNITS[_TILT_KEY]),
            (payload.tilt_db_per_octave,),
        ),
        "residual_rms": _shape_cost(
            _target(purpose, _RESIDUAL_KEY, TIMBRE_TARGET_UNITS[_RESIDUAL_KEY]),
            (payload.residual_rms_db,),
        ),
        "target_deviation": _shape_cost(
            _target(purpose, _DEVIATION_KEY, TIMBRE_TARGET_UNITS[_DEVIATION_KEY]),
            (payload.target_deviation_rms_db,),
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
    """把已量音色升成 ``costed``；逐聲道各算一次後取類代價算術平均。"""
    if evaluation.state is not EvaluationState.MEASURED:
        raise ValueError("音色代價只接 measured 評估")
    payload = evaluation.payload
    channels: tuple[tuple[str | None, TimbrePayload], ...]
    if isinstance(payload, TimbrePayload):
        channels = ((None, payload),)
    elif isinstance(payload, TimbreChannelsPayload):
        channels = tuple((item.role, item.payload) for item in payload.channels)
    else:
        raise TypeError("音色代價必須收到單支或逐聲道音色 payload")
    target_tilt = _scalar(
        _target(purpose, _TILT_KEY, TIMBRE_TARGET_UNITS[_TILT_KEY])
    )
    if any(item.target_tilt_db_per_octave != target_tilt for _, item in channels):
        raise ValueError("評估器用的目標傾斜與登記簿不同，代價沒有唯一答案")
    weights = principal_weights(purpose)
    per_channel = tuple(
        (role, _components(item, purpose)) for role, item in channels
    )
    channel_costs = tuple(
        sum(parts[name] * weight for name, weight in weights.items())
        for _, parts in per_channel
    )
    components = {
        name if role is None else f"{role}.{name}": value
        for role, parts in per_channel
        for name, value in parts.items()
    }
    category_cost = CategoryCost(
        value=sum(channel_costs) / len(channel_costs),
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
        (key, _target(purpose, key, TIMBRE_TARGET_UNITS[key]).status)
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
    channels: tuple[TimbrePayload, ...]
    if isinstance(payload, TimbrePayload):
        channels = (payload,)
    elif isinstance(payload, TimbreChannelsPayload):
        channels = tuple(item.payload for item in payload.channels)
    else:
        return ()
    protections = tuple(protection_costs(item, purpose) for item in channels)
    reasons: list[str] = []
    if any(item["peak"] > 0.0 for item in protections):
        reasons.append("timbre_peak_beyond_limit")
    if any(item["dip"] > 0.0 for item in protections):
        reasons.append("timbre_dip_beyond_limit")
    return tuple(reasons)


def comparison_support(evaluation: CategoryEvaluation) -> str:
    """回音色實際比較的聲道組與主位之可讀正規 JSON。"""
    payload = evaluation.payload
    if not isinstance(payload, TimbreChannelsPayload):
        return ""
    return json.dumps(
        {
            "channel_group_fingerprint": payload.channel_group_fingerprint,
            "primary_receiver_id": payload.primary_receiver_id,
            "timbre_evaluator_version": payload.timbre_evaluator_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


# category_registry 只靠這些共同名字載入各類；新增類時排名層不需要再加分支。
cost_evaluation = cost_timbre_evaluation
registry_sources = timbre_registry_sources
floor_reasons = timbre_floor_reasons
