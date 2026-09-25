"""音色的主要分項、峰谷複核警戒（review alert）、報表欄位與登記簿來源。"""

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
    QualityCategory,
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
from aosr.scoring.review_alert import PeakDipReviewAlert, ReviewAlert


_WEIGHTS_KEY: Final[str] = "timbre_balance.within_category_weights"
_TILT_KEY: Final[str] = "timbre_balance.target_tilt_db_per_octave"
_RESIDUAL_KEY: Final[str] = "timbre_balance.residual_rms_db"
_DEVIATION_KEY: Final[str] = "timbre_balance.target_deviation_rms_db"
_PEAK_KEY: Final[str] = "timbre_balance.peak_depth_db"
_DIP_KEY: Final[str] = "timbre_balance.dip_depth_db"

# 主要分項進類代價、對照只印；峰谷保護分項只印並另掛複核警戒（票 #445）。
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
    """峰或谷之中進分項代價的深度；太窄的保留在清單上、但不進代價。"""
    return tuple(
        feature.depth_db
        for feature in features
        if feature.kind == kind and Flag.FEATURE_TOO_NARROW not in feature.flags
    )


def protection_costs(
    payload: TimbrePayload, purpose: QualityPurpose
) -> dict[str, float]:
    """峰、谷各自超出複核警戒的分項代價；大於零不代表淘汰。"""
    costs: dict[str, float] = {}
    for kind, key in (("peak", _PEAK_KEY), ("dip", _DIP_KEY)):
        target = _target(purpose, key, TIMBRE_TARGET_UNITS[key])
        if target.cost_shape != "beyond_threshold_only":
            raise ValueError(
                f"{key} 是複核警戒，cost_shape 必須是 beyond_threshold_only"
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
    """把已量音色升成 ``costed``；逐聲道各算一次後取類代價算術平均。

    輸入不動。
    """
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
    """票 #445：音色今天沒有淘汰底線；峰谷只掛複核警戒並照算代價。"""
    del evaluation, purpose
    return ()


def _alert_note(feature: Feature, limit_db: float) -> str:
    """把一個峰谷警戒寫成人能直接讀的中文。"""
    name = "峰" if feature.kind == "peak" else "谷"
    depth = (
        f"+{feature.depth_db:g}"
        if feature.kind == "peak"
        else f"−{abs(feature.depth_db):g}"
    )
    width = (
        "寬度未知（邊界不完整）"
        if feature.width_octave is None
        else f"寬 {feature.width_octave:g} 八度"
    )
    note = (
        f"{name} {depth} dB 於 {feature.center_frequency_hz:g} Hz、{width}，"
        f"超過警戒 {limit_db:g} dB，待複核"
    )
    if Flag.FEATURE_NARROWER_THAN_AXIS in feature.flags:
        note += "；窄於軸解析度，待加密確認"
    return note


def _payload_alerts(
    payload: TimbrePayload,
    purpose: QualityPurpose,
    speaker_id: str,
    receiver_id: str,
) -> tuple[ReviewAlert, ...]:
    """一支聲道每一個超過登記簿警戒的峰或谷各回一筆。"""
    limits = {
        "peak": _scalar(_target(purpose, _PEAK_KEY, TIMBRE_TARGET_UNITS[_PEAK_KEY])),
        "dip": _scalar(_target(purpose, _DIP_KEY, TIMBRE_TARGET_UNITS[_DIP_KEY])),
    }
    return tuple(
        PeakDipReviewAlert(
            category=QualityCategory.TIMBRE_BALANCE,
            speaker_id=speaker_id,
            receiver_id=receiver_id,
            kind=feature.kind,
            center_frequency_hz=feature.center_frequency_hz,
            depth_db=feature.depth_db,
            width_octave=feature.width_octave,
            limit_db=limits[feature.kind],
            narrower_than_axis=Flag.FEATURE_NARROWER_THAN_AXIS in feature.flags,
            note=_alert_note(feature, limits[feature.kind]),
        )
        for feature in payload.features
        if Flag.FEATURE_TOO_NARROW not in feature.flags
        and abs(feature.depth_db) > limits[feature.kind]
    )


def timbre_review_alerts(
    evaluation: CategoryEvaluation, purpose: QualityPurpose
) -> tuple[ReviewAlert, ...]:
    """單支或逐聲道音色逐一產生複核警戒（review alert），不彙成最差一筆。"""
    payload = evaluation.payload
    if isinstance(payload, TimbrePayload):
        return _payload_alerts(
            payload,
            purpose,
            evaluation.provenance.speaker_id,
            evaluation.provenance.receiver_id,
        )
    if isinstance(payload, TimbreChannelsPayload):
        return tuple(
            alert
            for channel in payload.channels
            for alert in _payload_alerts(
                channel.payload,
                purpose,
                channel.speaker_id,
                channel.provenance.receiver_id,
            )
        )
    return ()


def comparison_support(evaluation: CategoryEvaluation) -> str:
    """回音色實際比較的聲道組與主位之可讀正規 JSON。

    #489 各聲道附實際頻率支撐的完整序列。
    """
    payload = evaluation.payload
    if not isinstance(payload, TimbreChannelsPayload):
        return ""
    return json.dumps(
        {
            "channel_group_fingerprint": payload.channel_group_fingerprint,
            "channels": [
                {"role": item.role, "speaker_id": item.speaker_id,
                 "frequencies_hz": item.payload.frequency_support_hz}
                # 聲道照角色排：聲道組指紋也照角色排，宣告順序不同不該分表。
                for item in sorted(payload.channels, key=lambda channel: channel.role)
            ],
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
review_alerts = timbre_review_alerts
