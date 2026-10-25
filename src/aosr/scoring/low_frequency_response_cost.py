"""低頻頻響類代價只呼叫第一階段合分；外層類權重由排名登記簿提供。"""
from __future__ import annotations

import hashlib

from aosr.config.quality_targets import EntryStatus, QualityPurpose, WeightTable
from aosr.scoring.contract import CategoryCost, CategoryEvaluation, EvaluationState
from aosr.scoring.cost_shapes import shape_cost
from aosr.scoring.low_frequency_response_contract import LowFrequencyResponsePayload
from aosr.scoring.stage_one_score import SETTING_KEYS, score_stage_one, stage_one_settings_from_section

# 這類的尺住第一階段那一段；排名註冊表據此選同用途的 section（設定段）。
settings_scope = "stage_one"
evaluator_keys = SETTING_KEYS


def cost_evaluation(
    evaluation: CategoryEvaluation, purpose: QualityPurpose, cost_settings_fingerprint: str,
) -> CategoryEvaluation:
    """用當次第一階段設定合分，不信上游代價；不另外抄公式或另設淘汰條件。"""
    if evaluation.state is not EvaluationState.MEASURED:
        raise ValueError("低頻頻響代價只接 measured 評估")
    payload = evaluation.payload
    if not isinstance(payload, LowFrequencyResponsePayload):
        raise TypeError("低頻頻響代價必須收到 LowFrequencyResponsePayload")
    settings = stage_one_settings_from_section(purpose)
    score = score_stage_one(payload.evaluation, settings)
    if score.screening_total is None:
        raise ValueError("可估的低頻頻響沒有第一階段合分")
    fingerprint = hashlib.sha256(
        (cost_settings_fingerprint + ":" + settings.fingerprint).encode("utf-8")).hexdigest()
    return CategoryEvaluation.model_validate({**evaluation.model_dump(),
        "state": EvaluationState.COSTED,
        "category_cost": CategoryCost(value=score.screening_total,
            # 分項跟別類同一個寫法：未乘權重的代價，權重另列；分項乘權重加總才是類代價。
            components={item.name: shape_cost(settings.target(item.name), (item.value_db,))
                        for item in score.components},
            cost_settings_fingerprint=fingerprint)})


def registry_sources(purpose: QualityPurpose) -> tuple[tuple[str, EntryStatus], ...]:
    """排名表頭列出真正使用的第一階段條目；沒有資格、底線或複核警戒。"""
    records: list[tuple[str, EntryStatus]] = []
    for key in SETTING_KEYS:
        entry = purpose.entry(key)
        if isinstance(entry, WeightTable):
            records.extend((f"{key}.{item.name}", item.status) for item in entry.item)
        else:
            records.append((key, entry.status))
    return tuple(records)
