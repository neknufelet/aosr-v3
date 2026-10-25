"""第二階段低頻頻響：沿用第一階段有限元素量法與聲道、座位彙整。"""
from __future__ import annotations

from aosr.config.quality_targets import QualityTargets
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CategoryEvaluation, EvaluationState, QualityCategory, RawQuantity,
)
from aosr.scoring.contract_base import Flag, InputProvenance, ReasonCode
from aosr.scoring.low_frequency_response_contract import LowFrequencyResponsePayload
from aosr.scoring.placement import Placement
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import StageOneLowFrequencyInput
from aosr.scoring.stage_one_score import (
    COHERENT_SUM_NOT_EVALUATED, COHERENT_SUM_NOT_EVALUATED_TEXT,
    STAGE_ONE_SCORE_VERSION, score_stage_one, stage_one_score_settings,
)

EVALUATOR_VERSION = "aosr.scoring.low_frequency_response.v1"


def evaluate_low_frequency_response(
    data: StageOneLowFrequencyInput | None, *, registry: QualityTargets, purpose: str,
    scene_fingerprint: str, placement: Placement, provenance: InputProvenance,
    input_reason: ReasonCode | None = None, candidate_id: str | None = None,
) -> CategoryEvaluation:
    """只量存下的有限元素能量；輸入組裝失敗帶原因，不把搜尋場景條件當量法前提。"""
    settings = stage_one_score_settings(registry, purpose)
    flags = (Flag.BASELINE_SETTINGS,) if settings.status == "baseline" else ()
    payload = None
    raw: tuple[RawQuantity, ...] = ()
    reasons: tuple[ReasonCode, ...]
    if data is None:
        if input_reason is None or candidate_id is None:
            raise ValueError("缺輸入時必須交候選代號與原因")
        reasons = (input_reason,)
    else:
        candidate_id = data.candidate_id
        evaluation = evaluate_stage_one_low_frequency(data, scoring_range_hz=settings.scored_range_hz)
        score = score_stage_one(evaluation, settings)
        reasons = tuple(ReasonCode(code.value) for code in score.reason_codes)
        if score.state == "scored":
            if score.worst_channel is None:
                raise ValueError("第一階段已計分卻沒有最差聲道")
            payload = LowFrequencyResponsePayload(
                category="low_frequency_response", evaluation=evaluation,
                scored_range_hz=settings.scored_range_hz, score_version=STAGE_ONE_SCORE_VERSION,
                settings_fingerprint=settings.fingerprint, worst_channel=score.worst_channel,
                channel_summation=COHERENT_SUM_NOT_EVALUATED,
                channel_summation_text=COHERENT_SUM_NOT_EVALUATED_TEXT,
            )
            raw = tuple(RawQuantity(name=item.name, value=item.value_db, unit="dB")
                        for item in score.components)
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id=candidate_id,
        scene_fingerprint=scene_fingerprint, source_model_fingerprint=None, placement=placement,
        category=QualityCategory.LOW_FREQUENCY_RESPONSE,
        state=EvaluationState.MEASURED if payload is not None else EvaluationState.UNAVAILABLE,
        payload=payload, raw_quantities=raw, category_cost=None, flags=flags, reason_codes=reasons,
        evaluator_version=EVALUATOR_VERSION, settings_fingerprint=settings.fingerprint, provenance=provenance,
    )
