"""第二階段低頻頻響的酬載；保留第一階段量測與合分身分，不複製量法。"""
from __future__ import annotations

from typing import Literal

from aosr.scoring.contract_base import FrequencyRange, FrozenModel, WorstChannel
from aosr.scoring.stage_one_low_frequency_contract import StageOneLowFrequencyEvaluation


class LowFrequencyResponsePayload(FrozenModel):
    """逐支分項、逐方向與最差方向留在 evaluation；最差聲道只提醒，不淘汰。"""

    category: Literal["low_frequency_response"]
    evaluation: StageOneLowFrequencyEvaluation
    scored_range_hz: FrequencyRange
    score_version: str
    settings_fingerprint: str
    worst_channel: WorstChannel
    channel_summation: str
    channel_summation_text: str
