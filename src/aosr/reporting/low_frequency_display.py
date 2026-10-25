"""低頻頻響的呈現：只讀評估原值與酬載，不重算量法或合分。"""
from aosr.config.quality_targets import QualityTargets
from aosr.reporting.display import LABELS, low_frequency_response_description, speaker_label
from aosr.scoring.contract import CategoryEvaluation
from aosr.scoring.low_frequency_response_contract import LowFrequencyResponsePayload


def low_frequency_response_lines(evaluation: CategoryEvaluation) -> tuple[str, ...]:
    """網頁、命令列與搜尋報告共用分項的逐聲道平均 dB 及最差聲道提醒。"""
    payload = evaluation.payload
    if not isinstance(payload, LowFrequencyResponsePayload):
        return ("不可估：" + "、".join(LABELS[reason.value] for reason in evaluation.reason_codes),)
    low, high = payload.scored_range_hz
    return (*(f"{LABELS[item.name]} {item.value:.3f} dB（逐聲道平均）" for item in evaluation.raw_quantities),
            f"最差聲道 {speaker_label(payload.worst_channel.speaker_id)}（只提醒、不淘汰）",
            f"計分範圍 {low:g}～{high:g} Hz", payload.channel_summation_text)


def low_frequency_response_note(evaluation: CategoryEvaluation | None,
                                registry: QualityTargets, purpose: str) -> str:
    """說明與詳情分行，缺評估時只列類別說明。"""
    return "\n".join((low_frequency_response_description(registry, purpose),
                      *(() if evaluation is None else low_frequency_response_lines(evaluation))))
