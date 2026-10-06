"""報表呈現共用的字句與純單位換算。"""
from __future__ import annotations

import math

from aosr.physics.fem_modal_check import FemModalCheck
from aosr.reporting.modal_diagnosis_model import CheckSummary, ModalDiagnosis


LOW_FREQUENCY_DECAY_NOTE = "低頻拖尾不計分，另有模態診斷報告"
SPATIAL_IMPRESSION_NOTE = "空間感：尚未評估"
REVERBERATION_ROOM_NOTE = "殘響是整間房的統計量，換座位不變"
BASELINE_NOTE = "尚未正式校準"
MODAL_STATE_TEXT = {"diagnosed_not_scored": "已診斷不計分", "not_computed": "未計算",
                    "failed": "失敗", "out_of_scope": "範圍外"}
MODAL_PLACEMENT_NOTE = "相對 dB：相對同一喇叭到同一座位最強的共振；前幾名只是版面長度，不是門檻；完整排序與整群大小剖面可展開。"
MODAL_GROUP_NOTE = ("重疊分組：相鄰共振的峰寬互相重疊就連成一組；成員數超過 1 的組不是只有一個共振，"
                    "而是一串互相重疊的共振；峰寬跟誰都不重疊的單獨共振自成一組（成員數 1）。"
                    "組內每個共振仍在模態表逐列列出")
DIRECTIONS = {"front": ("前", "主位前方"), "back": ("後", "主位後方"),
              "left": ("左", "主位左方"), "right": ("右", "主位右方"),
              "up": ("上", "主位上方"), "down": ("下", "主位下方")}
SPEAKERS = {"left": "左聲道喇叭", "right": "右聲道喇叭"}
LISTENING_POINTS = {"main": "主位", **{key: full for key, (_, full) in DIRECTIONS.items()}}


def modal_reason(diagnosis: ModalDiagnosis) -> str:
    return diagnosis.reason_text or {"unsupported_room": "房型不是長方形",
        "unsupported_impedance": "六面阻抗必須齊全、為有限正實數"}.get(diagnosis.reason_code or "", diagnosis.reason_code or "")


def modal_reference_text(excess: float | None) -> str:
    return "參考線範圍外" if excess is None else "超過" if excess > 0 else "未超過"


def speaker_label(channel_id: str) -> str:
    return SPEAKERS.get(channel_id, channel_id)


def listening_point_label(point_id: str) -> str:
    return LISTENING_POINTS.get(point_id, point_id)


def modal_guarantee_text(summary: CheckSummary) -> str:
    if summary.guaranteed_decay_rate_rad_s == 0:
        return "沒有保證（保證找齊的衰減高度為零，沒有 T60 門檻）"
    return (f"保證找齊的衰減高度：{summary.guaranteed_decay_rate_rad_s:.6g} rad/s（弧度／秒），"
            f"也就是 T60 不短於 {summary.guaranteed_min_t60_s:.6g} 秒的共振都保證找齊"
            "（這是門檻，不是找到的共振裡最短的 T60）")


def modal_check_text(summary: CheckSummary) -> str:
    weyl = ("Weyl 估計（幾何上該有幾個模態）只算體積項與表面積項，不含稜邊項"
            if summary.weyl_terms == FemModalCheck.__dataclass_fields__["weyl_terms"].default
            else f"求解器自報的 Weyl 估計公式：{summary.weyl_terms}")
    return (f"{weyl}；靜態 {summary.static_count}、零根延續 {summary.zero_mode_continuation_count}、"
            f"過阻尼 {summary.overdamped_count}、未確認衰減 {summary.unconfirmed_decay_count}、上限外返回 {summary.returned_above_limit_count}。"
            "保證以求解收斂且最近根選取完整為前提，只在開圓內成立；強阻尼完備性仍有未驗限制。")


def level_db(energy: float) -> float | None:
    """相對能量轉 dB；非正值沒有可顯示的對數。"""
    return 10.0 * math.log10(energy) if energy > 0 else None


def impedance_multiple(impedance: float, density: float, sound_speed: float) -> float | None:
    """牆阻抗相對於本方案 ρc 的顯示倍數。"""
    if impedance <= 0 or density <= 0 or sound_speed <= 0:
        return None
    return impedance / (density * sound_speed)
