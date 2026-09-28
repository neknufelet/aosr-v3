"""報表呈現共用的字句與純單位換算。"""
from __future__ import annotations

import math


LOW_FREQUENCY_DECAY_NOTE = "低頻拖尾：尚未評估"


def level_db(energy: float) -> float | None:
    """相對能量轉 dB；非正值沒有可顯示的對數。"""
    return 10.0 * math.log10(energy) if energy > 0 else None


def impedance_multiple(impedance: float, density: float, sound_speed: float) -> float | None:
    """牆阻抗相對於本方案 ρc 的顯示倍數。"""
    if impedance <= 0 or density <= 0 or sound_speed <= 0:
        return None
    return impedance / (density * sound_speed)
