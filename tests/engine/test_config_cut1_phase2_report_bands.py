"""``aosr.config.phase2_report_bands`` 的凍結常數（Phase-2 報告萃取頻帶與門檻）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``。值不是抄的：
``tests/engine/answers`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.phase2_report_bands as phase2_report_bands

from tests.engine._config_answers import constant, is_approx


def test_report_bands_and_thresholds_are_frozen() -> None:
    """報告要用的兩個頻帶與三個門檻／視窗。"""
    assert is_approx(phase2_report_bands.PHASE2_RFZ_THRESHOLD_DB, constant("phase2_report_bands", "PHASE2_RFZ_THRESHOLD_DB"))
    assert is_approx(phase2_report_bands.PHASE2_RFZ_BAND_HZ, constant("phase2_report_bands", "PHASE2_RFZ_BAND_HZ"))
    assert is_approx(phase2_report_bands.PHASE2_LATE_RT_BAND_HZ, constant("phase2_report_bands", "PHASE2_LATE_RT_BAND_HZ"))
    assert is_approx(phase2_report_bands.PHASE2_RFZ_WINDOW_S, constant("phase2_report_bands", "PHASE2_RFZ_WINDOW_S"))
    assert is_approx(phase2_report_bands.PHASE2_TARGET_SLOPE_DB_OCT, constant("phase2_report_bands", "PHASE2_TARGET_SLOPE_DB_OCT"))
