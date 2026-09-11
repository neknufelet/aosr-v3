"""``aosr.config.art_rt_guard`` 的凍結常數（art_rt 那一項的三個護欄旋鈕）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``。值不是抄的：
``blueprint`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.art_rt_guard as art_rt_guard

from tests.engine._config_answers import constant, is_approx


def test_art_rt_guard_knobs_are_frozen() -> None:
    """art_rt 那一項的三個護欄旋鈕（上限、軟化係數、render 預設取樣率）。"""
    assert is_approx(art_rt_guard.ART_RT_T_CAP_S, constant("art_rt_guard", "ART_RT_T_CAP_S"))
    assert is_approx(art_rt_guard.ART_RT_GUARD_SMOOTHNESS_S, constant("art_rt_guard", "ART_RT_GUARD_SMOOTHNESS_S"))
    assert is_approx(art_rt_guard.RENDER_DEFAULT_SAMPLE_RATE_HZ, constant("art_rt_guard", "RENDER_DEFAULT_SAMPLE_RATE_HZ"))
