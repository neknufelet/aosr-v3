"""``aosr.config.crossover_axis`` 的凍結常數與契約物件。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``。值不是抄的：
``blueprint`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.crossover_axis as crossover_axis

from tests.engine._config_answers import constant, is_approx


def test_default_seam_times_are_frozen() -> None:
    """接縫的預設時間常數（中心與淡出）。"""
    assert is_approx(crossover_axis.DEFAULT_T_C_CENTER_S, constant("crossover_axis", "DEFAULT_T_C_CENTER_S"))
    assert is_approx(crossover_axis.DEFAULT_T_C_FADE_S, constant("crossover_axis", "DEFAULT_T_C_FADE_S"))
