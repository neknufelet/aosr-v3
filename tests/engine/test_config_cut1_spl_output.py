"""``aosr.config.spl_output`` 的凍結常數（輸出層絕對 SPL 的參考）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``。值不是抄的：
``blueprint`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.spl_output as spl_output

from tests.engine._config_answers import constant, is_approx


def test_default_offsets_and_tolerance_are_frozen() -> None:
    """兩個預設值與平坦相等的容差。"""
    assert is_approx(spl_output.DEFAULT_SENSITIVITY_DB, constant("spl_output", "DEFAULT_SENSITIVITY_DB"))
    assert is_approx(spl_output.DEFAULT_PLAYBACK_LEVEL_DB, constant("spl_output", "DEFAULT_PLAYBACK_LEVEL_DB"))
    assert is_approx(spl_output.FLAT_EQUAL_TOL_DB, constant("spl_output", "FLAT_EQUAL_TOL_DB"))
