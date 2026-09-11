"""``aosr.config.receiver_grid`` 的凍結常數（scoring／report 的接收格點）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。值不是抄的：
``blueprint`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.receiver_grid as receiver_grid

from tests.engine._config_answers import constant, is_approx


def test_grid_defaults_are_frozen() -> None:
    """每軸點數、牆邊留白與耳朵高度。"""
    assert is_approx(receiver_grid.GRID_N, constant("receiver_grid", "GRID_N"))
    assert is_approx(receiver_grid.GRID_MARGIN_M, constant("receiver_grid", "GRID_MARGIN_M"))
    assert is_approx(receiver_grid.EAR_HEIGHT_M, constant("receiver_grid", "EAR_HEIGHT_M"))
