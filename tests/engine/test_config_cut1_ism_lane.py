"""``aosr.config.ism_lane`` 的凍結常數（HF-ISM 的per-face 格點）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``。值不是抄的：
``tests/engine/answers`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.ism_lane as ism_lane

from tests.engine._config_answers import constant, is_approx


def test_ism_grid_resolution_is_frozen() -> None:
    """HF-ISM 的 per-face 格點與每格取樣數。"""
    assert is_approx(ism_lane.ISM_FACE_GRID_N, constant("ism_lane", "ISM_FACE_GRID_N"))
    assert is_approx(ism_lane.ISM_FACE_GRID_SAMPLES, constant("ism_lane", "ISM_FACE_GRID_SAMPLES"))
