"""``aosr.config.default_geometry`` 的凍結常數（腳本用的鞋盒種子幾何）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``。值不是抄的：
``blueprint`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.default_geometry as default_geometry

from tests.engine._config_answers import constant, is_approx


def test_seed_geometry_is_frozen() -> None:
    """預設尺寸、聲源、接收點與耳朵高度。"""
    assert is_approx(default_geometry.DEFAULT_DIMS_M, constant("default_geometry", "DEFAULT_DIMS_M"))
    assert is_approx(default_geometry.DEFAULT_SOURCE_XYZ, constant("default_geometry", "DEFAULT_SOURCE_XYZ"))
    assert is_approx(default_geometry.DEFAULT_RECEIVER_XYZ, constant("default_geometry", "DEFAULT_RECEIVER_XYZ"))
    assert is_approx(default_geometry.DEFAULT_EAR_HEIGHT_M, constant("default_geometry", "DEFAULT_EAR_HEIGHT_M"))
