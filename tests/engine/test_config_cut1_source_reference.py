"""``aosr.config.source_reference`` 的凍結常數（所有 lane 共用的聲源參考）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。值不是抄的：
``blueprint`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.source_reference as source_reference

from tests.engine._config_answers import constant, is_approx


def test_monopole_strength_is_frozen() -> None:
    """單位振幅單極的強度（``4π``，答案檔以運算式記號存），以及它的舊名別稱。"""
    assert is_approx(source_reference.CANONICAL_MONOPOLE_STRENGTH, constant("source_reference", "CANONICAL_MONOPOLE_STRENGTH"))
    assert is_approx(source_reference.DIFFUSE_MONOPOLE_4PI, constant("source_reference", "DIFFUSE_MONOPOLE_4PI"))


def test_reference_anchor_and_sensitivities_are_frozen() -> None:
    """絕對位準的命名錨與兩組靈敏度參考。"""
    assert is_approx(source_reference.REFERENCE_ANCHOR_KIND, constant("source_reference", "REFERENCE_ANCHOR_KIND"))
    assert is_approx(source_reference.REFERENCE_SENSITIVITY_DB_SPL_1M_1W, constant("source_reference", "REFERENCE_SENSITIVITY_DB_SPL_1M_1W"))
    assert is_approx(source_reference.REFERENCE_SENSITIVITY_PRESSURE_PA, constant("source_reference", "REFERENCE_SENSITIVITY_PRESSURE_PA"))
    assert is_approx(source_reference.REFERENCE_DRIVE_POWER_W, constant("source_reference", "REFERENCE_DRIVE_POWER_W"))
    assert is_approx(source_reference.HIGH_SPL_REFERENCE_DB_SPL_1M_1W, constant("source_reference", "HIGH_SPL_REFERENCE_DB_SPL_1M_1W"))
    assert is_approx(source_reference.HIGH_SPL_REFERENCE_PRESSURE_PA, constant("source_reference", "HIGH_SPL_REFERENCE_PRESSURE_PA"))
