"""``aosr.config.speaker_directivity`` 的凍結常數（SCENE-1 解析式指向性）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``。值不是抄的：
``tests/engine/answers`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.speaker_directivity as speaker_directivity

from tests.engine._config_answers import constant, is_approx


def test_version_strings_are_frozen() -> None:
    """模型與 golden 的版本字串。"""
    assert is_approx(speaker_directivity.DIRECTIVITY_MODEL_VERSION, constant("speaker_directivity", "DIRECTIVITY_MODEL_VERSION"))
    assert is_approx(speaker_directivity.DIRECTIVITY_GOLDEN_VERSION, constant("speaker_directivity", "DIRECTIVITY_GOLDEN_VERSION"))


def test_setting_keys_are_frozen() -> None:
    """設定鍵（payload 裡那幾格的鍵名）。"""
    assert is_approx(speaker_directivity.DIRECTIVITY_ENABLED_KEY, constant("speaker_directivity", "DIRECTIVITY_ENABLED_KEY"))
    assert is_approx(speaker_directivity.SPEAKER_TYPE_KEY, constant("speaker_directivity", "SPEAKER_TYPE_KEY"))
    assert is_approx(speaker_directivity.SPEAKER_WIDTH_OVERRIDE_KEY, constant("speaker_directivity", "SPEAKER_WIDTH_OVERRIDE_KEY"))
    assert is_approx(speaker_directivity.PISTON_RADIUS_OVERRIDE_KEY, constant("speaker_directivity", "PISTON_RADIUS_OVERRIDE_KEY"))


def test_product_defaults_are_frozen() -> None:
    """產品預設值與型別清單。"""
    assert is_approx(speaker_directivity.DIRECTIVITY_DEFAULT_ENABLED, constant("speaker_directivity", "DIRECTIVITY_DEFAULT_ENABLED"))
    assert is_approx(speaker_directivity.DEFAULT_SPEAKER_TYPE, constant("speaker_directivity", "DEFAULT_SPEAKER_TYPE"))
    assert is_approx(speaker_directivity.SPEAKER_TYPES, constant("speaker_directivity", "SPEAKER_TYPES"))
    assert is_approx(speaker_directivity.DIRECTIVITY_REAR_GAIN_MIN, constant("speaker_directivity", "DIRECTIVITY_REAR_GAIN_MIN"))
    assert is_approx(speaker_directivity.DIRECTIVITY_NORMALIZATION_GL_ORDER, constant("speaker_directivity", "DIRECTIVITY_NORMALIZATION_GL_ORDER"))
    assert is_approx(speaker_directivity.DIRECTIVITY_TOE_IN_ANCHOR, constant("speaker_directivity", "DIRECTIVITY_TOE_IN_ANCHOR"))


def test_speaker_presets_are_frozen() -> None:
    """兩個喇叭預設（逐欄比對，不比筆數）。"""
    assert is_approx(speaker_directivity.SPEAKER_PRESETS, constant("speaker_directivity", "SPEAKER_PRESETS"))
