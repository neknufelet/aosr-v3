"""``aosr.config.authoring_defaults`` 的凍結常數（M13 authoring 的預設值與命名空間）。

判定與殘餘風險逐符號寫在 ``tests/engine/test_config_cut1_table``（case 表住 ``blueprint/config_cut1_cases.py``）。值不是抄的：
``blueprint`` 底下那一份是 ``blueprint/generate_config_cut1_answers.py``
在唯讀的 v2 工作樹上跑出來的。
"""
from __future__ import annotations

import aosr.config.authoring_defaults as authoring_defaults

from tests.engine._config_answers import constant, is_approx


def test_id_prefixes_are_frozen() -> None:
    """三個 id 命名空間的前綴。"""
    assert is_approx(authoring_defaults.APP_WALL_PREFIX, constant("authoring_defaults", "APP_WALL_PREFIX"))
    assert is_approx(authoring_defaults.APP_PATCH_PREFIX, constant("authoring_defaults", "APP_PATCH_PREFIX"))
    assert is_approx(authoring_defaults.RHINO_PREFIX, constant("authoring_defaults", "RHINO_PREFIX"))


def test_grid_presets_are_frozen() -> None:
    """GRID_PRESETS 那張表（逐鍵比對，不是比筆數）。"""
    assert is_approx(authoring_defaults.GRID_PRESETS, constant("authoring_defaults", "GRID_PRESETS"))


def test_template_and_material_ids_are_frozen() -> None:
    """模板與裸材質的 id。"""
    assert is_approx(authoring_defaults.BARE_MATERIAL_ID, constant("authoring_defaults", "BARE_MATERIAL_ID"))
    assert is_approx(authoring_defaults.BARE_TEMPLATE_ID, constant("authoring_defaults", "BARE_TEMPLATE_ID"))
    assert is_approx(authoring_defaults.BROADBAND_POROUS_TEMPLATE_ID, constant("authoring_defaults", "BROADBAND_POROUS_TEMPLATE_ID"))
    assert is_approx(authoring_defaults.RESONANT_PANEL_TEMPLATE_ID, constant("authoring_defaults", "RESONANT_PANEL_TEMPLATE_ID"))
    assert is_approx(authoring_defaults.CUSTOM_STACK_TEMPLATE_ID, constant("authoring_defaults", "CUSTOM_STACK_TEMPLATE_ID"))
    assert is_approx(authoring_defaults.FIXED_ALPHA_TEMPLATE_ID, constant("authoring_defaults", "FIXED_ALPHA_TEMPLATE_ID"))


def test_thickness_and_porous_bounds_are_frozen() -> None:
    """總厚度上限與多孔材料的參數範圍。"""
    assert is_approx(authoring_defaults.TOTAL_THICKNESS_MAX_M, constant("authoring_defaults", "TOTAL_THICKNESS_MAX_M"))
    assert is_approx(authoring_defaults.FLOW_RESISTIVITY_MIN, constant("authoring_defaults", "FLOW_RESISTIVITY_MIN"))
    assert is_approx(authoring_defaults.FLOW_RESISTIVITY_MAX, constant("authoring_defaults", "FLOW_RESISTIVITY_MAX"))
    assert is_approx(authoring_defaults.SPACING_MIN_M, constant("authoring_defaults", "SPACING_MIN_M"))
    assert is_approx(authoring_defaults.SPACING_MAX_M, constant("authoring_defaults", "SPACING_MAX_M"))
    assert is_approx(authoring_defaults.HOLE_RATIO_MIN, constant("authoring_defaults", "HOLE_RATIO_MIN"))
    assert is_approx(authoring_defaults.HOLE_RATIO_MAX, constant("authoring_defaults", "HOLE_RATIO_MAX"))
    assert is_approx(authoring_defaults.PERFORATED_FIXED_THICKNESS_M, constant("authoring_defaults", "PERFORATED_FIXED_THICKNESS_M"))
    assert is_approx(authoring_defaults.FABRIC_RS, constant("authoring_defaults", "FABRIC_RS"))
    assert is_approx(authoring_defaults.FABRIC_MS, constant("authoring_defaults", "FABRIC_MS"))


def test_layer_stack_specs_are_frozen() -> None:
    """三個分層堆疊樣板（逐鍵、逐值比對）。"""
    assert is_approx(authoring_defaults.BARE_LAYER_STACK_SPEC, constant("authoring_defaults", "BARE_LAYER_STACK_SPEC"))
    assert is_approx(authoring_defaults.BROADBAND_POROUS_LAYER_STACK_SPEC, constant("authoring_defaults", "BROADBAND_POROUS_LAYER_STACK_SPEC"))
    assert is_approx(authoring_defaults.RESONANT_PANEL_LAYER_STACK_SPEC, constant("authoring_defaults", "RESONANT_PANEL_LAYER_STACK_SPEC"))
