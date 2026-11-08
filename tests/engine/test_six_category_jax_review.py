"""支 A 審查修補：設計可變材料分支及正式路徑的邊界案例。"""
from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from tests.engine.test_six_category_jax import (
    _record, _tolerances, _x64_check, measured as measured,
    test_end_to_end_each_category_and_product_total as check_end_to_end,
)


def test_material_branch_is_not_frozen_in_constants() -> None:
    """抓：幾何常數包仍綁死原牆的材料積分分支。"""
    _x64_check('design-mask')


def test_changed_wall_uses_dynamic_paris_branch(measured: dict[str, object]) -> None:
    """抓：同一包幾何常數換構造後，晚期仍誤走原常數牆積分。"""
    record = _record(measured, 'changed_wall')
    check_end_to_end(measured, 'changed_wall')
    assert record['reused_constants'] is True
    assert record['constant_material'] == [True, True, False, True, True, True]
    previous = np.asarray(_record(measured, 'whole')['double'])
    changed = np.abs(np.asarray(record['double'])-previous)
    assert np.any(changed[[0,3]] > _tolerances(previous)[[0,3]])


def test_default_scattering_matches_numpy(measured: dict[str, object]) -> None:
    """抓：scattering_by_wall 缺席時漏接材料預設散射常數。"""
    check_end_to_end(measured, 'default_scattering')


def test_reflection_window_needs_no_extension(measured: dict[str, object]) -> None:
    """抓：已由三階證明窗完整時，仍誤補階或漏掉窗內過濾。"""
    check_end_to_end(measured, 'no_extension')


@pytest.mark.parametrize('index', (0,1))
def test_nonfinite_impedance_preserves_other_gradients(measured: dict[str, object], index: int) -> None:
    """抓：早期／Paris 真阻抗壞格未傳旗標，或污染其他有效項導數。"""
    record = cast(list[dict[str, object]], measured['nonfinite'])[index]
    flags = cast(list[bool], record['flags'])
    assert flags[2 if index==0 else 3] is False
    assert flags[5] is True
    assert flags[3 if index==0 else 2] is True
    assert record['nan'] == [not valid for valid in flags]
    assert record['product_nan'] and record['objective_nan']
    assert record['valid_gradient_finite'] is True


def test_decay_bands_must_match_numpy_report() -> None:
    """抓：目標帶中心漂移後，JAX 靜靜改算另一組殘響帶。"""
    _x64_check('band-guard')


def test_furniture_outside_reflection_window_is_excluded() -> None:
    """抓：家具固定列拿掉時間窗過濾，把窗外反射帶進早期類。"""
    _x64_check('furniture-window')
