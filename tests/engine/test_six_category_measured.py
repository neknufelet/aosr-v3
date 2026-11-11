"""實測阻抗表當固定材料進第三階段六類鏈（#1082 段 2-3d）：新舊對照——搜尋那套對產品（非搜尋）那套。

探針 ``_six_category_measured_probe`` 冷行程雙精度實量；兩張合成表（不是真實量測），一張有虛部整面保留、一張沒有虛部
照准用清單整塊換上一個分區。尺全部沿用現有的：阻抗用構造考卷那一把（IMPEDANCE_ABS＋REFERENCE_REL），晚期吸音用
晚期家族 JAX 對 Python 那一把（登記簿 late_family_jax_vs_python），六類與總分用六類 JAX 對 numpy 那一把（_tolerances）。
表內內插兩邊寫法不同（對數相減 vs log1p），不是逐位。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.runtime import child_process_env, configure_jax
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._precision_contracts import contract_value
from tests.engine._scheme_cache import shared_json
from tests.engine.test_six_category_jax import _tolerances
from tests.engine.test_transfer_matrix_reference import IMPEDANCE_ABS, REFERENCE_REL


@pytest.fixture(scope='module')
def measured(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str, object]:
    def produce() -> str:
        directory = tmp_path_factory.mktemp('six-category-measured')
        output = directory / 'measure.json'
        env = forward_cpu_pins(child_process_env(threads=1)).copy()
        configure_jax(platforms='cpu', enable_x64=True, env=env)
        env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'], str(Path(__file__).resolve().parents[2])))
        run = subprocess.run([sys.executable, '-m', 'tests.engine._six_category_measured_probe', str(output)],
                             cwd=directory, env=env, capture_output=True, text=True, timeout=1800)
        if run.returncode != 0:
            return json.dumps({'probe_error': run.stdout + run.stderr})
        return output.read_text()
    result = cast(dict[str, object], json.loads(shared_json(tmp_path_factory, worker_id, 'six-category-measured', produce)))
    assert 'probe_error' not in result, result.get('probe_error')
    return result


def test_case_covers_both_tables_inside_outside_and_the_seam(measured: dict[str, object]) -> None:
    """抓考卷自己沒考到：兩張表都要真的進鏈、表內外都有求值點、沒有虛部那張虛部是 0。"""
    assert measured['measured_patches'] == ['x0@a', 'y0']
    assert measured['imag_zero'] == [True, False]
    assert all(count > 0 for count in cast(list[int], measured['inside_counts']))
    assert all(count > 0 for count in cast(list[int], measured['outside_counts']))


def test_absorption_mask_follows_the_table_per_frequency(measured: dict[str, object]) -> None:
    """抓實測表整列當固定阻抗（或整列當構造）算吸音，而不是表內／表外逐頻分流；其他面旗標照舊。"""
    assert len(cast(list[int], measured['mask_shape'])) == 2
    assert measured['mask_rows'] == [True, True]
    assert all(cast(list[bool], measured['mask_others']))


@pytest.mark.parametrize('mode', ('eager', 'jit'))
@pytest.mark.parametrize('lane', range(4), ids=('fem', 'normal', 'oblique', 'paris'))
def test_four_impedance_lanes_match_product_entry(measured: dict[str, object], mode: str, lane: int) -> None:
    """抓表內內插、表外底構造補值、角度攤開或頻率軸接錯：搜尋那套四路阻抗對產品材料入口。"""
    errors = np.asarray(cast(list[object], measured[f'{mode}_errors'])[lane])
    scales = np.asarray(cast(list[object], measured['scales'])[lane])
    assert np.all(errors <= IMPEDANCE_ABS + REFERENCE_REL * scales), (mode, lane, errors, scales)


def test_late_absorption_matches_product_random_incidence(measured: dict[str, object]) -> None:
    """抓表內沒用固定阻抗那一支換算、表外沒走底構造 Paris：逐面片逐頻對產品 random_incidence_absorption。"""
    actual, expected = np.asarray(measured['alpha']), np.asarray(measured['alpha_expected'])
    assert actual.shape == expected.shape and np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=contract_value('late_family_jax_vs_python'), atol=0)


@pytest.mark.parametrize('mode', ('eager', 'jit'))
def test_categories_and_product_match_numpy_pipeline(measured: dict[str, object], mode: str) -> None:
    """抓實測材料進搜尋後六類與排名總分偏離正式 numpy 管線（匯出方案、同一份聲壓）。"""
    expected = np.asarray(measured['reference'])
    tolerance = _tolerances(expected)
    actual = np.asarray(measured[mode])
    assert np.all(np.isfinite(actual) & (np.abs(actual - expected) <= tolerance)), (mode, actual - expected, tolerance)
    bound = float(np.dot(np.asarray(measured['weights']), tolerance))
    assert abs(cast(float, measured[f'{mode}_product']) - cast(float, measured['reference_product'])) <= bound
    assert measured['flags'] == [True] * 6


def test_single_precision_chain_takes_measured_tables(measured: dict[str, object]) -> None:
    """抓單精度鏈（顯卡那條）的實測表 FEM 列或角度阻抗仍是雙精度、或出非有限值。"""
    row = cast(dict[str, object], measured['single'])
    assert row['inputs'] == ['complex64'] * 5
    assert row['gradient_dtype'] == 'float32'
    assert row['finite'] is True


def test_single_precision_table_ends_keep_one_side(measured: dict[str, object]) -> None:
    """抓單精度阻抗把端點頻率捨入後重判表內外、晚期吸音遮罩卻照雙精度判（審查 #1102）：兩端換成轉 float32 會落到表外的軸點，
    真的 single_precision=True 工廠、頻率軸動態傳入；阻抗要走表那一支、遮罩同判表內、吸音有限（無損底構造時錯邊會出 NaN）。"""
    seam = cast(dict[str, object], measured['seam_single'])
    assert seam['rounds_outside'] == [True, True]
    assert seam['mask_inside'] == [True, True]
    assert seam['nearer_table'] == [True, True]
    assert seam['alpha_finite'] == [True, True]
