"""六類接線的端到端考卷：正式 numpy 為答案，不從組裝器倒推答案。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aosr.runtime import child_process_env, configure_jax
from aosr.scoring.six_category_jax import category_totals
from tests.engine._precision_contracts import contract_value
from tests.engine._scheme_cache import shared_json
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL

CONTRACTS = ('timbre_cost_jax_vs_python', 'listening_area_cost_jax_vs_python',
             'reflections_cost_jax_vs_python', 'reverberation_cost_jax_vs_python',
             'channel_matching_cost_jax_vs_python', 'stage_one_objective_jax_vs_python')


@pytest.fixture(scope='module')
def measured(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str, object]:
    def produce() -> str:
        directory = tmp_path_factory.mktemp('six-category')
        output = directory/'measure.json'
        env = child_process_env(threads=1).copy()
        configure_jax(platforms='cpu', enable_x64=True, env=env)
        env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'], str(Path(__file__).resolve().parents[2])))
        run = subprocess.run([sys.executable, '-m', 'tests.engine._six_category_jax_probe', str(output)],
            cwd=directory, env=env, capture_output=True, text=True, timeout=1800)
        assert run.returncode == 0, run.stdout+run.stderr
        return output.read_text()
    return cast(dict[str, object], json.loads(shared_json(tmp_path_factory, worker_id, 'six-category-measure', produce)))


def _record(measured: dict[str, object], name: str) -> dict[str, object]:
    return cast(dict[str, dict[str, object]], measured['cases'])[name]


def _tolerances(expected: np.ndarray) -> np.ndarray:
    values = np.asarray([contract_value(name) for name in CONTRACTS])
    # 低頻既有尺是相對；類代價無單位，近零沿契約用 1 dB 當分母。
    values[-1] *= max(abs(float(expected[-1])), 1.)
    return values


@pytest.mark.parametrize('name', ('whole', 'zoned', 'furniture'))
def test_end_to_end_each_category_and_product_total(measured: dict[str, object], name: str) -> None:
    """抓：軸、晚期功率比、分區、反射補階、任一類權重與產品低頻誤混代理。"""
    record = _record(measured, name)
    expected = np.asarray(record['reference'], dtype=float)
    tolerances = _tolerances(expected)
    assert np.all(expected > 0), expected
    for key in ('double', 'compiled'):
        actual = np.asarray(record[key], dtype=float)
        assert np.all(np.isfinite(actual) & (np.abs(actual-expected) <= tolerances)), (name,key,actual-expected,tolerances)
    assert abs(float(cast(float, record['product_total']))-float(cast(float, record['reference_total']))) <= np.dot(np.asarray(record['weights']), tolerances)
    assert record['flags'] == [True]*6
    assert record['reflection_orders'] == ([3, 3] if name=='no_extension' else [4, 4])
    assert record['reflection_coverage'] == (['approximate']*2 if name=='furniture' else ['complete']*2)
    assert record['dtypes'] == ['float64']*6
    assert record['single_dtypes'] == ['float32']*6
    assert np.isfinite(np.asarray(record['single'], dtype=float)).all()


@pytest.mark.parametrize('name', ('whole', 'zoned', 'furniture'))
def test_low_frequency_objective_bitwise_direct(measured: dict[str, object], name: str) -> None:
    """抓：搜尋低頻誤用產品 20 Hz 常數或改寫代理算式。"""
    record = _record(measured, name)
    assert record['low_bitwise'] is True
    assert record['low_jit_bitwise'] is True
    assert float(cast(float, record['objective_total'])) != float(cast(float, record['product_total']))


def test_both_totals_gradient_matches_central_difference(measured: dict[str, object]) -> None:
    """抓：阻抗／聲壓任何一路斷梯度；避開最強路徑平手與交接可變權重。"""
    record = cast(dict[str, object], measured['gradients'])
    actual, expected = np.asarray(record['automatic']), np.asarray(record['central'])
    assert np.all(np.isfinite(actual) & (np.abs(actual-expected) <= GRADIENT_ABS+GRADIENT_REL*np.abs(expected))), (actual,expected)
    assert np.count_nonzero(np.abs(expected[:,1:]) > GRADIENT_ABS) >= 6
    assert cast(float, record['reflection_margin']) > 0
    assert np.any(np.abs(actual[0]-actual[1]) > GRADIENT_ABS)


def test_unprovable_reflections_unavailable_on_both_paths(measured: dict[str, object]) -> None:
    """抓：拿三階表冒充完整窗，或到補階上限仍標可估。"""
    record = cast(dict[str, object], measured['unavailable'])
    assert record == dict(jax_valid=False, jax_nan=True, product_nan=True, objective_nan=True,
                          numpy_state='unavailable', numpy_cost_missing=True)


@pytest.mark.parametrize('index', range(6))
def test_each_invalid_category_reaches_both_totals(index: int) -> None:
    """抓：任一類有效旗標漏傳總分；有效類的導數必須仍有限。"""
    weights = jnp.asarray([1., 1., 1., 1., 1., .25])
    costs = jnp.arange(1., 7.)
    valid = jnp.ones(6, dtype=jnp.bool_).at[index].set(False)
    values, product, objective = jax.jit(category_totals)(costs, valid, jnp.asarray(9.), weights)
    assert jnp.isnan(values[index]) & jnp.isnan(product) & jnp.isnan(objective)
    assert np.isfinite(np.asarray(values)[np.arange(6)!=index]).all()
    derivative = jax.grad(lambda x: jnp.nansum(category_totals(x, valid, jnp.asarray(9.), weights)[0]))(costs)
    assert np.isfinite(derivative).all()
    assert derivative[index] == 0
    assert np.array_equal(np.asarray(derivative)[np.arange(6)!=index], np.ones(5))


@pytest.mark.parametrize('index', range(6))
def test_real_invalid_inputs_reach_category_and_totals(measured: dict[str, object], index: int) -> None:
    """抓：真實組裝器忽略上游旗標；每種失效都經整個六類入口。"""
    record = cast(list[dict[str, object]], measured['invalid'])[index]
    assert cast(list[bool], record['flags'])[index] is False
    assert record['nan'] and record['product_nan'] and record['objective_nan']
    assert record['valid_gradient_finite'] is True


def _x64_check(*args: str) -> None:
    """建常數要雙精度行程（numpy 正式那組 Paris 角）；斷言在冷子行程跑，失敗原文帶回。"""
    env = child_process_env(threads=1).copy()
    configure_jax(platforms='cpu', enable_x64=True, env=env)
    env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'], str(Path(__file__).resolve().parents[2])))
    run = subprocess.run([sys.executable, '-m', 'tests.engine._six_category_jax_x64_checks', *args],
        env=env, capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stdout+run.stderr


def test_flat_constants_every_field_changes_fingerprint() -> None:
    """逐欄植錯：漏掉任一幾何、索引、設定或巢狀音色葉子就紅。"""
    _x64_check('fingerprint')


@pytest.mark.parametrize('name', ('zoned', 'reordered'))
def test_patch_order_matches_boundary_and_input_zone_order(name: str) -> None:
    """抓：面片排序抹掉方案區序，或幾何表與邊界面片表不一致。"""
    _x64_check('patch-order', name)


def test_mismatched_geometric_patch_order_rejected() -> None:
    """抓：建常數時面片順序守門被拿掉；舊逐對輸入不能套新區序。"""
    _x64_check('mismatched-order')
