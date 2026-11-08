"""θ 到六類的新工廠：舊鏈逐位控制、匯出材料答案、中央差分與失效旗標。"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from aosr.runtime import child_process_env, configure_jax
from tests.engine._scheme_cache import shared_json
from tests.engine.test_six_category_jax import _tolerances
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL, IMPEDANCE_ABS, REFERENCE_REL


@pytest.fixture(scope='module')
def measured(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str, object]:
    assert importlib.util.find_spec('aosr.gradient.six_category_chain') is not None, '缺少 θ → 六類工廠'
    def produce() -> str:
        directory = tmp_path_factory.mktemp('six-category-chain')
        output = directory/'measure.json'
        env = child_process_env(threads=1).copy()
        configure_jax(platforms='cpu', enable_x64=True, env=env)
        env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'], str(Path(__file__).resolve().parents[2])))
        run = subprocess.run([sys.executable, '-m', 'tests.engine._six_category_chain_probe', str(output)],
            cwd=directory, env=env, capture_output=True, text=True, timeout=1800)
        (directory/'probe.stdout').write_text(run.stdout)
        (directory/'probe.stderr').write_text(run.stderr)
        if run.returncode != 0:
            return json.dumps({'probe_error': run.stdout+run.stderr})
        return output.read_text()
    result = cast(dict[str, object], json.loads(shared_json(tmp_path_factory, worker_id, 'six-category-chain', produce)))
    assert 'probe_error' not in result, result.get('probe_error')
    return result


@pytest.mark.parametrize('mode', ('eager', 'jit'))
def test_low_frequency_control_is_bitwise(measured: dict[str, object], mode: str) -> None:
    """抓：低頻換常數、重寫代理，或求值方式使原鏈的位元改變。"""
    controls = cast(dict[str, dict[str, object]], measured['controls'])
    for name, row in controls.items():
        assert row[f'{mode}_bitwise'] is True, (name, row)
        assert row[f'{mode}_hex'] == row[f'{mode}_old_hex'], (name, row)
        assert row[f'{mode}_category_bitwise'] is True, (name, row)


@pytest.mark.parametrize('lane', range(4), ids=('fem', 'normal', 'oblique', 'paris'))
def test_material_inputs_match_exported_materials(measured: dict[str, object], lane: int) -> None:
    """抓：設計參數錯面、保留牆漏角度、把斜入射頂替成垂直入射。"""
    rows = cast(list[dict[str, object]], measured['materials'])
    for row in rows:
        for mode in ('eager', 'jit'):
            errors = np.asarray(row[f'{mode}_errors'])[lane]
            scales = np.asarray(row['scales'])[lane]
            assert np.all(errors <= IMPEDANCE_ABS+REFERENCE_REL*scales), (row['name'], mode, lane, errors, scales)
            assert np.all(np.asarray(row[f'{mode}_ratios'])[lane] <= 1), (row['name'], mode, lane, row[f'{mode}_ratios'])
        assert row['flags_match'] is True


@pytest.mark.parametrize('mode', ('eager', 'jit'))
def test_categories_and_product_match_export_path(measured: dict[str, object], mode: str) -> None:
    """抓：六類接線錯或總分混入低頻代理；獨立 numpy 答案由下一題守。"""
    for row in cast(list[dict[str, object]], measured['end_to_end']):
        expected = np.asarray(row['reference'])
        tolerance = _tolerances(expected)
        actual = np.asarray(row[mode])
        assert np.all(np.isfinite(actual) & (np.abs(actual-expected) <= tolerance)), (row['name'], mode, actual-expected)
        assert abs(cast(float, row[f'{mode}_product'])-cast(float, row['reference_product'])) <= np.dot(np.asarray(row['weights']), tolerance)
        assert abs(cast(float, row[f'{mode}_objective'])-cast(float, row['reference_objective'])) <= np.dot(np.asarray(row['weights']), tolerance)
        assert row['flags'] == [True]*6


@pytest.mark.parametrize('mode', ('eager', 'jit'))
def test_on_wall_categories_and_product_match_numpy(measured: dict[str, object], mode: str) -> None:
    """含 MountedOnWall（原牆上貼構造）的匯出方案，對正式 numpy 六類與排名總分。"""
    row = cast(dict[str, object], measured['numpy_on_wall'])
    expected = np.asarray(row['reference'])
    tolerance = _tolerances(expected)
    assert row['has_mounted_patch'] is True
    assert np.all(np.abs(np.asarray(row[mode])-expected) <= tolerance), (row, tolerance)
    assert abs(cast(float, row[f'{mode}_product'])-cast(float, row['reference_product'])) <= np.dot(np.asarray(row['weights']), tolerance)


def test_template_validation_axis_covers_all_four_lanes(measured: dict[str, object]) -> None:
    """抓：設計樣板僅在 FEM 軸驗合法性、漏掉報表／斜入射／Paris 軸。"""
    row = cast(dict[str, object], measured['template_axes'])
    assert row['design_coverage'] == [True]*4, row
    assert row['kept_coverage'] == [True]*4, row


@pytest.mark.parametrize('index', (0, 1), ids=('objective', 'product'))
def test_theta_gradients_match_central_difference(measured: dict[str, object], index: int) -> None:
    """逐參數比 θ_i·導數，沿用舊鏈相對步長；本題只在 capped=True（交接硬切）區。

    交接權重 stop_gradient（擋梯度）；硬切區真導數也是零，才能與中央差分相等。
    本題測不到平滑交接區少掉的 dw/dθ，那是決策紙第 11 條的已知限制。
    """
    row = cast(dict[str, object], measured['gradients'])
    actual, central = np.asarray(row['automatic'])[index], np.asarray(row['central'])[index]
    assert np.all(np.isfinite(actual) & (np.abs(actual-central) <= GRADIENT_ABS+GRADIENT_REL*np.abs(central))), (actual, central)
    assert np.all(np.abs(central) > GRADIENT_ABS)
    assert cast(float, row['reflection_margin']) > 0
    assert row['capped'] is True


def test_single_precision_unifies_all_five_inputs(measured: dict[str, object]) -> None:
    """抓：某一個阻抗或聲壓仍為 complex128，先污染六類再在輸出轉型。"""
    row = cast(dict[str, object], measured['single'])
    assert row['inputs'] == ['complex64']*5
    assert row['outputs'] == ['float32']*6
    assert row['aux_dtypes'] == ['float32']*5
    assert row['gradient_dtype'] == 'float32'
    assert row['finite'] is True


def test_invalid_category_preserves_valid_theta_gradients(measured: dict[str, object]) -> None:
    """抓：無效類未傳兩個總分，或有效類導數被 NaN 污染。"""
    for row in cast(list[dict[str, object]], measured['invalid']):
        assert row['product_nan'] and row['objective_nan'] and row['category_nan']
        assert row['gradient_finite'] is True


def test_shared_geometry_is_never_rebuilt(measured: dict[str, object]) -> None:
    """抓：每個候選工廠、eager 或 jit 偷偷重建六類幾何常數。"""
    assert measured['build_calls'] == 1
    assert measured['build_calls_after_candidates'] == 1


def test_material_flags_and_measured_wall_rejection(measured: dict[str, object]) -> None:
    """抓：旗標全 True、MountedOnWall 誤作常數，或保留實測牆靜默走不同公式。"""
    row = cast(dict[str, object], measured['rejections'])
    assert row['material_flags'] == [True, False, False, False]
    assert 'MeasuredImpedance' in cast(str, row['measured'])
    assert 'numpy' in cast(str, row['measured']) and 'JAX' in cast(str, row['measured'])
    assert '面片' in cast(str, row['order'])
    assert '頻率' in cast(str, row['axis'])
    assert '低頻' in cast(str, row['objective'])
    assert '材料' in cast(str, row['rigid'])
