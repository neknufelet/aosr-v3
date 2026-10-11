"""#560：只讀凍結外部答案；JAX 六十四位元只在隔離子行程開機。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest

from aosr.runtime import child_process_env, configure_jax


ROOT = Path(__file__).resolve().parents[2]
PROBLEM_FILE = ROOT / 'blueprint' / 'tmm_reference_problem_step1.json'
ANSWER_FILE = ROOT / 'blueprint' / 'tmm_reference_answers_step1.json'
PROBE = Path(__file__).with_name('_transfer_matrix_probe.py')
CASES = tuple(json.loads(PROBLEM_FILE.read_text())['cases'])
# 同公式雙精度，實量阻抗實部/虛部最大相對差 1.686e-12 / 8.075e-13。
# 三角函式、複數除法與近零消去誤差留約三倍空間，不是物理近似精度。
REFERENCE_REL = 5e-12
IMPEDANCE_ABS = 2e-10  # Pa*s/m；共振附近大阻抗由相對項判斷。
COEFFICIENT_ABS = 2e-14  # 無量綱；吸音全域只用絕對項，實量最大差 1.333e-15。
# AD−BC 有相消；32 個雙精度 epsilon 乘 max(1, |AD|+|BC|)，
# 涵蓋複數乘減與三角函式舍入，薄片的整體縮放不會被 A/C 消掉。
DETERMINANT_SCALED_ABS = 32 * np.finfo(np.float64).eps
DETERMINANT_ABS = 2e-14  # 1 mm 的各模型不嚴重相消，約 90 epsilon 留平台舍入空間。
# 中央差分相對步長 1e-5：O(h^2) 截斷與雙精度相減誤差；比無量綱 p*dy/dp。
# 實量最大相對差 4.064e-9、絕對差 4.575e-10；近零導數用絕對項。
GRADIENT_REL = 5e-8
GRADIENT_ABS = 2e-9


class Run(TypedDict):
    cases: dict[str, dict[str, list[str]]]
    dtype: str
    air: dict[str, float]


def _run(mode: str, directory: Path) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    configure_jax(platforms='cpu', enable_x64=True, env=env)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TMPDIR'] = str(directory)
    output = directory / f'{mode}.json'
    process = subprocess.run([sys.executable, str(PROBE), mode, str(PROBLEM_FILE), str(output)],
                             cwd=directory, env=env, capture_output=True, text=True, timeout=180)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, object], json.loads(output.read_text()))


def _decode(values: list[str]) -> np.ndarray:
    return np.asarray([float.fromhex(value) for value in values])


def _absolute_tolerance(key: str) -> float:
    return IMPEDANCE_ABS if key.startswith('surface_impedance') else COEFFICIENT_ABS


def _assert_reference_values(key: str, actual: list[str], expected: list[str], context: str) -> None:
    np.testing.assert_allclose(_decode(actual), _decode(expected),
        rtol=0 if key == 'absorption_hex' else REFERENCE_REL,
        atol=_absolute_tolerance(key), err_msg=context)


@pytest.fixture(scope='module')
def reference_run(tmp_path_factory: pytest.TempPathFactory) -> Run:
    return cast(Run, _run('reference', tmp_path_factory.mktemp('tmm-reference')))


@pytest.mark.parametrize('case_name', CASES)
def test_surface_impedance_and_normal_absorption_match_reference(reference_run: Run, case_name: str) -> None:
    """係數、分段、SI 單位、層順序、背牆或空氣條件改錯，逐頻實部與虛部必須紅。"""
    answers = json.loads(ANSWER_FILE.read_text())
    expected = cast(dict[str, list[str]], answers['cases'][case_name])
    actual = reference_run['cases'][case_name]
    assert reference_run['dtype'] == 'float64'
    for key, expected_values in expected.items():
        _assert_reference_values(key, actual[key], expected_values, f'{case_name}::{key}')


def test_absorption_comparison_rejects_absolute_drift_near_unity() -> None:
    """恢復共用相對容差時，α 接近 1 的 1e-12 偏差也必須抓到。"""
    with pytest.raises(AssertionError):
        _assert_reference_values('absorption_hex', [(0.8 + 1e-12).hex()], [0.8.hex()], '吸音絕對尺')


def test_each_step1_layer_has_unit_determinant(tmp_path: Path) -> None:
    """每個空氣/多孔模型與纖維種類的矩陣乘錯整體倍數時紅；不經背牆除法。"""
    result = _run('determinants', tmp_path)
    conditioned = cast(list[dict[str, object]], result['conditioned_rows'])
    assert conditioned
    for row in conditioned:
        determinant = _decode(cast(list[str], row['real_hex'])) + 1j * _decode(cast(list[str], row['imag_hex']))
        np.testing.assert_allclose(determinant, 1, rtol=0, atol=DETERMINANT_ABS, err_msg=str(row['layer']))
    rows = cast(list[dict[str, object]], result['rows'])
    assert rows
    for row in rows:
        determinant = _decode(cast(list[str], row['real_hex'])) + 1j * _decode(cast(list[str], row['imag_hex']))
        scale = np.maximum(1, _decode(cast(list[str], row['cancellation_scale_hex'])))
        np.testing.assert_allclose((determinant - 1) / scale, 0, rtol=0,
            atol=DETERMINANT_SCALED_ABS, err_msg=str(row['layer']))


def test_omitted_load_preserves_rigid_backing_bit_for_bit(reference_run: Run) -> None:
    """未給負載與明給 None 都必須逐位維持第一步 A/C 路徑。"""
    result = cast(dict[str, object], reference_run)
    assert result['cases'] == result['explicit_none_cases'] == result['rigid_cases']


def test_reference_run_uses_the_frozen_problem_air(reference_run: Run) -> None:
    """凍結案例記自己的物理條件；不以原套件的溫濕度預設取代產品單一來源。"""
    problem = json.loads(PROBLEM_FILE.read_text())
    assert reference_run['air'] == {key: value['value'] for key, value in problem['air_properties'].items()}
    assert set(reference_run['cases']) == set(problem['cases'])
    assert CASES


def test_jit_and_vmap_preserve_pointwise_results(tmp_path: Path) -> None:
    """編譯、向量化、四個連續參數被誤當 Python 值或截斷時必須紅。"""
    rows = cast(dict[str, dict[str, dict[str, list[str]]]], _run('transforms', tmp_path)['rows'])
    assert rows
    for name, row in rows.items():
        for transformation in ('jit', 'vmap'):
            for key, expected in row['pointwise'].items():
                _assert_reference_values(key, row[transformation][key], expected, f'{name}:{transformation}:{key}')


def test_parameter_gradients_match_central_difference(tmp_path: Path) -> None:
    """厚度、流阻、孔隙率、曲折度的導數斷線或 where 未選支污染梯度時必須紅。"""
    result = _run('gradients', tmp_path)
    rows = cast(list[dict[str, object]], result['rows'])
    assert rows
    for row in rows:
        automatic = float(cast(float, row['automatic_scaled']))
        central = float(cast(float, row['central_scaled']))
        assert np.isfinite(automatic) and np.isfinite(central), row
        np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS, err_msg=str(row))
    assert np.all(np.isfinite(np.asarray(result['boundary_gradients'], dtype=float)))


def test_static_options_and_empty_stack_are_rejected(tmp_path: Path) -> None:
    """錯模型、錯纖維或空層疊不能無聲改算其他模型。"""
    result = _run('options', tmp_path)
    assert set(cast(list[str], result['rejected'])) == {'unknown', 'mechel_grundmann', 'empty_stack'}


def test_transfer_matrix_stays_out_of_physics_and_modal_closures() -> None:
    """門面或計算模組提早匯入 TMM 時紅；只掃原始碼，不執行材料公式。"""
    from aosr.reporting.import_closure import scan_import_closure
    from aosr.reporting.modal_diagnosis import MODAL_ENTRY_MODULE
    from aosr.reporting.physics_identity import physics_import_closure

    physics = physics_import_closure()
    modal = scan_import_closure(ROOT / 'src' / 'aosr', MODAL_ENTRY_MODULE)[0]
    assert 'aosr.materials.transfer_matrix' not in physics.modules
    assert 'aosr.materials.transfer_matrix' not in modal.modules
