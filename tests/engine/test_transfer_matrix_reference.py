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
COEFFICIENT_ABS = 2e-14  # 無量綱；α≈0 時用絕對項，實量最大差 1.333e-15。
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
        np.testing.assert_allclose(_decode(actual[key]), _decode(expected_values),
                                   rtol=REFERENCE_REL, atol=_absolute_tolerance(key), err_msg=f'{case_name}::{key}')


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
                np.testing.assert_allclose(_decode(row[transformation][key]), _decode(expected),
                                           rtol=REFERENCE_REL, atol=_absolute_tolerance(key), err_msg=f'{name}:{transformation}:{key}')


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
