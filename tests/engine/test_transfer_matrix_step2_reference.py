"""第二步：凍結原套件答案、SciPy 比值與中央差分；六十四位元只在子行程。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.runtime import child_process_env, configure_jax


ROOT = Path(__file__).resolve().parents[2]
PROBLEM_FILE = ROOT / 'blueprint' / 'tmm_reference_problem_step2.json'
ANSWER_FILE = ROOT / 'blueprint' / 'tmm_reference_answers_step2.json'
PROBE = Path(__file__).with_name('_transfer_matrix_step2_probe.py')
CASES = tuple(json.loads(PROBLEM_FILE.read_text())['cases'])
# 同公式雙精度：阻抗實部/虛部實量最大相對差 1.689e-12 / 7.289e-13。
# 三角函式、複數除法與共振近零消去留約三倍空間，非物理誤差。
REFERENCE_REL = 5e-12
IMPEDANCE_ABS = 2e-10  # Pa*s/m 或 rad/m；相對項負責大值，絕對項負責近零。
COEFFICIENT_ABS = 2e-14  # 吸音全域只用絕對項，實量最大絕對差 4.441e-15。
# AD−BC 的相消誤差用 32 epsilon * max(1, |AD|+|BC|) 限制。
# 完整凍結頻域包含高衰減厚層；不可把病態相減當成物理行列式失效。
DETERMINANT_SCALED_ABS = 32 * np.finfo(np.float64).eps
DETERMINANT_ABS = 2e-14  # 第二步凍結層相消較小；約 90 epsilon 的直接絕對尺。
# 256 項反向連分式在 β<=1024 的截斷差低於雙精度運算誤差。
# SciPy jve 的共同縮放消掉，避免 jv 在大引數溢位；不是另一種近似。
BESSEL_REL = 5e-14  # 實量最大相對差 1.169e-15，留不同平台浮點運算空間。
BESSEL_ABS = 2e-17
# 中央差分相對步長 1e-5（層）/1e-4（SciPy 比值）；比 p*dy/dp。
# O(h²) 截斷與微小實部相減的誤差，近零導數由絕對項判。
# SciPy β 導數最大縮放絕對差 1.257e-8；層導數很大時由相對項判。
GRADIENT_REL = 3e-7
GRADIENT_ABS = 2e-8


def _run(mode: str, directory: Path) -> dict[str, object]:
    env = child_process_env(threads=1).copy()
    configure_jax(platforms='cpu', enable_x64=True, env=env)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['TMPDIR'] = str(directory)
    output = directory / f'{mode}.json'
    process = subprocess.run([sys.executable, str(PROBE), mode, str(PROBLEM_FILE), str(output)],
        cwd=directory, env=env, capture_output=True, text=True, timeout=240)
    assert process.returncode == 0, process.stdout + process.stderr
    return cast(dict[str, object], json.loads(output.read_text()))


def _decode(values: list[str]) -> np.ndarray:
    return np.asarray([float.fromhex(value) for value in values])


def _absolute_tolerance(key: str) -> float:
    return COEFFICIENT_ABS if key.startswith(('absorption', 'reflection')) else IMPEDANCE_ABS


def _assert_reference_values(key: str, actual: list[str], expected: list[str], context: str) -> None:
    np.testing.assert_allclose(_decode(actual), _decode(expected),
        rtol=0 if key == 'absorption_hex' else REFERENCE_REL,
        atol=_absolute_tolerance(key), err_msg=context)


@pytest.fixture(scope='module')
def reference_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return _run('reference', tmp_path_factory.mktemp('tmm-step2-reference'))


@pytest.fixture(scope='module')
def bessel_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    return _run('bessel', tmp_path_factory.mktemp('tmm-step2-bessel'))


@pytest.mark.parametrize('case_name', CASES)
def test_step2_spectra_match_frozen_reference(reference_run: dict[str, object], case_name: str) -> None:
    """逐案例逐頻率抓 SI、層序、末端修正、並聯質量、長縫開孔率和黏熱射線植錯。"""
    expected = json.loads(ANSWER_FILE.read_text())['cases'][case_name]
    actual = cast(dict[str, dict[str, list[str]]], reference_run['cases'])[case_name]
    assert reference_run['dtype'] == 'float64'
    assert set(actual) == set(expected)
    for key, values in expected.items():
        _assert_reference_values(key, actual[key], values, f'{case_name}::{key}')


def test_step2_absorption_comparison_rejects_absolute_drift_near_unity() -> None:
    """恢復相對項會錯放 α≈1 的 1e-12 偏差，這把吸音尺必須紅。"""
    with pytest.raises(AssertionError):
        _assert_reference_values('absorption_hex', [(0.8 + 1e-12).hex()], [0.8.hex()], '吸音絕對尺')


def test_each_step2_layer_has_unit_determinant(tmp_path: Path) -> None:
    """膜、面層各算法、圓孔四算法、長縫與空氣/多孔層的整體矩陣倍數錯誤時紅。"""
    rows = cast(list[dict[str, object]], _run('determinants', tmp_path)['rows'])
    assert rows
    for row in rows:
        determinant = _decode(cast(list[str], row['real_hex'])) + 1j * _decode(cast(list[str], row['imag_hex']))
        np.testing.assert_allclose(determinant, 1, rtol=0, atol=DETERMINANT_ABS, err_msg=str(row['layer']))
        scale = np.maximum(1, _decode(cast(list[str], row['cancellation_scale_hex'])))
        np.testing.assert_allclose((determinant - 1) / scale, 0, rtol=0,
            atol=DETERMINANT_SCALED_ABS, err_msg=str(row['layer']))


def test_step2_uses_frozen_air_and_covers_every_problem_case(reference_run: dict[str, object]) -> None:
    """移除案例或偷偷拿另一組空氣條件時紅，不釘案例數量。"""
    problem = json.loads(PROBLEM_FILE.read_text())
    assert reference_run['air'] == {key: value['value'] for key, value in problem['air_properties'].items()}
    assert set(cast(dict[str, object], reference_run['cases'])) == set(problem['cases'])
    assert CASES


def test_bessel_ratios_match_scipy_on_both_rays(bessel_run: dict[str, object]) -> None:
    """少掉連分式一項、比值共軛、黏熱射線弄錯，在對數間隔全範圍必須紅。"""
    rows = cast(list[dict[str, object]], bessel_run['rows'])
    assert rows
    assert bessel_run['dtype'] == 'float64'
    assert cast(float, bessel_run['tested_max_beta']) > cast(float, bessel_run['required_beta'])
    for row in rows:
        actual = _decode(cast(list[str], row['actual_real'])) + 1j * _decode(cast(list[str], row['actual_imag']))
        expected = _decode(cast(list[str], row['expected_real'])) + 1j * _decode(cast(list[str], row['expected_imag']))
        np.testing.assert_allclose(actual, expected, rtol=BESSEL_REL, atol=BESSEL_ABS,
            err_msg=f"ray={row['ray']} kind={row['kind']}")


def test_bessel_beta_gradients_match_scipy_central_difference(bessel_run: dict[str, object]) -> None:
    """對 β 的實部/虛部導數斷線或變 NaN，與獨立 SciPy 中央差分不符時紅。"""
    rows = cast(list[dict[str, object]], bessel_run['derivatives'])
    assert rows
    for row in rows:
        automatic = _decode(cast(list[str], row['automatic_scaled']))
        central = _decode(cast(list[str], row['central_scaled']))
        assert np.all(np.isfinite(automatic)) and np.all(np.isfinite(central)), row
        np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS,
            err_msg=f"ray={row['ray']} divided={row['divided']} component={row['component']}")


def test_bessel_d_beta_gradients_match_scipy_central_difference(bessel_run: dict[str, object]) -> None:
    """只在 |z|>200 切斷 D 梯度時，兩條射線到 β=1024 的直接中央差分必須紅。"""
    rows = cast(list[dict[str, object]], bessel_run['d_derivatives'])
    assert rows
    assert cast(float, bessel_run['tested_max_beta']) >= 1024.0
    for row in rows:
        automatic = _decode(cast(list[str], row['automatic_scaled']))
        central = _decode(cast(list[str], row['central_scaled']))
        assert np.all(np.isfinite(automatic)) and np.all(np.isfinite(central)), row
        np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS,
            err_msg=f"D ray={row['ray']} component={row['component']}")


def test_step2_parameter_gradients_match_central_difference(tmp_path: Path) -> None:
    """每種新層及幾何推導支的全部連續參數，中央差分抓斷線與 NaN。"""
    rows = cast(list[dict[str, object]], _run('gradients', tmp_path)['rows'])
    assert rows
    for row in rows:
        automatic, central = float(cast(float, row['automatic_scaled'])), float(cast(float, row['central_scaled']))
        assert np.isfinite(automatic) and np.isfinite(central), row
        np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS, err_msg=str(row))


def test_step2_jit_and_vmap_keep_continuous_inputs_dynamic(tmp_path: Path) -> None:
    """編譯/向量化頻率與所有連續參數，誤轉 Python 值或廣播維度錯誤時紅。"""
    rows = cast(dict[str, dict[str, dict[str, list[str]]]], _run('transforms', tmp_path)['rows'])
    assert rows
    for name, row in rows.items():
        for transformation in ('jit', 'vmap'):
            for key, expected in row['pointwise'].items():
                _assert_reference_values(key, row[transformation][key], expected, f'{name}:{transformation}:{key}')


def test_step2_static_options_and_constant_surface_impedance(tmp_path: Path) -> None:
    """未知選項/缺物性不能無聲替算；定值阻抗不是層，先加膜再放入要明確拒收。"""
    result = _run('options', tmp_path)
    assert set(cast(list[str], result['rejected'])) == {'perforated_method', 'end_correction',
        'slotted_method', 'facing_method', 'missing_pitch', 'missing_facing_mass',
        'missing_facing_resistance', 'missing_pressure', 'missing_prandtl',
        'constant_scalar_in_stack', 'constant_vector_in_stack', 'constant_as_only_layer',
        'constant_matrix_frequency', 'constant_array_impedance'}
    assert set(cast(list[str], result['required_rejected'])) == {'constant_z', 'membrane_mass',
        'panel_thickness', 'panel_diameter', 'slotted_thickness', 'slotted_width'}
    np.testing.assert_array_equal(cast(list[float], result['constant_z']), [180.0, 240.0])


def test_pressure_is_optional_excluded_frozen_and_positive() -> None:
    """新增大氣壓不能換四欄序列化契約或舊物理/模態身分；非法值要拒收。"""
    air = load_physics_constants(config_path('physics_constants.toml'))
    assert air.atmospheric_pressure_pa == json.loads(PROBLEM_FILE.read_text())['air_properties']['atmospheric_pressure_pa']['value']
    assert 'atmospheric_pressure_pa' not in air.model_dump()
    old = PhysicsConstants.model_validate(air.model_dump())
    assert old.atmospheric_pressure_pa is None
    with pytest.raises(ValidationError):
        PhysicsConstants.model_validate({**air.model_dump(), 'atmospheric_pressure_pa': 0.0})
    with pytest.raises(ValidationError):
        air.atmospheric_pressure_pa = 100000.0
