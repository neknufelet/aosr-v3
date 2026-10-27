"""晚期混響 JAX 版對 numpy 雙精度版：同公式的重算、對吸音率的梯度、jit 前後、單精度路。

雙精度只在隔離子行程開（``runtime.configure_jax`` 寫進子行程的環境）：同程式的 JAX 跑在
預設的單精度，那裡比雙精度等於沒比。子行程只算 JAX 那一份；numpy 參考、中央差分與判決
都在這個父行程，不靠 JAX 自己證明自己。
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from aosr.geometry.shoebox import Room, Wall
from aosr.physics import late_energy, late_energy_jax
from aosr.runtime import child_process_env, configure_jax
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value


_ROOT = Path(__file__).resolve().parents[2]
_PROBE = Path(__file__).with_name("_late_energy_jax_probe.py")
_REFERENCE_CASES = ("flat", "lowabs", "varied")
_RANDOM_CASES = ("random_reference_room", "random_asymmetric_room")
_CASES = _REFERENCE_CASES + _RANDOM_CASES
_QUANTITIES = ("alpha_bar", "raw_reverberant_energy", "eyring_ratio", "late_reverberant_energy")
_TOLERANCE_REL = contract_value("late_energy_jax_vs_numpy")
_INSIDE = (1.0 - MUTANT_MARGIN) * _TOLERANCE_REL
_OUTSIDE = (1.0 + MUTANT_MARGIN) * _TOLERANCE_REL
# 中央差分步長：對 α 的絕對步長。h=1e-4／1e-5／1e-6 實量最大相對差 8.6e-9／9.2e-11／6.0e-10
# （截斷誤差與相減捨入在 1e-5 交會），取 1e-5。
_DIFFERENCE_STEP = 1e-5
# 梯度對中央差分的相對界線：實量最大 9.2e-11，留約百倍給換機器；不是精度契約，是差分本身的誤差。
_GRADIENT_REL = 1e-8
# jit 前後不逐位相同：XLA 合併運算後加總順序會變，實量 varied 與隨機那間最後一位差一格
# （相對 2.6e-16）。這裡只准差到雙精度四個最小刻度，差更多就是 jit 改了算法。
_JIT_ULPS = 4.0


class JaxCaseRun(TypedDict):
    eager: dict[str, list[str]]
    jit: dict[str, list[str]]
    jit_again: dict[str, list[str]]
    single: dict[str, list[str]]
    gradient: list[str]
    single_gradient: list[str]
    double_dtypes: list[str]
    single_dtypes: list[str]


def _answer_path(case_name: str) -> Path:
    return _ROOT / "blueprint" / f"reference_art_{case_name}.json"


def _alpha(inputs: late_energy.LateEnergyInputs) -> NDArray[np.float64]:
    return np.asarray([inputs.absorption_by_wall[wall] for wall in Wall.wall_names()], dtype=np.float64)


def _with_alpha(inputs: late_energy.LateEnergyInputs, alpha: NDArray[np.float64]) -> late_energy.LateEnergyInputs:
    rows = {wall: tuple(float(value) for value in alpha[index]) for index, wall in enumerate(Wall.wall_names())}
    return replace(inputs, absorption_by_wall=rows)


def _random_inputs(name: str) -> late_energy.LateEnergyInputs:
    """固定種子的隨機吸音率；避開 0 與 1 兩端（夾點與 I−F 奇異都不在這裡考）。"""
    base = late_energy.load_late_energy_inputs(_answer_path("flat"))
    rng = np.random.default_rng(562 + _RANDOM_CASES.index(name))
    frequencies = tuple(100.0 * (index + 1) for index in range(9))
    alpha = rng.uniform(0.02, 0.95, size=(len(Wall.wall_names()), len(frequencies)))
    inputs = replace(base, frequencies_hz=frequencies)
    if name == "random_asymmetric_room":
        inputs = replace(inputs, room=Room(8.0, 3.0, 5.0), n_per_wall=4)
    return _with_alpha(inputs, alpha)


def _inputs(name: str) -> late_energy.LateEnergyInputs:
    if name in _REFERENCE_CASES:
        return late_energy.load_late_energy_inputs(_answer_path(name))
    return _random_inputs(name)


def _numpy_arrays(inputs: late_energy.LateEnergyInputs) -> dict[str, NDArray[np.float64]]:
    bands = late_energy.solve_late_energy(inputs).bands
    return {name: np.asarray([getattr(band, name) for band in bands], dtype=np.float64) for name in _QUANTITIES}


def _decode(values: list[str]) -> NDArray[np.float64]:
    return np.asarray([float.fromhex(value) for value in values], dtype=np.float64)


def _contract_fractions(
    actual: NDArray[np.float64], expected: NDArray[np.float64], tolerance_rel: float
) -> NDArray[np.float64]:
    """逐點「差÷界線」：``|JAX−numpy| / (T·|numpy|)``，不超過 1 才在契約內。"""
    return np.abs(actual - expected) / (tolerance_rel * np.abs(expected))


def _within_contract(actual: NDArray[np.float64], expected: NDArray[np.float64], tolerance_rel: float) -> bool:
    fractions = _contract_fractions(actual, expected, tolerance_rel)
    return bool(np.all(np.isfinite(actual)) and np.all(fractions <= 1.0))


@pytest.fixture(scope="module")
def jax_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, JaxCaseRun]:
    """一個雙精度子行程把五題的 JAX 結果一次算完。"""
    directory = tmp_path_factory.mktemp("late-energy-jax")
    problem = {}
    for name in _CASES:
        inputs = _inputs(name)
        room = inputs.room
        problem[name] = {
            "room_m": [float(length).hex() for length in (room.Lx, room.Ly, room.Lz)],
            "n_per_wall": inputs.n_per_wall,
            "alpha": [[float(value).hex() for value in row] for row in _alpha(inputs)],
        }
    problem_path = directory / "problem.json"
    output_path = directory / "output.json"
    problem_path.write_text(json.dumps(problem), encoding="utf-8")
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TMPDIR"] = str(directory)
    process = subprocess.run(
        [sys.executable, str(_PROBE), str(problem_path), str(output_path)],
        cwd=directory, env=env, capture_output=True, text=True, timeout=300,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    loaded = cast(dict[str, dict[str, JaxCaseRun]], json.loads(output_path.read_text(encoding="utf-8")))
    return loaded["cases"]


@pytest.mark.parametrize("case_name", _CASES)
def test_double_precision_matches_numpy_within_contract(jax_runs: dict[str, JaxCaseRun], case_name: str) -> None:
    """三份凍結題目與兩組隨機吸音率：四個逐頻量都在 ``late_energy_jax_vs_numpy`` 界線內。

    抓公式抄錯（反射率乘錯邊、源項、正規化、Eyring 夾點）、牆順序或分格索引對錯。
    """
    expected = _numpy_arrays(_inputs(case_name))
    run = jax_runs[case_name]
    for quantity in _QUANTITIES:
        actual = _decode(run["eager"][quantity])
        assert actual.shape == expected[quantity].shape, quantity
        assert _within_contract(actual, expected[quantity], _TOLERANCE_REL), (
            quantity, float(np.max(_contract_fractions(actual, expected[quantity], _TOLERANCE_REL))),
        )


@pytest.mark.parametrize("case_name", _CASES)
def test_output_precision_follows_input(jax_runs: dict[str, JaxCaseRun], case_name: str) -> None:
    """雙精度進、雙精度出；單精度進、單精度出（輸出與梯度都算），函式裡沒有偷轉型。"""
    run = jax_runs[case_name]
    assert set(run["double_dtypes"]) == {"float64"}
    assert set(run["single_dtypes"]) == {"float32"}


@pytest.mark.parametrize("case_name", _CASES)
def test_gradient_matches_central_difference_of_numpy(jax_runs: dict[str, JaxCaseRun], case_name: str) -> None:
    """``jax.grad`` 對 α 的梯度等於 numpy 版的中央差分。

    純量取全部頻帶晚期能量的和；頻帶之間互不相依，所以梯度第 (w, f) 格就是
    ``∂E_f/∂α_{w,f}``，一面牆一次推全部頻帶、兩次 numpy 精確解就得到那一整列。
    """
    inputs = _inputs(case_name)
    alpha = _alpha(inputs)
    gradient = _decode(jax_runs[case_name]["gradient"]).reshape(alpha.shape)
    difference = np.zeros_like(alpha)
    for wall_index in range(alpha.shape[0]):
        plus = alpha.copy()
        minus = alpha.copy()
        plus[wall_index] += _DIFFERENCE_STEP
        minus[wall_index] -= _DIFFERENCE_STEP
        upper = _numpy_arrays(_with_alpha(inputs, plus))["late_reverberant_energy"]
        lower = _numpy_arrays(_with_alpha(inputs, minus))["late_reverberant_energy"]
        difference[wall_index] = (upper - lower) / (2.0 * _DIFFERENCE_STEP)
    assert np.all(np.abs(difference) > 0.0)
    np.testing.assert_allclose(gradient, difference, rtol=_GRADIENT_REL, atol=0.0)


@pytest.mark.parametrize("case_name", _CASES)
def test_jit_matches_eager_to_last_bits(jax_runs: dict[str, JaxCaseRun], case_name: str) -> None:
    """jit 前後同一組輸入：逐點最多差四個雙精度最小刻度；同一份編譯跑兩次逐位相同。"""
    run = jax_runs[case_name]
    for quantity in _QUANTITIES:
        eager = _decode(run["eager"][quantity])
        compiled = _decode(run["jit"][quantity])
        again = _decode(run["jit_again"][quantity])
        assert np.array_equal(compiled, again), quantity
        allowed = _JIT_ULPS * np.spacing(np.abs(eager))
        assert np.all(np.abs(compiled - eager) <= allowed), quantity


@pytest.mark.parametrize("case_name", _CASES)
def test_single_precision_path_is_finite(jax_runs: dict[str, JaxCaseRun], case_name: str) -> None:
    """單精度那一路算得完、不報錯，輸出與梯度都是有限數、能量為正。單精度對雙精度的差只量不判。"""
    run = jax_runs[case_name]
    for quantity in _QUANTITIES:
        values = _decode(run["single"][quantity])
        assert np.all(np.isfinite(values)), quantity
        assert np.all(values > 0.0), quantity
    assert np.all(np.isfinite(_decode(run["single_gradient"])))


def test_rejects_wrong_shape_and_integer_absorption() -> None:
    """形狀不是 (6, F)、或吸音率不是浮點，在進 jit 之前就報錯，不默默廣播或轉型。"""
    geometry = late_energy_jax.late_energy_geometry(Room(6.0, 4.0, 3.0), 2)
    with pytest.raises(ValueError, match="形狀"):
        late_energy_jax.late_energy_arrays(geometry, jnp.full((5, 3), 0.2, dtype=jnp.float32))
    with pytest.raises(ValueError, match="形狀"):
        late_energy_jax.late_energy_arrays(geometry, jnp.full((6,), 0.2, dtype=jnp.float32))
    with pytest.raises(TypeError, match="浮點"):
        late_energy_jax.late_energy_arrays(geometry, jnp.zeros((6, 3), dtype=jnp.int32))


def test_mutant_beyond_tolerance_is_red() -> None:
    """numpy 真值在界線內推 δ 判綠、界線外推 δ 判紅（兩側各 δ）。

    判法是 ``|JAX−numpy| ≤ T·|numpy|``，所以 JAX 那一格放 ``numpy·(1+(1∓δ)·T)`` 讓
    「差÷界線」剛好是 ``1∓δ``。四個量各推一次，每一個都要在界外被記成越界。
    """
    expected = _numpy_arrays(_inputs("flat"))
    for quantity in _QUANTITIES:
        truth = expected[quantity]
        inside = truth * (1.0 + _INSIDE)
        outside = truth * (1.0 + _OUTSIDE)
        assert _within_contract(inside, truth, _TOLERANCE_REL), quantity
        assert not _within_contract(outside, truth, _TOLERANCE_REL), quantity
        assert math.isclose(
            float(np.max(_contract_fractions(outside, truth, _TOLERANCE_REL))), 1.0 + MUTANT_MARGIN,
            rel_tol=MUTANT_MARGIN / 4.0,
        ), quantity
    assert not _within_contract(np.full(3, math.nan), np.ones(3), _TOLERANCE_REL)
