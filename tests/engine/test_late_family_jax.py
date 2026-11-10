"""晚期一族與交接的 CPU 冷子行程考卷；參考、差分與契約裁判獨立於 JAX 實作。"""
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
from tests.engine._late_family_cases import CASES, air_fraction, alpha_rows, furniture_area, problem, reference, with_alpha
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL


def test_public_array_entries_exist() -> None:
    assert importlib.util.find_spec("aosr.physics.late_family_jax") is not None
    assert importlib.util.find_spec("aosr.physics.crossover_jax") is not None


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    directory = tmp_path_factory.mktemp("late-family-jax")
    env = child_process_env(threads=1)
    configure_jax(platforms="cpu", enable_x64=True, env=env)
    env["PYTHONDONTWRITEBYTECODE"], env["TMPDIR"] = "1", str(directory)
    env["PYTHONPATH"] = os.pathsep.join((env["PYTHONPATH"], str(Path(__file__).resolve().parents[2])))
    output = directory / "result.json"
    result = subprocess.run([sys.executable, str(Path(__file__).with_name("_late_family_jax_probe.py")), str(output)],
        cwd=directory, env=env, text=True, capture_output=True, timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    return cast(dict[str, object], json.loads(output.read_text()))


def _arrays(run: dict[str, object], case: str, mode: str) -> dict[str, np.ndarray]:
    cases = cast(dict[str, dict[str, dict[str, object]]], run["cases"])
    return {key: np.asarray(value) for key, value in cases[case][mode].items()}


def _check(actual: np.ndarray, expected: np.ndarray) -> None:
    # 一條逐格相對契約；零參考必須恰零。
    assert actual.shape == expected.shape
    assert np.all(np.isfinite(actual))
    np.testing.assert_allclose(actual, expected, rtol=contract_value("late_family_jax_vs_python"), atol=0)


@pytest.mark.parametrize("case", CASES)
def test_late_family_matches_numpy(run: dict[str, object], case: str) -> None:
    inputs = problem(case)
    expected = reference(inputs, furniture_area() if case in ("furniture", "furniture_air") else None,
                         air_fraction(inputs) if case in ("air", "furniture_air") else None)
    actual = _arrays(run, case, "jit")
    for key, value in expected.items():
        _check(actual[key], value)


@pytest.mark.parametrize("case", CASES)
def test_jit_and_single_precision(run: dict[str, object], case: str) -> None:
    eager, compiled, single = (_arrays(run, case, mode) for mode in ("eager", "jit", "single"))
    for key, value in eager.items():
        _check(compiled[key], value)
        assert np.all(np.isfinite(single[key]))
    dtypes = cast(dict[str, list[str]], run["dtypes"])
    assert set(dtypes[case + "64"]) == {"float64"}
    assert set(dtypes[case + "32"]) == {"float32"}


def _central_output_gradients(case: str, alpha: np.ndarray) -> dict[str, np.ndarray]:
    inputs = problem(case)
    area = furniture_area() if case in ("furniture", "furniture_air") else None
    epsilon = air_fraction(inputs) if case in ("air", "furniture_air") else None
    keys = ("energy_by_order", "tail_energy", "late_share_energy", "t60", "fs")
    baseline = reference(with_alpha(inputs, alpha), area, epsilon)
    expected = {key: np.zeros(baseline[key].shape + alpha.shape) for key in keys}
    # 沿用 stage_one_objective 的相對步長與 transfer_matrix 的無量綱梯度尺。
    for index in np.ndindex(alpha.shape):
        step = alpha[index] * 1e-5
        plus, minus = alpha.copy(), alpha.copy()
        plus[index] += step
        minus[index] -= step
        a, b = reference(with_alpha(inputs, plus), area, epsilon), reference(with_alpha(inputs, minus), area, epsilon)
        for key in keys:
            expected[key][(..., *index)] = (a[key] - b[key]) / (2 * step) * alpha[index]
    return expected


@pytest.mark.parametrize("case", CASES)
def test_each_absorption_gradient_matches_numpy_central_difference(run: dict[str, object], case: str) -> None:
    expected = _central_output_gradients(case, alpha_rows(problem(case)))
    gradients = cast(dict[str, dict[str, object]], run["gradients"])
    for key, central in expected.items():
        np.testing.assert_allclose(np.asarray(gradients[case][key]), central,
            rtol=GRADIENT_REL, atol=GRADIENT_ABS, err_msg=f"{case}: {key}")


def test_construction_parameters_through_paris_absorption(run: dict[str, object]) -> None:
    data = cast(dict[str, object], run["paris"])
    _check(np.asarray(data["actual"]), np.asarray(data["reference"]))
    np.testing.assert_allclose(np.asarray(data["automatic"]), np.asarray(data["central"]), rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    assert data["weights_bitwise"] is True
    assert data["angles_bitwise"] is True
    assert data["double_dtype"] == "float64"
    assert data["single_dtype"] == "float32"


@pytest.mark.parametrize("flag", ("paris", "constant", "alpha", "furniture", "energy", "scattering", "t60", "fs", "frequency", "missing_fem",
    "air_nan", "air_inf", "air_negative", "air_one"))
def test_invalid_items_are_nan_and_other_gradients_finite(run: dict[str, object], flag: str) -> None:
    rows = cast(dict[str, dict[str, object]], run["flags"])
    data = rows[flag]
    assert data["flag"] is True, flag
    assert data["nan"] is True, flag
    assert data["gradient_finite"] is True, flag
    assert data["valid_finite"] is True, flag


def test_crossover_weights_have_exactly_zero_gradient(run: dict[str, object]) -> None:
    gradients = cast(dict[str, object], run["weight_gradients"])
    for value in gradients.values():
        np.testing.assert_array_equal(np.asarray(value), 0)


def test_crossover_boundary_values_match_python(run: dict[str, object]) -> None:
    from aosr.physics.crossover import crossover_weights
    from tests.engine._late_family_cases import FREQUENCIES

    cases = cast(list[dict[str, object]], run["weight_cases"])
    for row in cases:
        expected = crossover_weights(FREQUENCIES, float(cast(float, row["fs"])))
        _check(np.asarray(row["w_fem"]), np.asarray(expected.w_fem))
        _check(np.asarray(row["w_geo"]), np.asarray(expected.w_geo))
        assert row["capped"] == expected.capped_by_upper_limit


def test_stitched_energy_matches_formal_report_points(run: dict[str, object]) -> None:
    row = cast(dict[str, object], run["stitched"])
    _check(np.asarray(row["actual"]), np.asarray(row["reference"]))
    np.testing.assert_allclose(np.asarray(row["gradient"]), np.asarray(row["central"]), rtol=GRADIENT_REL, atol=GRADIENT_ABS)


def test_scattering_gradient_matches_central_difference(run: dict[str, object]) -> None:
    from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
    from aosr.physics.geometric_lane import _late_share_energy
    from aosr.physics.late_energy import solve_late_energy_by_order

    bands = solve_late_energy_by_order(problem("zoned"), max_order=REFLECTION_ORDER_K).bands
    s = np.asarray(run["scattering_input"])
    central = np.zeros((len(s), len(s)))
    for index in range(len(s)):
        step = s[index] * 1e-5
        plus, minus = s.copy(), s.copy()
        plus[index] += step
        minus[index] -= step
        central[:, index] = (np.asarray(_late_share_energy(bands, tuple(plus))) -
            np.asarray(_late_share_energy(bands, tuple(minus)))) / (2 * step) * s[index]
    np.testing.assert_allclose(np.asarray(run["scattering_gradient"]), central, rtol=GRADIENT_REL, atol=GRADIENT_ABS)


def test_furniture_absorption_area_is_constant(run: dict[str, object]) -> None:
    np.testing.assert_array_equal(np.asarray(run["furniture_gradient"]), 0)


def test_equal_zones_have_area_weighted_gradients(run: dict[str, object]) -> None:
    """同材時凸組合仍是線性的；不能讓浮點上下界的 min/max 平分區域導數。"""
    assert run["uniform_forward_bitwise"] is True
    expected = _central_output_gradients("zoned", np.asarray(run["uniform_alpha"]))
    automatic = cast(dict[str, object], run["uniform_gradient"])
    for key, central in expected.items():
        np.testing.assert_allclose(np.asarray(automatic[key]), central,
            rtol=GRADIENT_REL, atol=GRADIENT_ABS, err_msg=key)


def test_capped_stitch_is_valid_and_has_hard_cut_energy_gradients(run: dict[str, object]) -> None:
    from aosr.physics.crossover import crossover_weights
    from tests.engine._late_family_cases import FREQUENCIES

    rows = cast(list[dict[str, object]], cast(dict[str, object], run["stitched"])["capped"])
    for row in rows:
        weights = crossover_weights(FREQUENCIES, float(cast(float, row["fs"])))
        assert row["capped"] is True
        assert not np.any(np.asarray(row["invalid"]))
        _check(np.asarray(row["actual"]), np.asarray(row["reference"]))
        fem_gradient, geo_gradient = cast(list[object], row["gradients"])
        np.testing.assert_array_equal(np.asarray(fem_gradient), np.diag(weights.w_fem))
        np.testing.assert_array_equal(np.asarray(geo_gradient), np.diag(weights.w_geo))


def test_crossing_cap_changes_only_exact_cap_frequency(run: dict[str, object]) -> None:
    from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    from tests.engine._late_family_cases import FREQUENCIES

    row = cast(dict[str, object], cast(dict[str, object], run["flags"])["capped"])
    assert row["capped"] is True
    assert not np.any(np.asarray(row["invalid"]))
    difference = np.asarray(row["boundary"]) - np.asarray(row["below"])
    cap = np.asarray(FREQUENCIES) == FEM_GEOMETRIC_CROSSOVER_CAP_HZ
    np.testing.assert_array_equal(difference[~cap], 0.0)
    np.testing.assert_array_equal(difference[cap], (np.asarray(row["fem"]) - np.asarray(row["geo"]))[cap])
    assert np.count_nonzero(difference) == 1


@pytest.mark.parametrize("axis_name", ("search_octave_24", "search_octave_48"))
def test_product_octave_axes_do_not_sample_exact_cap(axis_name: str) -> None:
    from aosr.config.frequency_axis import (
        FEM_GEOMETRIC_CROSSOVER_CAP_HZ, LowFrequencyAxis, low_frequency_axis_frequencies,
    )
    _, axis = low_frequency_axis_frequencies(LowFrequencyAxis(axis_name))
    assert FEM_GEOMETRIC_CROSSOVER_CAP_HZ not in axis


def test_omitted_patch_absorption_uses_each_wall_row(run: dict[str, object]) -> None:
    expected = reference(problem("whole"))
    modes = cast(dict[str, dict[str, object]], run["implicit_patches"])
    for row in modes.values():
        assert not np.any(np.asarray(row["invalid"]))
        for key in ("total_energy", "energy_by_order", "tail_energy"):
            _check(np.asarray(row[key]), expected[key])


def test_mutant_beyond_tolerance_is_red(run: dict[str, object]) -> None:
    expected = reference(problem("zoned"))
    for key, values in _arrays(run, "zoned", "jit").items():
        original = expected[key]
        flat = int(np.flatnonzero(original)[0])
        changed = values.copy()
        changed.flat[flat] = original.flat[flat] * (1 + (1 + MUTANT_MARGIN) * contract_value("late_family_jax_vs_python"))
        with pytest.raises(AssertionError):
            _check(changed, original)


def test_air_is_constant_and_zero_preserves_family_bitwise(run: dict[str, object]) -> None:
    """空氣不能參與反傳，零空氣不能改動既有晚期陣列。"""
    np.testing.assert_array_equal(np.asarray(run["air_gradient"]), 0.0)
    assert run["zero_air_bitwise"] is True
