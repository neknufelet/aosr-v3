"""晚期衰減 JAX 等價性、對稱精確根、逐格梯度及逐帶失敗旗標。"""
from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from aosr.config.art_lane import ART_NEUMANN_K_MAX
from aosr.physics import late_decay, late_decay_jax, late_energy
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine._t20_jax_cases import ALPHA_FAILURES, CASES, FLAGS, decode, inputs, mutant, numpy_t20, section, t20_runs
from tests.engine.test_stage_one_objective_jax import _DIFFERENCE_STEP
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL


@pytest.mark.parametrize("name", CASES)
def test_symmetric_largest_root_matches_exact_maximum_modulus(t20_runs: dict[str, object], name: str) -> None:
    """B=diag(A)F 對稱非負。W>0 時 W^-1/2 (WB) W^1/2 = sqrt(W)B sqrt(W)；
    Perron 定理保證最大模等於最大實特徵值。剛性格 α=0 仍有 W>0。
    """
    base, _speed, alpha = inputs(name)
    patches = late_energy._patch_geometry(base.room, base.n_per_wall)
    form = late_energy._form_factors(patches)
    exchange = patches.areas[:, None] * form
    np.testing.assert_allclose(exchange, exchange.T, rtol=contract_value("late_energy_physical_property"), atol=0.0)
    alpha[0] = 0.0
    w = (1.0 - alpha) / patches.areas[:, None]
    symmetric = np.sqrt(w).T[:, :, None] * exchange * np.sqrt(w).T[:, None, :]
    exact = late_decay._exact_roots((1.0 - alpha)[:, None, :] * form[:, :, None])
    np.testing.assert_allclose(np.linalg.eigvalsh(symmetric)[:, -1], exact,
                               rtol=contract_value("late_energy_physical_property"), atol=0.0)
    np.testing.assert_allclose(decode(section(t20_runs, "decay", name)["rigid_roots"]), exact,
                               rtol=contract_value("late_decay_t20_jax_vs_numpy"), atol=0.0)


@pytest.mark.parametrize("name", CASES)
def test_double_t20_matches_numpy(t20_runs: dict[str, object], name: str) -> None:
    expected = numpy_t20(name)
    run = section(t20_runs, "decay", name)
    for mode in ("eager", "jit"):
        np.testing.assert_allclose(decode(run[mode]), expected, rtol=contract_value("late_decay_t20_jax_vs_numpy"), atol=0.0)
    assert not np.any(decode(run["flags"]))
    assert run["double_dtype"] == "float64"
    assert run["single_dtype"] == "float32"
    assert np.all(np.isfinite(decode(run["single"])))
    assert np.array_equal(decode(run["jit"]), decode(run["again"]))


@pytest.mark.parametrize("name", ("patches", "tail"))
def test_patch_gradient_matches_numpy_central_difference(t20_runs: dict[str, object], name: str) -> None:
    """tail（低吸音尾巴）會抓到漏尾巴及 eigvalsh 根被擋掉梯度。"""
    _base, _speed, alpha = inputs(name)
    central = np.empty_like(alpha)
    for index in np.ndindex(alpha.shape):
        plus, minus = alpha.copy(), alpha.copy()
        plus[index] += _DIFFERENCE_STEP
        minus[index] -= _DIFFERENCE_STEP
        central[index] = (np.sum(numpy_t20(name, plus)) - np.sum(numpy_t20(name, minus))) / (2.0 * _DIFFERENCE_STEP)
    automatic = decode(section(t20_runs, "decay", name)["gradient"])
    np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS)
    assert np.all(np.isfinite(decode(section(t20_runs, "decay", name)["single_gradient"])))


@pytest.mark.parametrize("name", ("flat", "varied", "lowabs"))
def test_t20_matches_public_numpy_entry(t20_runs: dict[str, object], name: str) -> None:
    base, speed, _alpha = inputs(name)
    reference = late_decay.solve_late_decay_t20(base, sound_speed_m_s=speed)
    expected = np.asarray([band.t20_s for band in reference.bands])
    np.testing.assert_allclose(decode(section(t20_runs, "decay", name)["jit"]), expected,
                               rtol=contract_value("late_decay_t20_jax_vs_numpy"), atol=0.0)


@pytest.mark.parametrize("name", FLAGS + tuple(ALPHA_FAILURES))
def test_each_failure_is_nan_and_other_band_gradient_is_finite(t20_runs: dict[str, object], name: str) -> None:
    run = section(t20_runs, "failure", name)
    assert run["flag"] == [True, False]
    expected_flag = name if name in FLAGS else "invalid_absorption"
    assert run["flags"] == {flag: [flag == expected_flag, False] for flag in FLAGS}
    value = decode(run["value"])
    assert np.isnan(value[0]) and np.isfinite(value[1])
    gradient = decode(run["gradient"])
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient[:, 0] == 0.0)
    assert np.any(gradient[:, 1] != 0.0)


@pytest.mark.parametrize("name", ("zero_variance", "flat_slope"))
def test_zero_slope_with_enough_weight_has_only_nonnegative_slope(t20_runs: dict[str, object], name: str) -> None:
    run = section(t20_runs, "failure", name)
    assert run["flags"] == {flag: [flag == "nonnegative_slope", False] for flag in FLAGS}
    assert np.isnan(decode(run["value"])[0])


def test_flat_slope_with_insufficient_range_has_only_range_flag(t20_runs: dict[str, object]) -> None:
    run = section(t20_runs, "failure", "flat_range")
    assert run["flags"] == {flag: [flag == "insufficient_decay_range", False] for flag in FLAGS}
    assert np.isnan(decode(run["value"])[0])


def test_one_rigid_wall_is_valid(t20_runs: dict[str, object]) -> None:
    run = section(t20_runs, "failure", "rigid_wall")
    assert run["flag"] == [False, False]
    assert np.all(np.isfinite(decode(run["value"])))


def test_scan_really_uses_registered_order_count(t20_runs: dict[str, object]) -> None:
    assert t20_runs["scan_length"] == ART_NEUMANN_K_MAX


def test_nonuniform_fit_is_not_perron_shortcut(t20_runs: dict[str, object]) -> None:
    base, speed, _alpha = inputs("varied")
    problem = late_energy._reflection_problem(base)
    perron = late_decay._perron_t60(late_decay._exact_roots(problem.transfer), late_decay._collision_frequency(base, speed))
    actual = decode(section(t20_runs, "decay", "varied")["jit"])
    assert np.any(np.abs(actual / perron - 1.0) > contract_value("late_decay_t20_jax_vs_numpy"))


def test_mutant_beyond_tolerance_is_red() -> None:
    """直接推 NumPy 真值；正負兩側都考界內綠、界外紅，不混入實作差。"""
    expected = numpy_t20("tail")
    tolerance = contract_value("late_decay_t20_jax_vs_numpy")
    for direction in (-1.0, 1.0):
        inside = expected * (1.0 + direction * tolerance * (1.0 - MUTANT_MARGIN))
        outside = expected * (1.0 + direction * tolerance * (1.0 + MUTANT_MARGIN))
        np.testing.assert_allclose(inside, expected, rtol=tolerance, atol=0.0)
        with pytest.raises(AssertionError):
            np.testing.assert_allclose(outside, expected, rtol=tolerance, atol=0.0)


def test_removing_exact_tail_mutant_is_red(t20_runs: dict[str, object]) -> None:
    expected = numpy_t20("tail")
    actual = decode(mutant(t20_runs, "no_tail"))
    assert np.any(np.abs(actual - expected) > contract_value("late_decay_t20_jax_vs_numpy") * np.abs(expected))


def test_short_scan_mutant_is_red(t20_runs: dict[str, object]) -> None:
    assert mutant(t20_runs, "short_scan_length") != ART_NEUMANN_K_MAX


def test_perron_fallback_mutant_is_red(t20_runs: dict[str, object]) -> None:
    assert np.any(np.isfinite(decode(mutant(t20_runs, "perron_fallback"))))


def test_geometry_uses_existing_form_factors() -> None:
    base, speed, _alpha = inputs("patches")
    geometry = late_decay_jax.late_decay_geometry(base.room, base.n_per_wall, sound_speed_m_s=speed)
    problem = late_energy._reflection_problem(base)
    assert np.array_equal(geometry.form_factors, late_energy._form_factors(problem.patches))
    assert geometry.collision_frequency_hz == late_decay._collision_frequency(base, speed)


def test_mixed_failures_keep_flags_values_and_gradients_in_their_bands(t20_runs: dict[str, object]) -> None:
    """同時出現非法吸音、衰減不足與 NaN 時，好帶的值與梯度仍獨立。"""
    run = section(t20_runs, "failure", "mixed_failures")
    assert run["flag"] == [False, False, True, False]
    assert run["flags"] == {
        "invalid_absorption": [True, False, False, True],
        "insufficient_decay_range": [False, True, False, False],
        "insufficient_weight": [False, False, False, False],
        "nonnegative_slope": [False, False, False, False],
        "numerical_failure": [False, False, False, False],
    }
    values = decode(run["value"])
    assert np.all(np.isnan(values[[0, 1, 3]]))
    reference = decode(section(t20_runs, "failure", "mixed_failures_reference")["value"])
    np.testing.assert_allclose(values[2], reference[2], rtol=contract_value("late_decay_t20_jax_vs_numpy"), atol=0.0)
    gradient = decode(run["gradient"])
    assert np.all(np.isfinite(gradient))
    assert np.all(gradient[:, [0, 1, 3]] == 0.0)
    assert np.any(gradient[:, 2] != 0.0)


def test_decay_scan_has_no_operands_with_three_or_more_dimensions() -> None:
    """抓：把逐頻三維轉移表重新放進每階縮併；不以牆鐘設門檻。"""
    base, speed, alpha = inputs("patches")
    geometry = late_decay_jax.late_decay_geometry(base.room, base.n_per_wall, sound_speed_m_s=speed)
    expression = jax.make_jaxpr(late_decay_jax.late_decay_arrays)(geometry, jnp.asarray(alpha))
    scans = [equation for equation in expression.jaxpr.eqns if equation.primitive.name == "scan"]
    assert len(scans) == 1
    body = scans[0].params["jaxpr"].jaxpr
    for equation in body.eqns:
        for operand in equation.invars:
            assert len(operand.aval.shape) < 3, (equation.primitive.name, operand.aval.shape)
