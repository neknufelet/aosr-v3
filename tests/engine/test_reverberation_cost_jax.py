"""殘響兩項代價對正式 Python、旗標分母跳變與安全梯度。"""
from __future__ import annotations

import numpy as np
import pytest

from aosr.scoring import reverberation_cost_jax
from tests.engine._precision_contracts import contract_value
from tests.engine._t20_jax_cases import (
    COST_CASES, REGISTRY_CHANGES, changed_purpose, decode, mutant, python_cost, registry_values, section, t20_runs,
)
from tests.engine.test_stage_one_objective_jax import _DIFFERENCE_STEP
from tests.engine.test_transfer_matrix_reference import GRADIENT_ABS, GRADIENT_REL


@pytest.mark.parametrize("name", tuple(COST_CASES))
def test_cost_matches_python_and_uses_available_denominators(t20_runs: dict[str, object], name: str) -> None:
    values = np.asarray(COST_CASES[name])
    expected, components = python_cost(values)
    run = section(t20_runs, "cost", name)
    tolerance = contract_value("reverberation_cost_jax_vs_python")
    np.testing.assert_allclose(decode(run["value"]), expected, rtol=0.0, atol=tolerance, equal_nan=True)
    assert bool(np.isfinite(decode(run["single"]))) == bool(np.isfinite(expected))
    for key, field in (("t20_target_interval", "interval_mean"), ("adjacent_t20_jump", "adjacent_mean")):
        np.testing.assert_allclose(decode(run[field]), components.get(key, float("nan")), rtol=0.0, atol=tolerance, equal_nan=True)
    valid = np.isfinite(values)
    assert run["band_count"] == int(np.sum(valid))
    assert run["pair_count"] == int(np.sum(valid[:-1] & valid[1:]))
    assert run["unavailable"] == (not np.any(valid))
    assert run["double_dtype"] == "float64" and run["single_dtype"] == "float32"


@pytest.mark.parametrize("name", tuple(COST_CASES))
def test_t20_gradient_matches_python_central_difference(t20_runs: dict[str, object], name: str) -> None:
    values = np.asarray(COST_CASES[name])
    central = np.zeros_like(values)
    for i in np.flatnonzero(np.isfinite(values)):
        plus, minus = values.copy(), values.copy()
        plus[i] += _DIFFERENCE_STEP
        minus[i] -= _DIFFERENCE_STEP
        central[i] = (python_cost(plus)[0] - python_cost(minus)[0]) / (2.0 * _DIFFERENCE_STEP)
    automatic = decode(section(t20_runs, "cost", name)["gradient"])
    np.testing.assert_allclose(automatic, central, rtol=GRADIENT_REL, atol=GRADIENT_ABS)


def test_explicit_flag_excludes_finite_band_and_two_original_neighbors(t20_runs: dict[str, object]) -> None:
    run = section(t20_runs, "cost", "explicit_flag")
    values = np.asarray(COST_CASES["mixed"])
    flags = np.zeros(values.shape, dtype=np.bool_)
    flags[1] = True
    expected = python_cost(values, flags)[0]
    assert run["band_count"] == len(values) - 1
    assert run["pair_count"] == len(values) - 3
    assert np.isnan(decode(run["interval_costs"])[1])
    assert np.all(np.isnan(decode(run["adjacent_costs"])[:2]))
    assert np.isfinite(decode(run["value"]))
    assert abs(float(decode(run["value"])) - expected) <= contract_value("reverberation_cost_jax_vs_python")
    gradient = decode(run["gradient"])
    assert gradient[1] == 0.0 and np.all(np.isfinite(gradient))


@pytest.mark.parametrize("name", ("negative", "zero", "infinite"))
def test_invalid_t20_is_nan_without_poisoning_other_gradients(t20_runs: dict[str, object], name: str) -> None:
    run = section(t20_runs, "cost", name)
    assert decode(run["band_flags"])[1] == 1.0
    assert np.isnan(decode(run["interval_costs"])[1])
    assert np.isfinite(decode(run["value"]))
    assert np.all(np.isfinite(decode(run["gradient"])))


@pytest.mark.parametrize("name", REGISTRY_CHANGES)
def test_registry_weights_and_scales_are_read(t20_runs: dict[str, object], name: str) -> None:
    values = registry_values(name)
    selected = changed_purpose(name)
    expected, components = python_cost(values, quality=selected)
    run = section(t20_runs, "cost", f"registry_{name}")
    tolerance = contract_value("reverberation_cost_jax_vs_python")
    np.testing.assert_allclose(decode(run["value"]), expected, rtol=0.0, atol=tolerance)
    assert abs(expected - python_cost(values)[0]) > tolerance
    # 逐帶裁判也對，寫死任一帶下限或上限都會紅。
    from aosr.scoring.reverberation_cost import target_intervals

    centers = tuple(target_intervals(selected))
    expected_bands = [components[f"t20_target_interval.{center:g}Hz"] for center in centers]
    np.testing.assert_allclose(decode(run["interval_costs"]), expected_bands, rtol=0.0, atol=tolerance)
    np.testing.assert_allclose(decode(run["adjacent_mean"]), components["adjacent_t20_jump"], rtol=0.0, atol=tolerance)
    settings = reverberation_cost_jax.reverberation_cost_settings(selected)
    assert len(settings.lower_s) == len(COST_CASES["mixed"])


def test_mutant_beyond_tolerance_is_red(t20_runs: dict[str, object]) -> None:
    expected = float(decode(section(t20_runs, "cost", "explicit_flag")["value"]))
    actual = float(decode(mutant(t20_runs, "cost_shift")))
    tolerance = contract_value("reverberation_cost_jax_vs_python")
    assert abs(actual - expected) > tolerance


def test_fixed_denominator_mutant_is_red(t20_runs: dict[str, object]) -> None:
    expected = float(decode(section(t20_runs, "cost", "explicit_flag")["value"]))
    actual = float(decode(mutant(t20_runs, "fixed_denominator")))
    assert abs(actual - expected) > contract_value("reverberation_cost_jax_vs_python")
