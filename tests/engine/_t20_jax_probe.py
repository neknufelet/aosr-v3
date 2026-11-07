"""冷子行程只跑 JAX 公開入口；原始陣列交還父考卷的獨立裁判。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aosr.config.paths import config_path
from aosr.config.art_lane import ART_NEUMANN_K_MAX
from aosr.config.quality_targets import QualityPurpose, load_quality_targets
from aosr.geometry.shoebox import Room
from aosr.physics import late_decay_jax as decay
from aosr.scoring import reverberation_cost_jax as cost
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine._t20_jax_cases import FLAGS, REGISTRY_CHANGES, changed_purpose, registry_values


class DecayCase(TypedDict):
    room: list[float]
    n: int
    speed: float
    alpha: list[list[float]]


def _list(value: jax.Array) -> object:
    return np.asarray(value).tolist()


def _finite_total(values: jax.Array) -> jax.Array:
    return jnp.sum(jnp.where(jnp.isfinite(values), values, 0.0))


def _failure_result(run: decay.LateDecayArrays, name: str, gradient: jax.Array) -> dict[str, object]:
    return {"flag": _list(getattr(run, name)), "flags": {flag: _list(getattr(run, flag)) for flag in FLAGS},
            "value": _list(run.t20_s), "gradient": _list(gradient)}


def _decay_case(case: DecayCase) -> dict[str, object]:
    geometry = decay.late_decay_geometry(Room(*case["room"]), case["n"], sound_speed_m_s=case["speed"])
    alpha = jnp.asarray(case["alpha"], dtype=jnp.float64)
    compiled = jax.jit(decay.late_decay_arrays)
    eager, first = decay.late_decay_arrays(geometry, alpha), compiled(geometry, alpha)
    single = compiled(geometry, alpha.astype(jnp.float32))
    result = {"eager": _list(eager.t20_s), "jit": _list(first.t20_s), "again": _list(compiled(geometry, alpha).t20_s),
              "single": _list(single.t20_s), "flags": _list(~first.valid),
              "double_dtype": str(first.t20_s.dtype), "single_dtype": str(single.t20_s.dtype)}
    result["rigid_roots"] = _list(jax.jit(decay._exact_roots)(jnp.asarray(geometry.form_factors),
        jnp.asarray(geometry.patch_areas), 1.0 - alpha.at[0].set(0.0)))
    if case["n"] == 2:
        gradient = jax.jit(jax.grad(lambda a: jnp.sum(decay.late_decay_arrays(geometry, a).t20_s)))(alpha)
        result["gradient"] = _list(gradient)
        result["single_gradient"] = _list(jax.jit(jax.grad(lambda a: jnp.sum(decay.late_decay_arrays(geometry, a).t20_s)))(alpha.astype(jnp.float32)))
    return result


def _failures() -> dict[str, object]:
    geometry = decay.late_decay_geometry(Room(6.0, 4.0, 3.0), 1, sound_speed_m_s=343.0)
    alpha = jnp.full((6, 2), 0.3, dtype=jnp.float64)
    cases: dict[str, object] = {}
    for name, value in (("invalid_absorption", 1.0), ("insufficient_decay_range", 0.001)):
        values = alpha.at[:, 0].set(value)
        run = jax.jit(decay.late_decay_arrays)(geometry, values)
        gradient = jax.jit(jax.grad(lambda a: _finite_total(decay.late_decay_arrays(geometry, a).t20_s)))(values)
        cases[name] = _failure_result(run, name, gradient)
    for name, value in (("alpha_nan", float("nan")), ("alpha_inf", float("inf")),
                        ("alpha_negative", -0.1), ("alpha_above_one", 1.1), ("alpha_one_patch", 1.0),
                        ("alpha_rounded_away", float(np.finfo(np.float64).eps / 4.0))):
        values = alpha.at[0, 0].set(value)
        run = jax.jit(decay.late_decay_arrays)(geometry, values)
        gradient = jax.jit(jax.grad(lambda a: _finite_total(decay.late_decay_arrays(geometry, a).t20_s)))(values)
        cases[name] = _failure_result(run, "invalid_absorption", gradient)
    rigid = jax.jit(decay.late_decay_arrays)(geometry, alpha.at[0].set(0.0))
    cases["rigid_wall"] = {"flag": _list(rigid.invalid_absorption), "value": _list(rigid.t20_s)}
    order = jnp.arange(ART_NEUMANN_K_MAX, dtype=jnp.float64)
    valid = -order
    for name, bad in (("insufficient_weight", jnp.full_like(order, -400.0)),
                      ("nonnegative_slope", order - 128.0),
                      ("numerical_failure", jnp.full_like(order, jnp.nan))):
        levels = jnp.stack((bad, valid), axis=1)
        run = jax.jit(decay._fit_decay_arrays)(levels, jnp.asarray(100.0))
        gradient = jax.jit(jax.grad(lambda v: _finite_total(decay._fit_decay_arrays(v, jnp.asarray(100.0)).t20_s)))(levels)
        cases[name] = _failure_result(run, name, gradient)
    # 有限碰撞頻率的極值使時間變異數下溢為零；-32 是二次冪，常數曲線均值可恰算。
    for name, bad, collision in (("zero_variance", valid, jnp.asarray(np.finfo(np.float64).max)),
                                 ("flat_slope", jnp.full_like(order, -32.0), jnp.asarray(100.0)),
                                 ("flat_range", jnp.zeros_like(order), jnp.asarray(100.0))):
        run = jax.jit(decay._fit_decay_arrays)(bad[:, None], collision)
        good = jax.jit(decay._fit_decay_arrays)(valid[:, None], jnp.asarray(100.0))
        cases[name] = {"flags": {flag: _list(jnp.concatenate((getattr(run, flag), getattr(good, flag)))) for flag in FLAGS},
                       "value": _list(jnp.concatenate((run.t20_s, good.t20_s)))}
    return cases


def _purpose() -> QualityPurpose:
    return load_quality_targets(config_path("quality_targets.toml")).purpose("dedicated_two_channel_listening_room")


def _cost_case(values: list[float], purpose: QualityPurpose, *, flagged: bool = False) -> dict[str, object]:
    settings = cost.reverberation_cost_settings(purpose)
    t20 = jnp.asarray(values, dtype=jnp.float64)
    flags = jnp.zeros(t20.shape, dtype=jnp.bool_)
    if flagged:
        flags = flags.at[1].set(True)
    compiled = jax.jit(cost.reverberation_cost_arrays)
    run = compiled(settings, t20, flags)
    single = compiled(settings, t20.astype(jnp.float32), flags)
    gradient = jax.jit(jax.grad(lambda v: _finite_total(cost.reverberation_cost_arrays(settings, v, flags).category_cost)))(t20)
    return {"value": _list(run.category_cost), "single": _list(single.category_cost),
            "interval_mean": _list(run.interval_mean), "adjacent_mean": _list(run.adjacent_mean),
            "band_count": _list(run.band_count), "pair_count": _list(run.pair_count),
            "interval_costs": _list(run.interval_costs), "adjacent_costs": _list(run.adjacent_costs),
            "band_flags": _list(run.unavailable_bands), "unavailable": _list(run.unavailable),
            "double_dtype": str(run.category_cost.dtype), "single_dtype": str(single.category_cost.dtype),
            "gradient": _list(gradient)}


def _decay_mutants(case: DecayCase) -> dict[str, object]:
    geometry = decay.late_decay_geometry(Room(*case["room"]), case["n"], sound_speed_m_s=case["speed"])
    alpha = jnp.asarray(case["alpha"], dtype=jnp.float64)
    original_level = decay._decay_level
    result: dict[str, object] = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(decay, "_decay_level", lambda orders, roots: original_level(orders, jnp.zeros_like(roots)))
        result["no_tail"] = _list(jax.jit(lambda a: decay.late_decay_arrays(geometry, a).t20_s)(alpha))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(decay, "ART_NEUMANN_K_MAX", ART_NEUMANN_K_MAX // 2)
        expression = jax.make_jaxpr(lambda a: decay.late_decay_arrays(geometry, a))(alpha)
        result["short_scan_length"] = [e.params["length"] for e in expression.jaxpr.eqns if e.primitive.name == "scan"][0]
    invalid = jnp.full_like(alpha, 0.001)
    original_solve = decay.late_decay_arrays

    def perron_fallback(values: jax.Array) -> jax.Array:
        fit = original_solve(geometry, values)
        root = decay._exact_roots(jnp.asarray(geometry.form_factors), jnp.asarray(geometry.patch_areas), 1.0 - values)
        perron = jnp.log(1e-6) / jnp.log(root) / geometry.collision_frequency_hz
        return jnp.where(fit.valid, fit.t20_s, perron)

    result["perron_fallback"] = _list(jax.jit(perron_fallback)(invalid))
    return result


def _mutants(case: DecayCase, values: list[float]) -> dict[str, object]:
    result = _decay_mutants(case)
    settings = cost.reverberation_cost_settings(_purpose())
    t20 = jnp.asarray(values, dtype=jnp.float64)
    flags = jnp.zeros(t20.shape, dtype=jnp.bool_).at[1].set(True)
    original_cost = cost.reverberation_cost_arrays
    tolerance = contract_value("reverberation_cost_jax_vs_python")

    def shifted_cost(s: cost.ReverberationCostSettings, t: jax.Array, f: jax.Array) -> cost.ReverberationCostArrays:
        run = original_cost(s, t, f)
        return run._replace(category_cost=run.category_cost + tolerance * (1.0 + MUTANT_MARGIN))

    def fixed_mean(v: jax.Array, valid: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.sum(jnp.where(valid, v, 0.0)) / v.size, jnp.sum(valid)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cost, "reverberation_cost_arrays", shifted_cost)
        result["cost_shift"] = _list(jax.jit(lambda t: cost.reverberation_cost_arrays(settings, t, flags).category_cost)(t20))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cost, "_masked_mean", fixed_mean)
        result["fixed_denominator"] = _list(jax.jit(lambda t: cost.reverberation_cost_arrays(settings, t, flags).category_cost)(t20))
    return result


def main(problem: Path, output: Path) -> None:
    loaded = cast(dict[str, object], json.loads(problem.read_text(encoding="utf-8")))
    decay_cases = cast(dict[str, DecayCase], loaded["decay"])
    cost_cases = cast(dict[str, list[float]], loaded["cost"])
    costs = {name: _cost_case(values, _purpose()) for name, values in cost_cases.items()}
    costs["explicit_flag"] = _cost_case(cost_cases["mixed"], _purpose(), flagged=True)
    for name, value in (("negative", -1.0), ("zero", 0.0), ("infinite", float("inf"))):
        values = cost_cases["mixed"].copy()
        values[1] = value
        costs[name] = _cost_case(values, _purpose())
    for name in REGISTRY_CHANGES:
        costs[f"registry_{name}"] = _cost_case(registry_values(name).tolist(), changed_purpose(name))
    patch = decay_cases["patches"]
    geometry = decay.late_decay_geometry(Room(*patch["room"]), patch["n"], sound_speed_m_s=patch["speed"])
    expression = jax.make_jaxpr(decay.late_decay_arrays)(geometry, jnp.asarray(patch["alpha"]))
    scan_lengths = [equation.params["length"] for equation in expression.jaxpr.eqns if equation.primitive.name == "scan"]
    assert len(scan_lengths) == 1
    output.write_text(json.dumps({"decay": {name: _decay_case(case) for name, case in decay_cases.items()},
                                  "cost": costs, "failure": _failures(), "scan_length": scan_lengths[0],
                                  "mutants": _mutants(decay_cases["tail"], cost_cases["mixed"])}), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
