"""梯度目標子行程（#561 壬一）：只叫公開入口與 JAX 轉換，十六進位交回；不比對、不做判決。

每一份登記簿（正式那份、權重改 0 的暫存副本）各讀一次梯度目標設定，每一組（自己的頻率軸與能量）交：直接求值與編譯後求值的全部欄、
第一階段原函式的合分（直接與編譯後）、主位殘差、對能量的編譯後梯度（新合分、舊合分、代理、第一階段原函式的合分）、
溫度換成一串值時的代理，與上界 T·ln N。
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.gradient_objective_jax import (
    GradientObjectiveArrays, GradientObjectiveConstants, gradient_objective_constants, gradient_objective_from_energies,
    proxy_gap_bound_db,
)
from aosr.scoring.gradient_objective_settings import gradient_objective_settings
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_objective_jax import objective_from_energies, primary_residual_db

Record = dict[str, object]


def _hex(values: object) -> list[str]:
    return [float(value).hex() for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _energies(values: object) -> jax.Array:
    nested = cast(list[list[list[str]]], values)
    return jnp.asarray([[[float.fromhex(value) for value in row] for row in channel] for channel in nested],
                       dtype=jnp.float64)


def _arrays(result: GradientObjectiveArrays) -> dict[str, list[str]]:
    return {key: _hex(values) for key, values in result._asdict().items()}


def _gradient(pick: Callable[[jax.Array], jax.Array], energies: jax.Array) -> list[str]:
    return _hex(jax.jit(jax.grad(pick))(energies))


def _case(constants: GradientObjectiveConstants, energies: jax.Array, temperatures: list[float]) -> Record:
    stage_one = constants.stage_one()
    evaluate = jax.jit(lambda values: gradient_objective_from_energies(constants, values))
    original = jax.jit(lambda values: objective_from_energies(stage_one, values).screening_total)
    proxies = {}
    for temperature in temperatures:
        changed = constants._replace(proxy_temperature_db=np.asarray(temperature, dtype=np.float64))
        proxies[temperature.hex()] = _hex(gradient_objective_from_energies(changed, energies).big_peak_proxy_db)[0]
    return {
        "eager": _arrays(gradient_objective_from_energies(constants, energies)),
        "jit": _arrays(evaluate(energies)),
        "stage_one_eager": _hex(objective_from_energies(stage_one, energies).screening_total),
        "stage_one_jit": _hex(original(energies)),
        "residual": _hex(primary_residual_db(stage_one, energies)),
        "gradient": _gradient(lambda values: gradient_objective_from_energies(constants, values).objective_total,
                              energies),
        "screening_gradient": _gradient(
            lambda values: gradient_objective_from_energies(constants, values).screening_total, energies),
        "proxy_gradient": _gradient(
            lambda values: gradient_objective_from_energies(constants, values).big_peak_proxy_db, energies),
        "stage_one_gradient": _gradient(lambda values: objective_from_energies(stage_one, values).screening_total,
                                        energies),
        "temperatures": proxies,
        "bound": float(proxy_gap_bound_db(constants)).hex(),
    }


def main(problem_path: Path, output_path: Path) -> None:
    problem = cast(Record, json.loads(problem_path.read_text(encoding="utf-8")))
    receivers = ReceiverSet.model_validate(problem["receiver_set"])
    temperatures = [float.fromhex(value) for value in cast(list[str], problem["temperatures"])]
    cases = cast(dict[str, Record], problem["cases"])
    output: Record = {}
    for label, path in cast(dict[str, str], problem["registries"]).items():
        settings = gradient_objective_settings(load_quality_targets(path), str(problem["purpose"]))
        results: Record = {}
        for name, case in cases.items():
            # 每一組自己的頻率軸（48 點搜尋軸，或審查反例的 9 點軸）。
            frequencies = np.asarray([float.fromhex(value) for value in cast(list[str], case["frequencies_hz"])])
            constants = gradient_objective_constants(frequencies, settings, receivers, 2)
            results[name] = _case(constants, _energies(case["energies"]), temperatures)
        output[label] = {
            "fingerprint": settings.fingerprint, "stage_one_fingerprint": settings.stage_one.fingerprint,
            "weight": settings.proxy_weight.hex(), "cases": results,
        }
    output_path.write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
