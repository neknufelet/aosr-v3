"""低頻目標子行程：只叫公開入口與 JAX 轉換，十六進位交回結果；不比對、不做判決。"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import TypedDict, cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_objective_jax import (
    StageOneObjectiveArrays, StageOneObjectiveConstants, objective_constants, objective_from_energies, objective_from_pressures,
)
from aosr.scoring.stage_one_score import ComponentName, StageOneScoreSettings, stage_one_score_settings


class Case(TypedDict):
    energies: list[list[list[str]]]
    gradient_energies: list[list[list[str]]]
    frequencies_hz: list[str]
    receiver_set: dict[str, object]
    scored_range_hz: list[str]
    weights: dict[str, str]
    worse_references: dict[str, str]


def _hex(values: jax.Array) -> list[str]:
    return [float(value).hex() for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def _arrays(result: StageOneObjectiveArrays) -> dict[str, list[str]]:
    return {key: _hex(values) for key, values in result._asdict().items()}


def _energy_array(values: list[list[list[str]]]) -> jax.Array:
    return jnp.asarray([[[float.fromhex(value) for value in row] for row in channel] for channel in values],
                       dtype=jnp.float64)


def _settings(case: Case) -> StageOneScoreSettings:
    base = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")),
                                    "dedicated_two_channel_listening_room")
    lower, upper = (float.fromhex(value) for value in case["scored_range_hz"])
    return replace(base, scored_range_hz=(lower, upper),
                   weights=tuple((cast(ComponentName, key), float.fromhex(value))
                                 for key, value in case["weights"].items()),
                   targets=tuple((cast(ComponentName, key), base.target(cast(ComponentName, key)).model_copy(update={
                       "worse_reference": float.fromhex(value)})) for key, value in case["worse_references"].items()))


def _pressure_pair(constants: StageOneObjectiveConstants, energies: jax.Array
                   ) -> tuple[StageOneObjectiveArrays, StageOneObjectiveArrays]:
    # 非零而且逐格不同的相位，避免只考到純實數聲壓；能量定義由公開轉接器與直接入口各算一次。
    phase = jnp.arange(energies.size, dtype=energies.dtype).reshape(energies.shape) * 0.17
    pressures = jnp.transpose(jnp.sqrt(energies) * jnp.exp(1j * phase), (2, 0, 1))
    return (objective_from_pressures(constants, pressures),
            objective_from_energies(constants, jnp.transpose(jnp.abs(pressures) ** 2, (1, 2, 0))))


def _run_case(case: Case) -> dict[str, object]:
    energies = _energy_array(case["energies"])
    constants = objective_constants(np.asarray([float.fromhex(value) for value in case["frequencies_hz"]]),
                                    _settings(case), ReceiverSet.model_validate(case["receiver_set"]), energies.shape[0])
    compiled = jax.jit(objective_from_energies)
    eager, first, second = objective_from_energies(constants, energies), compiled(constants, energies), compiled(constants, energies)
    single = compiled(constants, energies.astype(jnp.float32))

    def total(values: jax.Array) -> jax.Array:
        return objective_from_energies(constants, values).screening_total

    def gained_total(gain: jax.Array, values: jax.Array) -> jax.Array:
        return objective_from_energies(constants, values * jnp.exp(gain)).screening_total

    single_eager = objective_from_energies(constants, energies.astype(jnp.float32))
    gain_gradient = jax.jit(jax.grad(gained_total))(jnp.zeros((), dtype=jnp.float64), energies)
    single_gain_gradient = jax.jit(jax.grad(gained_total))(jnp.zeros((), dtype=jnp.float32),
                                                           energies.astype(jnp.float32))
    gradient = jax.jit(jax.grad(total))(_energy_array(case["gradient_energies"]))
    single_gradient = jax.jit(jax.grad(total))(_energy_array(case["gradient_energies"]).astype(jnp.float32))
    pressure, pressure_energy = _pressure_pair(constants, energies)
    single_pressure, single_pressure_energy = _pressure_pair(constants, energies.astype(jnp.float32))
    batch = jnp.stack((energies, energies[:, :, ::-1]))
    vectorized = jax.jit(jax.vmap(objective_from_energies, in_axes=(None, 0)))(constants, batch)
    return {
        "eager": _arrays(eager), "jit": _arrays(first), "jit_again": _arrays(second), "single": _arrays(single),
        "single_eager": _arrays(single_eager),
        "gradient": _hex(gradient), "single_gradient": _hex(single_gradient),
        "gain_gradient": _hex(gain_gradient), "single_gain_gradient": _hex(single_gain_gradient),
        "double_dtypes": [str(value.dtype) for result in (eager, first, pressure) for value in result] +
                         [str(gradient.dtype), str(gain_gradient.dtype)],
        "single_dtypes": [str(value.dtype) for result in (single, single_eager, single_pressure) for value in result] +
                         [str(single_gradient.dtype), str(single_gain_gradient.dtype)],
        "pressure": _arrays(pressure), "pressure_energy": _arrays(pressure_energy),
        "single_pressure": _arrays(single_pressure), "single_pressure_energy": _arrays(single_pressure_energy),
        "vmap": _arrays(vectorized), "loop": [_arrays(compiled(constants, row)) for row in batch],
    }


def main(problem_path: Path, output_path: Path) -> None:
    problem = cast(dict[str, Case], json.loads(problem_path.read_text(encoding="utf-8")))
    output_path.write_text(json.dumps({name: _run_case(case) for name, case in problem.items()}), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
