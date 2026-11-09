"""分頻批次的雙精度探針：只交同程式原始值及編譯器記憶體帳，不判決。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aosr.physics.fem_reduced_jax import ReducedConstants, reduced_pressures
from tests.engine._fem_reduced_jax_probe import _bits


def problem(count: int) -> tuple[ReducedConstants, jax.Array, jax.Array]:
    """合成正定小系統，不建替身、不讀磁碟模型；稠密邊界避免對角解特例。"""
    rank, patches = 48, 6
    rng = np.random.default_rng(561)
    basis = rng.normal(size=(patches, rank, rank))
    boundary = np.einsum('pij,pkj->pik', basis, basis) / rank
    axis = np.linspace(20.0, 200.0, count)
    constants = ReducedConstants(
        stiffness=np.diag(np.linspace(2.0, 4.0, rank)), mass=np.eye(rank), boundary=boundary,
        sources=rng.normal(size=(2, rank)).astype(np.complex128), receivers=rng.normal(size=(3, rank)),
        frequencies_hz=axis, density_kg_m3=np.asarray(1.2), sound_speed_m_s=np.asarray(343.0),
        k_squared=np.linspace(0.1, 0.9, count), wall_numerator=np.asarray(1j * axis, dtype=np.complex128),
        partition_mask=np.zeros(patches, dtype=np.bool_))
    z = jnp.asarray(1000.0 + rng.uniform(0.0, 1000.0, (patches, count)) + 200j)
    return constants, z, jnp.asarray([False, True, False, False, True, False])


def main() -> None:
    output: dict[str, object] = {}
    constants, z, rigid = problem(14)
    for batch in (None, 1, 7, 5, 14, 20):
        def pressure(values: jax.Array) -> jax.Array:
            return reduced_pressures(constants, values, rigid_mask=rigid, frequency_batch=batch)
        def energy(values: jax.Array) -> jax.Array:
            return jnp.sum(jnp.abs(pressure(values)) ** 2)
        value, gradient = jax.jit(jax.value_and_grad(energy))(z)
        output[str(batch)] = {"pressure": _bits(jax.jit(pressure)(z)), "value": _bits(value), "gradient": _bits(gradient)}
    memory: dict[str, object] = {}
    for count in (14, 28):
        constants, z, rigid = problem(count)
        for batch in (None, 7):
            def energy(values: jax.Array) -> jax.Array:
                return jnp.sum(jnp.abs(reduced_pressures(constants, values, rigid_mask=rigid, frequency_batch=batch)) ** 2)
            for name, function in (("forward", energy), ("gradient", jax.value_and_grad(energy))):
                compiled = jax.jit(function).lower(z).compile()
                analysis = compiled.memory_analysis()
                if analysis is None:
                    raise RuntimeError("CPU 編譯器沒有記憶體分析數字")
                memory[f"{name}:{count}:{batch}"] = analysis.temp_size_in_bytes
    output["memory"] = memory
    sys.stdout.write(json.dumps(output) + "\n")


class _BatchReached(Exception):
    """在真正求解前停住，記下 project 往下傳的批量。"""


def wiring(folder: Path, mode: str) -> dict[str, object]:
    """只換數值求解邊界；讀檔、建鏈、終點編譯都走產品入口，回傳原始接線紀錄。"""
    from aosr.gradient import endpoint_run, material_gradient, run, six_category_chain

    batches: list[object] = []
    paths: list[str] = []

    def stop_at_solve(*args: object, **kwargs: object) -> NoReturn:
        batches.append(kwargs.get("frequency_batch"))
        raise _BatchReached

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(material_gradient, "reduced_pressures", stop_at_solve)
        patch.setattr(six_category_chain, "reduced_pressures", stop_at_solve)
        if mode == "resume":
            problem = run.load_problem(run.read_project(folder))
            candidate = run.designs_for((None,) * len(problem.designs), problem.settings, kept=problem.kept)
            for name, designs, objective in (
                ("initial", problem.designs, problem.objective),
                ("candidate", candidate, problem.templates.objective_for(candidate)),
            ):
                try:
                    objective(jnp.asarray(material_gradient.baseline_theta(designs)))
                except _BatchReached:
                    paths.append(name)
        elif mode == "endpoint":
            try:
                endpoint_run.run_endpoint(folder)
            except _BatchReached:
                paths.append("endpoint")
        else:
            raise ValueError(f"未知探針模式：{mode}")
    return {"batches": batches, "paths": paths}


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        sys.stdout.write(json.dumps(wiring(Path(sys.argv[2]), sys.argv[1])) + "\n")
