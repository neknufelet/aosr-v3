"""CPU 裝置上的共用常數、活陣列位元組與舊單精度鏈對照；不建替身。"""
from __future__ import annotations

import gc
import json
import sys
import weakref
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from aosr.geometry.shoebox import Wall
from aosr.gradient import material_gradient as search, run
from aosr.gradient.gpu import Kernel, SinglePrecisionObjective
from aosr.materials.construction_models import ConstantImpedance
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_jax import ReducedConstants, reduced_constants
from aosr.reporting.scheme import Scheme
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays
from tests.engine._material_gradient_gpu_probe import model
from tests.engine.test_material_gradient import _objective_constants, _settings
from tests.engine.test_material_gradient_mount import _project


class LegacyObjective(SinglePrecisionObjective):
    """414e8ee6 的建構式；忽略新增的共用入參，保留每個候選自行放置的獨立對照。"""

    def __init__(self, kernel: Kernel, constants: ReducedConstants, *,
                 device_constants: ReducedConstants | None = None) -> None:
        def put(value: np.ndarray | jax.Array) -> jax.Array:
            dtype = jnp.complex64 if np.issubdtype(value.dtype, np.complexfloating) else (
                jnp.bool_ if np.issubdtype(value.dtype, np.bool_) else jnp.float32)
            return cast(jax.Array, jax.device_put(jnp.asarray(value, dtype=dtype)))
        self.constants = cast(ReducedConstants, jax.tree.map(put, constants))

        def single(theta: jax.Array, arrays: ReducedConstants) -> tuple[jax.Array, GradientObjectiveArrays]:
            with jax.default_matmul_precision("highest"):
                value, aux = kernel(theta.astype(jnp.float32), arrays)
                return value.astype(jnp.float64), jax.tree.map(lambda x: x.astype(jnp.float64), aux)

        self.forward = jax.jit(single)
        self.compiled = jax.jit(jax.value_and_grad(single, has_aux=True))


def load_problem(room: ReducedModel, patch: pytest.MonkeyPatch, batch: int | None = 16) -> run.GradientProblem:
    """只替換磁碟輸入，使用真的 load_problem 與 objective_for 接線。"""
    scheme = cast(Scheme, SimpleNamespace(scene=SimpleNamespace(density_kg_m3=1.2, sound_speed_m_s=343.0)))
    kept = dict.fromkeys(Wall.all(), ConstantImpedance(impedance_pa_s_m=1500.0))
    patch.setattr(run, "check_inputs", lambda project: (scheme, room, _settings()))
    patch.setattr(run, "project_wall_materials", lambda project, scheme: kept)
    patch.setattr(run, "search_objective_constants", lambda project, scheme, model: _objective_constants(model))
    project = run.GradientProject.model_validate({**_project().model_dump(), "device": "gpu", "frequency_batch": batch})
    return run.load_problem(project)


def candidates(problem: run.GradientProblem) -> list[tuple[search.SurfaceDesign, ...]]:
    return [run.designs_for((preset,) * 6, problem.settings, kept=problem.kept)
            for preset in ("glass_wool_48k_50mm", "glass_wool_24k_50mm", "glass_wool_96k_50mm")]


def live_bytes() -> int:
    gc.collect()
    arrays = cast(Callable[[], list[jax.Array]], jax.live_arrays)()
    return sum(array.block_until_ready().nbytes for array in arrays)


def sharing() -> dict[str, object]:
    with pytest.MonkeyPatch.context() as patch:
        problem = load_problem(model(64), patch)
        initial = cast(SinglePrecisionObjective, problem.objective)
        before = live_bytes()
        others = [cast(SinglePrecisionObjective, problem.templates.objective_for(designs))
                  for designs in candidates(problem)]
        growth = live_bytes() - before
        leaves = jax.tree.leaves(initial.constants)
        shared = [all(a is b for a, b in zip(leaves, jax.tree.leaves(other.constants), strict=True))
                  for other in others]
        return {"shared": shared, "growth": growth, "constant_bytes": sum(a.nbytes for a in leaves),
                "initial_reused": problem.templates.objective_for(problem.designs) is initial}


def bitwise(batch: int | None) -> dict[str, object]:
    room = model()
    with pytest.MonkeyPatch.context() as patch:
        problem = load_problem(room, patch, batch)
        designs = [problem.designs, *candidates(problem)[:2]]
        actual = [cast(SinglePrecisionObjective, problem.templates.objective_for(item)) for item in designs]
        patch.setattr(search, "SinglePrecisionObjective", LegacyObjective)
        legacy = load_problem(room, patch, batch)
        expected = [cast(SinglePrecisionObjective, legacy.templates.objective_for(item)) for item in designs]
        matches = []
        for item, shared, separate in zip(designs, actual, expected, strict=True):
            theta = jnp.asarray(search.baseline_theta(item))
            left, right = shared.evaluate(theta), separate.evaluate(theta)
            matches.append(all(np.asarray(a).dtype == np.asarray(b).dtype and
                               np.asarray(a).tobytes() == np.asarray(b).tobytes()
                               for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)))
        return {"matches": matches}


def lifetime(accept: bool) -> dict[str, object]:
    from aosr.gradient import template_rounds as rounds
    from aosr.scoring.gradient_objective_jax import objective_reading
    from tests.engine._material_gradient_templates_probe import synthetic_factory

    with pytest.MonkeyPatch.context() as patch:
        problem = load_problem(model(), patch)
    settings = problem.settings
    target = settings.family[1].id
    offsets = {rounds.candidate_id(problem.designs[0].surface_id, target): -100.0 if accept else 100.0}
    make_kernel = synthetic_factory(settings, offsets, [0])
    refs: list[tuple[weakref.ReferenceType[SinglePrecisionObjective], weakref.ReferenceType[object],
                     weakref.ReferenceType[object]]] = []

    def factory(designs: tuple[search.SurfaceDesign, ...]) -> search.ChainObjective:
        kernel = make_kernel(designs)
        candidate = SinglePrecisionObjective(lambda theta, arrays: kernel(theta), reduced_constants(model()))
        refs.append((weakref.ref(candidate), weakref.ref(candidate.forward), weakref.ref(candidate.compiled)))
        return candidate

    context = problem.templates._replace(objective_for=factory)
    initial = factory(problem.designs)
    theta = search.snap_to_specs(search.baseline_theta(problem.designs), problem.designs).theta
    value, aux = initial(jnp.asarray(theta))
    reading = objective_reading(aux)
    point = rounds.TemplatePoint(problem.designs, theta, float(value), reading.screening_total, reading.components)
    result = rounds.template_round(initial, point, context, budget_left=10**6)
    gc.collect()
    discarded = [triple for triple in refs if triple[0]() is not initial and triple[0]() is not result.objective]
    return {"accepted": result.accepted is not None, "discarded": len(discarded),
            "retained": sum(any(ref() is not None for ref in triple) for triple in discarded),
            "chosen_callable": bool(np.isfinite(float(result.objective(jnp.asarray(result.point.theta))[0])))}


def main() -> None:
    mode = sys.argv[1]
    if mode in ("accept", "reject"):
        result = lifetime(mode == "accept")
    else:
        result = sharing() if mode == "sharing" else bitwise(None if mode == "full" else 16)
    sys.stdout.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
