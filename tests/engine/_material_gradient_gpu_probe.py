"""CPU 同程序單雙精度對照與 lowering（編譯前表示）；不建替身、不跑搜尋。"""
from __future__ import annotations

import dataclasses
import json
import io
from types import SimpleNamespace

import pytest
import sys
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.config.fem_reduced import load_reduced_model_settings
from aosr.config.frequency_axis import FEM_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.gradient import material_gradient as search
from aosr.physics.fem_reduced import BuildLedger, ReducedMatrices, ReducedModel
from tests.engine.test_material_gradient import _designs, _objective_constants, _receivers


def model() -> ReducedModel:
    rank = 16
    rng = np.random.default_rng(561)
    basis = rng.normal(size=(6, rank, rank))
    matrices = ReducedMatrices(
        stiffness=np.diag(np.linspace(30.0, 60.0, rank)), mass=np.eye(rank),
        boundary=np.einsum('pij,pkj->pik', basis, basis) / rank,
        sources=rng.normal(size=(2, rank)).astype(np.complex128), receivers=rng.normal(size=(7, rank)))
    return ReducedModel(matrices=matrices, source_names=("L", "R"),
        receiver_names=tuple(point.receiver_id for point in _receivers().points),
        frequencies_hz=FEM_LANE_FREQUENCIES_HZ, density_kg_m3=1.2, sound_speed_m_s=343.0,
        mesh_hash="synthetic", physics_identity="synthetic", input_hash="synthetic",
        ledger=BuildLedger(0, 0, rank, (), "synthetic", {}, load_reduced_model_settings(config_path("fem_reduced.toml"))))


def objective(single: bool, *, perturbed: bool = False) -> search.ChainObjective:
    room = model()
    if perturbed:
        changed = room.matrices.stiffness.copy()
        changed[0, 0] *= 2
        room = dataclasses.replace(room, matrices=dataclasses.replace(room.matrices, stiffness=changed))
    air, ranges = construction_environment(density_kg_m3=room.density_kg_m3, sound_speed_m_s=room.sound_speed_m_s)
    return search.chain_objective(room, _designs("all_lined"), _objective_constants(room), air=air, ranges=ranges,
                                   frequencies=room.frequencies_hz, frequency_batch=16, single_precision=single)


def wiring() -> dict[str, object]:
    """只替換磁碟輸入閘門，load_problem、候選工廠、CPU 參考入口都用產品接線。"""
    from aosr.gradient import gpu_reference, run
    from aosr.gradient.gpu import SinglePrecisionObjective
    from aosr.reporting.scheme import Scheme
    from aosr.materials.construction_models import ConstantImpedance
    from aosr.geometry.shoebox import Wall
    from tests.engine.test_material_gradient import _settings
    from tests.engine.test_material_gradient_mount import _project

    room = model()
    scheme = cast(Scheme, SimpleNamespace(scene=SimpleNamespace(density_kg_m3=1.2, sound_speed_m_s=343.0)))
    kept = dict.fromkeys(Wall.all(), ConstantImpedance(impedance_pa_s_m=1500.0))
    project = run.GradientProject.model_validate({**_project().model_dump(), "device": "gpu"})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "check_inputs", lambda project: (scheme, room, _settings()))
        patch.setattr(run, "project_wall_materials", lambda project, scheme: kept)
        patch.setattr(run, "search_objective_constants", lambda project, scheme, model: _objective_constants(model))
        problem = run.load_problem(project)
        candidate = run.designs_for(("glass_wool_48k_50mm",) * 6, _settings(), kept=kept)
        other = problem.templates.objective_for(candidate)
        assert isinstance(problem.objective, SinglePrecisionObjective)
        assert isinstance(other, SinglePrecisionObjective)
        theta = search.baseline_theta(candidate)
        (value, _), gradient = search.compiled_evaluator(other)(jnp.asarray(theta))
        request = project.model_dump_json() + "\n" + json.dumps([float(x).hex() for x in search.baseline_theta(problem.designs)])
        output = io.StringIO()
        patch.setattr(sys, "stdin", io.StringIO(request))
        patch.setattr(sys, "stdout", output)
        gpu_reference.main()
        reference = float.fromhex(output.getvalue().strip())
        single, _ = problem.objective(jnp.asarray(search.baseline_theta(problem.designs)))
    return {"candidate_value": float(value), "candidate_gradient_dtype": str(gradient.dtype),
            "reference": reference, "single": float(single)}


def main() -> None:
    from aosr.gradient.gpu import SinglePrecisionObjective

    theta = jnp.asarray(search.baseline_theta(_designs("all_lined")), dtype=jnp.float64)
    double, single = objective(False), objective(True)
    (reference, _), reference_gradient = search.compiled_evaluator(double)(theta)
    (value, aux), gradient = search.compiled_evaluator(single)(theta)
    mutated, _ = objective(True, perturbed=True)(theta)
    gpu = cast(SinglePrecisionObjective, single)
    text = gpu.compiled.lower(theta, gpu.constants).as_text()
    # 常數矩陣形狀不得出現在 stablehlo.constant（穩定編譯表示的字面常數）那一行。
    shapes = ("16x16", "6x16x16", "7x16", "2x16")
    embedded = [line for line in text.splitlines() if "stablehlo.constant" in line and
                any(f"tensor<{shape}x" in line for shape in shapes)]
    dots = [line for line in text.splitlines() if "stablehlo.dot_general" in line]
    sys.stdout.write(json.dumps({"wiring": wiring(), "double_gradient_hex": [float(x).hex() for x in reference_gradient], "double": float(reference), "single": float(value), "mutated": float(mutated),
        "dtype": str(value.dtype), "gradient_dtype": str(gradient.dtype), "embedded": embedded,
        "ledger_step": search._descent_step(0, theta, value, aux, float(jnp.linalg.norm(gradient)), 0.0, 0.0)._asdict(),
        "dots": dots, "argument_signature": text.split("func.func public @main", 1)[1].split("{", 1)[0],
        "array_dtypes": [str(x.dtype) for x in gpu.constants],
        "gradient_finite": bool(np.isfinite(gradient).all()),
        "gradient_nonzero": bool(np.any(np.asarray(gradient) != 0)),
        "double_gradient_finite": bool(np.isfinite(reference_gradient).all())}) + "\n")


if __name__ == "__main__":
    main()
