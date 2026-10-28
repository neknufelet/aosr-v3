"""牆面材料梯度鏈雙精度子行程；只叫公開入口、存原始數值，差分、比對與判決在父行程。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.quality_targets import load_quality_targets
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_models import Assembly
from aosr.physics.fem_reduced_store import load_reduced
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_objective_jax import objective_constants
from aosr.scoring.stage_one_score import stage_one_score_settings
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient import material_gradient_ledger as ledger


Record = dict[str, object]


def _hex(values: object) -> list[str]:
    return [float(value).hex() for value in np.asarray(values).ravel()]


def _complex(values: np.ndarray) -> Record:
    return {"real": _hex(values.real), "imag": _hex(values.imag), "shape": list(values.shape)}


def _steps(result: gradient_search.DescentResult) -> Record:
    return {
        "theta": [[value.hex() for value in step.theta] for step in result.steps],
        "value": [step.value.hex() for step in result.steps],
        "components": [[value.hex() for value in step.components] for step in result.steps],
        "gradient_norm": [step.gradient_norm.hex() for step in result.steps],
        "step_size": [step.step_size.hex() for step in result.steps],
        "final_theta": _hex(result.theta), "final_dtype": str(result.theta.dtype),
        "stop_reason": result.stop_reason, "final_trial_step": result.final_trial_step.hex(),
    }


def _impedances(designs: tuple[gradient_search.SurfaceDesign, ...], theta: np.ndarray, frequencies: np.ndarray) -> Record:
    """獨立重建每面構造、走既有非可微入口算阻抗；剛性面填零，父行程用遮罩跳過。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    rows = []
    offset = 0
    for design in designs:
        if design.material is None:
            rows.append(np.zeros(frequencies.shape, dtype=np.complex128))
            continue
        layers = list(design.material.layers)
        for index, name in enumerate(design.variables):
            layer_index = int(name.split(".")[0].removeprefix("layer")) - 1
            layers[layer_index] = layers[layer_index].model_copy(update={"thickness_m": float(theta[offset + index])})
        material = Assembly(layers=tuple(layers), backing=design.material.backing)
        rows.append(np.asarray(surface_impedance(material, jnp.asarray(frequencies), air=air, ranges=ranges)[0]))
        offset += len(design.variables)
    return _complex(np.asarray(rows))


def main(directory: Path) -> None:
    problem = cast(dict[str, object], json.loads((directory / "problem.json").read_text(encoding="utf-8")))
    model = load_reduced(directory / "model.npz")
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    presets = load_construction_presets(config_path("construction_presets.toml"))
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    receivers = ReceiverSet.model_validate(problem["receivers"])
    score = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), str(problem["purpose"]))
    frequencies = np.asarray(model.frequencies_hz, dtype=np.float64)
    constants = objective_constants(frequencies, score, receivers, len(model.source_names))
    output: Record = {}
    for name, case in cast(dict[str, Record], problem["cases"]).items():
        designs = gradient_search.wall_designs_from_presets(cast(list[str | None], case["presets"]), settings, presets)
        objective = gradient_search.chain_objective(model, designs, constants, air=air, ranges=ranges, frequencies=frequencies)
        theta = jnp.asarray([float.fromhex(value) for value in cast(list[str], case["theta"])])
        step = float(cast(float, case["step"]))
        (value, aux), automatic = jax.jit(jax.value_and_grad(objective, has_aux=True))(theta)
        eager_value, _ = objective(theta)
        points = {}
        for index in range(theta.shape[0]):
            for sign, label in ((1.0, "upper"), (-1.0, "lower")):
                moved = np.asarray(theta).copy()
                moved[index] += sign * step * moved[index]
                points[f"{index}.{label}"] = _impedances(designs, moved, frequencies)
        single = theta.astype(jnp.float32)
        (single_value, _), single_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(single)
        row: Record = {
            "names": list(gradient_search.design_names(designs)),
            "value": float(value).hex(), "eager_value": float(eager_value).hex(), "gradient": _hex(automatic),
            "components": [float(aux.mean_own_line_ripple_rms_db).hex(), float(aux.mean_seat_robustness_rms_db).hex()],
            "dtype": str(value.dtype), "gradient_dtype": str(automatic.dtype),
            "center": _impedances(designs, np.asarray(theta), frequencies), "points": points,
            "single": {"value": float(single_value).hex(), "gradient": _hex(single_gradient),
                       "dtype": str(single_value.dtype), "gradient_dtype": str(single_gradient.dtype)},
        }
        if case.get("descent"):
            start = gradient_search.baseline_theta(designs)
            bounds = gradient_search.design_bounds(designs)
            clock = time.perf_counter()
            first = gradient_search.projected_descent(objective, start, bounds, settings.descent)
            descent_seconds = time.perf_counter() - clock
            again = gradient_search.projected_descent(objective, start, bounds, settings.descent)
            # 每一步 θ 的梯度另外重算交回，父行程用它逐步核 Armijo 不等式。
            step_gradient = jax.jit(jax.grad(lambda values: objective(values)[0]))
            descent_gradients = [_hex(step_gradient(jnp.asarray(step.theta))) for step in first.steps]
            snapped = gradient_search.snap_to_specs(first.theta, designs)
            snapped_value, snapped_aux = objective(jnp.asarray(snapped.theta))
            retuned = gradient_search.retune_cavities(objective, snapped.theta, designs, settings)
            single_descent = gradient_search.projected_descent(objective, start.astype(np.float32), bounds, settings.descent)
            rows = (*ledger.descent_rows(first, "descent", 0),
                    ledger.snap_row(snapped, float(snapped_value), (float(snapped_aux.mean_own_line_ripple_rms_db),
                                    float(snapped_aux.mean_seat_robustness_rms_db)), 0.0, len(first.steps)),
                    *ledger.descent_rows(retuned, "retune", len(first.steps) + 1))
            ledger.write_ledger(directory / f"{name}.ledger.jsonl", ledger.header_for(model, designs, settings), rows)
            row.update({
                "start": _hex(start), "lower": _hex(bounds[0]), "upper": _hex(bounds[1]),
                "descent": _steps(first), "descent_again": _steps(again), "descent_seconds": descent_seconds,
                "descent_gradients": descent_gradients,
                "snap": {"theta": _hex(snapped.theta), "surfaces": [choice._asdict() for choice in snapped.surfaces],
                         "value": float(snapped_value).hex()},
                "retune": _steps(retuned), "single_descent": _steps(single_descent),
            })
        output[name] = row
    (directory / "output.json").write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
