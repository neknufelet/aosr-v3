"""牆面材料梯度鏈雙精度子行程；只叫公開入口、存原始數值，差分、比對與判決在父行程。"""
from __future__ import annotations

import gc
import json
import sys
import time
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.quality_targets import load_quality_targets
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_models import Assembly
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_store import load_reduced
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.gradient_objective_jax import (
    GradientObjectiveConstants, gradient_objective_constants, objective_reading,
)
from aosr.scoring.gradient_objective_settings import gradient_objective_settings
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
        "evaluations": result.evaluations,
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


def _box_only(objective: gradient_search.ChainObjective, start: np.ndarray,
              bounds: tuple[np.ndarray, np.ndarray], settings: MaterialGradientSettings,
              step_gradient: Callable[[jax.Array], jax.Array]) -> Record:
    """只守盒界（limits=()）的獨立一段：空腔會貼 0.150 上界被往外推，守的是盒界遮罩本身（#561 審查第 9 條）。"""
    box_only = gradient_search.projected_descent(objective, start, bounds, settings.descent, limits=())
    return {"box_only": _steps(box_only),
            "box_only_gradients": [_hex(step_gradient(jnp.asarray(step.theta))) for step in box_only.steps]}


def _evaluator_cache(model: ReducedModel, designs: tuple[gradient_search.SurfaceDesign, ...],
                     constants: GradientObjectiveConstants, theta: jax.Array) -> Record:
    """編譯快取（#561 審查）：真目標拿編譯版算一次，丟掉目標與編譯版、垃圾回收後量目標的弱參照死了沒；
    同一個目標連叫兩次是不是同一個物件、兩個不同目標是不是不同物件。目標各自新建，不跟後面的下降共用。"""
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))

    def build() -> gradient_search.ChainObjective:
        return gradient_search.chain_objective(model, designs, constants, air=air, ranges=ranges,
                                               frequencies=model.frequencies_hz)

    objective = build()
    evaluator = gradient_search.compiled_evaluator(objective)
    (value, _), gradient = evaluator(theta)
    alive = weakref.ref(objective)
    del objective, evaluator
    gc.collect()
    first, second = build(), build()
    shared = gradient_search.compiled_evaluator(first)
    return {"value": float(value).hex(), "gradient_finite": bool(np.isfinite(np.asarray(gradient)).all()),
            "released": alive() is None,
            "same_objective_same_evaluator": gradient_search.compiled_evaluator(first) is shared,
            "other_objective_other_evaluator": gradient_search.compiled_evaluator(second) is not shared}


def _difference_points(designs: tuple[gradient_search.SurfaceDesign, ...], theta: np.ndarray, step: float,
                       frequencies: np.ndarray) -> Record:
    """每個參數上下各推一個相對步長，交那一點的每面阻抗（父行程拿去做 numpy 差分）。"""
    points: Record = {}
    for index in range(theta.shape[0]):
        for sign, label in ((1.0, "upper"), (-1.0, "lower")):
            moved = theta.copy()
            moved[index] += sign * step * moved[index]
            points[f"{index}.{label}"] = _impedances(designs, moved, frequencies)
    return points


def _retune_with_room(objective: gradient_search.ChainObjective, snapped: np.ndarray,
                      designs: tuple[gradient_search.SurfaceDesign, ...], settings: MaterialGradientSettings) -> Record:
    """空腔留空間的起點：取過規格的 θ、每面空腔放在重調盒界的正中間再重調一次（取規格那一點可能貼在總深度上限，
    那裡投影梯度是零、空腔走不動；這一跑另外守「空腔真的會動」）。重調盒界的上界一起交回。"""
    lower, upper = gradient_search.retune_bounds(snapped, designs)
    start = np.where(lower < upper, (lower + upper) / 2.0, snapped)
    return {"retune_slack": _steps(gradient_search.retune_cavities(objective, start, designs, settings)),
            "retune_upper": _hex(upper)}


def main(directory: Path) -> None:
    problem = cast(dict[str, object], json.loads((directory / "problem.json").read_text(encoding="utf-8")))
    model = load_reduced(directory / "model.npz")
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    presets = load_construction_presets(config_path("construction_presets.toml"))
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    receivers = ReceiverSet.model_validate(problem["receivers"])
    objective_settings = gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")),
                                                     str(problem["purpose"]))
    frequencies = np.asarray(model.frequencies_hz, dtype=np.float64)
    constants = gradient_objective_constants(frequencies, objective_settings, receivers, len(model.source_names))
    output: Record = {}
    for name, case in cast(dict[str, Record], problem["cases"]).items():
        designs = gradient_search.wall_designs_from_presets(cast(list[str | None], case["presets"]), settings, presets)
        objective = gradient_search.chain_objective(model, designs, constants, air=air, ranges=ranges, frequencies=frequencies)
        theta = jnp.asarray([float.fromhex(value) for value in cast(list[str], case["theta"])])
        step = float(cast(float, case["step"]))
        (value, aux), automatic = jax.jit(jax.value_and_grad(objective, has_aux=True))(theta)
        eager_value, _ = objective(theta)
        points = _difference_points(designs, np.asarray(theta), step, frequencies)
        single = theta.astype(jnp.float32)
        (single_value, _), single_gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(single)
        row: Record = {
            "names": list(gradient_search.design_names(designs)),
            "value": float(value).hex(), "eager_value": float(eager_value).hex(), "gradient": _hex(automatic),
            "components": [value.hex() for value in objective_reading(aux).components],
            "screening_total": float(aux.screening_total).hex(),
            "dtype": str(value.dtype), "gradient_dtype": str(automatic.dtype),
            "center": _impedances(designs, np.asarray(theta), frequencies), "points": points,
            "single": {"value": float(single_value).hex(), "gradient": _hex(single_gradient),
                       "dtype": str(single_value.dtype), "gradient_dtype": str(single_gradient.dtype)},
        }
        if case.get("descent"):
            start = gradient_search.baseline_theta(designs)
            bounds = gradient_search.design_bounds(designs)
            clock, limits = time.perf_counter(), gradient_search.depth_limits(designs)
            first = gradient_search.projected_descent(objective, start, bounds, settings.descent, limits=limits)
            descent_seconds = time.perf_counter() - clock
            again = gradient_search.projected_descent(objective, start, bounds, settings.descent, limits=limits)
            # 每一步 θ 的梯度另外重算交回，父行程用它逐步核 Armijo 不等式。
            step_gradient = jax.jit(jax.grad(lambda values: objective(values)[0]))
            descent_gradients = [_hex(step_gradient(jnp.asarray(step.theta))) for step in first.steps]
            snapped = gradient_search.snap_to_specs(first.theta, designs)
            snapped_value, snapped_aux = objective(jnp.asarray(snapped.theta))
            retuned = gradient_search.retune_cavities(objective, snapped.theta, designs, settings)
            single_descent = gradient_search.projected_descent(objective, start.astype(np.float32), bounds, settings.descent,
                                                               limits=limits)
            rows = (*ledger.descent_rows(first, "descent", 0),
                    ledger.snap_row(snapped, float(snapped_value), objective_reading(snapped_aux).screening_total,
                                    objective_reading(snapped_aux).components, 0.0, len(first.steps)),
                    *ledger.descent_rows(retuned, "retune", len(first.steps) + 1))
            header = ledger.header_for(model, designs, settings, objective_constants=constants)
            ledger.write_ledger(directory / f"{name}.ledger.jsonl", header, rows)
            row.update({
                "start": _hex(start), "lower": _hex(bounds[0]), "upper": _hex(bounds[1]),
                "descent": _steps(first), "descent_again": _steps(again), "descent_seconds": descent_seconds,
                "descent_gradients": descent_gradients, "evaluator_cache": _evaluator_cache(model, designs, constants, theta),
                "snap": {"theta": _hex(snapped.theta), "surfaces": [choice._asdict() for choice in snapped.surfaces],
                         "value": float(snapped_value).hex()},
                "retune": _steps(retuned), "single_descent": _steps(single_descent),
                **_retune_with_room(objective, snapped.theta, designs, settings),
            })
            row.update(_box_only(objective, start, bounds, settings, step_gradient))
        output[name] = row
    (directory / "output.json").write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
