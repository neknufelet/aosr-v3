"""梯度搜尋終點雙精度子行程（#561 辛二）：只叫公開入口、存原始數值，比對與判決在父行程。

完整模型算子在這裡用小房間 fixture 同一組參數重建（網格種子固定、網格雜湊要等於替身的），座位列照
``test_material_gradient._chain_model`` 同一組七個線性組合換掉——父行程交進來的替身就是那一份衍生替身。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np

from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ, LowFrequencyAxis
from aosr.config.material_gradient import MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.porous_model_ranges import PorousModelRanges, load_porous_model_ranges
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_diff import build_template, surface_impedance_from_parameters
from aosr.physics.fem_reduced import ReducedModel, export_room_operators
from aosr.physics.fem_reduced_check import validate_reduced
from aosr.physics.fem_reduced_store import load_reduced
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.stage_one_objective_jax import StageOneObjectiveConstants, objective_constants, objective_from_pressures
from aosr.scoring.stage_one_score import stage_one_score_settings
from aosr.gradient import endpoint
from aosr.gradient import material_gradient as gradient_search
from tests.engine.test_fem_reduced import DENSITY, ROOM, SEATS, SOUND_SPEED, SOURCES
from tests.engine.test_material_gradient_endpoint import chain_operators, floor_case_pressures


Record = dict[str, object]


def _hex(values: object) -> list[str]:
    return [float(value).hex() for value in np.asarray(values).ravel()]


def _complex(values: object) -> Record:
    array = np.asarray(values)
    return {"real": _hex(array.real), "imag": _hex(array.imag), "shape": list(array.shape)}


def _theta(values: object) -> np.ndarray:
    return np.asarray([float.fromhex(value) for value in cast(list[str], values)], dtype=np.float64)


def _zero_cavity(case: Record, model: ReducedModel, constants: StageOneObjectiveConstants,
                 settings: MaterialGradientSettings, presets: tuple[ConstructionPreset, ...],
                 air: PhysicsConstants, ranges: PorousModelRanges) -> Record:
    """空腔 0：替身樣板（厚度 0 的空氣層）對「拿掉空氣層」的構造；阻抗層級與整條鏈層級各一份。"""
    axis = np.asarray(model.frequencies_hz, dtype=np.float64)
    with_gap = gradient_search.wall_designs_from_presets(cast(list[str | None], case["with_gap"]), settings, presets)
    without_gap = gradient_search.wall_designs_from_presets(cast(list[str | None], case["without_gap"]), settings,
                                                            presets)
    theta_gap, theta_plain = _theta(case["theta_with_gap"]), _theta(case["theta_without_gap"])
    design = with_gap[0]
    assert design.material is not None and design.thickness_variable and design.cavity_variable
    template = build_template(design.material, jnp.asarray(axis), air=air, ranges=ranges, variables=design.variables)
    parameters = {design.thickness_variable: jnp.asarray(theta_gap[0]), design.cavity_variable: jnp.asarray(0.0)}
    chain_impedance = surface_impedance_from_parameters(template, parameters, jnp.asarray(axis), air=air)
    materials = endpoint.materials_from_surfaces(with_gap, theta_gap, settings)
    exported = materials[Wall.all()[0]]
    assert exported is not None
    exported_impedance, _ = surface_impedance(exported, jnp.asarray(axis), air=air, ranges=ranges)
    gap_value, _ = gradient_search.chain_objective(model, with_gap, constants, air=air, ranges=ranges,
                                                   frequencies=axis)(jnp.asarray(theta_gap))
    plain_value, _ = gradient_search.chain_objective(model, without_gap, constants, air=air, ranges=ranges,
                                                     frequencies=axis)(jnp.asarray(theta_plain))
    return {"chain_impedance": _complex(chain_impedance), "exported_impedance": _complex(exported_impedance),
            "exported_layers": [layer.kind for layer in exported.layers],
            "chain_value_with_gap": float(gap_value).hex(), "chain_value_without_gap": float(plain_value).hex()}


def main(directory: Path) -> None:
    problem = cast(Record, json.loads((directory / "problem.json").read_text(encoding="utf-8")))
    model = load_reduced(directory / "model.npz")
    operators = chain_operators(export_room_operators(
        ROOM, SOURCES, SEATS, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED,
        physics_identity=model.physics_identity, max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ))
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    presets = load_construction_presets(config_path("construction_presets.toml"))
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    receivers = ReceiverSet.model_validate(problem["receivers"])
    score = stage_one_score_settings(load_quality_targets(config_path("quality_targets.toml")), str(problem["purpose"]))
    axis = tuple(model.frequencies_hz)
    constants = objective_constants(axis, score, receivers, len(model.source_names))
    case = cast(Record, problem["endpoint"])
    designs = gradient_search.wall_designs_from_presets(cast(list[str | None], case["presets"]), settings, presets)
    theta = _theta(case["theta"])
    result = endpoint.endpoint_check(
        operators, designs, theta, model=model, score_settings=score, receiver_set=receivers,
        low_frequency_axis=LowFrequencyAxis(str(problem["axis"])), settings=settings, air=air, ranges=ranges)
    materials = endpoint.materials_from_surfaces(designs, theta, settings)
    direct = endpoint.full_model_pressures(
        operators, materials, axis, air=air, ranges=ranges,
        max_relative_residual=model.ledger.settings.full_solve_max_relative_residual)
    reference = validate_reduced(model, operators, [endpoint.material_impedances(materials, axis, air=air,
                                                                                  ranges=ranges)])
    objective = gradient_search.chain_objective(model, designs, constants, air=air, ranges=ranges, frequencies=axis)
    output: Record = {
        "chain_at_search": float(objective(jnp.asarray(theta))[0]).hex(),
        "chain_at_expected_export": float(objective(jnp.asarray(_theta(case["expected_export"])))[0]).hex(),
        "mesh_hash_equal": operators.mesh_hash == model.mesh_hash,
        "report": json.loads(result.report.to_json()),
        "full": _complex(result.full_pressure), "reduced": _complex(result.reduced_pressure),
        "direct": _complex(direct), "reference": _complex(reference.full_pressure[0]),
        "direct_total": float(objective_from_pressures(constants, jnp.asarray(direct)).screening_total).hex(),
        "floor_case": [float(objective_from_pressures(constants, jnp.asarray(values)).screening_total).hex()
                       for values in floor_case_pressures(axis, receivers)],
        "zero_cavity": _zero_cavity(cast(Record, problem["zero_cavity"]), model, constants, settings, presets,
                                    air, ranges),
    }
    (directory / "output.json").write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
