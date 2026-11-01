"""安裝方式「貼在原牆上」（#990）接進梯度搜尋的雙精度子行程：只叫公開入口、存原始數值，比對與判決在父行程。

替身是父行程交進來的衍生替身（``test_material_gradient._chain_model``）。同一組六面 preset、同一組方案原牆：
- 鏈：``chain_objective`` 的值與梯度（貼在原牆上、同一組剛背各一份）；
- 鏈用的阻抗：``design_templates`` 那幾份樣板換進 θ 的垂直入射阻抗（``wall_impedances_from_templates``，跟鏈同一條軸）；
- 終點用的阻抗：``endpoint.materials_from_surfaces``（``_design_material``）那份材料走 ``material_impedances``；
  控制組：那幾列直接換進替身求值（遮罩全放開）交同一份目標的值；
- 差分點：考卷自己用 θ 換厚度重建 ``MountedOnWall``（不走產品的匯出），走既有非可微入口算阻抗；
- 三路報表：小方案的六面原牆（``scene_wall_materials``）當背後的牆，匯出（``scene_with_materials``＋``exported_scheme``）、
  存成 JSON 讀回、走報表求解入口（``checked_inputs`` → ``solver_inputs``；有限元素用假能量，收下每面阻抗）。
"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from aosr.config.capabilities import load_capabilities
from aosr.config.construction_environment import construction_environment
from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.material_gradient import (
    MOUNT_ON_WALL, MOUNT_RIGID, MaterialGradientSettings, load_material_gradient_settings,
)
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint
from aosr.gradient import material_gradient as gradient_search
from aosr.gradient.endpoint_run import exported_scheme
from aosr.gradient.run import scene_wall_materials
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_models import MountedOnWall, SurfaceMaterial
from aosr.physics import report_io, three_lane_report as report
from aosr.physics.fem_reduced_jax import reduced_constants, reduced_pressures, wall_impedances_from_templates
from aosr.physics.fem_reduced_store import load_reduced
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import checked_inputs
from aosr.scoring.gradient_objective_jax import gradient_objective_constants, gradient_objective_from_pressures
from aosr.scoring.gradient_objective_settings import gradient_objective_settings
from aosr.scoring.receiver_set import ReceiverSet
from tests.engine._source_model_control import fake_fem_energy


Record = dict[str, object]


def _hex(values: object) -> list[str]:
    return [float(value).hex() for value in np.asarray(values).ravel()]


def _complex(values: object) -> Record:
    array = np.asarray(values)
    return {"real": _hex(array.real), "imag": _hex(array.imag), "shape": list(array.shape)}


def _rebuilt(design: gradient_search.SurfaceDesign, theta: np.ndarray, offset: int) -> SurfaceMaterial:
    """考卷自己的構造：樣板的層換進 θ 的厚度、背後接原牆；不貼的面是原牆。不走產品的 ``_design_material``。"""
    if design.material is None:
        assert design.kept_material is not None
        return design.kept_material
    layers = list(design.material.layers)
    for index, name in enumerate(design.variables):
        layer = int(name.split(".")[0].removeprefix("layer")) - 1
        layers[layer] = layers[layer].model_copy(update={"thickness_m": float(theta[offset + index])})
    assert design.backing_material is not None
    return MountedOnWall(layers=tuple(layers), wall=design.backing_material)


def _rows(designs: tuple[gradient_search.SurfaceDesign, ...], theta: np.ndarray, axis: np.ndarray) -> Record:
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    rows, offset = [], 0
    for design in designs:
        rows.append(np.asarray(surface_impedance(_rebuilt(design, theta, offset), jnp.asarray(axis), air=air,
                                                 ranges=ranges)[0]))
        offset += len(design.variables)
    return _complex(np.asarray(rows))


def _three_lane(preset_ids: list[str | None], theta: np.ndarray, scheme_document: Record,
                settings: MaterialGradientSettings, presets: tuple[ConstructionPreset, ...]) -> Record:
    """匯出的方案讀回、報表求解入口；回每面讀回的材料、原牆、報表能力代號、x0 收到的阻抗是否等於非可微入口、兩路能量。"""
    scheme = Scheme.model_validate(scheme_document)
    capabilities = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    walls = scene_wall_materials(scheme, capabilities=capabilities, directivity=directivity)
    designs = gradient_search.wall_designs_from_presets(
        preset_ids, settings, presets, kept={wall.wall_name(): material for wall, material in walls.items()},
        mount=MOUNT_ON_WALL)
    materials = endpoint.materials_from_surfaces(designs, theta, settings)
    exported = exported_scheme(scheme, endpoint.scene_with_materials(scheme.scene, materials), "0" * 32)
    loaded = Scheme.model_validate(json.loads(exported.model_dump_json()))
    zones = loaded.scene.surface_zones_by_wall
    assert zones is not None
    _, documents = checked_inputs(loaded, capabilities=capabilities, directivity=directivity)
    solved = report_io.solver_inputs(next(iter(documents.values()))[1])
    scene_air, ranges = construction_environment(density_kg_m3=solved.density_kg_m3,
                                                 sound_speed_m_s=solved.sound_speed_m_s)
    fake = cast(Callable[..., tuple[float, ...]], fake_fem_energy)
    seen: list[dict[Wall, tuple[complex, ...] | None]] = []

    def fem(**kwargs: object) -> tuple[float, ...]:
        seen.append(cast(dict[Wall, tuple[complex, ...] | None], kwargs["wall_impedances"]))
        return fake(**kwargs)
    with patch.object(report, "_solve_fem_energy", fem):
        raw = report.solve_three_lane_report(**solved._asdict())
    x0 = solved.materials_by_wall[Wall.X0]
    expected = surface_impedance(x0, jnp.asarray(raw.fem_frequencies_hz), air=scene_air, ranges=ranges)[0]
    return {"zones": {name: zone[0].material.model_dump(mode="json") for name, zone in zones.items()},
            "walls": {wall.wall_name(): material.model_dump(mode="json") for wall, material in walls.items()},
            "exported": {wall.wall_name(): None if material is None else material.model_dump(mode="json")
                         for wall, material in materials.items()},
            "capability": raw.capability.materials,
            "x0_is_exported": seen[0][Wall.X0] == tuple(complex(z) for z in np.asarray(expected)),
            "geometric": [point.geometric_energy for point in raw.points],
            "late": [point.late_energy for point in raw.points]}


def main(directory: Path) -> None:
    problem = cast(Record, json.loads((directory / "problem.json").read_text(encoding="utf-8")))
    model = load_reduced(directory / "model.npz")
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    presets = load_construction_presets(config_path("construction_presets.toml"))
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    receivers = ReceiverSet.model_validate(problem["receivers"])
    objective_settings = gradient_objective_settings(load_quality_targets(config_path("quality_targets.toml")),
                                                     str(problem["purpose"]))
    axis = np.asarray(model.frequencies_hz, dtype=np.float64)
    constants = gradient_objective_constants(axis, objective_settings, receivers, len(model.source_names))
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    kept = {name: adapter.validate_python(material) for name, material in cast(Record, problem["kept"]).items()}
    preset_ids = cast(list[str | None], problem["presets"])
    theta = np.asarray([float.fromhex(value) for value in cast(list[str], problem["theta"])], dtype=np.float64)
    step = float(cast(float, problem["step"]))
    reduced = reduced_constants(model)
    output: Record = {}
    for mount in (MOUNT_ON_WALL, MOUNT_RIGID):
        designs = gradient_search.wall_designs_from_presets(preset_ids, settings, presets, kept=kept, mount=mount)
        objective = gradient_search.chain_objective(model, designs, constants, air=air, ranges=ranges,
                                                    frequencies=axis)
        (value, aux), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(jnp.asarray(theta))
        row: Record = {"names": list(gradient_search.design_names(designs)), "value": float(value).hex(),
                       "screening_total": float(aux.screening_total).hex(), "gradient": _hex(gradient),
                       "dtype": str(value.dtype)}
        if mount == MOUNT_ON_WALL:
            templates = gradient_search.design_templates(designs, jnp.asarray(axis), air=air, ranges=ranges)
            chain = wall_impedances_from_templates(
                templates, gradient_search.surface_parameters(designs, jnp.asarray(theta)),
                jnp.asarray(reduced.frequencies_hz), air=air)
            materials = endpoint.materials_from_surfaces(designs, theta, settings)
            rows = endpoint.material_impedances(materials, tuple(model.frequencies_hz), air=air, ranges=ranges)
            endpoint_rows = np.asarray([rows[wall] for wall in Wall.all()], dtype=np.complex128)
            kept_rows = gradient_search.kept_impedance_rows(designs, axis, air=air, ranges=ranges)
            control = gradient_objective_from_pressures(constants, reduced_pressures(
                reduced, jnp.asarray(endpoint_rows), rigid_mask=jnp.zeros(len(Wall.all()), dtype=bool)))
            points: Record = {}
            for index, center in enumerate(theta):
                for label, sign in (("upper", 1.0), ("lower", -1.0)):
                    moved = theta.copy()
                    moved[index] = center * (1.0 + sign * step)
                    points[f"{index}.{label}"] = _rows(designs, moved, axis)
            row.update({
                "chain_rows": _complex(chain.values), "chain_rigid_mask": [bool(item) for item in chain.rigid_mask],
                "endpoint_rows": _complex(endpoint_rows),
                "kept_rows": {design.surface_id: _complex(kept_row) for design, kept_row in zip(designs, kept_rows,
                                                                                               strict=True)
                              if kept_row is not None},
                "control_value": float(control.objective_total).hex(),
                "center": _rows(designs, theta, axis), "points": points,
                "three_lane": _three_lane(preset_ids, theta, cast(Record, problem["scheme"]), settings, presets)})
        output[mount] = row
    (directory / "output.json").write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
