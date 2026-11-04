"""分區替身雙精度求值探針；由父行程提供隔離後的 JAX 環境。"""
from __future__ import annotations

import sys
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_diff import build_template
from aosr.materials.construction_models import AirLayer, Assembly, PorousLayer
from aosr.physics.fem_reduced_jax import reduced_constants, reduced_pressures, wall_impedances_from_templates, rigid_mask_from_templates
from aosr.physics.fem_reduced_store import load_reduced


def _templates(directory: Path) -> None:
    model = load_reduced(directory / "model.npz")
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    material = Assembly(layers=(PorousLayer(material_class="fibrous", thickness_m=0.05, flow_resistivity_pa_s_m2=12000),
                                AirLayer(thickness_m=0.1)), backing="rigid")
    axis = jnp.asarray(model.frequencies_hz)
    template = build_template(material, axis, air=air, ranges=ranges)
    templates = tuple(None if index == 0 else template for index in range(len(model.patch_keys)))
    parameters = tuple({} if item is None else item.baseline_parameters() for item in templates)
    bundle = wall_impedances_from_templates(templates, parameters, axis, air=air, patch_keys=model.patch_keys)
    np.save(directory / "templates.npy", np.asarray(bundle.values), allow_pickle=False)
    reference = surface_impedance(material, axis, air=air, ranges=ranges)[0]
    np.save(directory / "template_reference.npy", np.asarray(reference), allow_pickle=False)
    masks = rigid_mask_from_templates(templates, patch_keys=model.patch_keys)
    errors = []
    for bad_templates, bad_parameters in ((templates[:6], parameters), (templates, parameters[:6]),
                                         ((*templates[:2], None, *templates[3:]), parameters)):
        try:
            wall_impedances_from_templates(bad_templates, bad_parameters, axis, air=air, patch_keys=model.patch_keys)
        except ValueError as error:
            errors.append(str(error))
    (directory / "templates.json").write_text(json.dumps({"errors": errors, "mask": np.asarray(bundle.rigid_mask).tolist(),
        "same_mask": np.array_equal(np.asarray(masks), np.asarray(bundle.rigid_mask))}), encoding="utf-8")


def main(directory: Path) -> None:
    model = load_reduced(directory / "model.npz")
    if len(model.patch_keys) == 7:
        _templates(directory)
    constants = reduced_constants(model)
    values = jnp.asarray(np.load(directory / "z.npy", allow_pickle=False))
    rigid = jnp.zeros(constants.boundary.shape[0], dtype=bool)
    pressure = reduced_pressures(constants, values, rigid_mask=rigid)
    np.save(directory / "pressure.npy", np.asarray(pressure), allow_pickle=False)
    np.save(directory / "jit.npy", np.asarray(jax.jit(reduced_pressures)(constants, values, rigid_mask=rigid)), allow_pickle=False)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
