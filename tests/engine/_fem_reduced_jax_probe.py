"""替身求值雙精度子行程；只叫公開入口、存原始數值，參考與判決在父行程。"""
from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from aosr.config.construction_presets import load_construction_presets
from aosr.config.paths import config_path
from aosr.config.physics_constants import load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.materials.construction import surface_impedance
from aosr.materials.construction_diff import ConstructionTemplate, build_template
from aosr.materials.construction_models import SurfaceMaterial
from aosr.physics import fem_reduced_jax as reduced
from aosr.physics.fem_reduced_store import load_reduced


Record = dict[str, object]
Parameters = tuple[Mapping[str, jax.Array], ...]
# 差分步長由父行程傳入；探針只交出擾動點的阻抗，不做差分或判決。


def _bits(value: jax.Array | np.ndarray) -> Record:
    array = np.asarray(value)
    return {"real": [float(x).hex() for x in array.real.ravel()],
            "imag": [float(x).hex() for x in array.imag.ravel()],
            "dtype": str(array.dtype), "shape": list(array.shape)}


def _case(constants: reduced.ReducedConstants, z: jax.Array, mask: jax.Array) -> Record:
    compiled = jax.jit(reduced.reduced_pressures)
    def energy(real: jax.Array, imag: jax.Array) -> jax.Array:
        return jnp.sum(jnp.abs(reduced.reduced_pressures(constants, jax.lax.complex(real, imag), rigid_mask=mask)) ** 2)
    def gradient(values: jax.Array) -> jax.Array:
        real, imag = jax.grad(energy, argnums=(0, 1))(values.real, values.imag)
        return jax.lax.complex(real, imag)
    batch = jnp.stack((z, z * (1.2 - 0.1j)))
    batched = jax.vmap(lambda values: reduced.reduced_pressures(constants, values, rigid_mask=mask))(batch)
    compiled_batch = jax.jit(jax.vmap(lambda values: reduced.reduced_pressures(constants, values, rigid_mask=mask)))
    single = z.astype(jnp.complex64)
    return {
        "eager": _bits(reduced.reduced_pressures(constants, z, rigid_mask=mask)),
        "jit": _bits(compiled(constants, z, rigid_mask=mask)),
        "jit_again": _bits(compiled(constants, z, rigid_mask=mask)),
        "vmap": _bits(batched[0]), "vmap_alternative": _bits(batched[1]),
        "vmap_jit": _bits(compiled_batch(batch)[0]), "vmap_jit_again": _bits(compiled_batch(batch)[0]),
        "coefficients": _bits(reduced.wall_coefficients(constants, z, rigid_mask=mask)),
        "gradient": _bits(gradient(z)),
        "single": _bits(compiled(constants, single, rigid_mask=mask)),
        "single_coefficients": _bits(reduced.wall_coefficients(constants, single, rigid_mask=mask)),
        "single_gradient": _bits(jax.jit(gradient)(single)),
    }


def _rejections(templates: tuple[ConstructionTemplate | None, ...], parameters: Parameters,
                frequencies: jax.Array) -> dict[str, str]:
    air = load_physics_constants(config_path("physics_constants.toml"))
    def run(t: tuple[ConstructionTemplate | None, ...], p: Parameters, f: jax.Array) -> jax.Array:
        return reduced.wall_impedances_from_templates(t, p, f, air=air)
    bad_keys = ({"wrong": jnp.asarray(0.05)}, *parameters[1:])
    bad_rigid = (*parameters[:1], {"wrong": jnp.asarray(0.05)}, *parameters[2:])
    calls: dict[str, Callable[[], object]] = {
        "walls": lambda: run(templates[:5], parameters, frequencies),
        "parameters": lambda: run(templates, parameters[:5], frequencies),
        "frequencies": lambda: run(templates, parameters, frequencies[:, None]),
        "keys": lambda: run(templates, bad_keys, frequencies),
        "rigid_parameters": lambda: run(templates, bad_rigid, frequencies),
    }
    errors = {}
    for name, call in calls.items():
        try:
            call()
        except ValueError as error:
            errors[name] = str(error)
        else:
            errors[name] = ""
    return errors


def _template_gradients(constants: reduced.ReducedConstants, templates: tuple[ConstructionTemplate | None, ...],
                        parameters: Parameters, step: float) -> list[Record]:
    air = load_physics_constants(config_path("physics_constants.toml"))
    frequencies = jnp.asarray(constants.frequencies_hz)
    mask = reduced.rigid_mask_from_templates(templates)
    def energy(values: Parameters) -> jax.Array:
        z = reduced.wall_impedances_from_templates(templates, values, frequencies, air=air)
        return jnp.sum(jnp.abs(reduced.reduced_pressures(constants, z, rigid_mask=mask)) ** 2)
    gradient = jax.jit(jax.grad(energy))(parameters)
    rows = []
    for wall, name in ((0, "layer2.thickness_m"), (4, "layer1.thickness_m")):
        template = templates[wall]
        assert template is not None
        variable = next(v for v in template.variables if v.name == name)
        impedances = []
        for sign in (1.0, -1.0):
            z = np.asarray(reduced.wall_impedances_from_templates(templates, parameters, frequencies, air=air)).copy()
            layers = list(template.material.layers)
            layer = layers[variable.layer_index - 1]
            layers[variable.layer_index - 1] = layer.model_copy(update={variable.field: variable.baseline * (1.0 + sign * step)})
            material = template.material.model_copy(update={"layers": tuple(layers)})
            # 獨立重建構造、走既有非可微入口；父行程用這兩組 Z 算 numpy 聲壓的差分。
            z[wall] = np.asarray(surface_impedance(material, frequencies, air=air,
                ranges=load_porous_model_ranges(config_path("porous_model_ranges.toml")))[0])
            impedances.append(_bits(z))
        rows.append({"wall": wall, "name": name, "baseline": variable.baseline,
                     "automatic": float(gradient[wall][name]), "upper": impedances[0], "lower": impedances[1]})
    return rows


def _templates(constants: reduced.ReducedConstants, step: float) -> Record:
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    adapter: TypeAdapter[SurfaceMaterial] = TypeAdapter(SurfaceMaterial)
    presets = {p.id: p.material for p in load_construction_presets(config_path("construction_presets.toml"))}
    frequencies = jnp.asarray(constants.frequencies_hz)
    glass = build_template(adapter.validate_python(presets["glass_wool_48k_50mm_gap50"]), frequencies, air=air, ranges=ranges)
    panel = build_template(adapter.validate_python(presets["perforated_0p8mm_d10_gw24k_50mm"]), frequencies, air=air, ranges=ranges)
    templates = (glass, None, None, None, panel, None)
    parameters = tuple({} if t is None else t.baseline_parameters() for t in templates)
    mask = reduced.rigid_mask_from_templates(templates)
    def pressures(values: Parameters, axis: jax.Array) -> jax.Array:
        z = reduced.wall_impedances_from_templates(templates, values, axis, air=air)
        return reduced.reduced_pressures(constants, z, rigid_mask=mask)
    compiled = jax.jit(pressures)
    single = jax.tree.map(lambda value: value.astype(jnp.float32), parameters)
    single_frequencies = frequencies.astype(jnp.float32)
    single_gradient = jax.jit(jax.grad(lambda p: jnp.sum(jnp.abs(pressures(p, single_frequencies)) ** 2)))(single)
    return {
        "mask": np.asarray(mask).tolist(),
        "impedances": _bits(reduced.wall_impedances_from_templates(templates, parameters, frequencies, air=air)),
        "pressure": _bits(pressures(parameters, frequencies)),
        "jit": _bits(compiled(parameters, frequencies)), "jit_again": _bits(compiled(parameters, frequencies)),
        "single": _bits(compiled(single, single_frequencies)),
        "single_gradient_dtype": str(jax.tree.leaves(single_gradient)[0].dtype),
        "single_gradient_finite": all(bool(jnp.isfinite(v)) for v in jax.tree.leaves(single_gradient)),
        "gradients": _template_gradients(constants, templates, parameters, step),
        "rejections": _rejections(templates, parameters, frequencies),
    }


def main(directory: Path, step: float) -> None:
    from aosr.geometry.shoebox import Wall
    from tests.engine.test_fem_reduced import HOLDOUT, TRAINING

    constants = reduced.reduced_constants(load_reduced(directory / "model.npz"))
    problem = cast(dict[str, Record], json.loads((directory / "problem.json").read_text(encoding="utf-8")))
    output = {}
    for name, row in problem.items():
        z = jnp.asarray(np.asarray(row["real"]) + 1j * np.asarray(row["imag"]))
        output[name] = _case(constants, z, jnp.asarray(row["mask"]))
    output["templates"] = _templates(constants, step)
    compiled = jax.jit(reduced.reduced_pressures)
    output["materials"] = {"pressures": [_bits(compiled(constants, jnp.asarray([walls[w] for w in Wall.all()])))
        for walls in [*TRAINING, *HOLDOUT]]}
    (directory / "output.json").write_text(json.dumps(output), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]), float(sys.argv[2]))
