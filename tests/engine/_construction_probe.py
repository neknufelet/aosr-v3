"""構造考卷的隔離算數子行程；精度由父行程透過執行環境層指定。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from pydantic import TypeAdapter

from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants, load_physics_constants
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.materials import construction as construction
from aosr.materials import transfer_matrix as tmm
from aosr.materials.construction_models import SurfaceMaterial


def _encode(value: jax.Array) -> dict[str, object]:
    assert isinstance(value, jax.Array), "計算結果必須是JAX陣列"
    array = np.asarray(value)
    return {"real": array.real.tolist(), "imag": array.imag.tolist(), "dtype": str(value.dtype),
            "shape": list(array.shape)}


def _direct(request: dict[str, object], f: jax.Array, air: PhysicsConstants, theta: jax.Array | None) -> jax.Array:
    """獨立轉接原始欄位；不經構造層的內部矩陣函式，抓單位或參數轉接錯誤。"""
    material = cast(dict[str, object], request["material"])
    matrices = []
    for layer in cast(list[dict[str, object]], material["layers"]):
        if layer["kind"] == "air":
            m = tmm.air_layer(f, jnp.asarray(layer["thickness_m"]), air=air, incidence_angles_deg=theta)
        elif layer["kind"] == "board":
            z = cast(float, layer["resistance_pa_s_m"]) + 1j * 2 * jnp.pi * f * cast(float, layer["surface_mass_kg_m2"])
            m = tmm._sheet_layer(tmm._broadcast_angles(z, theta))
        else:
            m = tmm.porous_layer(f, jnp.asarray(layer["thickness_m"]), jnp.asarray(layer["flow_resistivity_pa_s_m2"]),
                air=air, model=cast(str, layer.get("model", "komatsu")),
                porosity=cast(float, layer.get("porosity", 0.95)), tortuosity=cast(float, layer.get("tortuosity", 1.)),
                incidence_angles_deg=theta)
        matrices.append(m)
    z = tmm.compute(matrices, air=air, backing=cast(str, material["backing"]), incidence_angles_deg=theta)[0]
    return tmm.diffuse_absorption_coefficient(z, theta, air=air) if request["mode"] == "diffuse" and theta is not None else z


def _evaluate(request: dict[str, object]) -> dict[str, object]:
    air = load_physics_constants(config_path("physics_constants.toml"))
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    material: SurfaceMaterial = TypeAdapter(SurfaceMaterial).validate_python(request["material"])
    f = jnp.asarray(request["frequencies_hz"])
    mode = request["mode"]
    theta = jnp.asarray(request["angles_deg"]) if request.get("angles_deg") is not None else None
    if mode == "diffuse":
        value, notes = construction.random_incidence_absorption(material, f, air=air, ranges=ranges)
        theta = tmm.diffuse_incidence_angles_deg(method="paris")
    elif theta is None:
        value, notes = construction.surface_impedance(material, f, air=air, ranges=ranges)
    else:
        value, notes = construction.surface_impedance_at_angles(material, f, theta, air=air, ranges=ranges)
    result: dict[str, object] = {"value": _encode(value), "notes": [n.model_dump() for n in notes]}
    if request.get("direct"):
        result["direct"] = _encode(_direct(request, f, air, theta))
    if mode == "normal":
        result["absorption"] = _encode(tmm.reflection_and_absorption_coefficient(value, air=air)[1])
    return result


def main() -> None:
    requests = cast(dict[str, dict[str, object]], json.loads(Path(sys.argv[1]).read_text()))
    rows: dict[str, object] = {}
    for name, request in requests.items():
        try:
            rows[name] = _evaluate(request)
        except ValueError as exc:
            rows[name] = {"error": str(exc)}
    Path(sys.argv[2]).write_text(json.dumps(rows, allow_nan=False))


if __name__ == "__main__":
    main()
