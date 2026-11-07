"""冷子行程只求 JAX 值與梯度，數字以 hex 交回父行程裁判。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from aosr.physics.amplitude_jax import bounce_impedances
from aosr.scoring.reflections_jax import reflection_geometry, reflections_arrays, primary_category_cost
from tests.engine._reflections_jax_cases import CASES, Case, case, directional_inputs, purpose


def _hex(value: jax.Array) -> list[str]:
    return [float(v).hex() for v in np.asarray(value).reshape(-1)]


def run(data: Case, coordinates: jax.Array, scatter: jax.Array) -> dict[str, jax.Array]:
    """實／虛部以原始 |Z| 為尺度；每格的中央差分也用這個無量綱座標。"""
    real_dtype = coordinates.dtype
    z = (coordinates[0] + 1j * coordinates[1]) * jnp.asarray(np.abs(data.z), dtype=real_dtype)
    outputs = []
    for i, table in enumerate(data.tables):
        geometry = reflection_geometry(table, data.directions[i], data.axis, purpose())
        outputs.append(reflections_arrays(geometry, bounce_impedances(table, z), scatter, data.rho_c,
                                          jnp.asarray(True)))
    return {"path_energy": jnp.stack([v.path_energy for v in outputs]),
            "zone_excess_db": jnp.stack([v.zone_excess_db for v in outputs]),
            "zone_cost": jnp.stack([v.zone_cost for v in outputs]),
            "category_cost": primary_category_cost(jnp.stack([v.channel_cost for v in outputs])),
            "valid": jnp.stack([v.zone_valid for v in outputs])}


def _bounce_difference(data: Case, step: float) -> dict[str, list[str]]:
    """補考每一格獨立的 z_bounce；與編譯後同一函式的中央差分比，不共用牆面收集的梯度。"""
    table = data.tables[0]
    geometry = reflection_geometry(table, data.directions[0], data.axis, purpose())
    z = bounce_impedances(table, jnp.asarray(data.z, dtype=jnp.complex128))
    scale = jnp.abs(z)
    coordinates = jnp.stack([z.real, z.imag]) / scale
    scatter = jnp.asarray(data.scattering, dtype=jnp.float64)

    def total(values: jax.Array) -> jax.Array:
        impedances = (values[0] + 1j * values[1]) * scale
        return reflections_arrays(geometry, impedances, scatter, data.rho_c, jnp.asarray(True)).channel_cost

    compiled = jax.jit(total)
    automatic = jax.jit(jax.grad(total))(coordinates)
    array = np.asarray(coordinates)
    central = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        plus, minus = array.copy(), array.copy()
        plus[index] += step
        minus[index] -= step
        central[index] = (float(compiled(jnp.asarray(plus))) - float(compiled(jnp.asarray(minus)))) / (2 * step)
    return {"bounce_gradient": _hex(automatic), "bounce_central": _hex(jnp.asarray(central))}


def _case(name: str, step: float) -> dict[str, object]:
    data = case(name)
    coordinates = jnp.asarray(np.stack([data.z.real, data.z.imag]) / np.abs(data.z), dtype=jnp.float64)
    scatter = jnp.asarray(data.scattering, dtype=jnp.float64)
    compiled = jax.jit(lambda x, s: run(data, x, s))
    eager, jit = run(data, coordinates, scatter), compiled(coordinates, scatter)
    again = compiled(coordinates, scatter)
    single = compiled(coordinates.astype(jnp.float32), scatter.astype(jnp.float32))
    grad = jax.jit(jax.grad(lambda x, s: run(data, x, s)["category_cost"], argnums=(0, 1)))
    gradient, scattering_gradient = grad(coordinates, scatter)
    single_gradient = grad(coordinates.astype(jnp.float32), scatter.astype(jnp.float32))
    result: dict[str, object] = {"eager": {k: _hex(v) for k, v in eager.items()}, "jit": {k: _hex(v) for k, v in jit.items()},
            "again": {k: _hex(v) for k, v in again.items()}, "single": {k: _hex(v) for k, v in single.items()},
            "gradient": _hex(gradient), "scattering_gradient": _hex(scattering_gradient),
            "double_dtypes": [str(v.dtype) for k, v in jit.items() if k != "valid"],
            "single_dtypes": [str(v.dtype) for k, v in single.items() if k != "valid"],
            "gradient_dtypes": [str(gradient.dtype), str(scattering_gradient.dtype)],
            "single_gradient_dtypes": [str(v.dtype) for v in single_gradient],
            "single_gradient": [_hex(v) for v in single_gradient]}
    if name == "complex_frequency":
        result.update(_bounce_difference(data, step))
    if name == "room":
        result["zero_scattering_64"] = {k: _hex(v) for k, v in compiled(coordinates, jnp.zeros_like(scatter)).items()}
        for bits, dtype in ((64, jnp.float64), (32, jnp.float32)):
            outputs = []
            for i, table in enumerate(data.tables):
                factors, _ = directional_inputs(data, i)
                geometry = reflection_geometry(table, data.directions[i], data.axis, purpose())
                outputs.append(jax.jit(lambda z, s, f: reflections_arrays(
                    geometry, bounce_impedances(table, z), s, data.rho_c, jnp.asarray(True), f))(
                        jnp.asarray(data.z, dtype=jnp.complex128 if bits == 64 else jnp.complex64),
                        scatter.astype(dtype), jnp.asarray(factors, dtype=dtype)))
            result[f"directional_{bits}"] = {
                "path_energy": _hex(jnp.stack([v.path_energy for v in outputs])),
                "zone_excess_db": _hex(jnp.stack([v.zone_excess_db for v in outputs])),
                "zone_cost": _hex(jnp.stack([v.zone_cost for v in outputs])),
                "category_cost": _hex(primary_category_cost(jnp.stack([v.channel_cost for v in outputs])))}
    return result


def main() -> None:
    result = {"x64": bool(getattr(jax.config, "jax_enable_x64")), "backend": jax.default_backend(),
              "cases": {name: _case(name, float(sys.argv[2])) for name in CASES}}
    Path(sys.argv[1]).write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
