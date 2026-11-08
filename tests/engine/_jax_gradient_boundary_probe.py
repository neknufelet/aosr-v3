"""冷行程量兩個梯度接縫；中央差分沿用材料梯度考卷的步長。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aosr.runtime import configure_jax

configure_jax(platforms="cpu", enable_x64=True)

import jax
import jax.numpy as jnp
import numpy as np

from aosr.geometry.shoebox import Wall
from aosr.physics import geometric_early_jax as early, geometric_lane
from aosr.physics.late_energy_jax import zoned_late_energy_arrays, zoned_late_energy_geometry
from tests.engine import _geometric_early_jax_cases as case
from tests.engine.test_late_energy_zones import ROOM, synthetic
from tests.engine.test_material_gradient import DIFFERENCE_STEP


def late_gradients(dtype: str, quantity: str) -> dict[str, object]:
    """幾何必須是編譯函式的一般引數，不能先關進閉包掩蓋追蹤錯誤。"""
    problem = synthetic(Wall.FLOOR, 2, 1. / 3., .5)
    assert problem.absorption_by_zone is not None
    geometry = zoned_late_energy_geometry(ROOM, 2, problem.absorption_by_zone)
    alpha = jnp.full((10, 2), .25, dtype=dtype)
    objective = lambda g, a: jnp.sum(getattr(zoned_late_energy_arrays(g, a), quantity))
    inside = jax.jit(jax.grad(objective, argnums=1))(geometry, alpha)
    outside = jax.grad(jax.jit(objective), argnums=1)(geometry, alpha)
    return {"inside": np.asarray(inside).tolist(), "outside": np.asarray(outside).tolist(),
            "bitwise": np.asarray(inside).tobytes() == np.asarray(outside).tobytes(),
            "dtype": str(outside.dtype)}


def early_table() -> early.GeometricEarlyTable:
    return early.build_geometric_early_table(room=case.ROOM, sources=case.SOURCES[:1],
        receivers=case.RECEIVERS[:1], source_models=case.models()[:1], sound_speed_m_s=case.SPEED,
        rho_c_pa_s_per_m=case.RHO*case.SPEED, frequencies_hz=(125., 503., 1009.))


def early_inputs(table: early.GeometricEarlyTable, impedance: jax.Array) -> tuple[tuple[jax.Array, ...], jax.Array, jax.Array]:
    """地板同一個參數同時供法向與所有斜入射，其餘五面阻抗固定。"""
    normal = jnp.full((6, 3), 900.+0j, dtype=jnp.result_type(impedance, 1j)).at[0].set(impedance)
    rows = tuple(jnp.broadcast_to(normal[i, :, None], (3, len(a)))
                 for i, a in enumerate(table.incidence_angles_deg))
    scattering = jnp.ones(6, dtype=impedance.dtype).at[0].set(0.)
    return rows, normal, scattering


def early_reference(impedance: float) -> float:
    result = geometric_lane.solve_geometric_early_lane(room=case.ROOM, source=case.SOURCES[0],
        receiver=case.RECEIVERS[0], source_model=case.models()[0], sound_speed_m_s=case.SPEED,
        rho_c_pa_s_per_m=case.RHO*case.SPEED, frequencies_hz=(125., 503., 1009.),
        impedance_by_wall={w.wall_name(): impedance if w == Wall.FLOOR else 900. for w in Wall.all()},
        scattering_by_wall={w.wall_name(): 0. if w == Wall.FLOOR else 1. for w in Wall.all()})
    return float(np.sum(result.reflected_energy))


def early_gradients(dtype: str, compiled: bool) -> dict[str, object]:
    table = early_table()
    def solve(z: jax.Array) -> early.GeometricEarlyJaxResult:
        return early.solve_geometric_early(table, *early_inputs(table, z))
    objective = lambda z: jnp.sum(solve(z).reflected_energy)
    differentiate = jax.grad(objective)
    if compiled:
        differentiate = jax.jit(differentiate)
        solve = jax.jit(solve)
    records = []
    for offset in (0., .01, .001):
        z = jnp.asarray(table.rho_c*(1.+offset), dtype=dtype)
        result = solve(z)
        value = float(z)
        central = []
        for relative_step in (DIFFERENCE_STEP, DIFFERENCE_STEP / 2.):
            step = relative_step*value
            central.append((early_reference(value+step)-early_reference(value-step))/(2.*step))
        records.append({"impedance": value, "scattering": np.asarray(result.scattering).tolist(),
            "valid": all(np.asarray(v).all() for v in result[4:]),
            "energy": float(jnp.sum(result.reflected_energy)),
            "gradient": float(differentiate(z)), "central": central})
    return {"points": records}


if __name__ == "__main__":
    result = (late_gradients(sys.argv[3], sys.argv[4]) if sys.argv[2] == "late" else
              early_gradients(sys.argv[3], sys.argv[4] == "jit"))
    Path(sys.argv[1]).write_text(json.dumps(result))
