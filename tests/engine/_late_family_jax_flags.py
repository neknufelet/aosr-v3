"""每種數值失效都有一個有效鄰項；編譯後旗標、NaN 與反傳隔離。"""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.materials import transfer_matrix as tmm
from aosr.physics import crossover_jax as cj, late_family_jax as lf
from aosr.physics.late_energy_jax import zoned_late_energy_geometry
from tests.engine._late_family_cases import CONSTANTS, FREQUENCIES, alpha_rows, problem, scattering


def _record(fn: Callable[[jax.Array], cj.CheckedBands], data: jax.Array) -> dict[str, object]:
    result = jax.jit(fn)(data)
    def loss(values: jax.Array) -> jax.Array:
        output = fn(values).values
        return jnp.sum(jnp.where(jnp.isfinite(output), output, 0.0))
    gradient = jax.jit(jax.grad(loss))(data)
    return {"flag": bool(result.invalid[0]), "nan": bool(jnp.isnan(result.values[0])),
            "gradient_finite": bool(jnp.all(jnp.isfinite(gradient))), "valid_finite": bool(jnp.isfinite(result.values[-1]))}


def _family(which: str) -> dict[str, object]:
    inputs = problem("zoned")
    geometry = zoned_late_energy_geometry(inputs.room, inputs.n_per_wall, inputs.absorption_by_zone or {})
    alpha = jnp.asarray(alpha_rows(inputs))
    s = jnp.asarray(scattering())
    area = jnp.ones(len(FREQUENCIES))
    if which == "alpha":
        alpha = alpha.at[1, 0].set(jnp.nan)
    if which == "energy":
        alpha = alpha.at[:, 0].set(0.0)
    if which == "furniture":
        area = area.at[0].set(jnp.inf)
    if which == "scattering":
        s = s.at[0].set(jnp.nan)
    epsilon = jnp.zeros(len(FREQUENCIES))
    if which.startswith("air_"):
        epsilon = epsilon.at[0].set({"air_nan": jnp.nan, "air_inf": jnp.inf, "air_negative": -1.0, "air_one": 1.0}[which])
    def evaluate(a: jax.Array) -> cj.CheckedBands:
        fraction = epsilon if which.startswith("air_") else None
        result = lf.late_family_arrays(geometry, a, s, furniture_absorption_area_m2=area if which == "furniture" else None, air_absorption_fraction=fraction)
        invalid = result.invalid_absorption | result.invalid_furniture | result.invalid_energy | result.invalid_scattering
        return cj.CheckedBands(result.late_share_energy, invalid)
    return _record(evaluate, alpha)


def flags() -> dict[str, object]:
    angles = tmm.diffuse_incidence_angles_deg(method="paris")
    rows: dict[str, object] = {which: _family(which) for which in ("alpha", "energy", "furniture", "scattering", "air_nan", "air_inf", "air_negative", "air_one")}
    impedance = jnp.full((2, len(angles)), CONSTANTS.rho_c * 4.0 + 0j).at[0, 1].set(jnp.nan)
    # 對實部求導；複數本身不作 grad 的純量輸入。
    rows["paris"] = _record(lambda x: lf.paris_absorption_arrays(x + 0j, angles, CONSTANTS.rho_c), impedance.real)
    rows["constant"] = _record(lambda x: lf.constant_absorption_arrays(x + 0j, CONSTANTS.rho_c), jnp.asarray([-1.0, 8200.0]))
    areas = jnp.asarray(geometry_areas())
    alpha = jnp.full((len(areas), 2), 0.3).at[0, 0].set(1.0)
    rows["t60"] = _record(lambda x: cj.eyring_t60_arrays(areas, 72.0, x), alpha)
    rows["fs"] = _record(lambda x: jax.vmap(lambda band: cj.schroeder_frequency_arrays(72.0, band))(x),
                         jnp.asarray([[0.0, 1.0], [0.5, 0.6]]))
    def frequency(x: jax.Array) -> cj.CheckedBands:
        w = cj.crossover_weights_arrays(x, jnp.asarray(190.0))
        return cj.CheckedBands(w.w_geo, w.invalid)
    rows["frequency"] = _record(frequency, jnp.asarray([0.0, 240.0]))
    frequencies = jnp.asarray(FREQUENCIES)
    fem = jnp.arange(len(FREQUENCIES), dtype=frequencies.dtype) + 1.0
    geo = fem + 2.0
    def cap(fs: jax.Array) -> cj.CheckedBands:
        w = cj.crossover_weights_arrays(frequencies, fs)
        return cj.stitched_energy_arrays(w, fem, geo)
    boundary = jax.jit(cap)(jnp.asarray(FEM_GEOMETRIC_CROSSOVER_CAP_HZ))
    below = jax.jit(cap)(jnp.nextafter(jnp.asarray(FEM_GEOMETRIC_CROSSOVER_CAP_HZ), jnp.asarray(0.0)))
    rows["capped"] = {"capped": bool(cj.crossover_weights_arrays(frequencies,
        jnp.asarray(FEM_GEOMETRIC_CROSSOVER_CAP_HZ)).capped_by_upper_limit),
        "invalid": np.asarray(boundary.invalid).tolist(), "boundary": np.asarray(boundary.values).tolist(),
        "below": np.asarray(below.values).tolist(), "fem": np.asarray(fem).tolist(), "geo": np.asarray(geo).tolist()}
    w = cj.crossover_weights_arrays(jnp.asarray([240.0, 500.0]), jnp.asarray(190.0))
    rows["missing_fem"] = _record(lambda x: cj.stitched_energy_arrays(w, x, jnp.asarray([1.0, 2.0])),
                                  jnp.asarray([jnp.nan, jnp.nan]))
    return rows


def geometry_areas() -> np.ndarray:
    from aosr.physics.late_energy import _wall_areas
    return _wall_areas(problem("whole").room)
