"""晚期一族的 CPU 雙精度冷探針；只回原始數字，裁判住父考卷。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from aosr.runtime import configure_calculation_jax

configure_calculation_jax()

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.paths import config_path
from aosr.config.porous_model_ranges import load_porous_model_ranges
from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ
from aosr.materials import construction, transfer_matrix as tmm
from aosr.materials.catalog_absorption import PARIS_MAPPED_WEIGHTS, PARIS_MU, complex_random_incidence_absorption
from aosr.materials.construction_diff import build_template, surface_impedance_from_parameters
from aosr.materials.construction_models import AirLayer, Assembly, PorousLayer
from aosr.physics import crossover_jax as cj, late_family_jax as lf
from aosr.physics.late_energy_jax import _bounded_weighted_absorption, zoned_late_energy_geometry
from tests.engine._late_family_cases import CASES, CONSTANTS, FREQUENCIES, alpha_rows, furniture_area, problem, scattering


def evaluate(name: str, alpha: jax.Array) -> dict[str, jax.Array]:
    inputs = problem(name)
    geometry = zoned_late_energy_geometry(inputs.room, inputs.n_per_wall, inputs.absorption_by_zone or {})
    area = jnp.asarray(furniture_area(), dtype=alpha.dtype) if name == "furniture" else None
    late = lf.late_family_arrays(geometry, alpha, jnp.asarray(scattering(), dtype=alpha.dtype),
                                furniture_absorption_area_m2=area)
    volume = inputs.room.Lx * inputs.room.Ly * inputs.room.Lz
    t60 = cj.eyring_t60_arrays(geometry.whole.wall_areas, volume, late.alpha_by_wall)
    mids = jnp.asarray([FREQUENCIES.index(f) for f in SCHROEDER_T60_BANDS_HZ])
    fs = cj.schroeder_frequency_arrays(volume, t60.values[mids])
    weights = cj.crossover_weights_arrays(jnp.asarray(FREQUENCIES, dtype=alpha.dtype), fs.values)
    return {key: getattr(late, key) for key in ("alpha_by_wall", "alpha_by_patch", "total_energy", "energy_by_order",
        "tail_energy", "late_share_energy")} | {"t60": t60.values, "fs": fs.values,
                                                 "w_fem": weights.w_fem, "w_geo": weights.w_geo}


def loss(name: str, alpha: jax.Array) -> jax.Array:
    row = evaluate(name, alpha)
    return jnp.sum(jnp.stack([jnp.sum(row[key]) for key in ("energy_by_order", "tail_energy", "late_share_energy", "t60", "fs")]))


def output_gradients(name: str, alpha: jax.Array) -> dict[str, object]:
    """每個輸出各自反傳；Hz 的尺度不能遮掉能量鏈的導數誤差。"""
    keys = ("energy_by_order", "tail_energy", "late_share_energy", "t60", "fs")
    derivative = jax.jit(jax.jacrev(lambda a: {key: evaluate(name, a)[key] for key in keys}))(alpha)
    return {key: np.asarray(value * alpha).tolist() for key, value in derivative.items()}


def implicit_patches() -> dict[str, object]:
    """省略小塊列時應依牆索引取值，不可變成均勻吸音或交換牆序。"""
    inputs = problem("whole")
    geometry = zoned_late_energy_geometry(inputs.room, inputs.n_per_wall, {})
    alpha = jnp.asarray(alpha_rows(inputs))
    def evaluate(a: jax.Array) -> lf.LateOrderArrays:
        return lf.late_energy_by_order_arrays(geometry.whole, a)
    return {mode: {key: np.asarray(value).tolist() for key, value in result._asdict().items()}
            for mode, result in (("eager", evaluate(alpha)), ("jit", jax.jit(evaluate)(alpha)))}


def arrays() -> dict[str, object]:
    cases, gradients, dtypes = {}, {}, {}
    for name in CASES:
        alpha = jnp.asarray(alpha_rows(problem(name)))
        def fn(a: jax.Array) -> dict[str, jax.Array]:
            return evaluate(name, a)
        eager, compiled, single = fn(alpha), jax.jit(fn)(alpha), jax.jit(fn)(alpha.astype(jnp.float32))
        cases[name] = {mode: {key: np.asarray(value).tolist() for key, value in row.items()}
                       for mode, row in (("eager", eager), ("jit", compiled), ("single", single))}
        gradients[name] = output_gradients(name, alpha)
        for suffix, row in (("64", compiled), ("32", single)):
            dtypes[name + suffix] = [str(v.dtype) for v in row.values()]
        single_gradient = jax.jit(jax.grad(lambda a: loss(name, a)))(alpha.astype(jnp.float32))
        assert bool(jnp.all(jnp.isfinite(single_gradient)))
        dtypes[name + "32"].append(str(single_gradient.dtype))
    return {"cases": cases, "gradients": gradients, "dtypes": dtypes}


def paris() -> dict[str, object]:
    frequencies = jnp.asarray(FREQUENCIES)
    ranges = load_porous_model_ranges(config_path("porous_model_ranges.toml"))
    material = Assembly(kind="assembly", backing="rigid", layers=(PorousLayer(kind="porous", model="miki",
        thickness_m=0.05, flow_resistivity_pa_s_m2=19100.0, material_class="fibrous"),
        AirLayer(kind="air", thickness_m=0.1)))
    template = build_template(material, frequencies, air=CONSTANTS, ranges=ranges)
    parameters = template.baseline_parameters()
    angles = tmm.diffuse_incidence_angles_deg(method="paris")

    def evaluate(p: dict[str, jax.Array], f: jax.Array, theta: jax.Array) -> jax.Array:
        z = surface_impedance_from_parameters(template, p, f, air=CONSTANTS, incidence_angles_deg=theta)
        return lf.paris_absorption_arrays(z, theta, CONSTANTS.rho_c).values

    actual = jax.jit(evaluate)(parameters, frequencies, angles)
    reference = construction.random_incidence_absorption(material, frequencies, air=CONSTANTS, ranges=ranges)[0]
    automatic = jax.jit(jax.jacrev(lambda p: evaluate(p, frequencies, angles)))(parameters)
    central, scaled = [], []
    for key, value in parameters.items():
        step = value * 1e-5
        plus = material.model_copy(update={"layers": tuple(layer.model_copy(update={template.variables[
            template.names.index(key)].field: float(value + step)}) if i + 1 == template.variables[
            template.names.index(key)].layer_index else layer for i, layer in enumerate(material.layers))})
        minus = material.model_copy(update={"layers": tuple(layer.model_copy(update={template.variables[
            template.names.index(key)].field: float(value - step)}) if i + 1 == template.variables[
            template.names.index(key)].layer_index else layer for i, layer in enumerate(material.layers))})
        delta = (construction.random_incidence_absorption(plus, frequencies, air=CONSTANTS, ranges=ranges)[0] -
                 construction.random_incidence_absorption(minus, frequencies, air=CONSTANTS, ranges=ranges)[0]) / (2 * step)
        central.extend(np.asarray(delta * value).tolist())
        scaled.extend(np.asarray(automatic[key] * value).tolist())
    single_angles = angles.astype(jnp.float32).at[-1].set(jnp.nextafter(jnp.float32(90.0), jnp.float32(0.0)))
    single = evaluate({k: v.astype(jnp.float32) for k, v in parameters.items()}, frequencies.astype(jnp.float32), single_angles)
    constants = jnp.asarray([8200.0 + 400j, 1646.4, 45687.6, 950.0])
    fixed = lf.constant_absorption_arrays(constants, CONSTANTS.rho_c)
    expected_fixed = np.asarray([complex_random_incidence_absorption(complex(z) / CONSTANTS.rho_c) for z in constants])
    return {"actual": np.asarray(actual).tolist() + np.asarray(fixed.values).tolist(),
        "reference": np.asarray(reference).tolist() + expected_fixed.tolist(),
        "automatic": scaled, "central": central, "single": np.asarray(single).tolist(),
        "angles_bitwise": bool(jnp.array_equal(angles, tmm.diffuse_incidence_angles_deg(method="paris"))),
        "weights_bitwise": bool(jnp.array_equal(jax.jit(lf.paris_angular_weights)(angles),
            jnp.sin(jnp.deg2rad(angles)) * jnp.cos(jnp.deg2rad(angles)))),
        "gauss_nodes_bitwise": bool(np.array_equal(lf.PARIS_MU, PARIS_MU)),
        "gauss_weights_bitwise": bool(np.array_equal(lf.PARIS_MAPPED_WEIGHTS, PARIS_MAPPED_WEIGHTS)),
        "double_dtype": str(actual.dtype), "single_dtype": str(single.dtype)}


def weights() -> dict[str, object]:
    f = jnp.asarray(FREQUENCIES)
    cases = []
    for fs in (80.0, 150.0, 190.0, 299.0, 300.0, 301.0):
        w = jax.jit(cj.crossover_weights_arrays)(f, jnp.asarray(fs))
        cases.append({"fs": fs, "w_fem": np.asarray(w.w_fem).tolist(), "w_geo": np.asarray(w.w_geo).tolist(),
                      "capped": bool(w.capped_by_upper_limit)})
    return {"weight_cases": cases, "weight_gradients": {
        "fs": np.asarray(jax.jit(jax.jacrev(lambda x: cj.crossover_weights_arrays(f, x).w_geo))(jnp.asarray(190.0))).tolist(),
        "frequency": np.asarray(jax.jit(jax.jacrev(lambda x: cj.crossover_weights_arrays(x, jnp.asarray(190.0)).w_fem))(f)).tolist()}}


def stitched() -> dict[str, object]:
    from aosr.geometry.shoebox import Point, Wall
    from aosr.physics.crossover import crossover_weights
    from aosr.physics.geometric_lane import solve_geometric_lane
    from aosr.physics.report_source import SourceModelKind, SourceModelSpec
    from aosr.physics.three_lane_report import stitch_energy_points

    inputs = problem("whole")
    geo = solve_geometric_lane(room=inputs.room, source=Point(1.2, 1.3, 1.1), receiver=Point(4.7, 2.8, 1.4),
        sound_speed_m_s=CONSTANTS.sound_speed_m_s, rho_c_pa_s_per_m=CONSTANTS.rho_c,
        frequencies_hz=FREQUENCIES, impedance_by_wall={w.wall_name(): 4.0 * CONSTANTS.rho_c for w in Wall.all()},
        source_model=SourceModelSpec(SourceModelKind.OMNIDIRECTIONAL))
    fem = np.arange(len(FREQUENCIES), dtype=float) + 1.0
    old = crossover_weights(FREQUENCIES, 190.0)
    points = stitch_energy_points(frequencies_hz=FREQUENCIES, fem_energy_by_frequency=dict(zip(FREQUENCIES, fem, strict=True)),
        geometric_lane=geo, geometric_indices=tuple(range(len(FREQUENCIES))), weights=old)
    w = cj.crossover_weights_arrays(jnp.asarray(FREQUENCIES), jnp.asarray(190.0))
    early = jnp.asarray([geo.direct_energy, geo.reflected_energy, geo.interference_energy])
    late = jnp.asarray(geo.late_energy)
    def evaluate(values: jax.Array) -> jax.Array:
        return cj.stitched_energy_arrays(w, values[0], jnp.sum(values[1:4], axis=0) + values[4]).values
    data = jnp.concatenate((jnp.asarray(fem)[None, :], early, late[None, :]))
    derivative = jax.jit(jax.jacrev(evaluate))(data)
    step = 1e-5
    central = np.zeros(derivative.shape)
    for index in np.ndindex(data.shape):
        delta = np.zeros(data.shape)
        delta[index] = step
        central[(slice(None), *index)] = np.asarray((evaluate(data + delta) - evaluate(data - delta)) / (2 * step))
    capped = []
    for fs in (300.0, 302.0):
        hard = cj.crossover_weights_arrays(jnp.asarray(FREQUENCIES), jnp.asarray(fs))
        energy = jnp.sum(early, axis=0) + late
        actual = jax.jit(cj.stitched_energy_arrays)(hard, jnp.asarray(fem), energy)
        formal = stitch_energy_points(frequencies_hz=FREQUENCIES,
            fem_energy_by_frequency=dict(zip(FREQUENCIES, fem, strict=True)), geometric_lane=geo,
            geometric_indices=tuple(range(len(FREQUENCIES))), weights=crossover_weights(FREQUENCIES, fs))
        cap_derivative = jax.jit(jax.jacrev(lambda ef, eg: cj.stitched_energy_arrays(hard, ef, eg).values,
            argnums=(0, 1)))(jnp.asarray(fem), energy)
        capped.append({"fs": fs, "capped": bool(hard.capped_by_upper_limit),
            "invalid": np.asarray(actual.invalid).tolist(), "actual": np.asarray(actual.values).tolist(),
            "reference": [p.total_energy for p in formal],
            "gradients": [np.asarray(v).tolist() for v in cap_derivative]})
    return {"capped": capped, "actual": np.asarray(jax.jit(evaluate)(data)).tolist(), "reference": [p.total_energy for p in points],
            "gradient": np.asarray(derivative).tolist(), "central": central.tolist()}


def other_gradients() -> dict[str, object]:
    inputs = problem("zoned")
    geometry = zoned_late_energy_geometry(inputs.room, inputs.n_per_wall, inputs.absorption_by_zone or {})
    alpha = jnp.asarray(alpha_rows(inputs))
    s = jnp.linspace(0.1, 0.9, len(FREQUENCIES))
    def share(values: jax.Array) -> jax.Array:
        return lf.late_family_arrays(geometry, alpha, values).late_share_energy
    automatic = jax.jit(jax.jacrev(share))(s) * s[None, :]
    def furniture(area: jax.Array) -> jax.Array:
        return jnp.sum(lf.late_family_arrays(geometry, alpha, s, furniture_absorption_area_m2=area).total_energy)
    constant = jax.jit(jax.grad(furniture))(jnp.asarray(furniture_area()))
    uniform = jnp.full_like(alpha, 0.25)
    uniform_gradient = output_gradients("zoned", uniform)
    bitwise = True
    for weights in (geometry.wall_weights, geometry.patch_weights):
        def old(a: jax.Array) -> jax.Array:
            return _bounded_weighted_absorption(weights, a)
        def new(a: jax.Array) -> jax.Array:
            return lf._bounded_absorption_with_linear_gradient(weights, a)
        for dtype in (jnp.float64, jnp.float32):
            for values in (alpha, uniform, jnp.zeros_like(alpha), -jnp.zeros_like(alpha), jnp.ones_like(alpha)):
                before, after = jax.jit(old)(values.astype(dtype)), jax.jit(new)(values.astype(dtype))
                bitwise = bitwise and np.asarray(before).tobytes() == np.asarray(after).tobytes()
    return {"scattering_gradient": np.asarray(automatic).tolist(), "scattering_input": np.asarray(s).tolist(),
            "furniture_gradient": np.asarray(constant).tolist(), "uniform_alpha": np.asarray(uniform).tolist(),
            "uniform_gradient": uniform_gradient, "uniform_forward_bitwise": bitwise}


def main() -> None:
    from tests.engine._late_family_jax_flags import flags

    result = arrays() | weights() | other_gradients() | {"paris": paris(), "stitched": stitched(), "flags": flags(), "implicit_patches": implicit_patches()}
    Path(sys.argv[1]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
