"""#562 隔離雙精度探針：回原始前向與梯度，裁判留在父行程。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from dataclasses import replace
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from aosr.physics import geometric_early_jax as gj, geometric_lane as gl
from aosr.geometry.shoebox import Wall
from aosr.config.construction_environment import construction_environment
from aosr.config.three_lane_crossover import REFLECTION_ORDER_K
from aosr.materials.construction_diff import build_template, surface_impedance_from_parameters
from aosr.physics.late_energy import LateEnergyOrderBand, LateEnergyOrderResult
from aosr.geometry.shoebox import Point
from aosr.physics.report_source import SourceModelSpec
from aosr.physics.furniture_scene import FurnitureLaneInputs
from aosr.physics.wall_reflections import WallReflectionSources
from aosr.physics import amplitude_jax as aj
from aosr.physics.room_paths import image_source_paths
from tests.engine import _geometric_early_jax_cases as case, _furniture_energy_cases as furniture_case


class CaseRun(TypedDict):
    reference: dict[str, list[tuple[float, ...]]]
    double: dict[str, list[list[float]]]
    double_jit: dict[str, list[list[float]]]
    single: dict[str, list[list[float]]]
    single_jit: dict[str, list[list[float]]]
    double_dtypes: list[str]
    single_dtypes: list[str]


class GradientRun(TypedDict):
    automatic: list[float]
    central: list[float]


class ProbeRun(TypedDict):
    x64: bool
    cases: dict[str, CaseRun]
    gradients: dict[str, GradientRun]
    mutants: dict[str, float]
    legacy_bounces_equal: bool
    vmap_difference: float
    repeat_bitwise: bool


def reference_columns(
    points: tuple[Point, ...], receivers: tuple[Point, ...], models: tuple[SourceModelSpec, ...],
    sources: WallReflectionSources, furniture: FurnitureLaneInputs | None,
) -> dict[str, list[tuple[float, ...]]]:
    expected: dict[str, list[tuple[float, ...]]] = {q: [] for q in (
        "direct_energy", "reflected_energy", "interference_energy", "scattering")}
    late = LateEnergyOrderResult(REFLECTION_ORDER_K, tuple(
        LateEnergyOrderBand(f, (0.,)*REFLECTION_ORDER_K, 0., 0.) for f in case.AXIS))
    for source, model in zip(points, models, strict=True):
        for receiver in receivers:
            reference = gl.solve_geometric_early_lane(room=case.ROOM, source=source, receiver=receiver, source_model=model,
                sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO*case.SPEED,
                frequencies_hz=case.AXIS, impedance_by_wall={n: 900. for n in Wall.wall_names()},
                scattering_by_wall=case.scattering(), reflection_sources=sources, furniture=furniture)
            full = gl.solve_geometric_lane(room=case.ROOM, source=source, receiver=receiver, source_model=model,
                sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO*case.SPEED,
                frequencies_hz=case.AXIS, impedance_by_wall={n: 900. for n in Wall.wall_names()},
                scattering_by_wall=case.scattering(), reflection_sources=sources, furniture=furniture, late_result=late)
            for q in expected:
                assert getattr(reference, q) == getattr(full, q)
                expected[q].append(getattr(reference, q))
    return expected


def construction_gradient(name: str) -> GradientRun:
    """厚度、空腔逐格中央差分；參考仍是完整 numpy 早期入口，沒有用 JAX 自證。"""
    axis = (125., 503., 1009.)
    sources = case.reflection_sources(name, axis)
    table = gj.build_geometric_early_table(room=case.ROOM, sources=case.SOURCES[:1], receivers=case.RECEIVERS[:1],
        source_models=case.models()[:1], sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO*case.SPEED,
        frequencies_hz=axis, rectangle_zones_by_wall=sources.rectangle_zones_by_wall)
    rows, normal = case.patch_rows(sources, table.patch_ids, table.incidence_cosines)
    patch = table.patch_ids.index("x0@high" if name == "zoned" else "x0")
    air, ranges = construction_environment(density_kg_m3=case.RHO, sound_speed_m_s=case.SPEED)
    template = build_template(case.CONSTRUCTION, jnp.asarray(axis), air=air, ranges=ranges,
        variables=("layer1.thickness_m", "layer2.thickness_m"))
    scattering = jnp.asarray([case.scattering()[n.partition("@")[0]] for n in table.patch_ids])

    def objective(theta: jax.Array) -> jax.Array:
        params = dict(zip(template.names, theta, strict=True))
        z = list(jnp.asarray(r) for r in rows)
        z[patch] = surface_impedance_from_parameters(template, params, jnp.asarray(axis), air=air,
            incidence_angles_deg=jnp.asarray(table.incidence_angles_deg[patch]))
        n = jnp.asarray(normal).at[patch].set(surface_impedance_from_parameters(template, params,
            jnp.asarray(axis), air=air))
        result = gj.solve_geometric_early(table, z, n, scattering)
        return jnp.sum(result.reflected_energy+result.interference_energy+result.scattering)

    def reference(theta: np.ndarray) -> float:
        material = case.CONSTRUCTION.model_copy(update={"layers": tuple(layer.model_copy(
            update={"thickness_m": float(value)}) for layer, value in zip(case.CONSTRUCTION.layers, theta, strict=True))})
        if sources.rectangle_zones_by_wall:
            zones = sources.rectangle_zones_by_wall[Wall.X0]
            changed = replace(sources, rectangle_zones_by_wall={Wall.X0: (zones[0], zones[1].model_copy(update={"material": material}))})
        else:
            changed = replace(sources, materials_by_wall={**sources.materials_by_wall, Wall.X0: material})
        result = gl.solve_geometric_early_lane(room=case.ROOM, source=case.SOURCES[0], receiver=case.RECEIVERS[0],
            source_model=case.models()[0], sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=table.rho_c,
            frequencies_hz=axis, impedance_by_wall={w.wall_name(): 900. for w in Wall.all()},
            scattering_by_wall=case.scattering(), reflection_sources=changed)
        return float(np.sum(np.asarray(result.reflected_energy)+result.interference_energy+result.scattering))
    theta = np.array((.05, .17))
    automatic = np.asarray(jax.jit(jax.grad(objective))(jnp.asarray(theta)))
    central = []
    for i, value in enumerate(theta):
        step = 1e-6*abs(value)
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        central.append((reference(plus)-reference(minus))/(2.*step))
    return {"automatic": automatic.tolist(), "central": central}


def mutations(table: gj.GeometricEarlyTable, name: str, rows: tuple[jax.Array, ...], normal: jax.Array,
              scattering: jax.Array, expected: dict[str, list[tuple[float, ...]]]) -> dict[str, float]:
    """真算六種錯誤常數表；回對獨立 numpy 各欄的最大差供父行程裁判。"""
    variants = {}
    if name == "zoned":
        low, high = table.patch_ids.index("x0@low"), table.patch_ids.index("x0@high")
        wrong_zone = tuple(replace(pair, paths=replace(pair.paths,
            bounce_patch=np.where(pair.paths.bounce_patch == high, low, pair.paths.bounce_patch),
            bounce_angle=np.zeros_like(pair.paths.bounce_angle))) for pair in table.pairs)
        variants["wrong_zone"] = replace(table, pairs=wrong_zone)
        variants["normal_angle"] = replace(table, pairs=tuple(replace(pair, paths=replace(pair.paths,
            bounce_cos=np.ones_like(pair.paths.bounce_cos))) for pair in table.pairs))
        variants["lost_order"] = replace(table, pairs=tuple(replace(pair, paths=replace(pair.paths,
            order=np.minimum(pair.paths.order, 1))) for pair in table.pairs))
        variants["equal_area"] = replace(table, patch_areas_m2=np.ones_like(table.patch_areas_m2))
        variants["lost_directivity"] = replace(table, pairs=tuple(replace(pair,
            directivity=np.ones_like(pair.directivity)) for pair in table.pairs))
    if name == "furnished":
        variants["lost_furniture"] = replace(table, pairs=tuple(replace(pair,
            furniture_pressure=np.zeros_like(pair.furniture_pressure)) for pair in table.pairs))
    differences = {}
    for key, variant in variants.items():
        result = gj.solve_geometric_early(variant, rows, normal, scattering)
        differences[key] = max(float(np.max(np.abs(np.asarray(getattr(result, q)).reshape(-1, table.shape[-1])-
            np.asarray(ref)))) for q, ref in expected.items())
    return differences


def measure() -> ProbeRun:
    records: dict[str, CaseRun] = {}
    mutant_errors: dict[str, float] = {}
    legacy_equal, repeat_equal, vmap_difference = True, True, 0.
    for name in ("whole", "construction", "zoned", "furnished"):
        furnished = name == "furnished"
        sources = case.reflection_sources("whole" if furnished else name)
        points = (furniture_case.SOURCE,) if furnished else case.SOURCES
        receivers = (furniture_case.RECEIVER,) if furnished else case.RECEIVERS
        models = (furniture_case.analytic(),) if furnished else case.models()
        furniture = furniture_case.lane_inputs((furniture_case.desk(),), case.AXIS) if furnished else None
        table = gj.build_geometric_early_table(room=case.ROOM, sources=points, receivers=receivers,
            source_models=models, sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO*case.SPEED,
            rectangle_zones_by_wall=sources.rectangle_zones_by_wall, furniture=furniture)
        rows, normal = case.patch_rows(sources, table.patch_ids, table.incidence_cosines)
        scattering = np.array([case.scattering()[n.partition("@")[0]] for n in table.patch_ids])
        expected = reference_columns(points, receivers, models, sources, furniture)
        def solve(z: tuple[jax.Array, ...], n: jax.Array) -> gj.GeometricEarlyJaxResult:
            return gj.solve_geometric_early(table, z, n, jnp.asarray(scattering))
        record = CaseRun(reference=expected, double={}, double_jit={}, single={}, single_jit={}, double_dtypes=[], single_dtypes=[])
        for key, dtype in (("double", jnp.complex128), ("single", jnp.complex64)):
            z = tuple(jnp.asarray(r, dtype=dtype) for r in rows)
            n = jnp.asarray(normal, dtype=dtype)
            eager, compiled = solve(z, n), jax.jit(solve)(z, n)
            values = {q: np.asarray(getattr(eager, q)).reshape(-1, len(case.AXIS)).tolist() for q in expected}
            jit_values = {q: np.asarray(getattr(compiled, q)).reshape(-1, len(case.AXIS)).tolist() for q in expected}
            dtypes = [str(getattr(eager, q).dtype) for q in expected]
            if key == "double":
                record["double"], record["double_jit"], record["double_dtypes"] = values, jit_values, dtypes
            else:
                record["single"], record["single_jit"], record["single_dtypes"] = values, jit_values, dtypes
            assert np.asarray(compiled.valid).all()
            again = jax.jit(solve)(z, n)
            repeat_equal &= all(np.asarray(a).tobytes() == np.asarray(b).tobytes()
                for a, b in zip(compiled, again, strict=True))
            if key == "double":
                mutant_errors.update(mutations(table, name, z, n, jnp.asarray(scattering), expected))
        if name == "construction":
            paths = image_source_paths(case.ROOM, points[0], receivers[0], case.SPEED, 3, reflection_sources=sources)
            local = aj.build_path_table(paths, receivers[0].as_tuple(), case.AXIS)
            local_rows, _ = case.patch_rows(sources, local.patch_ids, local.incidence_cosines)
            gathered = aj.bounce_impedances_by_patch(local, tuple(jnp.asarray(r) for r in local_rows))
            original = aj.bounce_impedances_from_sources(local, paths, receivers[0].as_tuple(), sources)
            legacy_equal &= np.array_equal(gathered, original)
            z64 = tuple(jnp.asarray(r) for r in rows)
            normal64 = jnp.asarray(normal)
            batch = tuple(jnp.stack((z, z*1.1)) for z in z64)
            mapped = jax.jit(jax.vmap(lambda z: solve(z, normal64)))(batch)
            for i in range(2):
                sequential = solve(tuple(z[i] for z in batch), normal64)
                vmap_difference = max(vmap_difference, max(float(np.max(np.abs(np.asarray(a[i])-np.asarray(b))))
                    for a, b in zip(mapped[:4], sequential[:4], strict=True)))
        records[name] = record
    return {"x64": bool(getattr(jax.config, "jax_enable_x64")), "cases": records,
            "gradients": {name: construction_gradient(name) for name in ("construction", "zoned")},
            "mutants": mutant_errors, "legacy_bounces_equal": legacy_equal, "vmap_difference": vmap_difference,
            "repeat_bitwise": repeat_equal}


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(measure()))
