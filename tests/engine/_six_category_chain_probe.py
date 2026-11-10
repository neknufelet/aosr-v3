"""CPU 雙精度冷行程：同 θ 匯出材料、既有分區控制、實量與失效檢查。"""
from __future__ import annotations

import dataclasses
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.config.frequency_axis import low_frequency_axis_frequencies
from aosr.config.material_gradient import Mount
from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.gradient import endpoint, material_gradient as old, six_category_chain as chain
from aosr.materials.construction_diff import ConstructionTemplate, build_template
from aosr.materials.construction_models import Assembly, ConstantImpedance, MeasuredImpedance, MountedOnWall, SurfaceMaterial
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_build import build_reduced_model
from aosr.physics.fem_reduced_store import save_reduced
from aosr.physics.report_io import ReportInput
from aosr.reporting.fem_reduced_scheme import export_operators
from aosr.reporting.scheme import Scheme
from aosr.scoring import _six_category_build as build, six_category_jax as scoring
from aosr.scoring.six_category_jax import SixCategoryConstants, six_category_costs
from tests.engine import _six_category_jax_cases as cases
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._material_gradient_zones_case import scheme as zone_scheme
from tests.engine._material_gradient_zones_probe import _build, _problem
from tests.engine._six_category_jax_probe import reflection_margin
from tests.engine._zone_rom_case import SETTINGS
from tests.engine.test_material_gradient import DIFFERENCE_STEP, _presets, _probe_point, _settings
from tests.engine.test_transfer_matrix_reference import IMPEDANCE_ABS, REFERENCE_REL


def model_for(scheme: Scheme) -> ReducedModel:
    """沿用既有分區替身小案例的建模設定、訓練材料與 80 Hz 網格；只考接線，不宣稱替身精度。"""
    operators = export_operators(scheme, capabilities=CAPABILITIES, directivity=DIRECTIVITY, max_frequency_hz=80.)
    assert scheme.scene.low_frequency_axis is not None
    axis = low_frequency_axis_frequencies(scheme.scene.low_frequency_axis)[0]
    rng = np.random.default_rng(561)
    training = [{key: tuple(complex(z) for z in 411.6*(rng.uniform(2, 12)-1j*rng.uniform(.5, 2)*100/np.asarray(axis)))
                 for key in operators.patch_keys} for _ in range(8)]
    return build_reduced_model(operators, axis, training, settings=SETTINGS)


def factory(model: ReducedModel, designs: tuple[old.SurfaceDesign, ...], c: SixCategoryConstants) -> chain.SixCategoryChainObjective:
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    return chain.six_category_chain_objective(model, designs, c.objective(), c, air=air, ranges=ranges, frequencies=model.frequencies_hz)


def capture(objective: chain.SixCategoryChainObjective, theta: jax.Array) -> tuple[chain.SixCategoryChainAux, tuple[jax.Array, ...]]:
    calls: list[tuple[jax.Array, ...]] = []
    def observe(c: SixCategoryConstants, *args: jax.Array) -> object:
        calls.append(args)
        return six_category_costs(c, *args)
    with patch.object(chain, 'six_category_costs', observe):
        value, result = objective(theta)
    assert np.asarray(value).tobytes() == np.asarray(result.objective_total).tobytes()
    return result, calls[0]


def control(model: ReducedModel, designs: tuple[old.SurfaceDesign, ...], c: SixCategoryConstants, theta: jax.Array) -> dict[str, object]:
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    legacy = old.chain_objective(model, designs, c.objective(), air=air, ranges=ranges, frequencies=model.frequencies_hz)
    objective = factory(model, designs, c)
    result: dict[str, object] = {}
    for mode, transform in (('eager', lambda x: x), ('jit', jax.jit)):
        aux = transform(objective)(theta)[1]
        actual = aux.low_frequency.objective_total
        expected = transform(legacy)(theta)[1].objective_total
        result[f'{mode}_bitwise'] = np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
        result[f'{mode}_hex'], result[f'{mode}_old_hex'] = float(actual).hex(), float(expected).hex()
        result[f'{mode}_category_bitwise'] = np.asarray(aux.categories.low_frequency_objective).tobytes() == np.asarray(actual).tobytes()
    return result


def exported_scheme(scheme: Scheme, designs: tuple[old.SurfaceDesign, ...], theta: jax.Array) -> Scheme:
    materials = endpoint.materials_from_surfaces(designs, np.asarray(theta), _settings())
    scene = endpoint.scene_with_materials(scheme.scene, materials)
    return Scheme.model_validate({**scheme.model_dump(), 'scene': scene.model_dump()})


def reference(scheme: Scheme, designs: tuple[old.SurfaceDesign, ...], theta: jax.Array, c: SixCategoryConstants) -> tuple[jax.Array, ...]:
    return cases.array_inputs(exported_scheme(scheme, designs, theta), c)


def material_measure(name: str, model: ReducedModel, objective: chain.SixCategoryChainObjective, theta: jax.Array,
                     scheme: Scheme, designs: tuple[old.SurfaceDesign, ...], c: SixCategoryConstants) -> tuple[dict[str, object], dict[str, object], tuple[jax.Array, ...]]:
    result, inputs = capture(objective, theta)
    exported = reference(scheme, designs, theta, c)
    expected = six_category_costs(c, exported[0], exported[1], exported[2], exported[3], inputs[4], exported[5])
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    impedances = chain._impedance_function(designs, c, air=air, ranges=ranges)
    compiled_inputs = jax.jit(impedances)(theta)
    row: dict[str, object] = dict(name=name, flags_match=np.array_equal(inputs[5], exported[5]))
    scales = [[float(np.max(np.abs(part(np.asarray(x))), initial=0.)) for part in (np.real, np.imag)] for x in exported[:4]]
    row['scales'] = scales
    for mode, actual in (('eager', inputs[:4]), ('jit', compiled_inputs)):
        row[f'{mode}_errors'] = [[float(np.max(np.abs(part(np.asarray(a))-part(np.asarray(b))), initial=0.)) for part in (np.real, np.imag)]
            for a, b in zip(actual, exported[:4], strict=True)]
        row[f'{mode}_ratios'] = [[float(np.max(np.abs(part(np.asarray(a))-part(np.asarray(b)))/(IMPEDANCE_ABS+REFERENCE_REL*np.abs(part(np.asarray(b)))), initial=0.)) for part in (np.real, np.imag)]
            for a, b in zip(actual, exported[:4], strict=True)]
    compiled = jax.jit(objective)(theta)[1]
    end: dict[str, object] = dict(name=name, reference=np.asarray(expected[:6]).tolist(), reference_product=float(expected.product_total),
        reference_objective=float(expected.objective_total), eager_objective=float(result.objective_total), jit_objective=float(compiled.objective_total),
        eager=np.asarray(result.categories[:6]).tolist(), jit=np.asarray(compiled.categories[:6]).tolist(),
        eager_product=float(result.product_total), jit_product=float(compiled.product_total), weights=c.category_weights.tolist(),
        flags=np.asarray(result.categories[6:12]).tolist())
    return row, end, inputs


def numpy_measure(scheme: Scheme, designs: tuple[old.SurfaceDesign, ...], theta: jax.Array,
                  inputs: tuple[jax.Array, ...], end: dict[str, object]) -> dict[str, object]:
    exported = exported_scheme(scheme, designs, theta)
    zones = exported.scene.surface_zones_by_wall
    assert zones is not None
    has_mounted = any(isinstance(z.material, MountedOnWall) for cells in zones.values() for z in cells)
    result = cases.numpy_result(exported, np.asarray(inputs[4]))
    costs, total = cases.numpy_costs(result)
    return dict(reference=costs.tolist(), reference_product=total, has_mounted_patch=has_mounted,
        eager=end['eager'], jit=end['jit'], eager_product=end['eager_product'], jit_product=end['jit_product'], weights=end['weights'])


def template_axes_measure(model: ReducedModel, designs: tuple[old.SurfaceDesign, ...], c: SixCategoryConstants) -> dict[str, object]:
    """攔截真 build_template 的驗證軸，仍照原入口驗材料；不以產品函式自己的輸出倒推軸。"""
    captured: dict[str, list[np.ndarray]] = {'design': [], 'kept': []}
    def observe(material: SurfaceMaterial, frequencies_hz: jax.Array, *, air: PhysicsConstants,
                ranges: PorousModelRanges, variables: tuple[str, ...] | None = None) -> ConstructionTemplate:
        captured['design' if variables else 'kept'].append(np.asarray(frequencies_hz))
        return build_template(material, frequencies_hz, air=air, ranges=ranges, variables=variables)
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    with patch.object(old, 'build_template', observe), patch.object(chain, 'build_template', observe):
        chain._impedance_function(designs, c, air=air, ranges=ranges)
    axes = (c.fem_frequencies_hz, c.frequencies_hz, c.frequencies_hz, c.union_frequencies_hz)
    return {f'{kind}_coverage': [bool(captured[kind]) and all(np.isin(axis, used).all() for used in captured[kind]) for axis in axes]
        for kind in captured}


def gradient_measure(objective: chain.SixCategoryChainObjective, theta: jax.Array, c: SixCategoryConstants,
                     inputs: tuple[jax.Array, ...]) -> dict[str, object]:
    """只驗 capped=True 硬切區；測不到決策紙第 11 條在平滑區刻意省掉的 dw/dθ。"""
    def totals(x: jax.Array) -> jax.Array:
        r = objective(x)[1]
        return jnp.stack((r.objective_total, r.product_total))
    evaluate = jax.jit(totals)
    values = np.asarray(evaluate(theta))
    automatic = np.asarray(jax.jit(jax.jacrev(totals))(theta))*np.asarray(theta)
    central = np.empty_like(automatic)
    for i, h in enumerate(DIFFERENCE_STEP*np.asarray(theta)):
        delta = jnp.zeros_like(theta).at[i].set(h)
        central[:, i] = (np.asarray(evaluate(theta+delta))-np.asarray(evaluate(theta-delta)))/(2*h)*float(theta[i])
    return dict(automatic=automatic.tolist(), central=central.tolist(), values=values.tolist(), step=DIFFERENCE_STEP, theta=np.asarray(theta).tolist(),
        reflection_margin=reflection_margin(c, inputs), capped=bool(objective(theta)[1].categories.capped_by_upper_limit))


def single_measure(objective: chain.SixCategoryChainObjective, theta: jax.Array) -> dict[str, object]:
    r, inputs = capture(objective, theta.astype(jnp.float32))
    (_, _), gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))(theta.astype(jnp.float32))
    return dict(inputs=[str(x.dtype) for x in inputs[:5]], outputs=[str(x.dtype) for x in r.categories[:6]],
        aux_dtypes=[str(x.dtype) for x in (r.objective_total, r.product_total, r.low_frequency.screening_total,
            r.low_frequency.big_peak_proxy_db, r.low_frequency.objective_total)],
        gradient_dtype=str(gradient.dtype), finite=bool(jnp.all(jnp.isfinite(gradient)) & jnp.isfinite(r.objective_total)))


def invalid_measure(model: ReducedModel, designs: tuple[old.SurfaceDesign, ...], c: SixCategoryConstants, theta: jax.Array) -> list[dict[str, object]]:
    variants = [c._replace(response_insufficient_coverage=np.asarray(True)),
        c._replace(timbre_primary_pairs_weights=np.zeros_like(c.timbre_primary_pairs_weights)),
        c._replace(reflection_coverage=np.zeros_like(c.reflection_coverage)),
        c._replace(reverb_weights=np.zeros_like(c.reverb_weights)),
        c._replace(matching_weights=np.zeros_like(c.matching_weights))]
    records: list[dict[str, object]] = []
    for index, constants in enumerate(variants):
        objective = factory(model, designs, constants)
        r = objective(theta)[1]
        derivative = jax.jit(jax.grad(lambda x: jnp.nansum(jnp.stack(objective(x)[1].categories[:6]))))(theta)
        records.append(dict(product_nan=bool(jnp.isnan(r.product_total)), objective_nan=bool(jnp.isnan(r.objective_total)),
            category_nan=bool(jnp.isnan(r.categories[index])), gradient_finite=bool(jnp.all(jnp.isfinite(derivative)))))
    return records


def rejections(model: ReducedModel, designs: tuple[old.SurfaceDesign, ...], c: SixCategoryConstants) -> dict[str, object]:
    assembly = next(d.material for d in designs if d.material is not None)
    assert isinstance(assembly, Assembly)
    constant = ConstantImpedance(impedance_pa_s_m=2100.)
    measured = MeasuredImpedance.model_validate(dict(kind='measured_impedance', source_and_conditions='合成拒收考卷，非真實量測', base_assembly=assembly.model_dump(),
        points=[dict(frequency_hz=f, real_pa_s_m=2100., imag_pa_s_m=0.) for f in (100., 1000.)]))
    materials = (constant, assembly, MountedOnWall(layers=assembly.layers, wall=constant), measured)
    air, ranges = construction_environment(density_kg_m3=model.density_kg_m3, sound_speed_m_s=model.sound_speed_m_s)
    calls: dict[str, Callable[[], object]] = {
        'measured': lambda: factory(model, (dataclasses.replace(designs[0], kept_material=measured), *designs[1:]), c),
        'order': lambda: factory(model, designs, c._replace(patch_ids=c.patch_ids[::-1])),
        'axis': lambda: factory(model, designs, c._replace(fem_frequencies_hz=c.fem_frequencies_hz[:-1])),
        'objective': lambda: chain.six_category_chain_objective(model, designs, c.objective(),
            c._replace(objective_proxy_weight=c.objective_proxy_weight+1), air=air, ranges=ranges, frequencies=model.frequencies_hz),
        'rigid': lambda: factory(model, (dataclasses.replace(designs[0], kept_material=None), *designs[1:]), c)}
    result: dict[str, object] = dict(material_flags=scoring.constant_material_flags(materials).tolist())
    for name, call in calls.items():
        try:
            call()
        except ValueError as error:
            result[name] = str(error)
        else:
            result[name] = ''
    return result


def candidate_designs(scheme: Scheme, c: SixCategoryConstants, mount: Mount) -> tuple[old.SurfaceDesign, ...]:
    zones = scheme.scene.surface_zones_by_wall
    assert zones is not None
    kept = {str(name): next(z.material for z in zones[str(name).partition('@')[0]] if getattr(z, 'id', '') == str(name).partition('@')[2]) for name in c.patch_ids}
    return old.designs_from_presets(tuple((str(name), 'glass_wool_48k_50mm_gap50' if str(name)=='x0@b' else None) for name in c.patch_ids),
        _settings(), _presets(), kept=kept, mount=mount)


def gradient_designs(scheme: Scheme, c: SixCategoryConstants) -> tuple[old.SurfaceDesign, ...]:
    """#1057 空氣降低 T60 與 f_s；固定原牆阻抗提高一成，保留梯度題的硬切區前提。"""
    designs = candidate_designs(scheme, c, 'rigid')
    return tuple(dataclasses.replace(d, kept_material=ConstantImpedance(
        impedance_pa_s_m=1.1*d.kept_material.impedance_pa_s_m))
        if isinstance(d.kept_material, ConstantImpedance) else d for d in designs)


def main(output: Path) -> None:
    # 第一個控制用既有 _zone_rom_case 小房、同一份替身、同一組設計；不以六類可估性放寬低頻逐位要求。
    zone_model = _build(output.parent)
    problem = _problem(zone_model)
    zone_c = cases.constants(zone_scheme())
    zone_theta = jnp.asarray([.071, .063])
    controls = {'existing_zoned': control(zone_model, problem.designs, zone_c, zone_theta)}
    scheme = cases.scheme_case('zoned')
    zones = dict(scheme.scene.surface_zones_by_wall or {})
    assembly = cases.scheme_case('changed_wall').scene.surface_zones_by_wall
    assert assembly is not None
    zones['xL'] = assembly['x0']
    scheme = Scheme.model_validate({**scheme.model_dump(), 'scene':{**scheme.scene.model_dump(), 'surface_zones_by_wall':zones}})
    model = model_for(scheme)
    save_reduced(model, output.parent/'six-model.npz')
    calls = 0
    cached: list[SixCategoryConstants] = []
    original = build.build_constants
    def counted(scheme: build.SchemeInputs, *, pair_inputs: Mapping[tuple[str, str], ReportInput],
                quality_targets_path: Path = cases.TARGETS) -> SixCategoryConstants:
        nonlocal calls
        calls += 1
        if not cached:
            cached.append(original(scheme, pair_inputs=pair_inputs, quality_targets_path=quality_targets_path))
        return cached[0]
    materials, end_to_end = [], []
    numpy_on_wall: dict[str, object] = {}
    with (patch.object(build, 'build_constants', counted), patch.object(cases, 'build_constants', counted),
          patch.object(scoring, 'build_constants', counted), patch.object(chain, 'build_constants', counted, create=True)):
        c = cases.constants(scheme)
        build_calls = calls
        theta = jnp.asarray([.071, .063])
        rejected = rejections(model, candidate_designs(scheme, c, 'rigid'), c)
        template_axes = template_axes_measure(model, candidate_designs(scheme, c, 'rigid'), c)
        mounts: tuple[Mount, ...] = ('rigid', 'on_wall')
        for mount in mounts:
            designs = candidate_designs(scheme, c, mount)
            objective = factory(model, designs, c)
            row, end, inputs = material_measure(mount, model, objective, theta, scheme, designs, c)
            materials.append(row)
            end_to_end.append(end)
            controls[mount] = control(model, designs, c, theta)
            if mount == 'on_wall':
                numpy_on_wall = numpy_measure(scheme, designs, theta, inputs, end)
        # 梯度用既有 zoned 的固定阻抗原牆（#1057 起阻抗提高一成，見 gradient_designs），與上面的保留 Assembly 案例共用同一份幾何。
        # 原牆較反射，交接停在硬上限；不把 stop_gradient 擋掉的權重變化混進差分。
        designs = gradient_designs(cases.scheme_case('zoned'), c)
        objective = factory(model, designs, c)
        gradient_theta = jnp.asarray(old.baseline_theta(designs))
        # 厚度用既有基準，空腔用舊鏈第一個設計的梯度探點，不另訂探點或精度尺。
        gradient_theta = gradient_theta.at[1].set(_probe_point(tuple(d for d in designs if d.material is not None))[1])
        _, inputs = capture(objective, gradient_theta)
        gradients = gradient_measure(objective, gradient_theta, c, inputs)
        single = single_measure(objective, gradient_theta)
        invalid = invalid_measure(model, designs, c, gradient_theta)
        after = calls
    output.write_text(json.dumps(dict(controls=controls, materials=materials, end_to_end=end_to_end, gradients=gradients,
        single=single, invalid=invalid, rejections=rejected, build_calls=build_calls, build_calls_after_candidates=after,
        numpy_on_wall=numpy_on_wall, template_axes=template_axes), allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
