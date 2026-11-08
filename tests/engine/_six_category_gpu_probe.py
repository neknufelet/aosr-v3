"""CPU 演顯卡路徑：真六類與小矩陣，不使用顯卡或真房。"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.config.material_gradient import MaterialGradientSettings
from aosr.config.paths import config_path
from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.geometry.shoebox import Wall
from aosr.materials.construction_diff import build_template, surface_impedance_from_parameters
from aosr.materials.construction_models import Assembly, SurfaceMaterial
from aosr.materials.construction_diff import ConstructionTemplate
from aosr.gradient import material_gradient as g, material_gradient_ledger as ledger, run
from aosr.gradient import gpu_preflight, six_category_chain as chain
from aosr.gradient.gpu import PrecisionReading
from aosr.gradient.six_category_gpu import SixCategorySinglePrecisionObjective
from aosr.physics.fem_reduced import ReducedModel
from aosr.physics.fem_reduced_jax import ReducedConstants, reduced_pressures
from aosr.reporting.scheme import Scheme
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays
from aosr.scoring.six_category_jax import SixCategoryArrays, SixCategoryConstants, six_category_costs
from tests.engine import _six_category_jax_cases as cases
from tests.engine._material_gradient_gpu_probe import model
from tests.engine.test_material_gradient import _settings
from tests.engine.test_material_gradient_mount import _project


def problem_case() -> tuple[Scheme, ReducedModel, MaterialGradientSettings, run.GradientProject, SixCategoryConstants]:
    scheme = cases.scheme_case('changed_wall')
    c = cases.constants(scheme)
    room = dataclasses.replace(model(), frequencies_hz=tuple(float(x) for x in c.fem_frequencies_hz),
        source_names=tuple(ch.speaker_id for ch in scheme.channel_group.channels),
        receiver_names=tuple(p.receiver_id for p in scheme.receiver_set.points))
    settings = _settings()
    settings = settings.model_copy(update={
        'descent':settings.descent.model_copy(update={'max_steps':1}),
        'retune':settings.retune.model_copy(update={'steps':1}),
        'candidates':settings.candidates.model_copy(update={'max_rounds':1,'max_per_round':1,'max_evaluations':1}),
        'templates':settings.templates.model_copy(update={'max_rounds':1,'max_per_round':1,'max_evaluations':1})})
    presets = tuple('glass_wool_48k_50mm_gap50' if wall.wall_name()=='x0' else None for wall in Wall.all())
    project = _project().model_copy(update={'device':'gpu', 'presets':presets,
        'capabilities_path':str(config_path('capabilities.toml'))})
    return scheme, room, settings, project, c


def factory(room: ReducedModel, designs: tuple[g.SurfaceDesign, ...], c: SixCategoryConstants,
            single: bool) -> chain.SixCategoryChainObjective:
    air, ranges = construction_environment(density_kg_m3=room.density_kg_m3, sound_speed_m_s=room.sound_speed_m_s)
    return chain.six_category_chain_objective(room, designs, c.objective(), c, air=air, ranges=ranges,
        frequencies=room.frequencies_hz, frequency_batch=16, single_precision=single)


def kept_material_precision() -> dict[str, object]:
    scheme,room,settings,_,c = problem_case()
    air,ranges = construction_environment(density_kg_m3=room.density_kg_m3,sound_speed_m_s=room.sound_speed_m_s)
    kept = {w:(scheme.scene.surface_zones_by_wall or {})[w.wall_name()][0].material for w in Wall.all()}
    presets = tuple('glass_wool_48k_50mm_gap50' if wall.wall_name()=='xL' else None for wall in Wall.all())
    designs = run.designs_for(presets,settings,kept=kept)
    index = next(i for i,d in enumerate(designs) if d.surface_id=='x0')
    material = designs[index].kept_material
    assert isinstance(material,Assembly)
    axis = jnp.asarray(c.fem_frequencies_hz,dtype=jnp.float32)
    template = build_template(material,jnp.asarray(c.fem_frequencies_hz),air=air,ranges=ranges,variables=())
    expected = surface_impedance_from_parameters(template,{},axis,air=air)
    calls: list[list[str]] = []
    original = chain._angle_impedance
    def observe(material: SurfaceMaterial, template: ConstructionTemplate | None, parameters: dict[str,jax.Array],
                axis: jax.Array, angles: jax.Array | None, *, air: PhysicsConstants) -> jax.Array:
        value = original(material,template,parameters,axis,angles,air=air)
        if isinstance(material,Assembly) and angles is None and axis.size == c.fem_frequencies_hz.size:
            calls.append([str(axis.dtype),str(value.dtype)])
        return value
    with patch.object(chain,'_angle_impedance',observe):
        impedance = chain._impedance_function(designs,c,air=air,ranges=ranges,single_precision=True)
        result = impedance(jnp.asarray(g.baseline_theta(designs),dtype=jnp.float32),c)[0][index]
    return dict(bitwise=np.asarray(result).tobytes()==np.asarray(expected).tobytes(),tmm_calls=calls)


def boundary_types(room: ReducedModel, designs: tuple[g.SurfaceDesign, ...], c: SixCategoryConstants,
                   theta: jax.Array) -> dict[str, object]:
    inputs: list[str] = []
    materials: list[str] = []
    pressures: list[str] = []
    original_impedances = chain._impedance_function
    def observe_impedances(designs: tuple[g.SurfaceDesign, ...], c: SixCategoryConstants, *,
                           air: PhysicsConstants, ranges: PorousModelRanges, single_precision: bool = False) -> chain.ImpedanceFunction:
        impedance = original_impedances(designs, c, air=air, ranges=ranges, single_precision=single_precision)
        def call(theta: jax.Array, arrays: SixCategoryConstants) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
            values = impedance(theta, arrays)
            materials.extend(str(x.dtype) for x in values)
            return values
        return call
    def observe_score(c: SixCategoryConstants, *args: jax.Array) -> SixCategoryArrays:
        inputs.extend(str(x.dtype) for x in args[:5])
        return six_category_costs(c, *args)
    def observe_pressure(reduced: ReducedConstants, z: jax.Array, *, rigid_mask: jax.Array,
                         frequency_batch: int | None) -> jax.Array:
        result = reduced_pressures(reduced,z,rigid_mask=rigid_mask,frequency_batch=frequency_batch)
        pressures.append(str(result.dtype))
        return result
    with patch.object(chain,'_impedance_function',observe_impedances), patch.object(chain,'six_category_costs',observe_score), patch.object(chain,'reduced_pressures',observe_pressure):
        factory(room,designs,c,True)(theta)
    return dict(input_dtypes=inputs,material_dtypes=materials,pressure_dtype=pressures[0])


def dynamic_arrays(single: SixCategorySinglePrecisionObjective, theta: jax.Array,
                   value: jax.Array, c: SixCategoryConstants) -> dict[str, object]:
    text = single.compiled.lower(theta, single.constants, single.score_arrays).as_text()
    signature = text.split('func.func public @main',1)[1].split('{',1)[0]
    expected_names = [name for name, x in zip(c._fields,c,strict=True) if x.ndim and x.dtype.kind in 'fc']
    shapes = [f"{'x'.join(str(n) for n in x.shape)}x{'complex<f64>' if x.dtype.kind=='c' else 'f64'}"
              for x in (c.form_factors,c.phase,c.response_tilt_smoothing,c.response_ripple_smoothing)]
    assert all(shape in signature for shape in shapes)
    embedded = [line for line in text.splitlines() if 'stablehlo.constant' in line and any(f'tensor<{shape}>' in line for shape in shapes)]
    changed = list(single.score_arrays)
    index = single.dynamic_names.index('category_weights')
    changed[index] = changed[index]*2
    changed_value = single.forward(theta, single.constants, tuple(changed))[0]
    return dict(dynamic_names=list(single.dynamic_names),expected_dynamic_names=expected_names,
        dynamic_dtypes=[str(x.dtype) for x in single.score_arrays],embedded_large_arrays=embedded,
        all_dots_highest=all('HIGHEST' in line for line in text.splitlines() if 'stablehlo.dot_general' in line),
        dynamic_changed_output=float(changed_value)!=float(value))


def search_path(output: Path, scheme: Scheme, room: ReducedModel, settings: MaterialGradientSettings,
                project: run.GradientProject) -> dict[str, object]:
    with patch.object(run,'check_inputs',return_value=(scheme,room,settings)):
        problem = run.load_problem(project)
        candidate = problem.templates.objective_for(run.designs_for((None,)*6,settings,kept=problem.kept))
    assert isinstance(problem.objective,SixCategorySinglePrecisionObjective)
    assert isinstance(candidate,SixCategorySinglePrecisionObjective)
    assert candidate.constants is problem.objective.constants
    assert candidate.score_arrays is problem.objective.score_arrays
    folder = output.parent/'search'
    folder.mkdir()
    run.write_json_atomic(folder/run.PROJECT_FILE,project)
    def reference(project: run.GradientProject, point: tuple[float, ...]) -> PrecisionReading:
        c = cases.constants(scheme)
        value,aux = factory(room,problem.designs,c,False)(jnp.asarray(point))
        return PrecisionReading.model_validate(dict(objective_total=float(value),product_total=float(aux.product_total),
            categories=tuple(float(x) for x in aux.categories[:6])))
    with patch.object(run,'check_inputs',return_value=(scheme,room,settings)), patch.object(gpu_preflight,'cpu_six_category_reference',side_effect=reference):
        run.run_search(folder,should_stop=None)
    header,rows = ledger.read_ledger(folder/run.LEDGER_FILE)
    assert header.score_constants is not None
    ledger.check_scores(header.score_constants,rows)
    return dict(gpu_wrapper=True,candidate_gpu_wrapper=True,ledger_rows=len(rows),identities_checked=True,
                result_written=(folder/run.RESULT_FILE).is_file())


def numeric(output: Path) -> dict[str, object]:
    scheme, room, settings, project, c = problem_case()
    air, ranges = construction_environment(density_kg_m3=room.density_kg_m3, sound_speed_m_s=room.sound_speed_m_s)
    kept = {w:(scheme.scene.surface_zones_by_wall or {})[w.wall_name()][0].material for w in Wall.all()}
    designs = run.designs_for(project.presets, settings, kept=kept)
    single = factory(room,designs,c,True)
    assert isinstance(single, SixCategorySinglePrecisionObjective)
    theta = jnp.asarray(g.baseline_theta(designs))
    old = g.chain_objective(room, designs, c.objective(), air=air, ranges=ranges,
        frequencies=room.frequencies_hz, frequency_batch=16, single_precision=True)
    (old_value, old_aux), _ = g.compiled_evaluator(old)(theta)
    (value, aux), derivative = g.compiled_evaluator(single)(theta)
    (double, double_aux), _ = g.compiled_evaluator(factory(room,designs,c,False))(theta)
    assert isinstance(old_aux,GradientObjectiveArrays) and isinstance(aux,chain.SixCategoryChainAux)
    assert isinstance(double_aux,chain.SixCategoryChainAux)
    low = (aux.low_frequency.mean_own_line_ripple_rms_db, aux.low_frequency.mean_seat_robustness_rms_db, aux.low_frequency.big_peak_proxy_db)
    control = (old_aux.mean_own_line_ripple_rms_db, old_aux.mean_seat_robustness_rms_db, old_aux.big_peak_proxy_db)
    leaves, expected = jax.tree.leaves(aux), jax.tree.leaves(double_aux)
    mutated_matrix = room.matrices.stiffness.copy()
    mutated_matrix[0,0] *= 2
    mutant = dataclasses.replace(room,matrices=dataclasses.replace(room.matrices,stiffness=mutated_matrix))
    mutated = factory(mutant,designs,c,True)(theta)[0]
    return dict(control_equal=[np.asarray(a).tobytes()==np.asarray(b).tobytes() for a,b in zip(low,control,strict=True)],
        control_hex=[[float(a).hex(),float(b).hex()] for a,b in zip(low,control,strict=True)],
        absolute_differences=[abs(float(a)-float(b)) for a,b in zip((*aux.categories[:6],value,aux.product_total),(*double_aux.categories[:6],double,double_aux.product_total),strict=True)],
        double_j=float(double),single_j=float(value),mutated_j=float(mutated),
        gradient_dtype=str(derivative.dtype),gradient_finite=bool(jnp.all(jnp.isfinite(derivative))),
        aux_preserves_types=all(a.dtype == b.dtype for a,b in zip(leaves,expected,strict=True)),
        **boundary_types(room,designs,c,theta),**dynamic_arrays(single,theta,value,c),
        **search_path(output,scheme,room,settings,project))


if __name__ == '__main__':
    if sys.argv[1] == '--kept':
        sys.stdout.write(json.dumps(kept_material_precision())+'\n')
    else:
        output = Path(sys.argv[1])
        output.write_text(json.dumps(numeric(output),allow_nan=False))
