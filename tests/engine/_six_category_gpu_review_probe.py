"""B2b 修補控制：多點低頻同源、帳本身分、保留構造與四路材料編譯圖。"""
from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.construction_environment import construction_environment
from aosr.geometry.shoebox import Wall
from aosr.gradient import material_gradient as g, material_gradient_ledger as ledger, run
from aosr.gradient import six_category_chain as chain
from aosr.gradient.six_category_gpu import ScoreArrays, SixCategorySinglePrecisionObjective
from aosr.materials.construction_models import Assembly
from aosr.physics.fem_reduced import ReducedModel
from aosr.scoring.six_category_jax import SixCategoryConstants
from aosr.scoring import six_category_jax as scoring
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays, GradientObjectiveConstants, gradient_objective_from_pressures
from tests.engine._six_category_gpu_probe import factory, problem_case
from tests.engine import _six_category_jax_cases as cases


def wide_material_arithmetic(single: SixCategorySinglePrecisionObjective, designs: tuple[g.SurfaceDesign, ...],
                             c: SixCategoryConstants, theta: jax.Array, room: ReducedModel) -> list[str]:
    air, ranges = construction_environment(density_kg_m3=room.density_kg_m3,sound_speed_m_s=room.sound_speed_m_s)
    impedance = chain._impedance_function(designs,c,air=air,ranges=ranges,single_precision=True)
    def material(point: jax.Array, arrays: ScoreArrays) -> tuple[jax.Array, ...]:
        bind = cast(Callable[...,SixCategoryConstants],c._replace)
        scoring = bind(**dict(zip(single.dynamic_names,arrays,strict=True)))
        return impedance(point,scoring)
    # 可執行 HLO：lower 的未最佳化圖還有 1j*rho_c 的閉包常數乘法，編譯時即折成常數。
    text = jax.jit(material).lower(theta.astype(jnp.float32),single.score_arrays).compile().as_text()
    assert text is not None
    arithmetic = {'add','subtract','multiply','divide','remainder','power','exponential','exponential-minus-one',
        'log','log-plus-one','sqrt','rsqrt','sine','cosine','tan','tanh','atan2','abs','negate','sign',
        'maximum','minimum','clamp','compare','complex','real','imag','dot','convolution','reduce',
        'cbrt','ceil','floor','round-nearest-afz','round-nearest-even','logistic'}
    operations = [(op.group(1),op.group(2),line.strip()) for line in text.splitlines()
        if (op := re.search(r'= (f32|f64|c64|c128)\[[^]]*\](?:\{[^}]*\})? ([\w-]+)\(',line))]
    assert any(dtype in ('f32','c64') and op in arithmetic for dtype,op,_ in operations)
    return [line for dtype,op,line in operations if dtype in ('f64','c128') and op in arithmetic]


def score_error(c: SixCategoryConstants, theta: g.FloatVector, value: jax.Array, aux: chain.SixCategoryChainAux) -> str | None:
    reading = g.objective_reading(aux)
    row = ledger._row(phase='descent',step=0,theta_hex=tuple(float(x).hex() for x in theta),
        value=float(value),screening_total=reading.screening_total,components=reading.components,
        gradient_norm=0.,step_size=0.,seconds=0.)
    try:
        ledger.check_scores(ledger.score_constants(c),[row])
    except ValueError as invalid:
        return str(invalid)
    return None


def measure(kept_assembly: bool) -> dict[str, object]:
    scheme,room,settings,project,c = problem_case()
    kept = {w:(scheme.scene.surface_zones_by_wall or {})[w.wall_name()][0].material for w in Wall.all()}
    if kept_assembly:
        project = project.model_copy(update={'presets':tuple('glass_wool_48k_50mm_gap50' if w.wall_name()=='xL' else None for w in Wall.all())})
    designs = run.designs_for(project.presets,settings,kept=kept)
    single = factory(room,designs,c,True)
    assert isinstance(single,SixCategorySinglePrecisionObjective)
    double = jax.jit(factory(room,designs,c,False))
    lower,upper = g.design_bounds(designs)
    start = np.asarray(g.descent_start(g.baseline_theta(designs),(lower,upper),g.depth_limits(designs)))
    samples = [start,*(g.project_to_limits(point,lower,upper,g.depth_limits(designs))
        for point in np.random.default_rng(562).uniform(lower,upper,size=(8,len(start))))]
    low_calls = []
    def observe_low(constants: GradientObjectiveConstants, pressures: jax.Array) -> GradientObjectiveArrays:
        low_calls.append(True)
        return gradient_objective_from_pressures(constants,pressures)
    with patch.object(chain,'gradient_objective_from_pressures',observe_low), patch.object(scoring,'gradient_objective_from_pressures',observe_low):
        single(jnp.asarray(start))
    points=[]
    for theta in samples:
        (value,aux),_ = single.evaluate(jnp.asarray(theta))
        forward_value,forward_aux = single(jnp.asarray(theta))
        double_value,double_aux = double(jnp.asarray(theta))
        reading = g.objective_reading(aux)
        error = score_error(c,theta,value,aux)
        forward_error = score_error(c,theta,forward_value,forward_aux)
        points.append(dict(low_hex=[float(aux.low_frequency.objective_total).hex(),float(aux.categories.low_frequency_objective).hex()],
            low_bitwise=np.asarray(aux.low_frequency.objective_total).tobytes()==np.asarray(aux.categories.low_frequency_objective).tobytes(),
            forward_low_bitwise=np.asarray(forward_aux.low_frequency.objective_total).tobytes()==np.asarray(forward_aux.categories.low_frequency_objective).tobytes(),
            forward_scores_checked=forward_error is None,forward_score_error=forward_error,
            differences=[abs(float(a)-float(b)) for a,b in zip((*aux.categories[:6],value),(*double_aux.categories[:6],double_value),strict=True)],
            double_hex=[float(x).hex() for x in (*double_aux.categories[:6],double_value,*g.objective_reading(double_aux).components[:3])],
            low_components=[float(x) for x in reading.components[:3]],scores_checked=error is None,score_error=error))
    return dict(points=points,low_objective_traces=len(low_calls),kept_assembly=any(isinstance(d.kept_material,Assembly) and d.material is None for d in designs),
        wide_arithmetic=wide_material_arithmetic(single,designs,c,jnp.asarray(start),room))


def trace_count(kept_assembly: bool) -> int:
    scheme,room,settings,project,c = problem_case()
    kept = {w:(scheme.scene.surface_zones_by_wall or {})[w.wall_name()][0].material for w in Wall.all()}
    if kept_assembly:
        project = project.model_copy(update={'presets':tuple('glass_wool_48k_50mm_gap50' if w.wall_name()=='xL' else None for w in Wall.all())})
    designs = run.designs_for(project.presets,settings,kept=kept)
    single = factory(room,designs,c,True)
    assert isinstance(single,SixCategorySinglePrecisionObjective)
    calls = []
    def observe(constants: GradientObjectiveConstants, pressures: jax.Array) -> GradientObjectiveArrays:
        calls.append(True)
        return gradient_objective_from_pressures(constants,pressures)
    with patch.object(chain,'gradient_objective_from_pressures',observe), patch.object(scoring,'gradient_objective_from_pressures',observe):
        single.compiled.lower(jnp.asarray(g.baseline_theta(designs)),single.constants,single.score_arrays)
    return len(calls)


def invalid_pressure_control() -> dict[str, object]:
    scheme,_,_,_,c = problem_case()
    fem,normal,early,paris,base,flags = cases.array_inputs(scheme,c)
    def objective(scale: jax.Array) -> tuple[jax.Array,chain.SixCategoryChainAux]:
        pressure = base*scale[0]
        low = gradient_objective_from_pressures(c.objective(),jnp.transpose(pressure,(2,0,1)))
        categories = scoring._six_category_costs(c,fem,normal,early,paris,pressure,flags,low_frequency=lambda _:low.objective_total)
        aux = chain.SixCategoryChainAux(categories,low,categories.objective_total,categories.product_total)
        return aux.objective_total,aux
    evaluate = jax.jit(jax.value_and_grad(objective,has_aux=True))
    rows: dict[str,object] = {}
    for name,scale in (('valid',1.),('zero',0.),('nan',float('nan'))):
        (value,aux),gradient = evaluate(jnp.asarray([scale]))
        rows[name] = dict(estimable=g.estimable(value,aux),value_nan=bool(jnp.isnan(value)),
            gradient_finite=bool(jnp.all(jnp.isfinite(gradient))))
    return rows


if __name__=='__main__':
    trace,invalid = sys.argv[1]=='--trace',sys.argv[1]=='--invalid'
    data = invalid_pressure_control() if invalid else {name:trace_count(kept) if trace else measure(kept)
        for name,kept in (('normal',False),('kept',True))}
    Path(sys.argv[2] if trace or invalid else sys.argv[1]).write_text(json.dumps(data,allow_nan=False))
