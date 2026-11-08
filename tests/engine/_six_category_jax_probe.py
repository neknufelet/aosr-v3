"""冷行程雙精度實量；答案經 solve_reports、evaluate_parts、排名正式路徑。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

from aosr.scoring.gradient_objective_jax import gradient_objective_constants, gradient_objective_from_pressures
from aosr.scoring.gradient_objective_settings import gradient_objective_settings
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.six_category_jax import SixCategoryConstants, six_category_costs, _path, _early
from aosr.physics.amplitude_jax import bounce_impedances_by_patch
from aosr.physics.geometric_early_jax import room_scattering
from aosr.physics.reflection_energy_jax import reflection_path_energy
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from tests.engine import _six_category_jax_cases as cases
from tests.engine.test_material_gradient import DIFFERENCE_STEP


def measure(name: str) -> tuple[dict[str, object], object, tuple[jax.Array, ...]]:
    scheme = cases.scheme_case(name)
    c = cases.constants(scheme)
    inputs = cases.array_inputs(scheme, c)
    result = six_category_costs(c, *inputs)
    result.objective_total.block_until_ready()
    reference = cases.numpy_result(scheme, np.asarray(inputs[-1]))
    values, total = cases.numpy_costs(reference)
    reflection = next(e for e in reference.candidate.evaluations if e.category.value=='reflections_and_echo')
    assert isinstance(reflection.payload, ReflectionsAndEchoPayload)
    coverage = [channel.coverage for channel in reflection.payload.channels if channel.is_primary]
    run = jax.jit(lambda *args: six_category_costs(c, *args))
    start = perf_counter()
    compiled = run(*inputs)
    compiled.objective_total.block_until_ready()
    seconds = perf_counter()-start
    single = run(*(x.astype(jnp.complex64) for x in inputs))
    single.objective_total.block_until_ready()
    direct_constants = gradient_objective_constants(c.fem_frequencies_hz,
        gradient_objective_settings(load_quality_targets(cases.TARGETS), scheme.purpose), scheme.receiver_set, 2)
    pressure = jnp.transpose(inputs[-1], (2, 0, 1))
    direct = gradient_objective_from_pressures(direct_constants, pressure).objective_total
    direct_jit = jax.jit(lambda p: gradient_objective_from_pressures(direct_constants, p).objective_total)(pressure)
    record: dict[str, object] = dict(reference=values.tolist(), reference_total=total,
        double=np.asarray(result[:6]).tolist(), compiled=np.asarray(compiled[:6]).tolist(),
        single=np.asarray(single[:6]).tolist(), product_total=float(result.product_total),
        objective_total=float(result.objective_total), flags=np.asarray(result[6:12]).tolist(),
        dtypes=[str(x.dtype) for x in result[:6]], single_dtypes=[str(x.dtype) for x in single[:6]],
        low_bitwise=np.asarray(result.low_frequency_objective).tobytes()==np.asarray(direct).tobytes(),
        low_jit_bitwise=np.asarray(compiled.low_frequency_objective).tobytes()==np.asarray(direct_jit).tobytes(),
        compile_s=seconds, reflection_coverage=coverage, weights=c.category_weights.tolist(), fs=float(result.schroeder_frequency_hz),
        reflection_orders=[int(max(c.order[int(c.path_offsets[int(i)]):int(c.path_offsets[int(i)+1])])) for i in c.primary_pair_indices])
    return record, c, inputs


def reflection_margin(c: SixCategoryConstants, inputs: tuple[jax.Array, ...]) -> float:
    rows = tuple(inputs[2][:,int(a):int(b)] for a,b in zip(c.angle_offsets[:-1],c.angle_offsets[1:],strict=True))
    margins, offset = [], 0
    scattering, valid = room_scattering(_early(c), inputs[1], jnp.asarray(c.scattering_by_patch))
    assert bool(jnp.all(valid))
    for index in c.primary_pair_indices:
        table = _path(c, int(index))
        a,b = (int(v) for v in c.path_offsets[int(index):int(index)+2])
        energy = np.asarray(reflection_path_energy(table, bounce_impedances_by_patch(table, rows),
            scattering, float(c.rho_c), jnp.asarray(c.directivity[a:b])).energy)
        for mask in c.reflection_zone_mask[:,offset:offset+table.path_count]:
            if np.count_nonzero(mask) >= 2:
                ordered = np.sort(energy[mask], axis=0)
                used = (c.reflection_frequency_weights>0) & (ordered[-1]>0)
                margins.extend(((ordered[-1,used]-ordered[-2,used])/ordered[-1,used]).tolist())
        offset += table.path_count
    return float(min(margins))


def gradients() -> dict[str, object]:
    scheme = cases.scheme_case('zoned')
    c, inputs = cases.constants(scheme), None
    inputs = cases.array_inputs(scheme, c)
    def value(theta: jax.Array) -> jax.Array:
        # 各種阻抗與 FEM 聲壓各一個方向；法向 FEM 阻抗只作資格檢查，聲壓已由上游提供。
        perturbed = [x*(1.+theta[i]) for i,x in enumerate(inputs)]
        ripple = jnp.sin(jnp.log2(jnp.asarray(c.fem_frequencies_hz)))
        perturbed[-1] = inputs[-1]*(1.+theta[-1]*ripple[None,None,:])
        r = six_category_costs(c, *perturbed)
        return jnp.stack((r.objective_total, r.product_total))
    origin = jnp.zeros(len(inputs), dtype=jnp.float64)
    evaluate = jax.jit(value)
    automatic = np.asarray(jax.jit(jax.jacrev(value))(origin))
    central = np.empty_like(automatic)
    for i in range(len(inputs)):
        delta = origin.at[i].set(DIFFERENCE_STEP)
        central[:, i] = (np.asarray(evaluate(delta))-np.asarray(evaluate(-delta)))/(2*DIFFERENCE_STEP)
    return dict(automatic=automatic.tolist(), central=central.tolist(), step=DIFFERENCE_STEP, reflection_margin=reflection_margin(c, inputs))


def invalid_inputs() -> list[dict[str, object]]:
    scheme = cases.scheme_case('whole')
    c = cases.constants(scheme)
    inputs = cases.array_inputs(scheme, c)
    variants = [c._replace(response_insufficient_coverage=np.asarray(True)),
        c._replace(timbre_primary_pairs_weights=np.zeros_like(c.timbre_primary_pairs_weights)),
        c._replace(reflection_coverage=np.zeros_like(c.reflection_coverage)),
        c._replace(reverb_weights=np.zeros_like(c.reverb_weights)),
        c._replace(matching_weights=np.zeros_like(c.matching_weights)), c]
    records = []
    for index, constants in enumerate(variants):
        args = list(inputs)
        if index == 5:
            args[0] = args[0].at[0,0].set(jnp.nan)
        r = six_category_costs(constants, *args)
        def surviving(theta: jax.Array) -> jax.Array:
            varied = [arg*(1.+theta) for arg in args]
            evaluated = six_category_costs(constants, *varied)
            return jnp.nansum(jnp.stack(evaluated[:6]))
        derivative = jax.jit(jax.grad(surviving))(jnp.asarray(0., dtype=jnp.float64))
        records.append(dict(valid_gradient_finite=bool(jnp.isfinite(derivative)), flags=np.asarray(r[6:12]).tolist(), nan=bool(jnp.isnan(r[index])),
            product_nan=bool(jnp.isnan(r.product_total)), objective_nan=bool(jnp.isnan(r.objective_total))))
    return records


def main(output: Path) -> None:
    records = {}
    for name in ('whole', 'zoned', 'furniture'):
        record, _, _ = measure(name)
        records[name] = record
    scheme = cases.scheme_case('unavailable')
    c = cases.constants(scheme)
    inputs = cases.array_inputs(scheme, c)
    r = six_category_costs(c, *inputs)
    numpy = cases.numpy_result(scheme, np.asarray(inputs[-1]))
    reflection = next(e for e in numpy.candidate.evaluations if e.category.value=='reflections_and_echo')
    unavailable = dict(jax_valid=bool(r.reflections_valid), jax_nan=bool(jnp.isnan(r.reflections_cost)),
        product_nan=bool(jnp.isnan(r.product_total)), objective_nan=bool(jnp.isnan(r.objective_total)),
        numpy_state=reflection.state.value, numpy_cost_missing=reflection.category_cost is None)
    invalid = invalid_inputs()
    output.write_text(json.dumps(dict(cases=records, unavailable=unavailable, gradients=gradients(), invalid=invalid), allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
