"""不可估小題：真下降、候選確認與換樣板演算法，僅物理求值用可控純函式。"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from aosr.gradient import material_gradient as g, material_gradient_loop as loop, six_category_chain as chain
from aosr.scoring.gradient_objective_jax import GradientObjectiveArrays
from aosr.scoring.six_category_jax import SixCategoryArrays, category_totals
from tests.engine.test_material_gradient import _designs, _settings, DESCENT_CASE
from aosr.gradient import run, material_gradient_ledger as ledger, template_rounds as rounds, template_segment as segment, template_ledger as templates
from tests.engine._material_gradient_loop_probe import synthetic_header, _start_ledger
from tests.engine._material_gradient_templates_probe import synthetic_header as template_header
from tests.engine.test_material_gradient_templates import _context


def aux(value: jax.Array, valid: jax.Array) -> chain.SixCategoryChainAux:
    zero = value * 0.
    cost = jnp.where(valid, value, jnp.nan)
    low = GradientObjectiveArrays(zero[None], zero[None], zero, zero, zero, zero, zero, zero)
    categories = SixCategoryArrays(cost, zero, zero, zero, zero, zero, valid, valid, valid, valid, valid, valid,
        zero, cost, cost, jnp.asarray(1), jnp.asarray(1), zero, jnp.asarray(False))
    # 故意保留有限 J/product，證明逐類閘門會拒收，不能只靠總分的 NaN 比較。
    return chain.SixCategoryChainAux(categories, low, value, value)



def template_nan(directory: Path, *, all_quick: bool, retune: bool) -> dict[str, object]:
    context = _context()
    settings = context.settings.model_copy(update={
        'templates':context.settings.templates.model_copy(update={'max_rounds':1, 'max_per_round':2}),
        'retune':context.settings.retune.model_copy(update={'steps':1})})
    context = context._replace(settings=settings)
    surfaces = tuple((name, 'glass_wool_48k_50mm_gap50') for name in g.wall_surface_ids())
    designs = rounds.designs_for_surfaces(surfaces, context)
    counts: dict[rounds.Surfaces, int] = {}
    first_surface, first_target = rounds.alternatives(surfaces, settings)[0]
    first_bad, _ = rounds.swap(designs, g.baseline_theta(designs), first_surface, first_target, context)
    bad_surfaces = rounds.surfaces_of(first_bad)
    def factory(items: tuple[g.SurfaceDesign, ...]) -> g.ChainObjective:
        signature = rounds.surfaces_of(items)
        counts[signature] = counts.get(signature, 0)+1
        good = signature == surfaces or (not all_quick and signature != bad_surfaces)
        if retune:
            good = signature == surfaces or counts[signature] == 1
        def objective(theta: jax.Array) -> tuple[jax.Array, chain.SixCategoryChainAux]:
            result = aux(jnp.sum(theta**2), jnp.asarray(good))
            return result.objective_total, result
        return objective
    context = context._replace(objective_for=factory)
    theta = g.snap_to_specs(g.baseline_theta(designs), designs).theta
    value, result = factory(designs)(jnp.asarray(theta))
    reading = g.objective_reading(result)
    start = templates.TemplateStart(surfaces, tuple(float(x).hex() for x in theta), float(value), reading.screening_total, reading.components)
    header = template_header(settings, surfaces, dict(context.kept))
    path = directory / ('retune.jsonl' if retune else 'all.jsonl' if all_quick else 'some.jsonl')
    # 同次對照固定耗時計：整本帳與結果（含秒數）逐位比；真實成本另在考卷外量。
    with patch.object(time, 'perf_counter', return_value=0.0):
        full = segment.run_template_rounds(context, path, header, start)
    rows = templates.read_template_ledger(path, settings=settings, start=start)[1]
    decision = rows[-1].decision
    assert decision is not None
    resumed_bitwise = None
    resumed_result_bitwise = None
    def result_bytes(result: segment.TemplateResult) -> bytes:
        payload = result._asdict()
        payload['theta'] = result.theta.tobytes().hex()
        return json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    if retune:
        expected = path.read_bytes()
        encoded = expected.splitlines(keepends=True)
        # 首個 quick 已帶整輪失效證據，但被拒的候選 quick 還沒落帳。
        assert any(name != rows[0].candidate for name in rows[0].rejected_candidates)
        path.write_bytes(b''.join(encoded[:2]))
        counts.clear()
        with patch.object(time, 'perf_counter', return_value=0.0):
            resumed = segment.resume_template_rounds(context, path, header, start)
        resumed_bitwise = expected == path.read_bytes()
        resumed_result_bitwise = result_bytes(full) == result_bytes(resumed)
    return dict(resumed_bitwise=resumed_bitwise, resumed_result_bitwise=resumed_result_bitwise, rejected_quick=decision.rejected_quick,
        rejected_retune=[item.candidate for item in decision.outcomes if item.rejection is not None],
        accepted=decision.accepted, first_quick=rows[0].rejected_quick, first_retune=rows[0].rejected_candidates,
        rows_finite=all(np.isfinite(row.value) and all(x is not None and np.isfinite(x) for x in row.components) for row in rows))

def main(output: Path) -> None:
    _, product, objective = category_totals(jnp.asarray([1.,2.,3.,4.,5.,7.]), jnp.ones(6,dtype=bool),
        jnp.asarray(5.), jnp.asarray([1.,1.,1.,1.,1.,.25]))
    settings = _settings()
    bounds = (np.asarray([0.]), np.asarray([1.]))
    def invalid(theta: jax.Array) -> tuple[jax.Array, chain.SixCategoryChainAux]:
        value = jnp.sum((theta - 1.)**2)
        result = aux(value, jnp.asarray(False))
        return result.objective_total, result
    start = g.projected_descent(invalid, np.asarray([.5]), bounds, settings.descent, limits=())
    def intermediate(theta: jax.Array) -> tuple[jax.Array, chain.SixCategoryChainAux]:
        value = jnp.sum((theta - 1.)**2)
        result = aux(value, theta[0] == .5)
        return result.objective_total, result
    middle = g.projected_descent(intermediate, np.asarray([.5]), bounds, settings.descent, limits=())
    per_category_start_stopped: list[bool] = []
    for category in SixCategoryArrays._fields[:6]:
        def one_bad_category(theta: jax.Array) -> tuple[jax.Array, chain.SixCategoryChainAux]:
            value = jnp.sum(theta**2)
            result = aux(value, jnp.asarray(True))
            result = result._replace(categories=result.categories._replace(**{category:jnp.asarray(jnp.nan)}))
            return result.objective_total, result
        checked = g.projected_descent(one_bad_category, np.asarray([.5]), bounds, settings.descent, limits=())
        per_category_start_stopped.append(not checked.steps and checked.stop_reason == g.STOP_UNESTIMABLE)
    designs = _designs(DESCENT_CASE)
    start_path = output.parent / 'invalid-start.jsonl'
    start_header = synthetic_header(designs, settings)
    ledger.write_ledger(start_path, start_header, ())
    problem = run.GradientProblem(designs, settings, invalid, start_header, {}, _context())
    try:
        run._pre_loop(problem, start_path, None)
    except g.Unestimable as error:
        start_ledger_reason = str(error)
    else:
        raise AssertionError('不可估起點仍進入搜尋')
    start_rows = len(ledger.read_ledger(start_path)[1])
    theta = g.snap_to_specs(g.baseline_theta(designs), designs).theta
    def candidate(point: jax.Array) -> tuple[jax.Array, chain.SixCategoryChainAux]:
        value = jnp.sum((point - 1.)**2)
        result = aux(value, jnp.all(point[::2] == jnp.asarray(theta)[::2]))
        return result.objective_total, result
    confirmed = loop.confirmation_round(candidate, theta, designs, settings, budget_left=settings.candidates.max_evaluations)
    path = output.parent / 'confirmed.jsonl'
    header = synthetic_header(designs, settings)
    _start_ledger(path, header, designs, candidate, theta)
    loop.run_rounds(candidate, theta, designs, settings, path, header)
    rows = ledger.read_ledger(path)[1]
    decision = rows[-1].decision
    assert decision is not None
    output.write_text(json.dumps(dict(weighted_totals=[float(objective),float(product)], templates=[template_nan(output.parent, all_quick=all_bad, retune=retune)
        for all_bad,retune in ((True,False),(False,False),(False,True))],
        candidate_ledger_finite=all(np.isfinite(row.value) for row in rows),
        candidate_ledger_reasons=[item.rejection for item in decision.outcomes if item.rejection],
        per_category_start_stopped=per_category_start_stopped, start_steps=len(start.steps), start_reason=start.stop_reason, start_rows=start_rows, start_ledger_reason=start_ledger_reason,
        middle_steps=len(middle.steps), middle_reason=middle.stop_reason,
        middle_rejected=middle.unestimable_trials, candidate_accepted=confirmed.accepted is not None,
        candidate_rejected=[item.retune.stop_reason for item in confirmed.confirmed if not item.retune.steps]), allow_nan=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
