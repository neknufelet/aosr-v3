"""B2a 審查修補探針：真帳本、接續、worker 與 CLI；物理求值只在混合拒收題用合成函式。"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.material_gradient import MaterialGradientSettings
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gradient import cli, material_gradient as g, material_gradient_ledger as ledger
from aosr.gradient import material_gradient_loop as loop, report, run, template_ledger as tl
from aosr.gradient import template_rounds as tr, template_segment as ts, worker
from aosr.reporting.validation import checked_inputs
from aosr.scoring.gradient_objective_settings import legacy_gradient_objective_settings
from tests.engine._material_gradient_loop_probe import synthetic_header, _start_ledger
from tests.engine._material_gradient_templates_probe import synthetic_header as template_header
from tests.engine._six_search_nan_probe import aux
from tests.engine.test_material_gradient import DESCENT_CASE, _designs, _settings
from tests.engine.test_material_gradient_cli import _schemes, _start_args
from tests.engine.test_material_gradient_mount import _project
from tests.engine.test_material_gradient_templates import _context


def result_bytes(result: loop.LoopResult | ts.TemplateResult) -> bytes:
    payload = result._asdict()
    payload['theta'] = result.theta.tobytes().hex()
    return json.dumps(payload, sort_keys=True, allow_nan=False).encode()


def mixed_template(directory: Path) -> dict[str, object]:
    context = _context()
    settings = context.settings.model_copy(update={
        'templates':context.settings.templates.model_copy(update={'max_rounds':1,'max_per_round':2}),
        'retune':context.settings.retune.model_copy(update={'steps':1})})
    context = context._replace(settings=settings)
    surfaces = tuple((name,'glass_wool_48k_50mm_gap50') for name in g.wall_surface_ids())
    designs = tr.designs_for_surfaces(surfaces,context)
    signatures = [tr.surfaces_of(tr.swap(designs,g.baseline_theta(designs),face,target,context)[0])
                  for face,target in tr.alternatives(surfaces,settings)]
    counts: dict[tr.Surfaces,int] = {}
    def factory(items: tuple[g.SurfaceDesign,...]) -> g.ChainObjective:
        signature = tr.surfaces_of(items)
        counts[signature] = counts.get(signature,0)+1
        cost = 1. if signature==signatures[0] else 2. if signature==signatures[1] else 4.
        valid = not (signature==signatures[0] and counts[signature]>1)
        def objective(theta: jax.Array) -> tuple[jax.Array,g.ObjectiveAux]:
            value = jnp.sum(theta*0.)+cost
            return value,aux(value,jnp.asarray(valid))
        return objective
    context = context._replace(objective_for=factory)
    theta = g.snap_to_specs(g.baseline_theta(designs),designs).theta
    value,a = factory(designs)(jnp.asarray(theta))
    reading = g.objective_reading(a)
    start = tl.TemplateStart(surfaces,tuple(float(x).hex() for x in theta),float(value),reading.screening_total,reading.components)
    header = template_header(settings,surfaces,dict(context.kept))
    full_path, interrupted_path = directory/'template-full.jsonl',directory/'template-resumed.jsonl'
    original = tl.append_template_rows
    def append(path: Path, given: tl.TemplateHeader, rows: list[tl.TemplateRow], *,
               settings: MaterialGradientSettings | None=None, start: tl.TemplateStart | None=None) -> None:
        original(path,given,rows,settings=settings,start=start)
        if any(row.phase=='candidate_retune' for row in rows):
            raise g.SearchStopped('first candidate batch persisted')
    with patch.object(time,'perf_counter',return_value=0.):
        full = ts.run_template_rounds(context,full_path,header,start)
        counts.clear()
        try:
            with patch.object(ts,'append_template_rows',side_effect=append):
                ts.run_template_rounds(context,interrupted_path,header,start)
        except g.SearchStopped:
            pass
        else:
            raise AssertionError('candidate batch interruption did not occur')
        prefix = tl.read_template_ledger(interrupted_path,settings=settings,start=start)[1]
        counts.clear()
        resumed = ts.resume_template_rounds(context,interrupted_path,header,start)
    rows = tl.read_template_ledger(full_path,settings=settings,start=start)[1]
    decision = rows[-1].decision
    assert decision is not None
    return dict(first_rejected=decision.outcomes[0].rejection is not None,
        second_valid=decision.outcomes[1].rejection is None, selected=decision.accepted,
        second=decision.outcomes[1].candidate, prefix_last=prefix[-1].phase,
        candidate_batches=len({r.candidate for r in prefix if r.phase=='candidate_retune'}),
        ledger_bitwise=full_path.read_bytes()==interrupted_path.read_bytes(),
        result_bitwise=result_bytes(full)==result_bytes(resumed))


def mixed_material(directory: Path) -> dict[str,object]:
    settings = _settings()
    settings = settings.model_copy(update={
        'candidates':settings.candidates.model_copy(update={'max_rounds':1,'max_per_round':2}),
        'retune':settings.retune.model_copy(update={'steps':1})})
    designs = _designs(DESCENT_CASE)
    theta = g.snap_to_specs(g.baseline_theta(designs),designs).theta
    candidates = loop.candidates_from_gradient(theta,2*(theta-1.),designs,settings)
    assert len(candidates)==2
    bad = loop.apply_candidate(theta,candidates[0],designs)
    def objective(point: jax.Array) -> tuple[jax.Array,g.ObjectiveAux]:
        value = jnp.sum((point-1.)**2)
        valid = ~jnp.all(point[::2]==jnp.asarray(bad)[::2])
        return value,aux(value,valid)
    header = synthetic_header(designs,settings)
    full_path,interrupted_path = directory/'material-full.jsonl',directory/'material-resumed.jsonl'
    original = ledger.append_rows
    def append(path: Path, given: ledger.MaterialGradientHeader, rows: list[ledger.MaterialGradientRow]) -> None:
        original(path,given,rows)
        if any(row.phase=='candidate_retune' for row in rows):
            raise g.SearchStopped('first candidate batch persisted')
    with patch.object(time,'perf_counter',return_value=0.):
        _start_ledger(full_path,header,designs,objective,theta)
        full = loop.run_rounds(objective,theta,designs,settings,full_path,header)
        _start_ledger(interrupted_path,header,designs,objective,theta)
        try:
            with patch.object(loop,'append_rows',side_effect=append):
                loop.run_rounds(objective,theta,designs,settings,interrupted_path,header)
        except g.SearchStopped:
            pass
        else:
            raise AssertionError('candidate batch interruption did not occur')
        prefix = ledger.read_ledger(interrupted_path)[1]
        resumed = loop.resume_rounds(objective,designs,settings,interrupted_path,header)
    rows = ledger.read_ledger(full_path)[1]
    decision = rows[-1].decision
    assert decision is not None
    return dict(first_rejected=decision.outcomes[0].rejection is not None,
        second_valid=decision.outcomes[1].rejection is None, selected=decision.accepted,
        second=decision.outcomes[1].candidate, prefix_last=prefix[-1].phase,
        candidate_batches=len({r.candidate for r in prefix if r.phase=='candidate_retune'}),
        ledger_bitwise=full_path.read_bytes()==interrupted_path.read_bytes(),
        result_bitwise=result_bytes(full)==result_bytes(resumed))


def worker_case(directory: Path, phase: str) -> dict[str,object]:
    directory.mkdir()
    scheme_path = directory/'scheme.json'
    scheme_path.write_text(_schemes()['scheme.json'].model_dump_json())
    project = _project().model_copy(update={'scheme_path':str(scheme_path),
        'settings_path':str(config_path('material_gradient.toml')), 'capabilities_path':str(config_path('capabilities.toml'))})
    run.write_json_atomic(directory/run.PROJECT_FILE,project)
    designs,settings = _designs(DESCENT_CASE),_settings()
    settings = settings.model_copy(update={'retune':settings.retune.model_copy(update={'steps':1}),
        'candidates':settings.candidates.model_copy(update={'max_rounds':1,'max_per_round':2})})
    initial = jnp.asarray(g.baseline_theta(designs))
    def objective(theta: jax.Array) -> tuple[jax.Array,g.ObjectiveAux]:
        value = 4.+(jnp.sqrt(jnp.sum((theta-initial)**2)) if phase=='gradient' else jnp.sqrt(jnp.sum(theta-initial)) if phase=='gradient_inf' else jnp.sum(theta*0.))
        return value,aux(value,jnp.asarray(phase!='start'))
    problem = run.GradientProblem(designs,settings,objective,synthetic_header(designs,settings),{},_context())
    output = io.StringIO()
    codes=[]
    with patch.object(run,'load_problem',return_value=problem),contextlib.redirect_stderr(output):
        with contextlib.ExitStack() as stack:
            if phase=='template':
                stack.enter_context(patch.object(run,'_template_segment',side_effect=g.Unestimable('換樣板基準重調起點不可估')))
            for mode in ('start','resume'):
                codes.append(worker.run_worker(directory,mode,should_stop=lambda:False))
    status = run.GradientStatus.model_validate_json((directory/run.STATUS_FILE).read_bytes())
    rows = ledger.read_ledger(directory/run.LEDGER_FILE)[1]
    return dict(codes=codes,state=status.state,reason=status.stop_reason,message=status.message,
        stderr=output.getvalue(), report=report.render_report(directory),rows=len(rows))


def cli_cases(directory: Path) -> dict[str,object]:
    recorded: list[Path] = []
    calls: list[int] = []
    def capture(root: Path, project: run.GradientProject) -> int:
        folder = root/project.gradient_id
        folder.mkdir(parents=True)
        with patch.object(run,'checked_inputs',wraps=checked_inputs) as checked:
            problem = run.load_problem(project)
            calls.append(checked.call_count)
        scheme,model,settings = run.check_inputs(project)
        low = run.search_objective_constants(project,scheme,model)
        old_header = ledger.header_for(model,problem.designs,settings,objective_constants=low)
        (folder/run.LEDGER_FILE).write_text(json.dumps(old_header.model_dump(mode='json'),ensure_ascii=False)+'\n')
        previous = legacy_gradient_objective_settings(load_quality_targets(config_path('quality_targets.toml')),project.purpose)
        run.write_json_atomic(folder/run.PROJECT_FILE,project.model_copy(update={'objective_settings_fingerprint':previous.fingerprint}))
        recorded.append(folder)
        return 0
    with patch.object(cli,'_create_and_run',side_effect=capture):
        code = cli.main(_start_args(directory,directory/'captured')[3:])
    assert code==0
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        resumed = cli.main(['resume',str(recorded[0])])
    gpu_root = directory/'gpu-root'
    gpu_wrappers: list[bool] = []
    def child(folder: Path, mode: str, descriptor: int, *, device: str='cpu') -> int:
        from aosr.gradient.six_category_gpu import SixCategorySinglePrecisionObjective
        project = run.read_project(folder)
        problem = run.load_problem(project)
        gpu_wrappers.append(device == project.device == 'gpu' and isinstance(problem.objective,SixCategorySinglePrecisionObjective))
        return 0
    gpu_stderr = io.StringIO()
    with patch.object(cli,'_spawn_and_wait',side_effect=child),contextlib.redirect_stderr(gpu_stderr):
        gpu = cli.main([*_start_args(directory,gpu_root)[3:],'--device','gpu'])
    return dict(checked_calls=calls,resume_code=resumed,resume_stderr=stderr.getvalue(),gpu_code=gpu,
        gpu_stderr=gpu_stderr.getvalue(),gpu_root_exists=gpu_root.exists(),gpu_wrappers=gpu_wrappers,
        gpu_projects=[str(p) for p in gpu_root.glob(f'*/{run.PROJECT_FILE}')])


if __name__=='__main__':
    mode,location = sys.argv[1:]
    directory=Path(location)
    data: dict[str,object]
    if mode=='mixed': data={'template':mixed_template(directory),'material':mixed_material(directory)}
    elif mode=='worker': data={name:worker_case(directory/name,name) for name in ('start','gradient','gradient_inf','template')}
    elif mode=='cli': data=cli_cases(directory)
    else: raise ValueError(mode)
    (directory/(mode+'.json')).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False))
