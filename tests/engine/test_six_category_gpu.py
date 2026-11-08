"""六類單精度鏈的 CPU 考卷；評分邊界、動態常數、帳本與新尺。"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from aosr.runtime import child_process_env, configure_jax
from tests.engine._precision_contracts import contract_value
from tests.engine._scheme_cache import shared_json


@pytest.fixture(scope='module')
def measured(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> dict[str, object]:
    def produce() -> str:
        directory = tmp_path_factory.mktemp('six-category-gpu')
        output = directory/'measure.json'
        env = child_process_env(threads=1)
        configure_jax(platforms='cpu', enable_x64=True, env=env)
        env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'], str(Path(__file__).resolve().parents[2])))
        done = subprocess.run([sys.executable, '-m', 'tests.engine._six_category_gpu_probe', str(output)],
            cwd=directory, env=env, capture_output=True, text=True, timeout=300)
        if done.returncode != 0:
            return json.dumps({'probe_error':done.stdout+done.stderr})
        return output.read_text()
    data = cast(dict[str, object], json.loads(shared_json(tmp_path_factory, worker_id, 'six-category-gpu', produce)))
    assert 'probe_error' not in data,data.get('probe_error')
    return data


def test_single_low_frequency_control_is_bitwise(measured: dict[str, object]) -> None:
    assert measured['control_equal'] == [True]*3, measured['control_hex']


def test_kept_assembly_fem_tmm_is_single(tmp_path: Path) -> None:
    env = child_process_env(threads=1)
    configure_jax(platforms='cpu',enable_x64=True,env=env)
    env['PYTHONPATH'] = os.pathsep.join((env['PYTHONPATH'],str(Path(__file__).resolve().parents[2])))
    done = subprocess.run([sys.executable,'-m','tests.engine._six_category_gpu_probe','--kept'],
        cwd=tmp_path,env=env,capture_output=True,text=True,timeout=120)
    assert done.returncode == 0,done.stdout+done.stderr
    result = json.loads(done.stdout)
    assert result['bitwise'] is True and result['tmm_calls'] == [['float32','complex64']]


def test_single_six_categories_and_j_within_contract(measured: dict[str, object]) -> None:
    errors = cast(list[float], measured['absolute_differences'])
    assert all(value <= contract_value('six_category_search_gpu_single_vs_cpu_double') for value in errors), errors
    assert measured['gradient_dtype'] == 'float64' and measured['gradient_finite'] is True


def test_single_score_inputs_are_double_and_aux_preserves_flags(measured: dict[str, object]) -> None:
    assert measured['input_dtypes'] == ['complex128']*5
    assert measured['material_dtypes'] == ['complex64']*4
    assert measured['pressure_dtype'] == 'complex64'
    assert measured['aux_preserves_types'] is True


def test_single_large_score_constants_are_double_dynamic_arguments(measured: dict[str, object]) -> None:
    assert measured['dynamic_names'] == measured['expected_dynamic_names']
    assert set(cast(list[str], measured['dynamic_dtypes'])) == {'float64', 'complex128'}
    assert measured['embedded_large_arrays'] == []
    assert measured['all_dots_highest'] is True
    assert measured['dynamic_changed_output'] is True


def test_gpu_program_path_writes_six_category_search_ledger_and_result(measured: dict[str, object]) -> None:
    assert measured['gpu_wrapper'] is True
    assert measured['candidate_gpu_wrapper'] is True
    assert cast(int, measured['ledger_rows']) > 0
    assert measured['identities_checked'] is True and measured['result_written'] is True


def test_six_category_precision_gate_and_tampered_difference() -> None:
    from aosr.gradient.gpu import PrecisionComparison, require_precision, six_category_precision_comparison
    ruler = contract_value('six_category_search_gpu_single_vs_cpu_double')
    def comparison(delta: float) -> PrecisionComparison:
        return six_category_precision_comparison(cpu_double=0., gpu_single=delta, theta=(.05,),
            cpu_product=1., gpu_product=2., cpu_categories=(0.,)*6, gpu_categories=(1.,)*6)
    # product／六類的差只記錄，不參與閘門；J 的邊界是含等號。
    for delta in (0., ruler/2, ruler):
        item = comparison(delta)
        require_precision(item)
        assert item.contract == 'six_category_search_gpu_single_vs_cpu_double'
        assert item.absolute_difference == delta
        assert item.product_absolute_difference == 1. and item.category_absolute_differences == (1.,)*6
    for item in (comparison(ruler*2), comparison(ruler/2).model_copy(update={'absolute_difference':0.}),
                 comparison(0.).model_copy(update={'product_absolute_difference':0.}),
                 comparison(0.).model_copy(update={'category_absolute_differences':(0.,)*6})):
        with pytest.raises(ValueError):
            require_precision(item)


def test_mutant_beyond_six_category_precision_contract_is_red(measured: dict[str, object]) -> None:
    from aosr.gradient.gpu import require_precision, six_category_precision_comparison
    item = six_category_precision_comparison(cpu_double=cast(float, measured['double_j']),
        gpu_single=cast(float, measured['mutated_j']), theta=(.05,), cpu_product=0., gpu_product=0.,
        cpu_categories=(0.,)*6, gpu_categories=(0.,)*6)
    with pytest.raises(ValueError, match='objective_total'):
        require_precision(item)


def test_six_category_gate_never_reads_legacy_db_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.gradient import gpu
    from aosr.config.precision_contracts import load_precision_contracts, default_precision_contracts_path
    registry = load_precision_contracts(default_precision_contracts_path())
    # 兩把尺目前同值；移除舊鍵才真能抓錯讀，不另訂數字。
    available = {name:value for name,value in registry.items() if name != gpu.CONTRACT}
    monkeypatch.setattr(gpu,'load_precision_contracts',lambda path:available)
    gpu.require_precision(gpu.six_category_precision_comparison(cpu_double=0.,gpu_single=0.,theta=(),
        cpu_product=0.,gpu_product=0.,cpu_categories=(0.,)*6,gpu_categories=(0.,)*6))


@pytest.mark.parametrize('mode', ('start','resume'))
@pytest.mark.parametrize('accepted', (True,False))
def test_six_category_gate_precedes_ledger_and_persists_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            mode: str, accepted: bool) -> None:
    from types import SimpleNamespace
    from aosr.gradient import gpu_preflight, run
    from aosr.gradient.gpu import six_category_precision_comparison
    from tests.engine.test_material_gradient_mount import _project
    ruler = contract_value('six_category_search_gpu_single_vs_cpu_double')
    comparison = six_category_precision_comparison(cpu_double=0.,gpu_single=ruler/2 if accepted else ruler*2,
        theta=(.05,),cpu_product=1.,gpu_product=2.,cpu_categories=(0.,)*6,gpu_categories=(1.,)*6)
    project = _project().model_copy(update={'device':'gpu'})
    run.write_json_atomic(tmp_path/run.PROJECT_FILE,project)
    monkeypatch.setattr(gpu_preflight,'measure_start',lambda *args:comparison)
    monkeypatch.setattr(run,'load_problem',lambda project:cast(run.GradientProblem,SimpleNamespace(header=None)))
    class ReachedLedger(Exception):
        pass
    def reached(*args: object, **kwargs: object) -> None:
        raise ReachedLedger
    monkeypatch.setattr(run,'write_ledger',reached)
    entry = run.run_search if mode=='start' else run.resume_search
    with pytest.raises(ReachedLedger if accepted else ValueError):
        entry(tmp_path,should_stop=None)
    assert run.read_project(tmp_path).precision_check == comparison
    assert not (tmp_path/run.LEDGER_FILE).exists()


def test_old_precision_record_is_readable_but_six_search_cannot_resume() -> None:
    from aosr.gradient import gpu_preflight, run
    from aosr.gradient.gpu import precision_comparison
    from tests.engine.test_material_gradient_mount import _project
    from aosr.scoring.gradient_objective_settings import gradient_objective_settings
    from aosr.config.quality_targets import load_quality_targets
    from aosr.config.paths import config_path
    old = precision_comparison(cpu_double=0., gpu_single=0., theta=(.05,))
    project = _project().model_copy(update={'device':'gpu', 'precision_check':old,
        'objective_settings_fingerprint':gradient_objective_settings(load_quality_targets(config_path('quality_targets.toml')),
                                                                  _project().purpose).fingerprint})
    loaded = run.GradientProject.model_validate_json(project.model_dump_json())
    assert loaded.precision_check is not None and loaded.precision_check.contract == 'rom_search_gpu_single_vs_cpu_double'
    with pytest.raises(ValueError, match='舊.*契約.*重跑'):
        run.check_inputs(loaded)


@pytest.mark.parametrize('cached', (False,True))
def test_current_six_project_refuses_double_objective_even_with_cached_precision(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool) -> None:
    from types import SimpleNamespace
    from aosr.gradient import gpu_preflight, run
    from aosr.gradient.gpu import six_category_precision_comparison
    from tests.engine.test_material_gradient_mount import _project
    project = _project().model_copy(update={'device':'gpu',
        'objective_settings_fingerprint':run.search_objective_settings(_project().purpose).fingerprint})
    if cached:
        comparison = six_category_precision_comparison(cpu_double=0.,gpu_single=0.,theta=(),
            cpu_product=0.,gpu_product=0.,cpu_categories=(0.,)*6,gpu_categories=(0.,)*6)
        project = project.model_copy(update={'precision_check':comparison})
    def double(theta: object) -> tuple[float,object]:
        return 0.,object()
    problem = cast(run.GradientProblem,SimpleNamespace(objective=double))
    monkeypatch.setattr(gpu_preflight,'_start_theta',lambda problem:())
    monkeypatch.setattr(gpu_preflight,'cpu_reference',lambda *args:0.)
    with pytest.raises(ValueError,match='六類.*單精度包裝'):
        gpu_preflight.ensure_start_precision(tmp_path,project,problem)
    assert not (tmp_path/run.PROJECT_FILE).exists()
