"""並跑施工契約：沿用單機環境與檢查，pytest 命令及收尾登記必須對帳。"""
from __future__ import annotations

import shlex
import tomllib
from collections import Counter
from pathlib import Path
from typing import cast

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _workflow(name: str) -> dict[object, object]:
    """只讀版控設定，不執行 workflow 或寫進工作樹。"""
    path = ROOT / '.github' / 'workflows' / name
    assert path.is_file(), f'並跑契約缺少 {name}'
    return cast(dict[object, object], yaml.safe_load(path.read_text()))


def _jobs(workflow: dict[object, object]) -> dict[str, dict[str, object]]:
    """YAML 的工作表，只供本檔的契約斷言讀取。"""
    return cast(dict[str, dict[str, object]], workflow['jobs'])


def _steps(job: dict[str, object]) -> list[dict[str, object]]:
    """工作步驟保留所有欄位，供逐格對帳。"""
    return cast(list[dict[str, object]], job['steps'])


def _recorded(job: dict[str, object]) -> dict[str, dict[str, object]]:
    """用抄寫員的名字找原步驟；不得同一工作內重複登記。"""
    result: dict[str, dict[str, object]] = {}
    for step in _steps(job):
        argv = shlex.split(str(step.get('run', '')))
        if '--name' in argv:
            name = argv[argv.index('--name') + 1]
            assert name not in result, f'重複抄寫步驟 {name}'
            result[name] = step
    return result


def test_parallel_workflow_preserves_single_environment() -> None:
    single = _workflow('verify.yml')
    parallel = _workflow('verify-shards.yml')
    assert parallel['name'] == 'verify-shards'
    assert parallel[True] == single[True]
    assert parallel['permissions'] == single['permissions']
    for shard in ('shard-a', 'shard-b'):
        job = _jobs(parallel)[shard]
        assert job['env'] == _jobs(single)['verify']['env']
        setup = [step for step in _steps(job) if str(step.get('uses', '')).startswith(('actions/checkout@', 'astral-sh/setup-uv@')) or str(step.get('run', '')).startswith(('sudo ', 'uv sync'))]
        expected = [step for step in _steps(_jobs(single)['verify']) if str(step.get('uses', '')).startswith(('actions/checkout@', 'astral-sh/setup-uv@')) or str(step.get('run', '')).startswith(('sudo ', 'uv sync'))]
        assert setup == expected


def test_parallel_governance_steps_match_single_exactly_once() -> None:
    single = _recorded(_jobs(_workflow('verify.yml'))['verify'])
    parallel = _jobs(_workflow('verify-shards.yml'))
    expected = {name: step for name, step in single.items() if name not in {'pytest', 'install_browser', 'green_must_be_real_green'}}
    seen: Counter[str] = Counter()
    for shard in ('shard-a', 'shard-b'):
        records = _recorded(parallel[shard])
        for name, step in records.items():
            if name not in {'pytest', 'prepare', 'seal', 'install_browser'}:
                assert step == expected[name]
                seen[name] += 1
    assert seen == Counter(expected.keys())
    records = _recorded(parallel['shard-a'])
    assert records['install_browser'] == single['install_browser']
    assert 'install_browser' not in _recorded(parallel['shard-b'])


def test_parallel_pytest_and_finalizer_match_registries() -> None:
    parallel = _jobs(_workflow('verify-shards.yml'))
    settings = tomllib.loads((ROOT / 'governance/rules/green-must-be-real-green.toml').read_text())['settings']
    registry = tomllib.loads((ROOT / 'governance/verify-shards.toml').read_text())
    evidence = registry['evidence']
    junit = f"{evidence['dir']}/{evidence['junit']}"
    args = f"{evidence['dir']}/{evidence['args']}"
    for shard in registry['shards']:
        records = _recorded(parallel[shard])
        argv = shlex.split(str(records['pytest']['run']))
        assert argv[argv.index('--') + 1:] == shlex.split(settings['pytest_command']) + [f'@{args}', f'--junitxml={junit}']
        assert argv[argv.index('--output') + 1] == junit
        assert list(records).index('prepare') < list(records).index('pytest') < list(records).index('seal')
        assert not records['pytest'].get('env')
    ci = tomllib.loads((ROOT / 'governance/rules/ci-jobs-cannot-die-quietly.toml').read_text())['settings']
    finalizer = parallel['verify-sharded']
    registration = ci['finalizer_jobs']['verify-sharded']
    assert set(cast(list[str], finalizer['needs'])) == set(registry['shards'])
    assert finalizer['if'] == registration['job_if']
    assert all(step['if'] == registration['step_if'] for step in _steps(finalizer))
    verdict = _recorded(finalizer)['green_must_be_real_green']
    assert verdict['run'] == registration['verdict_run']
    assert cast(dict[str, str], verdict['env'])[registration['verdict_env_name']] == registration['verdict_env_value']
