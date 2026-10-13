"""分片跑法的逐格考卷；只改暫存控制樹，片名與錯誤欄位都要對上。"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import pytest
import yaml

from governance.checks import green_finalizer_mode as finalizer
from governance.checks import green_must_be_real_green as subject
from governance.exit_codes import TOOL_BROKEN, ToolBroken, run
from governance.loader import Card, load_card
from tests.conftest import GitSandbox
from tests.test_green_finalizer_inputs import SETTINGS_PATH
from tests.test_green_finalizer_mode import FIXTURES, files

CONTROL = FIXTURES / 'green-must-be-real-green-green' / 'shard-topology'
WORKFLOW = Path('.github') / 'workflows' / 'verify.yml'
REGISTRY = Path('governance') / 'verify-shards.toml'
FORMAL = 'governance/receipts/pytest.junit.xml'
ARGS = f'@{Path("governance") / "receipts" / "shard" / "pytest.args"}'
JUNIT = f'--junitxml={Path("governance") / "receipts" / "shard" / "junit.xml"}'
SCRIPT = Path('scripts') / 'runner.sh'
PYTEST = f'uv run pytest {ARGS} {JUNIT}'
SHARDS = ('shard-a', 'shard-b')


@pytest.fixture
def topology(tmp_path: Path) -> Path:
    """每題獨享可改寫的分片拓撲。"""
    return Path(shutil.copytree(CONTROL, tmp_path / 'topology'))


def card(root: Path) -> Card:
    return load_card(root / SETTINGS_PATH, root)


def jobs(root: Path) -> dict[str, dict[str, object]]:
    return cast(dict[str, dict[str, object]], yaml.safe_load((root / WORKFLOW).read_text())['jobs'])


def save_jobs(root: Path, value: dict[str, dict[str, object]]) -> None:
    (root / WORKFLOW).write_text(yaml.safe_dump({'jobs': value}, sort_keys=False))


def source(root: Path) -> list[str]:
    return subject._source_problems(card(root), root, files(root))


def job_steps(job: dict[str, object]) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], job['steps'])


def pytest_step(job: dict[str, object]) -> dict[str, str]:
    return cast(dict[str, str], next(step for step in job_steps(job) if 'pytest' in str(step.get('run', ''))))


def test_shard_topology_green_control_check() -> None:
    """PR2 收尾不寫正式收據，抄寫員外殼只算同一步 pytest。"""
    assert subject.check(CONTROL, files(CONTROL)) == []


@pytest.mark.parametrize('needs', [['shard-b', 'shard-a'], ['shard-a', 'shard-a', 'shard-b']])
def test_needs_compares_sets(topology: Path, needs: list[str]) -> None:
    value = jobs(topology)
    value['verify']['needs'] = needs
    save_jobs(topology, value)
    assert source(topology) == []


@pytest.mark.parametrize('needs', ['shard-a', ['shard-a'], ['shard-a', 'unknown'], ['shard-a', 'shard-b', 'unknown']])
def test_needs_requires_registered_set(topology: Path, needs: str | list[str]) -> None:
    value = jobs(topology)
    value['verify']['needs'] = needs
    save_jobs(topology, value)
    actual = [needs] if isinstance(needs, str) else sorted(needs)
    assert any(f"job=verify needs={actual!r}，預期 ['shard-a', 'shard-b']" in item for item in source(topology))


def test_string_needs_accepts_single_registered_shard(topology: Path) -> None:
    path = topology / REGISTRY
    path.write_text(path.read_text().replace('shards = ["shard-a", "shard-b"]', 'shards = ["shard-a"]')
                    .replace('default_shard = "shard-b"', 'default_shard = "shard-a"'))
    value = jobs(topology)
    value['verify']['needs'] = 'shard-a'
    value.pop('shard-b')
    save_jobs(topology, value)
    assert source(topology) == []


@pytest.mark.parametrize('shard', SHARDS)
def test_registered_job_must_be_in_same_workflow(topology: Path, shard: str) -> None:
    value = jobs(topology)
    other = value.pop(shard)
    save_jobs(topology, value)
    (topology / WORKFLOW.parent / 'elsewhere.yml').write_text(yaml.safe_dump({'jobs': {shard: other}}))
    assert any(f'片 {shard} job={shard}: 同一份 workflow 缺分片 job' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('extra', [False, True], ids=['none', 'multiple'])
def test_each_shard_has_exactly_one_pytest_step(topology: Path, shard: str, extra: bool) -> None:
    value = jobs(topology)
    step = pytest_step(value[shard])
    if extra:
        job_steps(value[shard]).append({'run': PYTEST})
    else:
        step['run'] = 'uv run python -m governance.status.shard_prepare'
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}: pytest 步驟' in item and '預期恰好一步' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('old,new,word', [(ARGS, '', '缺參數檔'), (JUNIT, '--junitxml=other.xml', '缺分片收據參數'),
                                       (JUNIT, f'--junitxml={FORMAL}', '出現正式收據路徑'),
                                       (JUNIT, f'{JUNIT} {FORMAL}', '出現正式收據路徑')])
def test_shard_pytest_paths_are_bound(topology: Path, shard: str, old: str, new: str, word: str) -> None:
    value = jobs(topology)
    step = pytest_step(value[shard])
    step['run'] = step['run'].replace(old, new)
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}' in item and word in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('flag,_why', subject.FORBIDDEN_FLAGS)
def test_shard_forbidden_flags(topology: Path, shard: str, flag: str, _why: str) -> None:
    value = jobs(topology)
    pytest_step(value[shard])['run'] += ' ' + flag + ' target'
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}' in item and f'帶了 {flag.strip()}' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('snippet', subject.SWALLOW_SNIPPETS)
def test_shard_swallowed_exit(topology: Path, shard: str, snippet: str) -> None:
    value = jobs(topology)
    pytest_step(value[shard])['run'] += ' ' + snippet
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}' in item and f'把離開碼吞掉了（{snippet!r}）' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('level', ['job', 'step'])
@pytest.mark.parametrize('setting', [True, 'true', 'True'])
def test_shard_continue_on_error(topology: Path, shard: str, level: str, setting: bool | str) -> None:
    value = jobs(topology)
    target = value[shard] if level == 'job' else job_steps(value[shard])[0]
    target['continue-on-error'] = setting
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}' in item and 'continue-on-error' in item and level in item for item in source(topology))


@pytest.mark.parametrize('level', ['job', 'step'])
def test_finalizer_continue_on_error(topology: Path, level: str) -> None:
    value = jobs(topology)
    target = value['verify'] if level == 'job' else job_steps(value['verify'])[-1]
    target['continue-on-error'] = True
    save_jobs(topology, value)
    assert any('收尾 job=verify' in item and 'continue-on-error' in item and level in item for item in source(topology))


@pytest.mark.parametrize('script', [False, True], ids=['run', 'called-script'])
def test_finalizer_never_runs_pytest(topology: Path, script: bool) -> None:
    value = jobs(topology)
    if script:
        (topology / 'scripts').mkdir()
        (topology / 'scripts' / 'runner.sh').write_text('uv run pytest\n')
    job_steps(value['verify']).append({'run': f'bash {SCRIPT}' if script else 'uv run pytest'})
    save_jobs(topology, value)
    assert any('收尾 job=verify' in item and '不准跑 pytest' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
def test_shard_called_script_counts_as_its_step(topology: Path, shard: str) -> None:
    value = jobs(topology)
    (topology / 'scripts').mkdir()
    (topology / 'scripts' / 'runner.sh').write_text(PYTEST + '\n')
    pytest_step(value[shard])['run'] = f'bash {SCRIPT}'
    save_jobs(topology, value)
    assert source(topology) == []
    (topology / 'scripts' / 'runner.sh').write_text(PYTEST + ' --deselect test\n')
    assert any(f'片 {shard} job={shard}' in item and '帶了 --deselect' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
def test_script_step_binds_arguments_and_checks_caller(topology: Path, shard: str) -> None:
    """呼叫端參數不能冒充 pytest 段自己的參數；呼叫端正式路徑仍須擋。"""
    value = jobs(topology)
    (topology / 'scripts').mkdir()
    (topology / 'scripts' / 'runner.sh').write_text('uv run pytest "$@"\n')
    pytest_step(value[shard])['run'] = f'bash {SCRIPT} {ARGS} {JUNIT}'
    save_jobs(topology, value)
    assert any(f'片 {shard}' in item and '缺參數檔' in item for item in source(topology))
    step = cast(dict[str, str], job_steps(value[shard])[3])
    step['run'] += f' {FORMAL}'
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}' in item and '出現正式收據路徑' in item for item in source(topology))


@pytest.mark.parametrize('name,marker', subject.RIVAL_CONFIGS)
def test_shard_topology_still_rejects_rival_configs(topology: Path, name: str, marker: str) -> None:
    (topology / name).write_text(marker + '\n')
    assert any(f'多了一份 pytest 設定 {name}' in item for item in source(topology))


@pytest.mark.parametrize('name', ['tox.ini', 'setup.cfg'])
def test_other_tool_config_is_not_pytest_config(topology: Path, name: str) -> None:
    (topology / name).write_text('[other-tool]\n')
    assert source(topology) == []


@pytest.mark.parametrize('missing', ['settings', 'registry'])
def test_topology_without_registry_authority_is_broken(topology: Path, missing: str) -> None:
    if missing == 'settings':
        path = topology / SETTINGS_PATH
        path.write_text(path.read_text().split('[settings]')[0])
    else:
        (topology / REGISTRY).unlink()
    with pytest.raises(ToolBroken, match='settings|registry.*讀不開'):
        source(topology)


@pytest.mark.parametrize('needs', [None, 7, ['shard-a', 7]])
def test_unreadable_needs_is_tool_broken(topology: Path, needs: object) -> None:
    value = jobs(topology)
    value['verify']['needs'] = needs
    save_jobs(topology, value)
    with pytest.raises(ToolBroken, match='job=verify.*needs'):
        source(topology)


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('wrong', [None, {}, [7], [{'run': 7}]])
def test_unreadable_shard_steps_is_tool_broken(topology: Path, shard: str, wrong: object) -> None:
    value = jobs(topology)
    value[shard]['steps'] = wrong
    save_jobs(topology, value)
    with pytest.raises(ToolBroken, match=f'片 {shard} job={shard}.*steps'):
        source(topology)


def real_root_env(git_sandbox: GitSandbox, topology: Path, job_name: str = 'verify') -> dict[str, str]:
    """真根只由 git_sandbox 造；不把樣本身分檔複製進真根。"""
    shutil.copytree(topology / WORKFLOW.parent, git_sandbox.root / WORKFLOW.parent)
    (git_sandbox.root / 'governance').mkdir()
    shutil.copyfile(topology / REGISTRY, git_sandbox.root / REGISTRY)
    value = jobs(git_sandbox.root)
    value[job_name] = value.pop('verify')
    save_jobs(git_sandbox.root, value)
    git_sandbox.git('commit', '--allow-empty', '-m', '拓撲守門沙箱')
    return {'GITHUB_JOB': job_name, 'GITHUB_REPOSITORY': 'owner/repo',
            'GITHUB_WORKFLOW_REF': f'owner/repo/{WORKFLOW}@refs/pull/9/merge'}


@pytest.mark.parametrize('job_name', ['verify', 'verify-sharded'])
def test_real_finalizer_missing_trigger_returns_tool_broken(git_sandbox: GitSandbox, topology: Path, job_name: str,
                                                           monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    env = real_root_env(git_sandbox, topology, job_name)
    monkeypatch.setattr(finalizer, 'process_env', lambda: env)
    settings = finalizer.read_settings(card(topology))
    assert settings is not None
    with pytest.raises(ToolBroken, match=f'收尾 job={job_name} 沒帶觸發變數.*退回單機模式拿正式收據判綠'):
        finalizer.resolve_run(git_sandbox.root, settings)
    result = run(lambda root, _files: [] if finalizer.resolve_run(root, settings) is None else ['收尾'],
                 ['--scan-root', str(git_sandbox.root)])
    assert result == TOOL_BROKEN
    assert 'FAIL(2)' in capsys.readouterr().err


@pytest.mark.parametrize('local', [False, True], ids=['single-job', 'no-github-job'])
def test_non_finalizer_without_trigger_stays_single(git_sandbox: GitSandbox, topology: Path,
                                                   monkeypatch: pytest.MonkeyPatch, local: bool) -> None:
    env = real_root_env(git_sandbox, topology)
    if local:
        env.pop('GITHUB_JOB')
    else:
        value = jobs(git_sandbox.root)
        value['verify'].pop('needs')
        save_jobs(git_sandbox.root, value)
        (git_sandbox.root / REGISTRY).unlink()
    monkeypatch.setattr(finalizer, 'process_env', lambda: env)
    settings = finalizer.read_settings(card(topology))
    assert settings is not None
    assert finalizer.resolve_run(git_sandbox.root, settings) is None
