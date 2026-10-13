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
ARGS = f'@{Path("governance") / "receipts" / "shard" / "pytest.args"}'
JUNIT = f'--junitxml={Path("governance") / "receipts" / "shard" / "junit.xml"}'
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
