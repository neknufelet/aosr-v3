"""封存身分与位元組；所有環境與 HEAD 都由考卷注入。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken
from governance.status.shard_seal import seal
from governance.verify_shards import load_registry
from tests.test_verify_shards import write_registry

ENV = {
    'GITHUB_REPOSITORY': 'owner/repo',
    'GITHUB_WORKFLOW_REF': f'owner/repo/{Path(".github") / "workflows" / "verify.yml"}@refs/pull/9/merge',
    'GITHUB_EVENT_NAME': 'pull_request',
    'GITHUB_RUN_ID': '123',
    'GITHUB_RUN_ATTEMPT': '2',
    'GITHUB_JOB': 'shard-a',
    'GITHUB_SHA': 'a' * 40,
}
CONTENTS = {'expected.txt': b'tests/test_verify_shards.py::test_a\n', 'pytest.args': b'tests/test_verify_shards.py\n',
            'junit.xml': b'<testsuites/>\n'}


def evidence(tmp_path: Path) -> Path:
    out = tmp_path / 'evidence'
    out.mkdir()
    for file, content in CONTENTS.items():
        (out / file).write_bytes(content)
    return out


def test_manifest_identity_and_exact_byte_hashes(tmp_path: Path) -> None:
    registry = load_registry(write_registry(tmp_path))
    out = evidence(tmp_path)
    seal(registry, 'shard-a', out, env=ENV, head=lambda: 'b' * 40)
    manifest = json.loads((out / registry.evidence.manifest).read_text())
    expected: dict[str, object] = {
        'schema': 1, 'shard': 'shard-a', 'repository': ENV['GITHUB_REPOSITORY'],
        'workflow_path': f'owner/repo/{Path(".github") / "workflows" / "verify.yml"}', 'workflow_ref': 'refs/pull/9/merge',
        'event': ENV['GITHUB_EVENT_NAME'], 'run_id': ENV['GITHUB_RUN_ID'],
        'run_attempt': ENV['GITHUB_RUN_ATTEMPT'], 'job': ENV['GITHUB_JOB'],
        'github_sha': ENV['GITHUB_SHA'], 'tested_sha': 'b' * 40,
    }
    for key, file in [('expected', 'expected.txt'), ('args', 'pytest.args'), ('junit', 'junit.xml')]:
        content = CONTENTS[file]
        expected[key] = {'path': file, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    assert manifest == expected


@pytest.mark.parametrize('key', list(ENV))
@pytest.mark.parametrize('value', [None, '', '  '])
def test_env_required(tmp_path: Path, key: str, value: str | None) -> None:
    env = dict(ENV)
    if value is None:
        env.pop(key)
    else:
        env[key] = value
    with pytest.raises(ToolBroken, match=key):
        seal(load_registry(write_registry(tmp_path)), 'shard-a', evidence(tmp_path), env=env, head=lambda: 'b' * 40)


@pytest.mark.parametrize('key', ['GITHUB_RUN_ID', 'GITHUB_RUN_ATTEMPT'])
@pytest.mark.parametrize('value', ['0', '-1', '1.5', 'x', ' 1', '1\n', '01', '１'])
def test_run_numbers_are_positive_decimal_strings(tmp_path: Path, key: str, value: str) -> None:
    with pytest.raises(ToolBroken, match=key):
        seal(load_registry(write_registry(tmp_path)), 'shard-a', evidence(tmp_path),
             env=ENV | {key: value}, head=lambda: 'b' * 40)


@pytest.mark.parametrize('key', ['github_sha', 'tested_sha'])
@pytest.mark.parametrize('value', ['a' * 39, 'a' * 41, 'A' * 40, 'g' * 40, 'a' * 40 + '\n', ''])
def test_shas_are_full_lowercase_hex(tmp_path: Path, key: str, value: str) -> None:
    env = ENV | {'GITHUB_SHA': value} if key == 'github_sha' else ENV
    head = value if key == 'tested_sha' else 'b' * 40
    with pytest.raises(ToolBroken):
        seal(load_registry(write_registry(tmp_path)), 'shard-a', evidence(tmp_path), env=env, head=lambda: head)


@pytest.mark.parametrize('file', list(CONTENTS))
def test_all_evidence_files_required(tmp_path: Path, file: str) -> None:
    out = evidence(tmp_path)
    (out / file).unlink()
    with pytest.raises(ToolBroken, match=file):
        seal(load_registry(write_registry(tmp_path)), 'shard-a', out, env=ENV, head=lambda: 'b' * 40)


def test_manifest_cannot_be_overwritten(tmp_path: Path) -> None:
    out = evidence(tmp_path)
    target = out / 'manifest.json'
    target.write_bytes(b'old evidence')
    with pytest.raises(ToolBroken, match='已經存在'):
        seal(load_registry(write_registry(tmp_path)), 'shard-a', out, env=ENV, head=lambda: 'b' * 40)
    assert target.read_bytes() == b'old evidence'


@pytest.mark.parametrize('value', ['workflow-without-ref', '@refs/heads/main', 'workflow@', 'a@b@c',
                                   'owner/repo/verify.yml@heads/main', 'owner/repo/verify.yml@'])
def test_workflow_ref_must_split_unambiguously(tmp_path: Path, value: str) -> None:
    with pytest.raises(ToolBroken, match='GITHUB_WORKFLOW_REF'):
        seal(load_registry(write_registry(tmp_path)), 'shard-a', evidence(tmp_path),
             env=ENV | {'GITHUB_WORKFLOW_REF': value}, head=lambda: 'b' * 40)


def test_workflow_ref_keeps_legal_at_inside_ref(tmp_path: Path) -> None:
    """分支名可以合法帶 @（git check-ref-format refs/heads/review@format 回 0）；路徑在 .yml 為止。"""
    registry = load_registry(write_registry(tmp_path))
    out = evidence(tmp_path)
    path = f'owner/repo/{Path(".github") / "workflows" / "verify.yml"}'
    seal(registry, 'shard-a', out, env=ENV | {'GITHUB_WORKFLOW_REF': f'{path}@refs/heads/review@format'},
         head=lambda: 'b' * 40)
    manifest = json.loads((out / registry.evidence.manifest).read_text())
    assert (manifest['workflow_path'], manifest['workflow_ref']) == (path, 'refs/heads/review@format')


def test_unknown_shard_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ToolBroken, match='shard'):
        seal(load_registry(write_registry(tmp_path)), 'missing', evidence(tmp_path), env=ENV, head=lambda: 'b' * 40)


@pytest.mark.parametrize('bad', ['env', 'root'])
def test_cli_errors_return_two_with_injected_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    from governance.status.shard_seal import main
    registry = write_registry(tmp_path)
    out = evidence(tmp_path)
    monkeypatch.chdir(tmp_path)
    if bad != 'root':
        (tmp_path / 'pyproject.toml').write_text('')
        (tmp_path / 'tests').mkdir()
    assert main(['--registry', str(registry), '--shard', 'shard-a', '--out', str(out)],
                env={}, head=lambda: 'b' * 40) == TOOL_BROKEN


def test_cli_default_out_uses_registry_and_injected_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from governance.status.shard_seal import main
    registry = load_registry(write_registry(tmp_path))
    (tmp_path / 'pyproject.toml').write_text('')
    (tmp_path / 'tests').mkdir()
    out = tmp_path / registry.evidence.dir
    out.mkdir(parents=True)
    for file, content in CONTENTS.items():
        (out / file).write_bytes(content)
    monkeypatch.chdir(tmp_path)
    assert main(['--registry', 'verify-shards.toml', '--shard', 'shard-a'], env=ENV, head=lambda: 'b' * 40) == CLEAN
    assert json.loads((out / registry.evidence.manifest).read_text())['tested_sha'] == 'b' * 40
