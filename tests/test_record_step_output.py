"""正式子程序結束後量輸出；所有檔案與行程身分都由本題提供。"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION
from governance.status import record_step

IDENTITY = {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2', 'GITHUB_JOB': 'shard-b'}


@pytest.fixture(autouse=True)
def isolated_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in IDENTITY:
        monkeypatch.delenv(key, raising=False)


def invoke(tmp_path: Path, code: str, output: str | None) -> tuple[int, dict[str, object]]:
    options = ['--output', output] if output is not None else []
    rc = record_step.main(['--name', 'pytest', *options, '--out-dir', str(tmp_path / 'steps'),
                           '--', sys.executable, '-c', code])
    return rc, json.loads((tmp_path / 'steps' / 'pytest.json').read_text())


@pytest.mark.parametrize('exit_code', [CLEAN, VIOLATION, TOOL_BROKEN])
def test_output_is_measured_after_child_overwrites_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     exit_code: int) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'junit.xml').write_bytes(b'before')
    for key, value in IDENTITY.items():
        monkeypatch.setenv(key, value)
    output = './' + 'junit.xml'
    rc, row = invoke(tmp_path, f"from pathlib import Path; Path('junit.xml').write_bytes(b'after'); exit({exit_code})",
                     output)
    assert rc == exit_code
    assert row['exit_code'] == exit_code
    assert row['output'] == {'path': output, 'sha256': hashlib.sha256(b'after').hexdigest(), 'bytes': len(b'after')}
    assert {key: row[key] for key in ('run_id', 'run_attempt', 'job', 'schema')} == {
        'run_id': '123', 'run_attempt': '2', 'job': 'shard-b', 'schema': 2}


@pytest.mark.parametrize('exit_code', [CLEAN, VIOLATION, TOOL_BROKEN])
def test_missing_output_preserves_child_exit(tmp_path: Path, exit_code: int) -> None:
    path = str(tmp_path / 'missing.xml')
    rc, row = invoke(tmp_path, f'exit({exit_code})', path)
    assert rc == exit_code
    assert row['exit_code'] == exit_code
    assert row['output'] == {'path': path, 'sha256': None, 'bytes': None}


def test_no_output_option_and_absent_identity_are_null(tmp_path: Path) -> None:
    rc, row = invoke(tmp_path, 'pass', None)
    assert rc == CLEAN
    assert {key: row[key] for key in ('run_id', 'run_attempt', 'job', 'output')} == {
        'run_id': None, 'run_attempt': None, 'job': None, 'output': None}


@pytest.mark.parametrize('failure', ['directory', 'permission', 'nul'])
def test_unreadable_output_is_tool_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        failure: str) -> None:
    path = str(tmp_path) if failure == 'directory' else str(tmp_path / 'junit.xml')
    if failure == 'permission':
        def denied(target: Path) -> bytes:
            raise PermissionError('EACCES')
        monkeypatch.setattr(Path, 'read_bytes', denied)
    elif failure == 'nul':
        path += '\0'
    rc = record_step.main(['--name', 'pytest', '--output', path, '--out-dir', str(tmp_path / 'steps'),
                           '--', sys.executable, '-c', 'exit(1)'])
    assert rc == TOOL_BROKEN
    assert not (tmp_path / 'steps' / 'pytest.json').exists()
