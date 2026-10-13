"""命令與執行參數的反例；每題只改 tmp_path 的樣本副本。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from governance.checks import green_must_be_real_green as subject
from tests.test_green_finalizer_inputs import part_path
from tests.test_green_finalizer_mode import files
from tests.test_green_shard_topology import (
    ARGS, JUNIT, SHARDS, job_steps, jobs, pytest_step,
    save_jobs, topology as topology,
)


TEST_FILE = str(Path("tests") / "test_a.py")


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('argv', [
    ['uv', 'run', 'pytest', JUNIT],
    ['uv', 'run', 'pytest', ARGS, JUNIT, TEST_FILE],
    ['uv', 'run', 'pytest', JUNIT, ARGS],
    ['uv', 'run', 'pytest', ARGS + '.bak', JUNIT],
    ['uv', 'run', 'pytest', ARGS, JUNIT + '.bak'],
    ['uv', 'run', 'py.test', ARGS, JUNIT],
    ['python', '-m', 'pytest', ARGS, JUNIT],
    ['uv', 'run', 'pytest', ARGS, JUNIT, '--junitxml=other.xml'],
], ids=['missing', 'extra', 'reorder', 'args-bak', 'junit-bak', 'py.test', 'module', 'duplicate-junit'])
def test_fragment_argv_equals_registered_command(topology: Path, shard: str, argv: list[str]) -> None:
    path = part_path(topology, shard) / 'steps' / 'pytest.json'
    row = json.loads(path.read_text())
    row['argv'] = argv
    path.write_text(json.dumps(row))
    bad = subject.check(topology, files(topology))
    assert any(f'片 {shard} ' in item and f'{Path("steps") / "pytest.json"}.argv=' in item and
               f'預期 {["uv", "run", "pytest", ARGS, JUNIT]!r}' in item for item in bad)


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('extra', [False, True], ids=['zero', 'two'])
def test_fragment_pytest_command_cardinality(topology: Path, shard: str, extra: bool) -> None:
    path = part_path(topology, shard) / 'steps' / 'pytest.json'
    row = json.loads(path.read_text())
    if extra:
        row['name'] = 'again'
        (path.parent / 'again.json').write_text(json.dumps(row))
        value = jobs(topology)
        step = pytest_step(value[shard])
        job_steps(value[shard]).append({'run': step['run'].replace('--name pytest ', '--name again ')})
        save_jobs(topology, value)
    else:
        row['argv'] = ['echo', 'pytest']
        path.write_text(json.dumps(row))
    bad = subject.check(topology, files(topology))
    assert any(f'片 {shard} ' in item and '.pytest 片段=' in item and '預期恰好一筆' in item for item in bad)


@pytest.mark.parametrize('shard', SHARDS)
def test_registered_prefix_drives_runtime(topology: Path, shard: str) -> None:
    from tests.test_green_finalizer_inputs import rewrite_setting
    rewrite_setting(topology, 'pytest_command', '"python3 -m pytest"')
    for name in SHARDS:
        path = part_path(topology, name) / 'steps' / 'pytest.json'
        row = json.loads(path.read_text())
        row['argv'] = ['python3', '-m', 'pytest', ARGS, JUNIT]
        path.write_text(json.dumps(row))
    assert subject.check(topology, files(topology)) == []
    path = part_path(topology, shard) / 'steps' / 'pytest.json'
    row = json.loads(path.read_text())
    row['argv'] = ['uv', 'run', 'pytest', ARGS, JUNIT]
    path.write_text(json.dumps(row))
    assert any(f'片 {shard}' in item and 'pytest.json.argv=' in item for item in subject.check(topology, files(topology)))
