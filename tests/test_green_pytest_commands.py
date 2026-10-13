"""命令與執行參數的反例；每題只改 tmp_path 的樣本副本。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from governance.checks import green_must_be_real_green as subject
from tests.test_green_finalizer_inputs import part_path
from tests.test_green_finalizer_mode import files
from tests.test_green_shard_topology import (
    ARGS, FORMAL, JUNIT, PYTEST, SCRIPT, SHARDS, job_steps, jobs, pytest_step,
    save_jobs, source, topology as topology,
)


TEST_FILE = str(Path("tests") / "test_a.py")


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('old,new,word', [
    (ARGS, TEST_FILE, '缺參數檔'),
    (ARGS, ARGS + '.bak', '缺參數檔'),
    (JUNIT, JUNIT + '.bak', '缺分片收據參數'),
    (JUNIT, JUNIT + ' --junitxml=other.xml', '收據參數重複'),
])
def test_pytest_requires_exact_path_tokens(topology: Path, shard: str, old: str, new: str, word: str) -> None:
    value = jobs(topology)
    step = pytest_step(value[shard])
    step['run'] = step['run'].replace(old, new) + f' # {ARGS} {JUNIT}'
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}' in item and word in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('separator', ['\n', '; ', ' && ', ' || ', ' | '])
def test_two_pytest_commands_in_same_step(topology: Path, shard: str, separator: str) -> None:
    value = jobs(topology)
    pytest_step(value[shard])['run'] += separator + PYTEST
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}: pytest 步驟=2' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('command', ['uv run py.test', 'python -m pytest', 'python3 -m pytest', '/usr/bin/pytest',
                                     'uv run --with x --no-sync pytest', 'uv run --with=x python3 -m pytest',
                                     'uv run -w x --exclude-newer 2020-01-01 pytest'])
def test_other_pytest_entry_adds_a_command(topology: Path, shard: str, command: str) -> None:
    value = jobs(topology)
    job_steps(value[shard]).append({'run': f'{command} {ARGS} {JUNIT}'})
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}: pytest 步驟=2' in item for item in source(topology))


@pytest.mark.parametrize('job_name', ['shard-a', 'shard-b', 'verify'])
def test_local_action_is_uninspectable(topology: Path, job_name: str) -> None:
    value = jobs(topology)
    job_steps(value[job_name]).append({'uses': './.github/actions/extra-tests'})
    save_jobs(topology, value)
    assert any(f'job={job_name}' in item and '本地 action' in item for item in source(topology))


@pytest.mark.parametrize('job_name', ['shard-a', 'shard-b', 'verify'])
@pytest.mark.parametrize('command', ["# pytest 已下載", "echo 'pytest 的證據稍後封存'", 'printf "%s" "pytest"',
                                    "echo 'uv run pytest && pytest'", 'echo "${literal}"', "echo '$(pytest)'"],
                         ids=['comment', 'echo', 'printf', 'quoted-separator', 'expansion', 'literal-expansion'])
def test_pytest_words_are_not_execution(topology: Path, job_name: str, command: str) -> None:
    value = jobs(topology)
    job_steps(value[job_name]).append({'run': command})
    save_jobs(topology, value)
    bad = source(topology)
    if command == 'echo "${literal}"':
        assert any(f'job={job_name}' in item and '看不懂的命令不准出現在分片拓撲' in item for item in bad)
    else:
        assert bad == []


@pytest.mark.parametrize('job_name', ['shard-a', 'shard-b', 'verify'])
@pytest.mark.parametrize('command', ['echo "unterminated', 'echo $(pytest)', 'echo ${value}', 'echo $[1]',
                                    'echo `pytest`', 'cat <(pytest)', 'cat >(pytest)'])
def test_unreadable_command_fails_closed(topology: Path, job_name: str, command: str) -> None:
    value = jobs(topology)
    job_steps(value[job_name]).append({'run': command})
    save_jobs(topology, value)
    assert any(f'job={job_name}' in item and '看不懂的命令不准出現在分片拓撲' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('flag', ['-k', '-kfoo', '-k=foo', '--deselect', '--deselect=foo', '--ignore', '--ignore=foo',
                                '--collect-only'])
def test_forbidden_flags_use_tokens(topology: Path, shard: str, flag: str) -> None:
    value = jobs(topology)
    pytest_step(value[shard])['run'] += '\t' + flag
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}' in item and
               '帶了 ' + (flag.partition('=')[0] if flag.startswith('--') else '-k') in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('snippet', subject.SWALLOW_SNIPPETS)
def test_called_script_swallowed_exit(topology: Path, shard: str, snippet: str) -> None:
    value = jobs(topology)
    (topology / SCRIPT.parent).mkdir()
    (topology / SCRIPT).write_text(PYTEST + ' ' + snippet + '\n')
    pytest_step(value[shard])['run'] = f'bash {SCRIPT}'
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}' in item and '腳本' in item and
               f'把離開碼吞掉了（{snippet!r}）' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
@pytest.mark.parametrize('step_name', ['pytest', 'seal'])
def test_continue_on_error_after_first_step(topology: Path, shard: str, step_name: str) -> None:
    value = jobs(topology)
    step = next(step for step in job_steps(value[shard]) if f'--name {step_name} ' in str(step.get('run')))
    step['continue-on-error'] = True
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard} step[' in item and 'continue-on-error' in item for item in source(topology))


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
def test_formal_path_in_pytest_argument(topology: Path, shard: str) -> None:
    value = jobs(topology)
    pytest_step(value[shard])['run'] += ' ' + FORMAL
    save_jobs(topology, value)
    assert any(f'片 {shard}' in item and '出現正式收據路徑' in item for item in source(topology))


@pytest.mark.parametrize('job_name', ['shard-a', 'shard-b', 'verify'])
@pytest.mark.parametrize('script_body', ["echo 'unterminated", 'echo $(pytest)'])
def test_called_script_unreadable_command(topology: Path, job_name: str, script_body: str) -> None:
    value = jobs(topology)
    (topology / SCRIPT.parent).mkdir()
    (topology / SCRIPT).write_text(script_body + '\n')
    job_steps(value[job_name]).append({'run': f'bash {SCRIPT}'})
    save_jobs(topology, value)
    assert any(f'job={job_name}' in item and '腳本' in item and
               '看不懂的命令不准出現在分片拓撲' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
def test_script_two_commands_count_separately(topology: Path, shard: str) -> None:
    value = jobs(topology)
    (topology / SCRIPT.parent).mkdir()
    (topology / SCRIPT).write_text(PYTEST + '\n' + PYTEST + '\n')
    pytest_step(value[shard])['run'] = f'bash {SCRIPT}'
    save_jobs(topology, value)
    assert any(f'片 {shard} job={shard}: pytest 步驟=2' in item for item in source(topology))


@pytest.mark.parametrize('shard', SHARDS)
def test_pytest_path_on_other_command_is_not_argument(topology: Path, shard: str) -> None:
    value = jobs(topology)
    pytest_step(value[shard])['run'] = 'uv run pytest; echo ' + ARGS + ' ' + JUNIT
    save_jobs(topology, value)
    bad = source(topology)
    assert any(f'片 {shard}' in item and '缺參數檔' in item for item in bad)
    assert any(f'片 {shard}' in item and '缺分片收據參數' in item for item in bad)


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


@pytest.mark.parametrize('shard', SHARDS)
def test_forbidden_substring_second_defense(topology: Path, shard: str) -> None:
    value = jobs(topology)
    pytest_step(value[shard])['run'] += " # --ignore-hidden"
    save_jobs(topology, value)
    assert any(f'片 {shard}' in item and '帶了 --ignore' in item for item in source(topology))
