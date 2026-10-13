"""全套清點與互斥分片參數；子程序只在迷你暫存 repo 跑。"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken
from governance.status.shard_prepare import ENVIRONMENT_OPTIONS, Collector, collect_nodeids, prepare, render_inputs
from governance.verify_shards import Registry, load_registry
from tests.test_verify_shards import REGISTRY, write_registry

NODES = ('tests/test_shard_seal.py::test_z', 'tests/test_verify_shards.py::test_y', 'tests/test_verify_shards.py::test_x[escaped\\n]')


def test_outputs_are_sorted_and_default_is_complement(tmp_path: Path) -> None:
    registry = load_registry(write_registry(tmp_path))
    first = render_inputs(NODES, registry, 'shard-a')
    second = render_inputs(reversed(NODES), registry, 'shard-b')
    assert first.expected == second.expected == '\n'.join(sorted(NODES)) + '\n'
    assert first.args == 'tests/test_verify_shards.py\n'
    assert second.args == 'tests/test_shard_seal.py\n'


@pytest.mark.parametrize('nodes,shard,message', [
    ((), 'shard-a', 'nodeid'),
    ((NODES[0], NODES[0]), 'shard-b', '重複'),
    (('tests/test_verify_shards.py::test_x[\n]',), 'shard-a', '換行'),
    (('tests/test_verify_shards.py::test_x[\r]',), 'shard-a', '換行'),
    (('tests/test_shard_seal.py::test_x',), 'shard-b', '收集不到'),
    (NODES, 'unknown', '^shard: 未登記'),
    (('tests/test_verify_shards.py::test_x',), 'shard-b', '空'),
])
def test_bad_preparation(tmp_path: Path, nodes: tuple[str, ...], shard: str, message: str) -> None:
    with pytest.raises(ToolBroken, match=message):
        render_inputs(nodes, load_registry(write_registry(tmp_path)), shard)


@pytest.mark.parametrize('nodeid', ['governance/exit_codes.py::test_x', 'tests/conftest.py::test_x', 'tests/test_verify_shards.py'])
def test_collected_address_must_be_test_file_and_item(tmp_path: Path, nodeid: str) -> None:
    with pytest.raises(ToolBroken, match='nodeid'):
        render_inputs((nodeid,), load_registry(write_registry(tmp_path)), 'shard-b')


def test_default_can_also_have_explicit_groups(tmp_path: Path) -> None:
    registry = load_registry(write_registry(tmp_path))
    registry = replace(registry, groups=(replace(registry.groups[0], shard='shard-b'),))
    result = render_inputs(NODES, registry, 'shard-b')
    assert result.args == '\n'.join(sorted(('tests/test_verify_shards.py', 'tests/test_shard_seal.py'))) + '\n'


@pytest.mark.parametrize('error', [OSError('模擬 pytest 起不來'), ImportError('模擬外掛載入失敗')], ids=['os', 'import'])
def test_collect_startup_error_is_tool_broken(error: Exception) -> None:
    def runner(args: list[str], *, plugins: list[object]) -> int:
        raise error
    with pytest.raises(ToolBroken, match='起不來'):
        collect_nodeids(runner=runner)


def test_collect_rejects_deselected_items() -> None:
    def runner(args: list[str], *, plugins: list[object]) -> int:
        collector = cast(Collector, plugins[0])
        collector.pytest_deselected([cast(pytest.Item, SimpleNamespace(nodeid=NODES[0]))])
        session = cast(pytest.Session, SimpleNamespace(items=[SimpleNamespace(nodeid=node) for node in NODES[1:]]))
        collector.pytest_collection_finish(session)
        return 0
    with pytest.raises(ToolBroken, match='篩掉'):
        collect_nodeids(runner=runner)


@pytest.mark.parametrize('code', [1, 2, 3, 4, 5])
def test_nonzero_collect_is_tool_broken(code: int) -> None:
    def runner(args: list[str], *, plugins: list[object]) -> int:
        return code
    with pytest.raises(ToolBroken, match=f'離開碼 {code}'):
        collect_nodeids(runner=runner)


def test_collector_records_items_and_collect_arguments() -> None:
    def runner(args: list[str], *, plugins: list[object]) -> int:
        assert args == ['--collect-only', '-n', '0', '-q', '-p', 'no:cacheprovider']
        collector = cast(Collector, plugins[0])
        session = cast(pytest.Session, SimpleNamespace(items=[SimpleNamespace(nodeid=node) for node in NODES]))
        collector.pytest_collection_finish(session)
        return 0
    assert collect_nodeids(runner=runner) == NODES


def test_prepare_creates_outputs_and_does_not_overwrite(tmp_path: Path) -> None:
    registry = load_registry(write_registry(tmp_path))
    out = tmp_path / 'evidence'
    prepare(NODES, registry, 'shard-a', out)
    before = {file.name: file.read_bytes() for file in out.iterdir()}
    with pytest.raises(ToolBroken, match='非空'):
        prepare(NODES, registry, 'shard-b', out)
    assert {file.name: file.read_bytes() for file in out.iterdir()} == before


def test_output_directory_can_start_empty(tmp_path: Path) -> None:
    out = tmp_path / 'empty'
    out.mkdir()
    registry = load_registry(write_registry(tmp_path))
    prepare(NODES, registry, 'shard-b', out)
    assert (out / registry.evidence.args).read_bytes() == b'tests/test_shard_seal.py\n'


def _mini_repo(tmp_path: Path, addopts: str = '-n logical') -> tuple[dict[str, str], list[str]]:
    """迷你 repo：兩支各一題的考卷、一份登記簿；回傳子行程的環境與命令（PYTHONPATH 只追加）。"""
    (tmp_path / 'pyproject.toml').write_text(f'[tool.pytest.ini_options]\ntestpaths = ["tests"]\naddopts = "{addopts}"\n')
    tests = tmp_path / 'tests'
    tests.mkdir()
    (tests / 'test_verify_shards.py').write_text('def test_a():\n    assert True\n')
    (tests / 'test_shard_seal.py').write_text('def test_b():\n    assert True\n')
    registry = write_registry(tmp_path, REGISTRY)
    source = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items() if key not in ENVIRONMENT_OPTIONS}
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (env.get('PYTHONPATH'), str(source))))
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env, [sys.executable, '-m', 'governance.status.shard_prepare', '--registry', str(registry)]


def test_cli_two_shards_in_mini_repo(tmp_path: Path) -> None:
    env, command = _mini_repo(tmp_path)
    outputs = [tmp_path / 'a', tmp_path / 'b']
    for shard, out in zip(('shard-a', 'shard-b'), outputs, strict=True):
        result = subprocess.run([*command, '--shard', shard, '--out', str(out)], cwd=tmp_path,
                                env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        assert '::test_' not in result.stdout
        assert 'scan_root=' not in result.stdout
    assert (outputs[0] / 'expected.txt').read_bytes() == (outputs[1] / 'expected.txt').read_bytes()
    first, second = (set((out / 'pytest.args').read_text().splitlines()) for out in outputs)
    assert not first & second
    assert first | second == {'tests/test_verify_shards.py', 'tests/test_shard_seal.py'}
    again = subprocess.run([*command, '--shard', 'shard-a', '--out', str(outputs[0])],
                           cwd=tmp_path, env=env, capture_output=True, text=True)
    assert again.returncode == 2, again.stdout + again.stderr
    assert '非空' in again.stderr


def test_output_path_cannot_be_a_file(tmp_path: Path) -> None:
    out = tmp_path / 'file'
    out.write_text('old')
    with pytest.raises(ToolBroken, match='out'):
        prepare(NODES, load_registry(write_registry(tmp_path)), 'shard-a', out)
    assert out.read_text() == 'old'


def test_collector_rejects_parent_root(tmp_path: Path) -> None:
    session = cast(pytest.Session, SimpleNamespace(config=SimpleNamespace(rootpath=tmp_path.parent)))
    with pytest.raises(pytest.UsageError, match='只准在 repo 根'):
        Collector(tmp_path).pytest_sessionstart(session)


@pytest.mark.parametrize('bad', ['root', 'shard', 'collection'])
def test_cli_tool_broken_becomes_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    from governance.status import shard_prepare
    registry = write_registry(tmp_path)
    for option in ENVIRONMENT_OPTIONS:
        monkeypatch.delenv(option, raising=False)
    monkeypatch.chdir(tmp_path)
    if bad != 'root':
        (tmp_path / 'pyproject.toml').write_text('')
        (tmp_path / 'tests').mkdir()
    def collector(*, root: Path) -> tuple[str, ...]:
        raise ToolBroken('清點故障')
    monkeypatch.setattr(shard_prepare, 'collect_nodeids', collector)
    shard = 'missing' if bad == 'shard' else 'shard-a'
    assert shard_prepare.main(['--registry', str(registry), '--shard', shard]) == TOOL_BROKEN


def test_cli_default_out_uses_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from governance.status import shard_prepare
    registry = load_registry(write_registry(tmp_path))
    (tmp_path / 'pyproject.toml').write_text('')
    (tmp_path / 'tests').mkdir()
    for option in ENVIRONMENT_OPTIONS:
        monkeypatch.delenv(option, raising=False)
    monkeypatch.chdir(tmp_path)
    def collector(*, root: Path) -> tuple[str, ...]:
        assert root == tmp_path
        return NODES
    monkeypatch.setattr(shard_prepare, 'collect_nodeids', collector)
    assert shard_prepare.main(['--registry', 'verify-shards.toml', '--shard', 'shard-a']) == CLEAN
    assert (tmp_path / registry.evidence.dir / registry.evidence.args).read_text() == 'tests/test_verify_shards.py\n'


@pytest.mark.parametrize('addopts,environment,message', [
    ('-n logical -k test_a', {}, '篩掉'),
    ('-n logical --deselect=tests/test_shard_seal.py::test_b', {}, '篩掉'),
    ('-n logical --ignore=tests/test_shard_seal.py', {}, '--ignore'),
    ('-n logical tests/test_verify_shards.py', {}, 'testpaths'),
    ('-n logical', {'PYTEST_ADDOPTS': '-k test_a'}, 'PYTEST_ADDOPTS'),
    ('-n logical', {'PYTEST_PLUGINS': 'missing_step6_plugin'}, 'PYTEST_PLUGINS'),
], ids=['keyword', 'deselect', 'ignore', 'positional', 'options', 'plugins'])
def test_cli_refuses_narrowed_collection(tmp_path: Path, addopts: str, environment: dict[str, str],
                                         message: str) -> None:
    """縮小收集面的清點兩片一樣少、實跑也一樣少，收尾看不出來，所以清點當下就回 2、不寫證據。"""
    env, command = _mini_repo(tmp_path, addopts)
    env.update(environment)
    out = tmp_path / 'out'
    result = subprocess.run([*command, '--shard', 'shard-a', '--out', str(out)], cwd=tmp_path,
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert message in result.stderr
    assert not out.exists()
