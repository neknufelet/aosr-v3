"""合成收據驗配平；每題指明它能咬的產品植錯。"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN
from governance.verify_shards import load_registry, render_inputs
from tests.test_verify_shards import REGISTRY

if TYPE_CHECKING:
    from governance.status.shard_balance import Balance


def _assignment(file: str) -> str:
    """獨立於產品的摘要答案。"""
    digest = hashlib.sha256(file.encode()).digest()
    return ('shard-a', 'shard-b')[int.from_bytes(digest[:8], 'big') % 2]


def _tree(tmp_path: Path) -> tuple[Path, list[str]]:
    """只在 tmp_path 製造現在存在的考卷與登記簿。"""
    files = ['tests/test_verify_shards.py'] + [f'tests/test_probe_{index}.py' for index in range(12)]
    for file in files:
        path = tmp_path / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('def test_one():\n    assert True\n')
    registry = tmp_path / 'registry.toml'
    registry.write_text(REGISTRY)
    return registry, files


def _sample(tmp_path: Path, run: str, shard: str, cpu: str | None,
            times: Mapping[str, float], wall: float, fixed: float) -> Path:
    """逐題時間與每步秒數均為預先指定的合成值。"""
    root = tmp_path / 'hist' / run / shard
    steps = root / 'steps'
    steps.mkdir(parents=True)
    machine = '' if cpu is None else f'開跑\n機器：{cpu}；avx512f：False\n'
    for name, seconds, tail in [('pytest', wall, machine), ('prepare', fixed / 2, ''), ('seal', fixed / 2, '')]:
        (steps / f'{name}.json').write_text(json.dumps({'name': name, 'seconds': seconds, 'stdout_tail': tail.splitlines()}))
    suite = ET.Element('testsuite')
    for classname, seconds in times.items():
        ET.SubElement(suite, 'testcase', classname=classname, name='test_one', time=str(seconds))
    ET.ElementTree(suite).write(root / 'junit.xml')
    return root


def _analyze(tmp_path: Path, registry: Path) -> Balance:
    from governance.status.shard_balance import analyze
    return analyze(tmp_path / 'hist', registry, tmp_path, '參考處理器')


def test_reference_cpu_only_and_per_file_median(tmp_path: Path) -> None:
    """抓異型混入、逐份漏加總、平均值取代中位數、類別尾綴映射錯。"""
    registry, files = _tree(tmp_path)
    target = files[1].removesuffix('.py').replace('/', '.')
    for run, seconds in [('r1', 10), ('r2', 30), ('r3', 200)]:
        _sample(tmp_path, run, 'shard-a', '參考處理器', {target: seconds, target + '.Case': 2}, 20, 4)
    _sample(tmp_path, 'other', 'shard-a', '別種處理器', {target: 9000}, 90, 900)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {target: 32}, 20, 6)
    result = _analyze(tmp_path, registry)
    assert result.weights[files[1]] == 32
    assert ('other', 'shard-a', '別種處理器') not in result.accepted
    assert ('r2', 'shard-a', '參考處理器') in result.accepted
    assert files[2] in result.unmeasured
    assert result.weights[files[2]] == 0


def test_missing_machine_and_unreadable_junit_are_reported(tmp_path: Path) -> None:
    """抓缺機器行／壞 XML 默默丟掉與未知 classname 靜默消失。"""
    registry, _ = _tree(tmp_path)
    _sample(tmp_path, 'valid', 'shard-a', '參考處理器', {'tests.test_verify_shards': 10, 'tests.unknown.Case': 5}, 3, 2)
    _sample(tmp_path, 'valid', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 3, 2)
    _sample(tmp_path, 'missing', 'shard-a', None, {}, 3, 2)
    broken = _sample(tmp_path, 'broken', 'shard-b', '參考處理器', {}, 3, 2)
    (broken / 'junit.xml').write_text('<broken')
    result = _analyze(tmp_path, registry)
    assert any('missing/shard-a' in item and '機器行' in item for item in result.skipped)
    assert any('broken/shard-b' in item and 'junit' in item for item in result.skipped)
    assert any('tests.unknown.Case' in item for item in result.unmapped)


def test_fixed_costs_and_ratio_use_correct_populations(tmp_path: Path) -> None:
    """抓非參考固定成本混入、pytest 重複算固定成本、比例只取參考或平均。"""
    registry, _ = _tree(tmp_path)
    for run, shard, cpu, wall, fixed in [
        ('r1', 'shard-a', '參考處理器', 2, 10),
        ('r2', 'shard-a', '參考處理器', 4, 30),
        ('r3', 'shard-a', '參考處理器', 6, 200),
        ('r1', 'shard-b', '參考處理器', 8, 40),
        ('other', 'shard-a', '別種處理器', 100, 900),
    ]:
        _sample(tmp_path, run, shard, cpu, {'tests.test_verify_shards': 10}, wall, fixed)
    result = _analyze(tmp_path, registry)
    assert result.fixed == {'shard-a': 30, 'shard-b': 40}
    assert result.ratio == pytest.approx(0.6)
    assert result.predictions(result.hashed)['shard-a'] == pytest.approx(36)
    assert result.predictions(result.hashed)['shard-b'] == pytest.approx(40)


def test_pinned_group_and_greedy_improvement(tmp_path: Path) -> None:
    """抓不能動群被搬、重片判錯、不挑最佳檔、停止門檻失效。"""
    registry, files = _tree(tmp_path)
    candidates = [file for file in files[1:] if _assignment(file) == 'shard-a']
    first, second = candidates[:2]
    times = {'tests.test_verify_shards': 10, first.removesuffix('.py').replace('/', '.'): 100,
             second.removesuffix('.py').replace('/', '.'): 60}
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器', times, 170, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    result = _analyze(tmp_path, registry)
    assert result.suggested['tests/test_verify_shards.py'] == 'shard-a'
    assert result.suggested[first] == 'shard-b'
    assert result.suggested[second] == 'shard-a'
    assert max(result.predictions(result.suggested).values()) == pytest.approx(102)
    assert max(result.predictions(result.suggested).values()) < max(result.predictions(result.hashed).values())


def test_old_balance_groups_discarded_and_write_roundtrips(tmp_path: Path) -> None:
    """抓舊建議固定化、原文遭重寫、輸出群重疊或資料遺漏。"""
    from governance.status.shard_balance import write_suggestion
    registry, files = _tree(tmp_path)
    file = next(file for file in files[1:] if _assignment(file) == 'shard-a')
    original = registry.read_text()
    registry.write_text(original + f'\n[[group]]\nname = "balance-stale"\nshard = "shard-b"\nwhy = "舊"\nfiles = ["{file}"]\n')
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    result = _analyze(tmp_path, registry)
    assert result.actual[file] == 'shard-b'
    assert result.hashed[file] == result.suggested[file] == 'shard-a'
    output = tmp_path / 'suggestion.toml'
    write_suggestion(registry, output, result)
    assert output.read_text().startswith(original)
    assert 'balance-stale' not in output.read_text()
    rewritten = load_registry(output)
    parts = [set(render_inputs([f'{file}::test_one' for file in files], rewritten, shard).args.splitlines()) for shard in rewritten.shards]
    assert not parts[0] & parts[1]
    assert parts[0] | parts[1] == set(files)


def test_import_tracking_helpers_cycles_and_paths(tmp_path: Path) -> None:
    """抓只追直接匯入、忽略輔助模組、迴圈失控與群定位錯。"""
    registry, files = _tree(tmp_path)
    helper = tmp_path / 'tests' / 'engine' / '_helper.py'
    helper.parent.mkdir()
    helper.write_text('import tests.engine._loop\nfrom tests.test_verify_shards import control\n')
    (helper.parent / '_loop.py').write_text('from tests.engine import _helper\n')
    (tmp_path / files[1]).write_text('from tests.engine._helper import value\n')
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    result = _analyze(tmp_path, registry)
    assert (files[1], 'control', (files[1], helper.relative_to(tmp_path).as_posix(), 'tests/test_verify_shards.py')) in result.hints


@pytest.mark.parametrize('cpu', ['別種處理器', None])
def test_zero_reference_cpu_returns_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu: str | None) -> None:
    """抓零參考份數仍印出成功預測。"""
    from governance.status.shard_balance import main
    registry, _ = _tree(tmp_path)
    _sample(tmp_path, 'r1', 'shard-a', cpu, {'tests.test_verify_shards': 10}, 10, 2)
    monkeypatch.chdir(tmp_path)
    assert main(['--history', str(tmp_path / 'hist'), '--registry', str(registry), '--reference-cpu', '參考處理器']) == TOOL_BROKEN


@pytest.mark.parametrize('weight,should_move', [(0.5, False), (1.0, True)])
def test_move_requires_at_least_one_second(tmp_path: Path, weight: float, should_move: bool) -> None:
    """抓小於一秒仍搬與恰好一秒卻不搬的邊界植錯。"""
    registry, files = _tree(tmp_path)
    file = next(file for file in files[1:] if _assignment(file) == 'shard-a')
    times = {'tests.test_verify_shards': 10, file.removesuffix('.py').replace('/', '.'): weight}
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器', times, 10 + weight, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    result = _analyze(tmp_path, registry)
    assert (result.suggested[file] == 'shard-b') == should_move


def test_cli_report_and_nonempty_written_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                              capsys: pytest.CaptureFixture[str]) -> None:
    """抓標準輸出漏三組預測、寫出搬檔反向、把 run id 或預測寫進登記簿。"""
    from governance.status.shard_balance import main
    registry, files = _tree(tmp_path)
    file = next(file for file in files[1:] if _assignment(file) == 'shard-a')
    _sample(tmp_path, 'synthetic-run', 'shard-a', '參考處理器',
            {'tests.test_verify_shards': 10, file.removesuffix('.py').replace('/', '.'): 100}, 110, 2)
    _sample(tmp_path, 'synthetic-run', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    output = tmp_path / 'new.toml'
    monkeypatch.chdir(tmp_path)
    command = ['--history', str(tmp_path / 'hist'), '--registry', str(registry),
               '--reference-cpu', '參考處理器', '--write', str(output)]
    assert main(command) == CLEAN
    report = capsys.readouterr().out
    for marker in ('算數的跑', 'synthetic-run/shard-a', '跳過的與原因', '牆鐘比例', '每片固定成本',
                   '照今天實際落片預測', '只用雜湊預測', '建議預測', '推測', '分鐘', '未量', '共用前置提醒'):
        assert marker in report
    rewritten = load_registry(output)
    balance = next(group for group in rewritten.groups if group.name == 'balance-shard-b')
    assert balance.files == (file,)
    assert 'synthetic-run' not in output.read_text()
    assert '110' not in output.read_text()
    nodes = [f'{file}::test_one' for file in files]
    parts = {shard: set(render_inputs(nodes, rewritten, shard).args.splitlines()) for shard in rewritten.shards}
    assert file in parts['shard-b'] and file not in parts['shard-a']
    assert not parts['shard-a'] & parts['shard-b']
    assert parts['shard-a'] | parts['shard-b'] == set(files)
    command[command.index('--registry') + 1] = str(output)
    first = output.read_bytes()
    assert main(command) == CLEAN
    assert output.read_bytes() == first


@pytest.mark.parametrize('replacement', ['', 'unregistered = "other"\n', 'unregistered = 7\n'])
def test_bad_policy_cli_returns_two(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str) -> None:
    """抓政策缺鍵、錯值或錯型別未走命令列回二。"""
    from governance.status.shard_balance import main
    registry, _ = _tree(tmp_path)
    registry.write_text(registry.read_text().replace('unregistered = "path-hash"\n', replacement))
    monkeypatch.chdir(tmp_path)
    assert main(['--history', str(tmp_path / 'hist'), '--registry', str(registry), '--reference-cpu', '參考處理器']) == TOOL_BROKEN


def test_reference_cpu_requires_complete_equality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓前綴比對錯收不同完整型號；不讀任何真歷史資料。"""
    from governance.status.shard_balance import main
    registry, _ = _tree(tmp_path)
    for shard in ('shard-a', 'shard-b'):
        _sample(tmp_path, 'r1', shard, '參考處理器 額外尾綴', {'tests.test_verify_shards': 10}, 10, 2)
    monkeypatch.chdir(tmp_path)
    assert main(['--history', str(tmp_path / 'hist'), '--registry', str(registry), '--reference-cpu', '參考處理器']) == TOOL_BROKEN


def test_pinned_group_stays_even_when_it_is_the_best_move(tmp_path: Path) -> None:
    """抓不能動的群被當成能搬：讓固定群那一檔正好是唯一最划算的一搬。"""
    registry, files = _tree(tmp_path)
    pinned = 'tests/test_verify_shards.py'
    loose = [file for file in files[1:] if _assignment(file) == 'shard-a'][:2]
    times = {pinned.removesuffix('.py').replace('/', '.'): 60}
    times.update({file.removesuffix('.py').replace('/', '.'): 50 for file in loose})
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器', times, 160, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {}, 0, 2)
    result = _analyze(tmp_path, registry)
    assert result.ratio == pytest.approx(1)
    assert result.suggested[pinned] == 'shard-a'
    assert [result.suggested[file] for file in loose] == ['shard-b', 'shard-b']


@pytest.mark.parametrize('broken', ['overlap', 'incomplete'])
def test_write_refuses_partitions_that_overlap_or_miss_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, broken: str,
) -> None:
    """抓寫檔前不驗兩片互斥與聯集：分片結果壞掉時一個字都不准寫出去。"""
    from governance.exit_codes import ToolBroken
    from governance.status import shard_balance
    from governance.verify_shards import Prepared
    registry, files = _tree(tmp_path)
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    result = _analyze(tmp_path, registry)
    picked = {'overlap': {'shard-a': files, 'shard-b': files},
              'incomplete': {'shard-a': files[:1], 'shard-b': files[1:2]}}[broken]
    monkeypatch.setattr(shard_balance, 'render_inputs',
                        lambda nodes, registry, shard: Prepared('', '\n'.join(picked[shard]) + '\n'))
    output = tmp_path / 'suggestion.toml'
    with pytest.raises(ToolBroken, match='互斥'):
        shard_balance.write_suggestion(registry, output, result)
    assert not output.exists()


_STALE = 'name = "balance-stale"\nshard = "shard-b"\nwhy = "舊"\nfiles = ["tests/test_shard_prepare.py"]\n'
_FIXED = '[[group]]\nname = "fixed-tail"\nshard = "shard-a"\nwhy = "固定"\nfiles = ["tests/test_shard_seal.py"]\n'


def test_balance_header_with_trailing_comment_is_still_removed() -> None:
    """抓表頭帶行尾註解就認不出來、舊 balance 群留下來跟新群撞名（審查 B 第 1 條）。"""
    from governance.status.shard_balance import _without_balance_groups
    assert _without_balance_groups(REGISTRY + '\n[[group]] # 自動配平群\n' + _STALE) == REGISTRY


def test_middle_balance_group_keeps_the_next_groups_comment() -> None:
    """抓刪中間那一群時把下一群前面的說明註解一起吞掉（審查 B 第 2 條）；其餘逐位元相同。"""
    from governance.status.shard_balance import _without_balance_groups
    note = '\n# 以下這一群請勿拆散：既有註解。\n'
    assert _without_balance_groups(REGISTRY + '\n[[group]]\n' + _STALE + note + _FIXED) == REGISTRY + note + _FIXED


@pytest.mark.parametrize('source', [
    # 多行字串裡長得像表頭的一行：前一段切出來讀不成 TOML。
    REGISTRY + '\n' + _FIXED.replace('why = "固定"', 'why = """\n[[group]]\n"""'),
    # 假群的收尾引號落在註解行、被歸給下一段：刪完讀得成 TOML，但內容變了。
    '[evidence]\na = """\n[[group]]\nname = "balance-fake"\nshard = "shard-b"\n# """\n',
    # 假群讀得成、刪完的檔讀不成。
    '[evidence]\na = """\n[[group]]\nname = "balance-fake"\nfiles = [] # """\n',
], ids=['segment-unreadable', 'content-changed', 'result-unreadable'])
def test_cut_that_goes_wrong_is_tool_broken(source: str) -> None:
    """抓文字切段失手卻照樣寫出（多行字串裡的假表頭）：一律回 2，不寫壞登記簿。"""
    from governance.exit_codes import ToolBroken
    from governance.status.shard_balance import _without_balance_groups
    with pytest.raises(ToolBroken):
        _without_balance_groups(source)


def test_source_without_any_group_comes_back_unchanged() -> None:
    """抓原檔沒有群時硬塞一格空群、把沒動過的檔判成切段失手。"""
    from governance.status.shard_balance import _without_balance_groups
    assert _without_balance_groups('[evidence]\na = 1\n') == '[evidence]\na = 1\n'
