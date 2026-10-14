"""合成收據驗配平；每題指明它能咬的產品植錯。"""
from __future__ import annotations

from collections.abc import Callable, Mapping
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


def test_old_balance_groups_discarded_and_nothing_to_paste(tmp_path: Path) -> None:
    """抓舊建議固定化：登記簿裡舊的 balance 群先丟掉、從雜湊重算；這次沒得搬，就沒有群可貼、也報「跟建議不同」。"""
    from governance.status.shard_balance import same_as_registry, suggested_groups, suggestion_text
    registry, files = _tree(tmp_path)
    file = next(file for file in files[1:] if _assignment(file) == 'shard-a')
    registry.write_text(registry.read_text() + f'\n[[group]]\nname = "balance-stale"\nshard = "shard-b"\nwhy = "舊"\nfiles = ["{file}"]\n')
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    result = _analyze(tmp_path, registry)
    assert result.actual[file] == 'shard-b'
    assert result.hashed[file] == result.suggested[file] == 'shard-a'
    assert suggested_groups(result) == ()
    assert suggestion_text(()) == ''
    assert not same_as_registry(load_registry(registry), ())


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


def _pasted(report: str) -> str:
    """報告尾巴那段建議原文（標題行之後到結尾）。"""
    return report.split('（換掉登記簿裡現有的 balance-* 群；改登記簿走合併請求，這支不改檔）：\n', 1)[1]


def _snapshot(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


def test_cli_prints_a_pasteable_group_and_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                         capsys: pytest.CaptureFixture[str]) -> None:
    """抓標準輸出漏三組預測、印出的群搬反了或貼上讀不回、把 run id 或預測印進群、工具偷改任何檔、貼上後重算不穩。"""
    from governance.status.shard_balance import main
    registry, files = _tree(tmp_path)
    file = next(file for file in files[1:] if _assignment(file) == 'shard-a')
    _sample(tmp_path, 'synthetic-run', 'shard-a', '參考處理器',
            {'tests.test_verify_shards': 10, file.removesuffix('.py').replace('/', '.'): 100}, 110, 2)
    _sample(tmp_path, 'synthetic-run', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    monkeypatch.chdir(tmp_path)
    command = ['--history', str(tmp_path / 'hist'), '--reference-cpu', '參考處理器']
    before = _snapshot(tmp_path)
    assert main([*command, '--registry', str(registry)]) == CLEAN
    assert _snapshot(tmp_path) == before
    report = capsys.readouterr().out
    for marker in ('算數的跑', 'synthetic-run/shard-a', '跳過的與原因', '牆鐘比例', '每片固定成本',
                   '照今天實際落片預測', '只用雜湊預測', '建議預測', '推測', '分鐘', '未量', '共用前置提醒',
                   '登記簿現有的 balance 群：跟建議不同'):
        assert marker in report
    text = _pasted(report)
    assert 'synthetic-run' not in text
    assert '110' not in text
    pasted = tmp_path / 'pasted.toml'
    pasted.write_text(registry.read_text() + '\n' + text)
    rewritten = load_registry(pasted)
    balance = next(group for group in rewritten.groups if group.name == 'balance-shard-b')
    assert balance.files == (file,)
    nodes = [f'{file}::test_one' for file in files]
    parts = {shard: set(render_inputs(nodes, rewritten, shard).args.splitlines()) for shard in rewritten.shards}
    assert file in parts['shard-b'] and file not in parts['shard-a']
    assert not parts['shard-a'] & parts['shard-b']
    assert parts['shard-a'] | parts['shard-b'] == set(files)
    assert main([*command, '--registry', str(pasted)]) == CLEAN
    again = capsys.readouterr().out
    assert '登記簿現有的 balance 群：跟建議相同，不用改' in again
    assert _pasted(again) == text


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
def test_suggestion_check_refuses_partitions_that_overlap_or_miss_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], broken: str,
) -> None:
    """抓印建議前不驗兩片互斥與聯集：分片結果壞掉時回 2，標準輸出一個字都不印。"""
    from governance.status import shard_balance
    from governance.verify_shards import Prepared
    registry, files = _tree(tmp_path)
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    picked = {'overlap': {'shard-a': files, 'shard-b': files},
              'incomplete': {'shard-a': files[:1], 'shard-b': files[1:2]}}[broken]
    monkeypatch.setattr(shard_balance, 'render_inputs',
                        lambda nodes, registry, shard: Prepared('', '\n'.join(picked[shard]) + '\n'))
    monkeypatch.chdir(tmp_path)
    assert shard_balance.main(['--history', str(tmp_path / 'hist'), '--registry', str(registry),
                               '--reference-cpu', '參考處理器']) == TOOL_BROKEN
    captured = capsys.readouterr()
    assert captured.out == ''
    assert '互斥' in captured.err


def _moved(tmp_path: Path) -> tuple[Balance, Path, str]:
    """一檔從第一台搬去第二台的合成結果、登記簿與那一檔。"""
    registry, files = _tree(tmp_path)
    file = next(file for file in files[1:] if _assignment(file) == 'shard-a')
    _sample(tmp_path, 'r1', 'shard-a', '參考處理器',
            {'tests.test_verify_shards': 10, file.removesuffix('.py').replace('/', '.'): 100}, 110, 2)
    _sample(tmp_path, 'r1', 'shard-b', '參考處理器', {'tests.test_verify_shards': 10}, 10, 2)
    return _analyze(tmp_path, registry), registry, file


@pytest.mark.parametrize('tamper,message', [
    (lambda text, file: text.replace('[[group]]', '[[group]', 1), '讀不回來'),
    (lambda text, file: 'group = 7\n', '不是群清單'),
    (lambda text, file: text.replace('"shard-b"', '"shard-a"'), '不一樣'),
    (lambda text, file: '', '不一樣'),
    (lambda text, file: text + text.replace('balance-shard-b', 'balance-shard-b-again'), '兩群'),
    (lambda text, file: text.replace(f'"{file}"', '"tests/test_verify_shards.py"'), '兩群'),
], ids=['not-toml', 'not-a-list', 'other-shard', 'missing', 'file-twice', 'steals-a-pinned-file'])
def test_suggestion_check_reads_the_printed_text_back(tmp_path: Path, tamper: Callable[[str, str], str],
                                                      message: str) -> None:
    """抓印出來的原文沒讀回核對：讀不回、讀回來跟算的不一樣、或跟固定群撞檔，都回 2。"""
    from governance.exit_codes import ToolBroken
    from governance.status.shard_balance import check_suggestion, suggested_groups, suggestion_text
    result, registry, file = _moved(tmp_path)
    text = suggestion_text(suggested_groups(result))
    check_suggestion(result, load_registry(registry), text)
    with pytest.raises(ToolBroken, match=message):
        check_suggestion(result, load_registry(registry), tamper(text, file))


def test_same_as_registry_compares_groups_not_prose(tmp_path: Path) -> None:
    """抓「跟建議相同」判錯：說明改字仍算相同；檔、片、群名任一不同就不同；固定群不算進來。"""
    from dataclasses import replace
    from governance.status.shard_balance import same_as_registry, suggested_groups
    result, registry, _ = _moved(tmp_path)
    groups = suggested_groups(result)
    loaded = load_registry(registry)
    assert not same_as_registry(loaded, groups)
    assert same_as_registry(replace(loaded, groups=loaded.groups + groups), groups)
    reworded = tuple(replace(group, why='人改過的說明') for group in groups)
    assert same_as_registry(replace(loaded, groups=loaded.groups + reworded), groups)
    for changed in (replace(groups[0], shard='shard-a'), replace(groups[0], name='balance-other'),
                    replace(groups[0], files=groups[0].files + ('tests/test_verify_shards.py',))):
        assert not same_as_registry(replace(loaded, groups=loaded.groups + (changed,)), groups)
