"""歷史秒數分配的決定性、整群與退回契約；只用暫存收據。"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from governance import shard_timing as subject
from governance.exit_codes import ToolBroken
from governance.verify_shards import Group, Registry, load_registry, render_inputs
from tests.test_verify_shards import write_registry

FILES = tuple(f'tests/test_{letter}.py' for letter in 'abcdefghij')


def registry_at(tmp_path: Path) -> Registry:
    """五片、單一共用前置群，群名與檔名刻意不同序。"""
    registry = load_registry(write_registry(tmp_path))
    return replace(registry, shards=tuple(f'shard-{letter}' for letter in 'abcde'),
                   groups=(Group('shared', 'shard-b', '共用前置', FILES[:2]),))


def test_largest_first_keeps_group_and_uses_median_for_new_files(tmp_path: Path) -> None:
    registry = registry_at(tmp_path)
    weights = dict(zip(FILES[:6], [20.0, 20.0, 30.0, 10.0, 10.0, 10.0], strict=True))
    assigned = subject.assign(FILES, registry, weights)
    assert assigned[FILES[0]] == assigned[FILES[1]] == 'shard-a'
    assert assigned[FILES[2]] == 'shard-b'
    # 已知中位數為 15，新檔先於十秒檔落片。
    assert [assigned[file] for file in FILES[6:]] == ['shard-c', 'shard-d', 'shard-e', 'shard-c']
    assert set(assigned) == set(FILES)


def test_equal_cost_ties_use_path_and_ignore_input_order(tmp_path: Path) -> None:
    registry = replace(registry_at(tmp_path), groups=())
    weights = dict.fromkeys(FILES, 1.0)
    expected = dict(zip(FILES, registry.shards * 2, strict=True))
    assert subject.assign(FILES, registry, weights) == expected
    assert subject.assign(tuple(reversed(FILES)), registry, dict(reversed(list(weights.items())))) == expected


@pytest.mark.parametrize('seconds', [-1.0, float('nan'), float('inf')])
def test_invalid_seconds_are_rejected(tmp_path: Path, seconds: float) -> None:
    with pytest.raises(ToolBroken, match='秒數'):
        subject.assign(FILES, registry_at(tmp_path), {FILES[0]: seconds})


def test_fallback_is_the_original_registry_and_path_hash(tmp_path: Path) -> None:
    registry = registry_at(tmp_path)
    nodes = [f'{file}::test_one' for file in FILES]
    snapshot = subject.fallback('a' * 40, '找不到成功那一跑')
    assert snapshot['timing_source'] == 'fallback'
    assert snapshot['reason'] == '找不到成功那一跑'
    assignments = subject.from_snapshot(FILES, registry, snapshot)
    for shard in registry.shards:
        assert render_inputs(nodes, registry, shard, assignments=assignments) == render_inputs(nodes, registry, shard)


def test_same_snapshot_yields_same_plan_in_five_preparations(tmp_path: Path) -> None:
    registry = registry_at(tmp_path)
    snapshot = {'timing_source': 'history', 'run_id': '123', 'sha': 'a' * 40, 'reason': '',
                'times': dict.fromkeys(FILES, 1.0)}
    plans = [subject.from_snapshot(tuple(reversed(FILES)) if i % 2 else FILES, registry, snapshot) for i in range(5)]
    assert all(plan == plans[0] for plan in plans)
    assert len({subject.assignment_hash(plan) for plan in plans}) == 1
    changed = plans[0] | {FILES[0]: 'shard-e'}
    assert subject.assignment_hash(changed) != subject.assignment_hash(plans[0])


@pytest.mark.parametrize('event,expected', [({'pull_request': {'base': {'sha': 'a' * 40}}}, 'a' * 40),
                                          ({'before': 'b' * 40}, 'b' * 40)])
def test_event_pins_history_commit(event: dict[str, object], expected: str) -> None:
    assert subject.history_sha(event) == expected


def test_event_without_history_commit_is_rejected() -> None:
    with pytest.raises(ToolBroken, match='提交'):
        subject.history_sha({})


def test_history_requires_exact_successful_mainline_run(tmp_path: Path) -> None:
    rows: list[dict[str, object]] = [
        {'databaseId': 3, 'headSha': 'a' * 40, 'status': 'in_progress', 'conclusion': '', 'event': 'push', 'headBranch': 'main'},
        {'databaseId': 2, 'headSha': 'b' * 40, 'status': 'completed', 'conclusion': 'success', 'event': 'push', 'headBranch': 'main'},
        {'databaseId': 5, 'headSha': 'a' * 40, 'status': 'completed', 'conclusion': 'success', 'event': 'pull_request', 'headBranch': 'main'},
        {'databaseId': 4, 'headSha': 'a' * 40, 'status': 'completed', 'conclusion': 'success', 'event': 'push', 'headBranch': 'main'},
    ]
    assert subject.select_run(rows, 'a' * 40, 'main') == '4'
    assert subject.select_run(list(reversed(rows)), 'a' * 40, 'main') == '4'
    with pytest.raises(ToolBroken, match='成功'):
        subject.select_run(rows[:3], 'a' * 40, 'main')


def test_history_junit_reuses_per_test_reader_and_rejects_partial_receipts(tmp_path: Path) -> None:
    from governance.status import shard_balance
    root = tmp_path / 'history'
    root.mkdir()
    for shard, seconds in [('shard-a', 2), ('shard-b', 3)]:
        folder = root / shard
        folder.mkdir()
        (folder / 'junit.xml').write_text(
            f'<testsuite><testcase classname="tests.test_a" name="first" time="{seconds}"/>'
            f'<testcase classname="tests.test_b.TestClass" name="second" time="{seconds}"/></testsuite>')
    assert shard_balance.read_timing_junits(root, FILES) == {FILES[0]: 5.0, FILES[1]: 5.0}
    (root / 'shard-b' / 'junit.xml').write_text('<broken')
    with pytest.raises(ToolBroken, match='收據'):
        shard_balance.read_timing_junits(root, FILES)


@pytest.mark.parametrize('broken', ['missing-run', 'network', 'corrupt'])
def test_snapshot_read_failure_returns_explicit_whole_run_fallback(tmp_path: Path, broken: str) -> None:
    def reader(sha: str) -> tuple[str, dict[str, float]]:
        assert sha == 'a' * 40
        raise ToolBroken(broken)
    result = subject.resolve_snapshot({'before': 'a' * 40}, reader)
    assert result == subject.fallback('a' * 40, broken)
    plans = [subject.from_snapshot(FILES, registry_at(tmp_path), result) for _ in range(5)]
    assert all(plan == plans[0] for plan in plans)


def test_successful_snapshot_keeps_run_and_seconds() -> None:
    result = subject.resolve_snapshot({'before': 'a' * 40}, lambda sha: ('42', {FILES[0]: 3.0}))
    assert result == {'timing_source': 'history', 'run_id': '42', 'sha': 'a' * 40, 'reason': '', 'times': {FILES[0]: 3.0}}


def test_prepare_and_seal_preserve_timing_plan(tmp_path: Path) -> None:
    import json
    from governance.status.shard_prepare import prepare
    from governance.status.shard_seal import seal
    from tests.test_shard_seal import ENV
    registry = registry_at(tmp_path)
    snapshot = {'timing_source': 'history', 'run_id': '42', 'sha': 'a' * 40, 'reason': '',
                'times': dict.fromkeys(FILES, 1.0)}
    hashes = set()
    for shard in registry.shards:
        out = tmp_path / shard / 'shard'
        prepare([f'{file}::test_one' for file in FILES], registry, shard, out, timing=snapshot)
        seal(registry, shard, out, env=ENV | {'GITHUB_JOB': shard}, head=lambda: 'b' * 40)
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['schema'] == 3
        assert manifest['timing_source'] == 'history'
        assert manifest['timing']['run_id'] == '42'
        hashes.add(manifest['assignment_sha256'])
    assert len(hashes) == 1


@pytest.mark.parametrize('mutation', ['source', 'hash', 'weights', 'missing'])
def test_finalizer_rejects_mixed_or_tampered_timing(tmp_path: Path, mutation: str) -> None:
    from governance.checks import green_shard_evidence as green
    registry = registry_at(tmp_path)
    snapshot = {'timing_source': 'history', 'run_id': '42', 'sha': 'a' * 40, 'reason': '',
                'times': dict.fromkeys(FILES, 1.0)}
    assigned = subject.from_snapshot(FILES, registry, snapshot)
    manifests = {shard: {'timing_source': 'history', 'timing': snapshot.copy(),
                         'assignment_sha256': subject.assignment_hash(assigned)} for shard in registry.shards}
    assert green.timing_consensus(manifests, FILES, registry) == (assigned, [])
    target = manifests['shard-e']
    if mutation == 'source':
        target['timing'] = subject.fallback('a' * 40, '網路失敗')
        target['timing_source'] = 'fallback'
    elif mutation == 'hash':
        target['assignment_sha256'] = '0' * 64
    elif mutation == 'weights':
        target['timing'] = snapshot | {'times': {FILES[0]: 100.0}}
    else:
        del manifests['shard-e']
    _, bad = green.timing_consensus(manifests, FILES, registry)
    assert bad and any('耗時' in item or '分配' in item for item in bad)


def test_finalizer_fallback_is_reported_and_not_red(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from governance.checks import green_shard_evidence as green
    registry = registry_at(tmp_path)
    snapshot = subject.fallback('a' * 40, '沒有成功收據')
    assigned = subject.from_snapshot(FILES, registry, snapshot)
    manifests = {shard: {'timing_source': 'fallback', 'timing': snapshot,
                         'assignment_sha256': subject.assignment_hash(assigned)} for shard in registry.shards}
    assert green.timing_consensus(manifests, FILES, registry) == (assigned, [])
    assert '沒有成功收據' in capsys.readouterr().err


def test_local_snapshot_preview_uses_the_same_assignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        capsys: pytest.CaptureFixture[str]) -> None:
    import json
    from governance.status import shard_balance
    registry = registry_at(tmp_path)
    text = (tmp_path / 'verify-shards.toml').read_text().replace('["shard-a", "shard-b"]',
                                                               '["shard-a", "shard-b", "shard-c", "shard-d", "shard-e"]')
    # 使用此考卷的兩檔群，其他檔為未登記。
    text = text.replace('files = ["tests/test_verify_shards.py"]', f'files = {json.dumps(list(FILES[:2]))}')
    (tmp_path / 'verify-shards.toml').write_text(text)
    for file in FILES:
        path = tmp_path / file
        path.parent.mkdir(exist_ok=True)
        path.write_text('def test_one():\n    assert True\n')
    (tmp_path / 'pyproject.toml').write_text('[tool.pytest.ini_options]\ntestpaths = ["tests"]\n')
    (tmp_path / 'tests' / 'test_empty.py').write_text('"""空的輔助模組不是考卷。"""\n')
    snapshot = {'timing_source': 'history', 'run_id': '42', 'sha': 'a' * 40, 'reason': '', 'times': dict.fromkeys(FILES, 1.0)}
    (tmp_path / 'snapshot.json').write_text(json.dumps(snapshot))
    monkeypatch.chdir(tmp_path)
    assert shard_balance.main(['--snapshot', 'snapshot.json', '--registry', 'verify-shards.toml']) == 0
    printed = capsys.readouterr().out
    assert 'test_empty.py' not in printed
    for file, shard in subject.from_snapshot(FILES, registry, snapshot).items():
        assert f'{file} → {shard}' in printed


@pytest.mark.parametrize('case', ['five-history', 'five-fallback'])
def test_five_shard_finalizer_controls(case: str) -> None:
    from governance.checks import green_must_be_real_green as green
    root = Path(__file__).resolve().parents[1] / 'governance/fixtures/green-must-be-real-green-green' / case
    assert green.check(root, [path for path in root.rglob('*') if path.is_file()]) == []


@pytest.mark.parametrize('case', ['case-five-timing-source-mismatch', 'case-five-assignment-hash-mismatch'])
def test_five_shard_finalizer_negative_fixtures(case: str) -> None:
    from governance.checks import green_must_be_real_green as green
    root = Path(__file__).resolve().parents[1] / 'governance/fixtures/green-must-be-real-green' / case
    bad = green.check(root, [path for path in root.rglob('*') if path.is_file()])
    assert bad and all('耗時來源或分配雜湊不同' in item for item in bad), bad


@pytest.mark.parametrize('fault,word', [('missing', '漏題'), ('overlap', '跨片重複')])
def test_five_shard_union_and_disjointness(tmp_path: Path, fault: str, word: str) -> None:
    import hashlib
    import json
    import shutil
    from xml.etree import ElementTree as ET
    from governance.checks import green_must_be_real_green as green
    source = Path(__file__).resolve().parents[1] / 'governance/fixtures/green-must-be-real-green-green/five-history'
    root = tmp_path / 'control'
    shutil.copytree(source, root)
    parts = root / 'governance/receipts/parts'
    part = parts / 'receipts-123-2-shard-e'
    path = part / 'shard' / 'junit.xml'
    tree = ET.parse(path)
    suite = next(tree.iter('testsuite'))
    if fault == 'missing':
        suite.remove(next(suite.iter('testcase')))
        suite.set('tests', '8')
    else:
        other = ET.parse(parts / 'receipts-123-2-shard-a' / 'shard' / 'junit.xml')
        suite.append(next(other.iter('testcase')))
        suite.set('tests', '10')
    tree.write(path)
    step = part / 'steps' / 'pytest.json'
    fragment = json.loads(step.read_text())
    content = path.read_bytes()
    fragment['output'].update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
    step.write_text(json.dumps(fragment))
    bad = green.check(root, [path for path in root.rglob('*') if path.is_file()])
    assert any(word in item for item in bad), bad


@pytest.mark.parametrize('result', ['failure', 'skipped', 'cancelled'])
def test_five_shard_finalizer_rejects_upstream_failure(tmp_path: Path, result: str) -> None:
    import json
    from governance.checks import green_shard_evidence as green
    registry = registry_at(tmp_path)
    needs = {shard: {'result': 'success'} for shard in registry.shards}
    assert green.needs_problems(json.dumps(needs), registry) == []
    needs['shard-e']['result'] = result
    assert green.needs_problems(json.dumps(needs), registry) == [f"片 shard-e needs.result={result!r}，預期 'success'"]


@pytest.mark.parametrize('group_files,owners', [
    ((), ('a', 'b', 'c', 'd', 'e', 'a', 'b', 'c', 'd', 'e')),
    (FILES[:4], ('a', 'a', 'a', 'a', 'b', 'c', 'd', 'e', 'b', 'c')),
], ids=['ten-independent-files', 'count-files-in-shared-group'])
def test_zero_seconds_ties_prefer_fewer_files_before_shard_name(
    tmp_path: Path, group_files: tuple[str, ...], owners: tuple[str, ...],
) -> None:
    """全零秒仍分到五片；共用群中的每個檔都要計數，不能只計群數。"""
    groups = (Group('shared', 'shard-b', '共用前置', group_files),) if group_files else ()
    registry = replace(registry_at(tmp_path), groups=groups)
    expected = dict(zip(FILES, (f'shard-{letter}' for letter in owners), strict=True))
    assigned = subject.assign(FILES, registry, dict.fromkeys(FILES, 0.0))
    assert assigned == expected
    assert subject.assign(tuple(reversed(FILES)), registry, dict.fromkeys(reversed(FILES), 0.0)) == expected
    nodes = [f'{file}::test_one' for file in FILES]
    for shard in registry.shards:
        assert render_inputs(nodes, registry, shard, assignments=assigned).args.splitlines() == sorted(
            file for file, owner in expected.items() if owner == shard)


@pytest.mark.parametrize('time', [None, '-1', 'nan', 'inf'])
def test_missing_or_invalid_junit_seconds_fall_back_for_the_whole_run(tmp_path: Path, time: str | None) -> None:
    """前片有效、後片缺秒數或秒數壞掉時，不能留下局部歷史資料。"""
    from governance.status.shard_balance import read_timing_junits
    for shard, attribute in [('shard-a', ' time="3"'), ('shard-b', '' if time is None else f' time="{time}"')]:
        folder = tmp_path / shard
        folder.mkdir()
        (folder / 'junit.xml').write_text(
            f'<testsuite><testcase classname="tests.test_a" name="test_one"{attribute}/></testsuite>')
    def reader(sha: str) -> tuple[str, dict[str, float]]:
        return '42', read_timing_junits(tmp_path, FILES)
    snapshot = subject.resolve_snapshot({'before': 'a' * 40}, reader)
    assert snapshot['timing_source'] == 'fallback'
    assert snapshot['times'] == {}
    assert 'time' in str(snapshot['reason'])
    if time is None:
        assert '缺' in str(snapshot['reason'])
