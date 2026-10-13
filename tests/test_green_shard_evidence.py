"""分片收尾的合法控制與逐格違規；資料只在各題的暫存樹。"""
from __future__ import annotations

import hashlib
import json
from importlib import metadata
import xml.etree.ElementTree as ET
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path

import pytest

from governance.checks import green_shard_evidence as green
from governance.checks.green_must_be_real_green import _judge_receipt
from governance.loader import Card
from governance.verify_shards import Registry, load_registry
from tests.test_shard_seal import ENV
from tests.test_verify_shards import write_registry

NODES = ('tests/test_verify_shards.py::test_a', 'tests/test_shard_seal.py::test_b')
EXPECTED = ('\n'.join(sorted(NODES)) + '\n').encode()
WRAPPER = 'uv run python -m governance.status.record_step'
STEPS = [{'run': WRAPPER + ' --name pytest --out-dir receipts -- uv run pytest'}]


@dataclass
class Bundle:
    """考卷的兩片證據與注入身分。"""
    registry: Registry
    settings: green.FinalizerSettings
    download: Path
    parts: dict[str, Path]
    own: dict[str, object]

    def root(self, shard: str = 'shard-a') -> Path:
        """取得本題的證據目錄。"""
        return self.parts[shard] / 'shard'


def fragment(shard: str = 'shard-a') -> dict[str, object]:
    """手造下一版片段，沒有呼叫仍是舊版的抄寫員。"""
    return {'schema': 2, 'name': 'pytest', 'argv': ['uv', 'run', 'pytest', f'@{Path("governance") / "receipts" / "shard" / "pytest.args"}',
                                                   f'--junitxml={Path("governance") / "receipts" / "shard" / "junit.xml"}'],
            'started': '2026-10-13T00:00:00+00:00', 'seconds': 0.1, 'exit_code': 0,
            'signal': None, 'report': None, 'stdout_tail': [], 'stderr_tail': [],
            'run_id': '123', 'run_attempt': '2', 'job': shard}


def suite(shard: str = 'shard-a', tag: str = '') -> ET.Element:
    """小型 junit，classname 與 name 由人核對。"""
    module, name = ('test_verify_shards', 'test_a') if shard == 'shard-a' else ('test_shard_seal', 'test_b')
    child = f'<{tag}>植入的錯</{tag}>' if tag else ''
    return ET.fromstring(f'<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
                         f'<testcase classname="tests.{module}" name="{name}">{child}</testcase></testsuite></testsuites>')


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Bundle:
    """建立完整兩片；manifest 雜湊只取手寫資料，不從受測裁判推答案。"""
    registry = load_registry(write_registry(tmp_path))
    settings = green.FinalizerSettings('governance/receipts', 'receipts-', 'steps', 1, 2, '9.1.1', WRAPPER, 'uv run pytest')
    own = green.finalizer_identity(ENV | {'GITHUB_JOB': 'verify'}, lambda: ENV['GITHUB_SHA'])
    download = tmp_path / 'download'
    download.mkdir()
    parts: dict[str, Path] = {}
    for shard in registry.shards:
        part = download / f'receipts-123-2-{shard}'
        root = part / 'shard'
        root.mkdir(parents=True)
        args = b'tests/test_verify_shards.py\n' if shard == 'shard-a' else b'tests/test_shard_seal.py\n'
        values = {'expected': EXPECTED, 'args': args, 'junit': ET.tostring(suite(shard))}
        manifest: dict[str, object] = {'schema': 1, **own, 'job': shard, 'shard': shard}
        for key, content in values.items():
            filename = getattr(registry.evidence, key)
            (root / filename).write_bytes(content)
            manifest[key] = {'path': filename, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        (root / registry.evidence.manifest).write_text(json.dumps(manifest))
        steps = part / settings.steps_subdir
        steps.mkdir()
        (steps / 'pytest.json').write_text(json.dumps(fragment(shard)))
        parts[shard] = part
    monkeypatch.setattr(metadata, 'version', lambda package: '9.1.1')
    return Bundle(registry, settings, download, parts, own)


def roots(bundle: Bundle) -> dict[str, ET.Element]:
    """每片有自己的可改樹。"""
    return {shard: suite(shard) for shard in bundle.registry.shards}


def args_bytes(bundle: Bundle) -> dict[str, bytes]:
    """取本題手寫的參數證據。"""
    return {shard: (bundle.root(shard) / bundle.registry.evidence.args).read_bytes() for shard in bundle.registry.shards}


def expected(bundle: Bundle) -> green.Expected:
    """餵兩片的清單。"""
    return green.expected_set({shard: EXPECTED for shard in bundle.registry.shards}, bundle.settings, bundle.registry)


def provenance(bundle: Bundle) -> list[str]:
    """只考出處，不讓檔案集合或覆蓋率遮蔽這一格。"""
    return green.provenance_problems(bundle.own, 'shard-a', bundle.root(), bundle.registry, bundle.settings)


def rewrite_manifest(bundle: Bundle, key: str, value: object) -> None:
    """只改一格 manifest。"""
    path = bundle.root() / bundle.registry.evidence.manifest
    raw = json.loads(path.read_text())
    raw[key] = value
    path.write_text(json.dumps(raw))


def judge_card(tmp_path: Path) -> Card:
    """只供現有判樹函式的小卡；不載入真 repo 卡或改地板。"""
    return Card('green', '', [], 'files', '', '', '', 'blocking', 'github-actions-job', 'verify', [], {}, tmp_path,
                junit={'path': 'junit.xml', 'engine_bucket': 'tests.engine.', 'floor_stale_ratio': 2,
                       'bucket': [{'prefix': 'tests.', 'collected_floor': 2}]})


def test_all_legal_evidence_is_clean(bundle: Bundle, tmp_path: Path) -> None:
    assert green.needs_problems(json.dumps({s: {'result': 'success'} for s in bundle.registry.shards}), bundle.registry) == []
    parts, bad = green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)
    assert parts == bundle.parts
    assert bad == []
    for shard in bundle.registry.shards:
        assert green.provenance_problems(bundle.own, shard, bundle.root(shard), bundle.registry, bundle.settings) == []
        assert green.step_fragment_problems(STEPS, parts[shard], shard, bundle.own, bundle.settings, bundle.registry) == []
    trees = roots(bundle)
    assert green.coverage_problems(expected(bundle), trees, args_bytes(bundle), bundle.registry, bundle.settings) == []
    assert green.part_attr_problems(trees, bundle.registry.evidence.junit) == []
    single_card = replace(judge_card(tmp_path), junit={'path': 'junit.xml', 'engine_bucket': 'tests.engine.',
                          'floor_stale_ratio': 2, 'bucket': [{'prefix': 'tests.', 'collected_floor': 1}]})
    for shard, root in trees.items():
        assert _judge_receipt(single_card, shard, root) == []
    assert _judge_receipt(judge_card(tmp_path), 'merged', green.merged_receipt(trees)) == []


def test_symlink_above_download_root_is_runner_path(bundle: Bundle, tmp_path: Path) -> None:
    alias = tmp_path / 'alias'
    alias.symlink_to(tmp_path, target_is_directory=True)
    parts, bad = green.part_dirs(alias / 'download', bundle.registry, '123', '2', bundle.settings)
    assert bad == []
    for shard, part in parts.items():
        assert green.provenance_problems(bundle.own, shard, part / 'shard', bundle.registry, bundle.settings) == []
        assert green.step_fragment_problems(STEPS, part, shard, bundle.own, bundle.settings, bundle.registry) == []


def test_settings_are_frozen_and_not_defaults(bundle: Bundle) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(bundle.settings, 'artifact_prefix', 'other-')
    settings = replace(bundle.settings, artifact_prefix='other-')
    _, bad = green.part_dirs(bundle.download, bundle.registry, '123', '2', settings)
    assert any('other-123-2-shard-a' in item for item in bad)


@pytest.mark.parametrize('result', ['failure', 'cancelled', 'skipped', 'Success', 'SUCCESS', ''])
def test_needs_unsuccessful_shard(bundle: Bundle, result: str) -> None:
    bad = green.needs_problems(json.dumps({'shard-a': {'result': result}, 'shard-b': {'result': 'success'}}), bundle.registry)
    assert any('shard-a' in item and 'result' in item and repr(result) in item for item in bad)


@pytest.mark.parametrize('extra', [False, True], ids=['missing', 'extra'])
def test_needs_key_set(bundle: Bundle, extra: bool) -> None:
    raw = {s: {'result': 'success'} for s in bundle.registry.shards}
    if extra:
        raw['foreign'] = {'result': 'success'}
    else:
        raw.pop('shard-b')
    assert any('鍵集合' in item and ('foreign' if extra else 'shard-b') in item for item in green.needs_problems(json.dumps(raw), bundle.registry))


@pytest.mark.parametrize('change', ['missing', 'attempt', 'run', 'foreign', 'file'])
def test_artifact_names_are_exact(bundle: Bundle, change: str) -> None:
    first = bundle.parts['shard-a']
    if change == 'missing':
        first.rename(bundle.download.parent / first.name)
    elif change in {'attempt', 'run'}:
        first.rename(bundle.download / ('receipts-123-1-shard-a' if change == 'attempt' else 'receipts-124-2-shard-a'))
    elif change == 'foreign':
        (bundle.download / 'receipts-123-2-foreign').mkdir()
    else:
        (bundle.download / 'loose').write_text('x')
    _, bad = green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)
    assert any(('未登記的包' if change in {'foreign', 'file'} else '缺包') in item for item in bad)
    if change in {'attempt', 'run'}:
        wrong = 'receipts-123-1-shard-a' if change == 'attempt' else 'receipts-124-2-shard-a'
        assert any('未登記的包' in item and wrong in item for item in bad)


def test_stale_attempt_beside_full_set_is_only_unregistered(bundle: Bundle) -> None:
    """兩包俱全再混一包上一次嘗試的：只報未登記，不靠缺包碰巧紅。"""
    (bundle.download / 'receipts-123-1-shard-a').mkdir()
    _, bad = green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)
    assert any('未登記的包' in item and 'receipts-123-1-shard-a' in item for item in bad)
    assert not any('缺包' in item for item in bad)


@pytest.mark.parametrize('change', ['missing', 'extra', 'xml'])
def test_evidence_inventory(bundle: Bundle, change: str) -> None:
    if change == 'missing':
        (bundle.root() / 'expected.txt').unlink()
    elif change == 'extra':
        (bundle.root() / 'extra.txt').write_text('x')
    else:
        (bundle.parts['shard-a'] / 'pytest.junit.xml').write_text('<testsuites/>')
    _, bad = green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)
    word = {'missing': '缺證據', 'extra': '多證據', 'xml': '額外 XML'}[change]
    assert any('shard-a' in item and word in item for item in bad)


def test_other_artifact_subtrees_are_ignored(bundle: Bundle) -> None:
    cloud = bundle.parts['shard-a'] / 'cloud'
    cloud.mkdir()
    (cloud / 'mirror.json').write_text('{}')
    assert green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)[1] == []


@pytest.mark.parametrize('key', ['repository', 'workflow_path', 'workflow_ref', 'event', 'run_id', 'run_attempt',
                                 'github_sha', 'tested_sha', 'job', 'shard'])
def test_manifest_identity_mismatch(bundle: Bundle, key: str) -> None:
    value = 'b' * 40 if key.endswith('sha') else 'different'
    rewrite_manifest(bundle, key, value)
    assert any('shard-a' in item and f'.{key}=' in item and repr(value) in item for item in provenance(bundle))


def test_finalizer_github_sha_must_equal_head(bundle: Bundle) -> None:
    bundle.own['tested_sha'] = 'b' * 40
    assert any('收尾' in item and 'tested_sha' in item for item in provenance(bundle))


@pytest.mark.parametrize('key', ['expected', 'args', 'junit'])
@pytest.mark.parametrize('field,value', [('path', 'other'), ('sha256', '0' * 64), ('bytes', 999)])
def test_manifest_file_fields(bundle: Bundle, key: str, field: str, value: object) -> None:
    raw = json.loads((bundle.root() / 'manifest.json').read_text())
    row = raw[key]
    row[field] = value
    rewrite_manifest(bundle, key, row)
    assert any(f'{key}.{field}=' in item and repr(value) in item for item in provenance(bundle))


def test_file_replaced_after_seal(bundle: Bundle) -> None:
    (bundle.root() / 'junit.xml').write_bytes(b'<testsuites />')
    assert any('junit.sha256' in item for item in provenance(bundle))


@pytest.mark.parametrize('field,value', [('name', 'wrong'), ('exit_code', 1), ('exit_code', 143), ('signal', 15),
                                        ('run_id', '124'), ('run_attempt', '1'), ('job', 'shard-b')])
def test_fragment_fields_mismatch(bundle: Bundle, field: str, value: object) -> None:
    raw = fragment()
    raw[field] = value
    (bundle.parts['shard-a'] / 'steps' / 'pytest.json').write_text(json.dumps(raw))
    bad = green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings, bundle.registry)
    assert any('pytest.json' in item and f'.{field}=' in item and repr(value) in item for item in bad)


def test_missing_steps_directory_is_missing_fragments(bundle: Bundle) -> None:
    for path in (bundle.parts['shard-a'] / 'steps').iterdir():
        path.unlink()
    (bundle.parts['shard-a'] / 'steps').rmdir()
    bad = green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings, bundle.registry)
    assert any('缺片段' in item and 'pytest' in item for item in bad)


@pytest.mark.parametrize('extra', [False, True], ids=['missing', 'extra'])
def test_fragment_names_match_workflow(bundle: Bundle, extra: bool) -> None:
    path = bundle.parts['shard-a'] / 'steps' / 'pytest.json'
    if extra:
        path.with_name('foreign.json').write_text(json.dumps(fragment() | {'name': 'foreign'}))
    else:
        path.unlink()
    bad = green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings, bundle.registry)
    assert any(('多片段' if extra else '缺片段') in item for item in bad)


def test_recorder_uses_shared_shell_segmentation(bundle: Bundle) -> None:
    steps = [{'run': '# ' + WRAPPER + ' --name imaginary\n' + WRAPPER + ' \\\n --name "pytest" --out-dir receipts -- uv run pytest && echo "--name foreign"'}]
    assert green.step_fragment_problems(steps, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings, bundle.registry) == []


def test_child_recorder_string_is_not_another_step(bundle: Bundle) -> None:
    steps = [{'run': WRAPPER + ' --name pytest --out-dir receipts -- uv run python -m governance.status.record_step --name imaginary'}]
    assert green.step_fragment_problems(steps, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings, bundle.registry) == []


@pytest.mark.parametrize('change', ['missing', 'unknown', 'within', 'across', 'wrong-shard', 'args', 'args-extra', 'args-empty'])
def test_coverage_defects(bundle: Bundle, change: str) -> None:
    trees, args = roots(bundle), args_bytes(bundle)
    first = next(trees['shard-a'].iter('testsuite'))
    case = next(first.iter('testcase'))
    if change == 'missing':
        first.remove(case)
    elif change == 'unknown':
        case.set('name', 'test_unknown')
    elif change == 'within':
        first.append(ET.fromstring(ET.tostring(case)))
    elif change == 'across':
        next(trees['shard-b'].iter('testsuite')).append(ET.fromstring(ET.tostring(case)))
    elif change == 'wrong-shard':
        trees = {'shard-a': suite('shard-b'), 'shard-b': suite('shard-a')}
    elif change == 'args':
        args['shard-a'] = b'tests/test_shard_seal.py\n'
    elif change == 'args-extra':
        args['shard-a'] = b'tests/test_shard_seal.py\n' + b'tests/test_verify_shards.py\n'
    else:
        args['shard-a'] = b''
    bad = green.coverage_problems(expected(bundle), trees, args, bundle.registry, bundle.settings)
    word = {'missing': '漏題', 'unknown': '不明題', 'within': '片內重複', 'across': '跨片重複',
            'wrong-shard': '不屬本片', 'args': 'pytest.args', 'args-extra': 'pytest.args', 'args-empty': 'pytest.args'}[change]
    assert any(word in item for item in bad)


def test_collection_error_in_shard_is_red_not_broken(bundle: Bundle) -> None:
    """pytest 收集期爆掉交的就是 classname 空字串加 error；形狀照 pytest 9.1.1 實跑抄。"""
    trees = roots(bundle)
    trees['shard-a'] = ET.fromstring('<testsuites><testsuite tests="1"><testcase classname="" name="tests.test_verify_shards">'
                                     '<error message="collection failure">ImportError</error></testcase></testsuite></testsuites>')
    bad = green.coverage_problems(expected(bundle), trees, args_bytes(bundle), bundle.registry, bundle.settings)
    assert 'error' in bad[0] and 'collection failure' in bad[0]
    assert any('不明題' in item for item in bad)
    assert any('漏題' in item and 'test_a' in item for item in bad)


@pytest.mark.parametrize('option', ['-k test_absent', '--deselect=tests/test_verify_shards.py::test_a'])
def test_runtime_filtering_leaves_missing_tests(bundle: Bundle, option: str) -> None:
    """清點完整，手造實跑篩題後的空 junit；不用重新考清點守衛。"""
    trees = roots(bundle)
    trees['shard-a'] = ET.fromstring('<testsuites><testsuite tests="0"/></testsuites>')
    bad = green.coverage_problems(expected(bundle), trees, args_bytes(bundle), bundle.registry, bundle.settings)
    assert any('漏題' in item and 'test_a' in item for item in bad), option
    assert any(item.startswith('片 shard-a ') and 'tests/test_verify_shards.py' in item for item in bad if '漏題' in item)


def test_testsuite_root_is_accepted(bundle: Bundle) -> None:
    trees = roots(bundle)
    trees['shard-a'] = next(trees['shard-a'].iter('testsuite'))
    assert green.coverage_problems(expected(bundle), trees, args_bytes(bundle), bundle.registry, bundle.settings) == []
    assert green.part_attr_problems(trees, bundle.registry.evidence.junit) == []


@pytest.mark.parametrize('tag', ['failure', 'error'])
def test_failed_duplicate_is_reported_first(bundle: Bundle, tag: str) -> None:
    trees = roots(bundle)
    first = next(trees['shard-a'].iter('testsuite'))
    first.append(next(suite(tag=tag).iter('testcase')))
    bad = green.coverage_problems(expected(bundle), trees, args_bytes(bundle), bundle.registry, bundle.settings)
    assert tag in bad[0] and 'test_a' in bad[0]
    assert any('片內重複' in item for item in bad)


def test_part_suite_attribute_cannot_cancel_other_part(bundle: Bundle) -> None:
    trees = roots(bundle)
    next(trees['shard-a'].iter('testsuite')).set('tests', '2')
    next(trees['shard-b'].iter('testsuite')).set('tests', '0')
    bad = green.part_attr_problems(trees, 'junit.xml')
    assert any('shard-a' in item and 'tests=2' in item for item in bad)
    assert any('shard-b' in item and 'tests=0' in item for item in bad)


@pytest.mark.parametrize('tag', ['skipped', 'failure', 'error'])
def test_merged_junit_keeps_existing_red_verdict(bundle: Bundle, tmp_path: Path, tag: str) -> None:
    trees = roots(bundle)
    trees['shard-a'] = suite(tag=tag)
    before = {shard: ET.tostring(root) for shard, root in trees.items()}
    merged = green.merged_receipt(trees)
    bad = _judge_receipt(judge_card(tmp_path), 'merged', merged)
    assert any(('skipped' if tag == 'skipped' else 'failures=') in item for item in bad)
    assert {shard: ET.tostring(root) for shard, root in trees.items()} == before
    assert [ET.tostring(root) for root in merged] == [ET.tostring(next(root.iter('testsuite'))) for root in trees.values()]


def test_mapped_addresses_match_pytest_xml_names(bundle: Bundle) -> None:
    raw = b'tests/test_verify_shards.py::TestCase::test_a[a::b\\n]\n'
    result = green.expected_set({'shard-a': raw, 'shard-b': raw}, bundle.settings, bundle.registry)
    assert dict(result.files) == {('tests.test_verify_shards.TestCase', 'test_a[a::b\\n]'): 'tests/test_verify_shards.py'}


def test_xml_control_character_in_name_is_escaped(bundle: Bundle) -> None:
    raw = b'tests/test_verify_shards.py::test_a[\x01]\n'
    result = green.expected_set({'shard-a': raw, 'shard-b': raw}, bundle.settings, bundle.registry)
    assert ('tests.test_verify_shards', 'test_a[#x01]') in dict(result.files)


def test_multiple_suites_are_kept_verbatim(bundle: Bundle) -> None:
    tree = ET.fromstring('<testsuites><testsuite tests="0" name="a"/><testsuite tests="0" name="b"/></testsuites>')
    assert green.part_attr_problems({'shard-a': tree}, 'junit.xml') == []
    assert ET.tostring(green.merged_receipt({'shard-a': tree})) == ET.tostring(tree)
