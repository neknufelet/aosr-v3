"""輸入不可讀或形狀不明必須 ToolBroken；每题只壞指定那一格。"""
from __future__ import annotations

import json
from importlib import metadata
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from collections.abc import Callable
from typing import cast

import pytest

from governance.checks import green_shard_evidence as green
from governance.exit_codes import ToolBroken
from tests.test_green_shard_evidence import bundle as bundle
from tests.test_green_shard_evidence import EXPECTED, STEPS, Bundle, args_bytes, expected, fragment, provenance, roots, rewrite_manifest


@pytest.mark.parametrize('raw', ['{', '[]', 'null', '{"shard-a": {}}', '{"shard-a": {"result": 0}}'])
def test_needs_malformed(bundle: Bundle, raw: str) -> None:
    word = 'JSON' if raw == '{' else ('頂層' if raw in {'[]', 'null'} else 'result')
    with pytest.raises(ToolBroken, match='needs.*' + word):
        green.needs_problems(raw, bundle.registry)


@pytest.mark.parametrize('target', ['download', 'part', 'evidence', 'file', 'other-subtree', 'steps'])
def test_symlinks_never_followed(bundle: Bundle, tmp_path: Path, target: str) -> None:
    path = {'download': bundle.download, 'part': bundle.parts['shard-a'], 'evidence': bundle.root(),
            'file': bundle.root() / 'expected.txt', 'other-subtree': bundle.parts['shard-a'] / 'cloud',
            'steps': bundle.parts['shard-a'] / 'steps'}[target]
    real = tmp_path / 'real'
    if path.exists():
        path.rename(real)
    else:
        real.mkdir()
    path.symlink_to(real, target_is_directory=real.is_dir())
    with pytest.raises(ToolBroken, match='符號連結'):
        green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)


def test_provenance_alone_rejects_symlinked_evidence_dir(bundle: Bundle, tmp_path: Path) -> None:
    """不先過 part_dirs 單獨叫出處，證據資料夾本身是連結也要拒收。"""
    real = tmp_path / 'real'
    bundle.root().rename(real)
    bundle.root().symlink_to(real, target_is_directory=True)
    with pytest.raises(ToolBroken, match='符號連結'):
        provenance(bundle)


@pytest.mark.parametrize('target', ['text', 'subdir', 'link'])
def test_fragment_entry_shape_is_broken(bundle: Bundle, tmp_path: Path, target: str) -> None:
    steps = bundle.parts['shard-a'] / 'steps'
    if target == 'text':
        (steps / 'notes.txt').write_text('x')
    elif target == 'subdir':
        (steps / 'nested').mkdir()
    else:
        real = tmp_path / 'real.json'
        (steps / 'pytest.json').rename(real)
        (steps / 'pytest.json').symlink_to(real)
    with pytest.raises(ToolBroken, match='符號連結' if target == 'link' else 'JSON 一般檔案'):
        green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings)


@pytest.mark.parametrize('judge,query', [('inventory', 'is_dir'), ('inventory', 'exists'), ('inventory', 'is_file'),
                                         ('fragments', 'exists'), ('fragments', 'is_file')])
def test_kind_query_unreadable_is_broken(bundle: Bundle, monkeypatch: pytest.MonkeyPatch, judge: str, query: str) -> None:
    def unreadable(path: Path, *args: object, **kwargs: object) -> bool:
        raise PermissionError('模擬 stat EACCES')
    monkeypatch.setattr(Path, query, unreadable)
    with pytest.raises(ToolBroken, match='檔案種類讀不開.*EACCES'):
        if judge == 'inventory':
            green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)
        else:
            green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings)


def test_fragment_seconds_huge_integer_does_not_overflow(bundle: Bundle) -> None:
    (bundle.parts['shard-a'] / 'steps' / 'pytest.json').write_text(json.dumps(fragment() | {'seconds': 10 ** 400}))
    assert green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings) == []


def test_tests_attribute_too_long_is_broken() -> None:
    root = ET.fromstring(f'<testsuites><testsuite tests="{"9" * 5000}"><testcase classname="tests.x" name="t"/></testsuite></testsuites>')
    with pytest.raises(ToolBroken, match='tests.*數字讀不懂'):
        green.part_attr_problems({'shard-a': root}, 'junit.xml')


def test_testcase_without_classname_is_broken(bundle: Bundle) -> None:
    trees = roots(bundle)
    trees['shard-a'] = ET.fromstring('<testsuites><testsuite tests="1"><testcase name="test_a"/></testsuite></testsuites>')
    with pytest.raises(ToolBroken, match='classname.*缺屬性'):
        green.coverage_problems(expected(bundle), trees, args_bytes(bundle), bundle.registry, bundle.settings)


def test_evidence_dir_outside_artifact_root(bundle: Bundle) -> None:
    registry = replace(bundle.registry, evidence=replace(bundle.registry.evidence, dir='elsewhere'))
    with pytest.raises(ToolBroken, match='evidence.dir'):
        green.part_dirs(bundle.download, registry, '123', '2', bundle.settings)


def test_download_unreadable(bundle: Bundle) -> None:
    with pytest.raises(ToolBroken, match='讀不開'):
        green.part_dirs(bundle.download / 'absent', bundle.registry, '123', '2', bundle.settings)


def test_expected_filename_is_directory(bundle: Bundle) -> None:
    path = bundle.root() / 'expected.txt'
    path.unlink()
    path.mkdir()
    with pytest.raises(ToolBroken, match='一般檔案'):
        green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)


@pytest.mark.parametrize('raw', ['{', '[]', '{}', '\udcff'])
def test_manifest_unparseable_or_shape_wrong(bundle: Bundle, raw: str) -> None:
    (bundle.root() / 'manifest.json').write_bytes(raw.encode('utf-8', 'surrogatepass'))
    word = 'JSON' if raw in {'{', '\udcff'} else '鍵'
    with pytest.raises(ToolBroken, match='manifest.json.*' + word):
        provenance(bundle)


def test_manifest_unreadable(bundle: Bundle) -> None:
    (bundle.root() / 'manifest.json').unlink()
    with pytest.raises(ToolBroken, match='manifest.json.*檔案讀不開'):
        provenance(bundle)


@pytest.mark.parametrize('field,value', [('schema', 0), ('schema', 2), ('schema', True), ('job', 7),
                                        ('expected', []), ('expected', {'path': 'x', 'bytes': 0, 'sha256': 7})])
def test_manifest_field_shape_or_schema(bundle: Bundle, field: str, value: object) -> None:
    rewrite_manifest(bundle, field, value)
    with pytest.raises(ToolBroken, match=field):
        provenance(bundle)


@pytest.mark.parametrize('change', ['missing', 'extra'])
def test_manifest_keys_exact(bundle: Bundle, change: str) -> None:
    path = bundle.root() / 'manifest.json'
    raw = json.loads(path.read_text())
    if change == 'missing':
        raw.pop('event')
    else:
        raw['extra'] = 'x'
    path.write_text(json.dumps(raw))
    with pytest.raises(ToolBroken, match='鍵'):
        provenance(bundle)


def test_evidence_read_failure_does_not_become_empty_bytes(bundle: Bundle) -> None:
    (bundle.root() / 'junit.xml').unlink()
    with pytest.raises(ToolBroken, match='junit.xml.*檔案讀不開'):
        provenance(bundle)


@pytest.mark.parametrize('field,value', [('schema', 1), ('schema', 3), ('schema', True), ('exit_code', '0'),
                                        ('exit_code', False), ('run_id', 123), ('signal', '15'), ('argv', 'pytest'),
                                        ('name', None), ('seconds', True), ('stdout_tail', [1]), ('report', [])])
def test_fragment_bad_field(bundle: Bundle, field: str, value: object) -> None:
    raw = fragment()
    raw[field] = value
    (bundle.parts['shard-a'] / 'steps' / 'pytest.json').write_text(json.dumps(raw))
    with pytest.raises(ToolBroken, match=field):
        green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings)


@pytest.mark.parametrize('raw', ['{', '[]', '{}'])
def test_fragment_bad_json_or_shape(bundle: Bundle, raw: str) -> None:
    (bundle.parts['shard-a'] / 'steps' / 'pytest.json').write_text(raw)
    word = 'JSON' if raw == '{' else '鍵'
    with pytest.raises(ToolBroken, match='pytest.json.*' + word):
        green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings)


@pytest.mark.parametrize('steps', [None, [{}], [{'run': 7}], [{'run': 'uv run python -m governance.status.record_step --name'}]])
def test_steps_shape_cannot_be_guessed(bundle: Bundle, steps: object) -> None:
    with pytest.raises(ToolBroken, match='steps'):
        green.step_fragment_problems(steps, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings)


@pytest.mark.parametrize('raw,word', [(b'', '空'), (b'\xff\n', 'UTF-8'), (EXPECTED.replace(b'\n', b'\r\n'), 'CR'),
                                    (EXPECTED[:-1], '換行'), (b'\n' + EXPECTED, '空行'),
                                    (EXPECTED + EXPECTED, '重複'), (b'\n'.join(reversed(EXPECTED.splitlines())) + b'\n', '排序')])
def test_expected_bad_format(bundle: Bundle, raw: bytes, word: str) -> None:
    with pytest.raises(ToolBroken, match=word):
        green.expected_set({'shard-a': raw, 'shard-b': raw}, bundle.settings)


def test_expected_lists_differ_bytewise(bundle: Bundle) -> None:
    with pytest.raises(ToolBroken, match='逐位元組'):
        green.expected_set({'shard-a': EXPECTED, 'shard-b': EXPECTED + b'x\n'}, bundle.settings)


def test_expected_mapping_collision(bundle: Bundle) -> None:
    raw = ('\n'.join(('tests/test_verify_shards.py::TestCase.Nested::test_a',
                      'tests/test_verify_shards.py::TestCase::Nested::test_a')) + '\n').encode()
    with pytest.raises(ToolBroken, match='撞'):
        green.expected_set({'shard-a': raw, 'shard-b': raw}, bundle.settings)


def test_pytest_version_must_be_revalidated(bundle: Bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metadata, 'version', lambda package: 'future')
    with pytest.raises(ToolBroken, match='pytest.*future'):
        expected(bundle)


def test_version_lookup_os_error_is_broken(bundle: Bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(package: str) -> str:
        raise PermissionError('模擬 metadata EACCES')
    monkeypatch.setattr(metadata, 'version', unreadable)
    with pytest.raises(ToolBroken, match='pytest: 版本讀不到.*EACCES'):
        expected(bundle)


def test_version_lookup_unavailable(bundle: Bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(package: str) -> str:
        raise metadata.PackageNotFoundError(package)
    monkeypatch.setattr(metadata, 'version', missing)
    with pytest.raises(ToolBroken, match='pytest'):
        expected(bundle)


@pytest.mark.parametrize('raw', [b'<', b'<testsuites/>', b'\xff'])
def test_junit_unparseable_or_no_suites(bundle: Bundle, raw: bytes) -> None:
    word = 'testsuite' if raw == b'<testsuites/>' else '解不開'
    with pytest.raises(ToolBroken, match='junit.xml.*' + word):
        green.read_junit(raw, 'shard-a', 'junit.xml')


@pytest.mark.parametrize('value', [None, 'x', '-1'])
def test_junit_tests_attribute_bad_shape(bundle: Bundle, value: str | None) -> None:
    trees = roots(bundle)
    attrs = next(trees['shard-a'].iter('testsuite')).attrib
    if value is None:
        attrs.pop('tests')
    else:
        attrs['tests'] = value
    with pytest.raises(ToolBroken, match='tests'):
        green.part_attr_problems(trees, 'junit.xml')


def test_coverage_case_without_address(bundle: Bundle) -> None:
    trees = roots(bundle)
    next(trees['shard-a'].iter('testcase')).attrib.pop('name')
    with pytest.raises(ToolBroken, match='name'):
        green.coverage_problems(expected(bundle), trees, args_bytes(bundle), bundle.registry, bundle.settings)


def test_coverage_args_not_utf8(bundle: Bundle) -> None:
    args = args_bytes(bundle)
    args['shard-a'] = b'\xff'
    with pytest.raises(ToolBroken, match='pytest.args'):
        green.coverage_problems(expected(bundle), roots(bundle), args, bundle.registry, bundle.settings)


def test_no_suites_in_memory_is_also_broken(bundle: Bundle) -> None:
    with pytest.raises(ToolBroken, match='testsuite'):
        green.merged_receipt({'shard-a': ET.fromstring('<testsuites/>')})


@pytest.mark.parametrize('value', ['0', True, -1])
def test_manifest_file_bytes_wrong_type(bundle: Bundle, value: object) -> None:
    raw = json.loads((bundle.root() / 'manifest.json').read_text())['expected']
    raw['bytes'] = value
    rewrite_manifest(bundle, 'expected', raw)
    with pytest.raises(ToolBroken, match='expected.bytes'):
        provenance(bundle)


@pytest.mark.parametrize('raw', ['{"shard-a":{"result":"success","result":"failure"}}',
                               '{"shard-a":{"result":"success"},"shard-a":{"result":"success"}}'])
def test_duplicate_json_keys_cannot_hide_evidence(bundle: Bundle, raw: str) -> None:
    with pytest.raises(ToolBroken, match='鍵.*重複'):
        green.needs_problems(raw, bundle.registry)


@pytest.mark.parametrize('field,value', [('artifact_root', '../elsewhere'), ('steps_subdir', '/outside'),
                                        ('artifact_prefix', 'a/b'), ('manifest_schema', True),
                                        ('fragment_schema', 0), ('pytest_version', ''), ('wrapper_command', '')])
def test_invalid_settings_are_broken(bundle: Bundle, field: str, value: object) -> None:
    construct = cast(Callable[..., green.FinalizerSettings], replace)
    with pytest.raises(ToolBroken, match=f'settings.{field}'):
        construct(bundle.settings, **{field: value})


def test_empty_expected_inputs_cannot_be_judged(bundle: Bundle) -> None:
    with pytest.raises(ToolBroken, match='expected'):
        green.expected_set({}, bundle.settings)


def test_missing_part_in_coverage_is_not_empty_coverage(bundle: Bundle) -> None:
    trees = roots(bundle)
    trees.pop('shard-b')
    with pytest.raises(ToolBroken, match='coverage'):
        green.coverage_problems(expected(bundle), trees, args_bytes(bundle), bundle.registry, bundle.settings)


def test_fragment_path_is_not_a_regular_file(bundle: Bundle) -> None:
    path = bundle.parts['shard-a'] / 'steps' / 'pytest.json'
    path.unlink()
    path.mkdir()
    with pytest.raises(ToolBroken, match='pytest.json.*JSON 一般檔案'):
        green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings)


def test_duplicate_recorder_names_are_ambiguous(bundle: Bundle) -> None:
    with pytest.raises(ToolBroken, match='同名'):
        green.step_fragment_problems(STEPS + STEPS, bundle.parts['shard-a'], 'shard-a', bundle.own, bundle.settings)


def test_expected_nodeid_without_item_is_broken(bundle: Bundle) -> None:
    raw = b'tests/test_verify_shards.py\n'
    with pytest.raises(ToolBroken, match='nodeid'):
        green.expected_set({'shard-a': raw, 'shard-b': raw}, bundle.settings)


def test_empty_junit_part_attributes_are_broken() -> None:
    with pytest.raises(ToolBroken, match='缺分片樹'):
        green.part_attr_problems({}, 'junit.xml')


def test_empty_junit_merge_is_broken() -> None:
    with pytest.raises(ToolBroken, match='缺分片樹'):
        green.merged_receipt({})


def test_symlink_inspection_unreadable_is_broken(bundle: Bundle, monkeypatch: pytest.MonkeyPatch) -> None:
    def unreadable(path: Path) -> bool:
        raise PermissionError('模擬無法查看檔案種類')
    monkeypatch.setattr(Path, 'is_symlink', unreadable)
    with pytest.raises(ToolBroken, match='讀不開'):
        green.part_dirs(bundle.download, bundle.registry, '123', '2', bundle.settings)


@pytest.mark.parametrize('change', ['missing', 'extra', 'run-type', 'unused-sha-type'])
def test_step_fragment_own_identity_shape(bundle: Bundle, change: str) -> None:
    own = dict(bundle.own)
    if change == 'missing':
        own.pop('event')
    elif change == 'extra':
        own['extra'] = 'x'
    elif change == 'run-type':
        own['run_id'] = 123
    else:
        own['github_sha'] = None
    with pytest.raises(ToolBroken, match='收尾身分'):
        green.step_fragment_problems(STEPS, bundle.parts['shard-a'], 'shard-a', own, bundle.settings)
