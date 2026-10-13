"""設定、模式來源與輸入故障；環境只由指定小函式注入，真 git 只在沙箱。"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections.abc import Mapping

import pytest

from governance.checks import green_must_be_real_green as subject
from governance.checks import green_shard_evidence as evidence
from governance.loader import load_card
from governance.exit_codes import ToolBroken
from tests.conftest import GitSandbox
from tests.test_green_finalizer_mode import control as control
from tests.test_green_finalizer_mode import files

SETTINGS_PATH = Path('governance') / 'rules' / 'minimal-card.toml'
PARTS_PATH = Path('governance') / 'receipts' / 'parts'
WORKFLOW_PATH = Path('.github') / 'workflows' / 'verify-shards.yml'
RUN_PATH = 'governance/fixture-finalizer-run.json'
KEYS = ('finalizer_needs_env', 'fixture_finalizer_run_file', 'shard_registry', 'parts_dir',
        'artifact_root', 'artifact_prefix', 'steps_subdir', 'manifest_schema', 'fragment_schema',
        'pytest_version', 'wrapper_command')


def part_path(root: Path, shard: str) -> Path:
    """此題某片證據包；測試地址以路徑段表達，不當真樹引用。"""
    return root / PARTS_PATH / f'receipts-123-2-{shard}'


def rewrite_setting(root: Path, key: str, value: str | None) -> None:
    """改暫存樣本卡的一格 TOML 原文，None 代表刪那格。"""
    path = root / SETTINGS_PATH
    lines = path.read_text().splitlines()
    path.write_text('\n'.join((f'{key} = {value}' if value is not None else '')
                             if line.startswith(f'{key} = ') else line for line in lines) + '\n')


@pytest.mark.parametrize('key', KEYS)
def test_setting_missing_key_is_broken(control: Path, key: str) -> None:
    rewrite_setting(control, key, None)
    with pytest.raises(ToolBroken, match='settings.*' + re.escape(f"缺 ['{key}']；多 []")):
        subject.check(control, files(control))


@pytest.mark.parametrize('key', KEYS)
def test_setting_wrong_type_is_broken(control: Path, key: str) -> None:
    rewrite_setting(control, key, '"one"' if key.endswith('_schema') else '7')
    with pytest.raises(ToolBroken, match=f'settings.{key}'):
        subject.check(control, files(control))


@pytest.mark.parametrize('key', [key for key in KEYS if not key.endswith('_schema')])
@pytest.mark.parametrize('value', ['""', '"  "'])
def test_setting_blank_string_is_broken(control: Path, key: str, value: str) -> None:
    rewrite_setting(control, key, value)
    with pytest.raises(ToolBroken, match=f'settings.{key}'):
        subject.check(control, files(control))


@pytest.mark.parametrize('key', ['manifest_schema', 'fragment_schema'])
@pytest.mark.parametrize('value', ['0', '-1', 'true', '1.5'])
def test_setting_schema_requires_positive_integer(control: Path, key: str, value: str) -> None:
    rewrite_setting(control, key, value)
    with pytest.raises(ToolBroken, match=f'settings.{key}'):
        subject.check(control, files(control))


def test_setting_extra_key_is_broken(control: Path) -> None:
    path = control / SETTINGS_PATH
    path.write_text(path.read_text() + 'extra = "unexpected"\n')
    with pytest.raises(ToolBroken, match='settings.*extra'):
        subject.check(control, files(control))


def test_table_names_sorted_missing_and_extra_keys() -> None:
    """缺／多清單各自排序，預期鍵仍保留；不能靠完整鍵表冒充缺哪一格。"""
    from governance.verify_shards import _table
    with pytest.raises(ToolBroken) as caught:
        _table({'z-extra': 'x', 'a-extra': 'x', 'kept': 'x'}, '此題表',
               frozenset({'z-missing', 'a-missing', 'kept'}))
    message = str(caught.value)
    assert "預期鍵 ['a-missing', 'kept', 'z-missing']" in message
    assert "缺 ['a-missing', 'z-missing']；多 ['a-extra', 'z-extra']" in message


def test_setting_array_table_is_broken(control: Path) -> None:
    path = control / SETTINGS_PATH
    path.write_text(path.read_text().replace('[settings]', '[[settings]]'))
    with pytest.raises(ToolBroken, match='settings.*表'):
        subject.check(control, files(control))


def test_absent_settings_preserves_single_machine(control: Path) -> None:
    path = control / SETTINGS_PATH
    path.write_text(path.read_text().split('[settings]')[0])
    bad = subject.check(control, files(control))
    assert bad and all('收據' in item and '不存在' in item for item in bad)


def inputs(root: Path) -> tuple[Mapping[str, str], str]:
    """只讀此題樣本身分。"""
    row = json.loads((root / RUN_PATH).read_text())
    return row['env'], row['head']


def fail_env() -> Mapping[str, str]:
    """樣本樹不准讀行程環境。"""
    raise AssertionError('樣本樹讀了行程環境')


def test_fixture_with_run_file_ignores_process_env(control: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from governance.checks import green_finalizer_mode as finalizer
    monkeypatch.setattr(finalizer, 'process_env', fail_env)
    assert subject.check(control, files(control)) == []


def test_fixture_without_run_file_ignores_process_env(control: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from governance.checks import green_finalizer_mode as finalizer
    (control / RUN_PATH).unlink()
    monkeypatch.setattr(finalizer, 'process_env', fail_env)
    bad = subject.check(control, files(control))
    assert bad and all('收據' in item and '不存在' in item for item in bad)


def test_real_git_root_rejects_fixture_file(git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    from governance.checks import green_finalizer_mode as finalizer
    root = git_sandbox.root
    (root / 'governance').mkdir()
    (root / RUN_PATH).write_text('{}')
    monkeypatch.setattr(finalizer, 'process_env', fail_env)
    settings = finalizer.read_settings(load_card(Path(__file__).resolve().parents[1] /
                                               'governance/fixtures/green-must-be-real-green-green/control' / SETTINGS_PATH,
                                               Path(__file__).resolve().parents[1] /
                                               'governance/fixtures/green-must-be-real-green-green/control'))
    assert settings is not None
    with pytest.raises(ToolBroken, match='真的 git 工作樹的根'):
        finalizer.resolve_run(root, settings)


@pytest.mark.parametrize('present', [True, False])
def test_real_git_root_uses_trigger_key(git_sandbox: GitSandbox, control: Path,
                                       monkeypatch: pytest.MonkeyPatch, present: bool) -> None:
    from governance.checks import green_finalizer_mode as finalizer
    git_sandbox.git('commit', '--allow-empty', '-m', '收尾觸發沙箱')
    env, _head = inputs(control)
    env = dict(env, GITHUB_SHA=git_sandbox.git('rev-parse', 'HEAD').stdout.strip())
    if not present:
        env.pop('AOSR_FINALIZER_NEEDS')
    monkeypatch.setattr(finalizer, 'process_env', lambda: env)
    settings = finalizer.read_settings(load_card(control / SETTINGS_PATH, control))
    assert settings is not None
    resolved = finalizer.resolve_run(git_sandbox.root, settings)
    if present:
        assert resolved is not None
        assert resolved.env == env
        assert resolved.own['tested_sha'] == env['GITHUB_SHA']
    else:
        assert resolved is None


def test_empty_trigger_enters_finalizer_then_is_broken(git_sandbox: GitSandbox, control: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    from governance.checks import green_finalizer_mode as finalizer
    git_sandbox.git('commit', '--allow-empty', '-m', '空觸發沙箱')
    env, _head = inputs(control)
    env = dict(env, AOSR_FINALIZER_NEEDS='', GITHUB_SHA=git_sandbox.git('rev-parse', 'HEAD').stdout.strip())
    monkeypatch.setattr(finalizer, 'process_env', lambda: env)
    settings = finalizer.read_settings(load_card(control / SETTINGS_PATH, control))
    assert settings is not None
    resolved = finalizer.resolve_run(git_sandbox.root, settings)
    assert resolved is not None
    with pytest.raises(ToolBroken, match='needs.*JSON'):
        finalizer.judge(load_card(control / SETTINGS_PATH, control), control, settings, resolved, subject._judge_receipt)


@pytest.mark.parametrize('field,value,word', [('top', [], '頂層'), ('extra', 'x', '鍵'), ('env', [], 'env'),
                                            ('head', 'ABC', 'head'), ('head', 7, 'head'),
                                            ('missing-head', None, 'head'), ('env-value', 7, 'GITHUB_RUN_ID'),
                                            ('missing-env-key', None, 'GITHUB_RUN_ID'), ('env-extra', 'x', 'extra')])
def test_fixture_identity_shape_is_broken(control: Path, field: str, value: object, word: str) -> None:
    path = control / RUN_PATH
    row = json.loads(path.read_text())
    if field == 'top':
        row = value
    elif field == 'missing-head':
        row.pop('head')
    elif field == 'env-value':
        row['env']['GITHUB_RUN_ID'] = value
    elif field == 'missing-env-key':
        row['env'].pop('GITHUB_RUN_ID')
    elif field == 'env-extra':
        row['env']['extra'] = value
    else:
        row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ToolBroken, match=word):
        subject.check(control, files(control))


@pytest.mark.parametrize('content', ['{', '\udcff'])
def test_fixture_identity_unparseable(control: Path, content: str) -> None:
    (control / RUN_PATH).write_bytes(content.encode('utf-8', 'surrogatepass'))
    with pytest.raises(ToolBroken, match='fixture-finalizer-run.json.*JSON'):
        subject.check(control, files(control))


def test_part_problems_stop_before_unreadable_manifest(control: Path) -> None:
    parts = control / PARTS_PATH
    (parts / 'receipts-123-2-shard-b').rename(parts / 'receipts-123-1-shard-b')
    (parts / 'receipts-123-2-shard-a' / 'shard' / 'manifest.json').write_text('{')
    bad = subject.check(control, files(control))
    assert bad and all('未登記的包' in item or '缺包' in item for item in bad)


def test_after_inventory_all_problems_are_collected(control: Path) -> None:
    fragment = part_path(control, 'shard-a') / 'steps' / 'style_guard.json'
    row = json.loads(fragment.read_text())
    row['exit_code'] = 1
    fragment.write_text(json.dumps(row))
    manifest = part_path(control, 'shard-b') / 'shard' / 'manifest.json'
    row = json.loads(manifest.read_text())
    row['job'] = 'wrong-job'
    manifest.write_text(json.dumps(row))
    (control / 'governance/receipts/pytest.junit.xml').write_text('<')
    bad = subject.check(control, files(control))
    assert any('.exit_code' in item for item in bad)
    assert any('.job' in item for item in bad)
    assert any('正式收據' in item for item in bad)


@pytest.mark.parametrize('value', [f'owner/else/{WORKFLOW_PATH}',
                                 'owner/repo//outside.yml', '/'.join(('owner', 'repo', '..', 'outside.yml'))])
def test_workflow_path_must_be_within_repository(control: Path, value: str) -> None:
    path = control / RUN_PATH
    row = json.loads(path.read_text())
    row['env']['GITHUB_WORKFLOW_REF'] = value + '@refs/pull/9/merge'
    path.write_text(json.dumps(row))
    with pytest.raises(ToolBroken, match='workflow_path'):
        subject.check(control, files(control))


@pytest.mark.parametrize('content,word', [('[', 'YAML'), ('[]', '表'), ('jobs: []', 'jobs'),
                                       ('jobs: {elsewhere: {steps: []}}', 'shard-a.*steps'),
                                       ('jobs: {shard-a: {steps: 7}, shard-b: {steps: []}}', 'shard-a.*steps')])
def test_workflow_cannot_be_guessed(control: Path, content: str, word: str) -> None:
    (control / WORKFLOW_PATH).write_text(content)
    with pytest.raises(ToolBroken, match=word):
        subject.check(control, files(control))


def test_workflow_missing_is_broken(control: Path) -> None:
    (control / WORKFLOW_PATH).unlink()
    with pytest.raises(ToolBroken, match='verify-shards.yml'):
        subject.check(control, files(control))


@pytest.mark.parametrize('kind', ['file', 'link'])
def test_parts_directory_wrong_kind_is_broken(control: Path, tmp_path: Path, kind: str) -> None:
    import shutil
    path = control / PARTS_PATH
    if kind == 'file':
        shutil.rmtree(path)
        path.write_text('not a directory')
    else:
        other = tmp_path / 'download'
        path.rename(other)
        path.symlink_to(other, target_is_directory=True)
    with pytest.raises(ToolBroken, match='parts_dir' if kind == 'file' else '符號連結'):
        subject.check(control, files(control))


@pytest.mark.parametrize('target', ['card', 'fixture', 'registry', 'expected', 'args', 'junit', 'manifest', 'fragment'])
def test_finalizer_input_read_error_is_broken(control: Path, monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    paths = {'card': control / SETTINGS_PATH, 'fixture': control / RUN_PATH,
             'registry': control / 'governance/verify-shards.toml',
             'fragment': part_path(control, 'shard-a') / 'steps' / 'pytest.json'}
    for key, filename in [('expected', 'expected.txt'), ('args', 'pytest.args'), ('junit', 'junit.xml'), ('manifest', 'manifest.json')]:
        paths[key] = part_path(control, 'shard-a') / 'shard' / filename
    original_bytes, original_text = Path.read_bytes, Path.read_text
    def read_bytes(path: Path) -> bytes:
        if path == paths[target]:
            raise PermissionError('此題不可讀')
        return original_bytes(path)
    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        if path == paths[target]:
            raise PermissionError('此題不可讀')
        return original_text(path, encoding='utf-8')
    # 卡載入器本來會跳過壞卡；單獨讀設定才能驗出此一讀檔邊界。
    from governance.checks import green_finalizer_mode as finalizer
    card = load_card(control / SETTINGS_PATH, control)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    monkeypatch.setattr(Path, 'read_text', read_text)
    with pytest.raises(ToolBroken, match=paths[target].name + '.*(讀不開|讀不到)'):
        finalizer.finalizer_problems(card, control, subject._judge_receipt)


@pytest.mark.parametrize('change', ['missing', 'not-json'])
def test_registry_input_is_broken(control: Path, change: str) -> None:
    path = control / 'governance/verify-shards.toml'
    if change == 'missing':
        path.unlink()
    else:
        path.write_text('[')
    with pytest.raises(ToolBroken, match='registry.*verify-shards.toml'):
        subject.check(control, files(control))


def test_finalizer_source_group_still_runs(control: Path) -> None:
    path = control / '.github/workflows/verify.yml'
    path.write_text(path.read_text().replace('uv run pytest --junitxml=governance/receipts/pytest.junit.xml', 'uv run echo checked'))
    bad = subject.check(control, files(control))
    assert any('沒有任何一步在跑 pytest' in item for item in bad)


def test_finalizer_placement_group_still_runs(control: Path) -> None:
    path = control / SETTINGS_PATH
    path.write_text(path.read_text().replace('engine_bucket = "tests."', 'engine_bucket = "tests.engine."')
                    + '\n[[junit.bucket]]\nprefix = "tests.engine."\ncollected_floor = 1\n')
    (control / 'tests' / 'test_a.py').write_text('import aosr\n')
    bad = subject.check(control, files(control))
    assert any(str(Path('tests') / 'test_a.py') in item and '放錯籃子' in item for item in bad)


@pytest.mark.parametrize('failure', ['unavailable', 'nonzero'])
def test_head_read_failure_is_broken(git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    import subprocess
    from governance.checks import green_finalizer_mode as finalizer
    def broken_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if failure == 'unavailable':
            raise FileNotFoundError('git 缺席')
        return subprocess.CompletedProcess(['git', 'rev-parse', 'HEAD'], 128, '', 'HEAD 無法解析')
    monkeypatch.setattr(subprocess, 'run', broken_run)
    with pytest.raises(ToolBroken, match='tested_sha.*git rev-parse HEAD'):
        finalizer.git_head(git_sandbox.root)


@pytest.mark.parametrize('field', ['repository', 'workflow_path', 'workflow_ref', 'event', 'run_id',
                                 'run_attempt', 'job', 'shard', 'github_sha', 'tested_sha'])
@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_manifest_each_identity_field_compares_to_this_run(control: Path, field: str, shard: str) -> None:
    path = part_path(control, shard) / 'shard' / 'manifest.json'
    row = json.loads(path.read_text())
    row[field] = 'other-identity'
    path.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert bad and all(f'片 {shard} {path}.{field}=' in item for item in bad), bad


@pytest.mark.parametrize('file', ['expected', 'args', 'junit'])
@pytest.mark.parametrize('field', ['sha256', 'bytes'])
@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_finalizer_recomputes_each_file_seal(control: Path, file: str, field: str, shard: str) -> None:
    path = part_path(control, shard) / 'shard' / 'manifest.json'
    row = json.loads(path.read_text())
    row[file][field] = 'b' * 64 if field == 'sha256' else 0
    path.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert bad and all(f'片 {shard} {path}.{file}.{field}=' in item for item in bad), bad


def test_registry_is_in_targets(control: Path) -> None:
    assert control / 'governance/verify-shards.toml' in subject.targets(control, files(control))


def test_finalizer_bucket_assertion_still_runs(control: Path) -> None:
    # tests.test_a. 前綴把模組當成資料夾；收尾不能繞過分籃唯一性守門。
    path = control / SETTINGS_PATH
    path.write_text(path.read_text().replace('engine_bucket = "tests."', 'engine_bucket = "tests.test_a."')
                    + '\n[[junit.bucket]]\nprefix = "tests.test_a."\ncollected_floor = 1\n')
    with pytest.raises(ToolBroken, match=str(Path('tests') / 'test_a.py')):
        subject.check(control, files(control))


def rewrite_junit(root: Path, shard: str, old: str, new: str) -> None:
    """改一片 junit 原文並重封 manifest 的雜湊與大小，讓唯一的錯是改的那一格。"""
    shard_dir = part_path(root, shard) / 'shard'
    junit = shard_dir / 'junit.xml'
    text = junit.read_text()
    assert old in text
    raw = text.replace(old, new, 1).encode()
    junit.write_bytes(raw)
    manifest = shard_dir / 'manifest.json'
    row = json.loads(manifest.read_text())
    row['junit'].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest.write_text(json.dumps(row))


@pytest.mark.parametrize('attr', ['skipped', 'failures', 'errors'])
@pytest.mark.parametrize('negative,positive', [('shard-a', 'shard-b'), ('shard-b', 'shard-a')])
def test_negative_count_cannot_cancel_other_shard(control: Path, attr: str, negative: str, positive: str) -> None:
    """一片寫 -1、另一片寫 1，加總是 0；每一格要先單獨核非負，兩個方向都不能互相抵掉。"""
    rewrite_junit(control, negative, f'{attr}="0"', f'{attr}="-1"')
    rewrite_junit(control, positive, f'{attr}="0"', f'{attr}="1"')
    with pytest.raises(ToolBroken, match=f"{attr}='-1'.*非負整數"):
        subject.check(control, files(control))


@pytest.mark.parametrize('value', ['-1', '+1', ' 1', '1.0'])
@pytest.mark.parametrize('second', [False, True], ids=['first-suite', 'second-suite'])
def test_single_receipt_count_must_be_plain_integer(value: str, second: bool) -> None:
    """單機收據多個 testsuite 也一樣：不論壞的在第幾個，負數或怪寫法回 2，不加進總數。"""
    suites = [f'<testsuite skipped="{value}" />', '<testsuite skipped="1" />']
    root = ET.fromstring('<testsuites>' + ''.join(reversed(suites) if second else suites) + '</testsuites>')
    with pytest.raises(ToolBroken, match='非負整數'):
        subject._suite_attr_totals(root, 'r.xml')


@pytest.mark.parametrize('tag', ['error', 'failure', 'skipped', 'properties'])
def test_node_outside_testsuite_is_broken(control: Path, tag: str) -> None:
    """合併只抄 testsuite：b 片根層的失敗、錯誤、跳過一抄就不見，要在合併前回 2；訊息點名是根層多了這一種節點。"""
    rewrite_junit(control, 'shard-b', '</testsuites>', f'<{tag} /></testsuites>')
    with pytest.raises(ToolBroken, match=re.escape("片 shard-b junit.xml: ") + '.*'
                       + re.escape(f"實際根 'testsuites'、外面 {[tag]!r}、套疊 False")):
        subject.check(control, files(control))


@pytest.mark.parametrize('raw,actual', [
    ('<testsuites><testsuite><testsuite /></testsuite></testsuites>', "實際根 'testsuites'、外面 []、套疊 True"),
    ('<report><testsuite /></report>', "實際根 'report'、外面 []、套疊 False"),
], ids=['nested', 'other-root'])
def test_junit_shape_outside_merge_is_broken(raw: str, actual: str) -> None:
    with pytest.raises(ToolBroken, match='預期根是 testsuites.*' + re.escape(actual)):
        evidence.read_junit(raw.encode(), 'shard-a', 'junit.xml')


@pytest.mark.parametrize('raw', ['<testsuites><testsuite /><testsuite /></testsuites>', '<testsuite><testcase /></testsuite>'],
                         ids=['two-suites', 'bare-suite'])
def test_junit_plain_shapes_still_read(raw: str) -> None:
    assert ET.tostring(evidence.read_junit(raw.encode(), 'shard-a', 'junit.xml')) == ET.tostring(ET.fromstring(raw))


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
@pytest.mark.parametrize('result', ['failure', 'cancelled', 'skipped'])
def test_needs_each_shard_requires_success(control: Path, shard: str, result: str) -> None:
    """另一片成功不能蓋掉本片的上游失敗、取消或跳過。"""
    path = control / RUN_PATH
    row = json.loads(path.read_text())
    needs = json.loads(row['env']['AOSR_FINALIZER_NEEDS'])
    needs[shard]['result'] = result
    row['env']['AOSR_FINALIZER_NEEDS'] = json.dumps(needs)
    path.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert set(bad) == {f'片 {shard} needs.result={result!r}，預期 \'success\''}, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_each_shard_missing_package_is_red(control: Path, shard: str) -> None:
    """只缺本片；略過第二片的包清點會讓這題不再報缺包。"""
    shutil.rmtree(part_path(control, shard))
    bad = subject.check(control, files(control))
    assert set(bad) == {f"片 {shard} {control / PARTS_PATH}.缺包=None，預期 'receipts-123-2-{shard}'；請整個 workflow 重跑"}, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_each_shard_previous_attempt_is_not_current(control: Path, shard: str) -> None:
    """只把本片改成上一次嘗試：未登記舊包與本次缺包要同時指出。"""
    part_path(control, shard).rename(control / PARTS_PATH / f'receipts-123-1-{shard}')
    bad = subject.check(control, files(control))
    assert set(bad) == {
        f"{control / PARTS_PATH}.未登記的包='receipts-123-1-{shard}'，預期 ['receipts-123-2-shard-a', 'receipts-123-2-shard-b']",
        f"片 {shard} {control / PARTS_PATH}.缺包=None，預期 'receipts-123-2-{shard}'；請整個 workflow 重跑",
    }, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
@pytest.mark.parametrize('filename', ['expected.txt', 'manifest.json'])
def test_each_shard_missing_evidence_is_red(control: Path, shard: str, filename: str) -> None:
    """包目錄在，但本片少證據，應先回缺證據紅而非繼續開不存在的檔回 2。"""
    directory = part_path(control, shard) / 'shard'
    (directory / filename).unlink()
    bad = subject.check(control, files(control))
    assert set(bad) == {
        f"片 {shard} {directory}.缺證據={[filename]!r}，預期 ['expected.txt', 'junit.xml', 'manifest.json', 'pytest.args']",
    }, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_each_shard_extra_xml_is_red(control: Path, shard: str) -> None:
    """只有本片多 XML；不能只掃第一包。"""
    part = part_path(control, shard)
    (part / 'extra.xml').write_text('<testsuites />')
    bad = subject.check(control, files(control))
    assert set(bad) == {f"片 {shard} {part}.額外 XML='extra.xml'，預期 'junit.xml'"}, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_each_shard_missing_governance_fragment_is_red(control: Path, shard: str) -> None:
    """另一片片段齊全，不代表本片治理步驟有跑完。"""
    directory = part_path(control, shard) / 'steps'
    (directory / 'style_guard.json').unlink()
    bad = subject.check(control, files(control))
    assert set(bad) == {
        f"片 {shard} {directory}.缺片段=['style_guard']，預期 ['prepare', 'pytest', 'refs_and_links_resolve', 'seal', 'style_guard']",
    }, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
@pytest.mark.parametrize('field,value,wanted', [('exit_code', 1, 0), ('name', 'other-name', 'style_guard'),
                                             ('run_id', '456', '123'), ('run_attempt', '1', '2'),
                                             ('job', 'other-job', None)])
def test_each_shard_fragment_field_is_checked(control: Path, shard: str, field: str,
                                             value: object, wanted: object) -> None:
    """本片的離開碼與片段身分逐格比，另一片正確不能放行。"""
    path = part_path(control, shard) / 'steps' / 'style_guard.json'
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    expected = shard if field == 'job' else wanted
    bad = subject.check(control, files(control))
    assert set(bad) == {f'片 {shard} {path}.{field}={value!r}，預期 {expected!r}'}, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_each_shard_tests_attribute_is_checked(control: Path, shard: str) -> None:
    """只改本片 tests 屬性並重封，另一片的對帳完全正確。"""
    rewrite_junit(control, shard, 'tests="9"', 'tests="10"')
    bad = subject.check(control, files(control))
    assert f'片 {shard} junit.xml.tests=10，預期 9' in bad, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_each_shard_args_bytes_are_checked(control: Path, shard: str) -> None:
    """只改本片參數檔並重封；雜湊正確仍須與清點重算的本片參數逐位元組相同。"""
    directory = part_path(control, shard) / 'shard'
    args = directory / 'pytest.args'
    wanted = args.read_bytes()
    raw = wanted + (str(Path('tests') / 'test_unknown.py') + '\n').encode()
    args.write_bytes(raw)
    manifest = directory / 'manifest.json'
    row = json.loads(manifest.read_text())
    row['args'].update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    manifest.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert set(bad) == {f'片 {shard} pytest.args.位元組={raw!r}，預期 {wanted!r}'}, bad


@pytest.mark.parametrize('shard,module', [('shard-a', 'tests.test_a'), ('shard-b', 'tests.test_b')])
@pytest.mark.parametrize('replacement', ['test_unknown', 'test_0'])
def test_each_shard_unknown_or_duplicate_case_is_red(control: Path, shard: str, module: str, replacement: str) -> None:
    """同總數替换本片最後一題：漏題之外，須另抓不明題或片內重複。"""
    rewrite_junit(control, shard, 'name="test_8"', f'name="{replacement}"')
    bad = subject.check(control, files(control))
    assert bad and all(item.startswith(f'片 {shard} junit.xml.') for item in bad), bad
    assert any(f"testcase({module!r}, 'test_8').漏題=[]" in item for item in bad), bad
    if replacement == 'test_unknown':
        assert any(f"不明題=({module!r}, 'test_unknown')，預期 " in item for item in bad), bad
    else:
        assert f"片 {shard} junit.xml.testcase({module!r}, 'test_0').片內重複=2，預期 '恰好一次'" in bad, bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
@pytest.mark.parametrize('tag,word', [('skipped', 'skipped=1，必須是 0'),
                                    ('failure', 'failures=1、errors=0'), ('error', 'failures=0、errors=1')])
def test_merged_judge_checks_each_shard_case_flag(control: Path, shard: str, tag: str, word: str) -> None:
    """只加本片 testcase 的元素，計數屬性維持零；必須由合併後舊裁判認出元素。"""
    rewrite_junit(control, shard, 'name="test_0" />', f'name="test_0"><{tag} /></testcase>')
    bad = subject.check(control, files(control))
    assert any(item.startswith("2 片合併 ['shard-a', 'shard-b'] junit.xml") and word in item for item in bad), bad
    assert any(item.startswith(f"片 {shard} junit.xml.testcase") and f".{tag}=" in item for item in bad), bad


@pytest.mark.parametrize('shard', ['shard-a', 'shard-b'])
def test_each_shard_fragment_signal_is_checked(control: Path, shard: str) -> None:
    """本片治理步驟被訊號殺掉、離開碼卻記 0：另一片正常不能放行。"""
    path = part_path(control, shard) / 'steps' / 'style_guard.json'
    row = json.loads(path.read_text())
    row['signal'] = 15
    path.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert set(bad) == {f'片 {shard} {path}.signal=15，預期 None；離開碼記 0，被訊號殺掉不是跑完'}, bad


def test_expected_blank_line_is_broken(control: Path) -> None:
    """清點清單裡的空行跟空清單是兩種錯：空行另有專屬訊息。"""
    for shard in ('shard-a', 'shard-b'):
        directory = part_path(control, shard) / 'shard'
        (directory / 'expected.txt').write_bytes(b'\n')
        manifest = directory / 'manifest.json'
        row = json.loads(manifest.read_text())
        row['expected'].update(sha256=hashlib.sha256(b'\n').hexdigest(), bytes=1)
        manifest.write_text(json.dumps(row))
    with pytest.raises(ToolBroken, match=re.escape("不准空行，實際 ('',)")):
        subject.check(control, files(control))
