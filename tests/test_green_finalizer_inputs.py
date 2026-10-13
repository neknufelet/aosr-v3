"""設定、模式來源與輸入故障；環境只由指定小函式注入，真 git 只在沙箱。"""
from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

import pytest

from governance.checks import green_must_be_real_green as subject
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
    with pytest.raises(ToolBroken, match=f'settings.*{key}'):
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
def test_manifest_each_identity_field_compares_to_this_run(control: Path, field: str) -> None:
    path = part_path(control, 'shard-a') / 'shard' / 'manifest.json'
    row = json.loads(path.read_text())
    row[field] = 'other-identity'
    path.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert bad and all(f'.{field}=' in item and 'manifest.json' in item for item in bad)


@pytest.mark.parametrize('file', ['expected', 'args', 'junit'])
@pytest.mark.parametrize('field', ['sha256', 'bytes'])
def test_finalizer_recomputes_each_file_seal(control: Path, file: str, field: str) -> None:
    path = part_path(control, 'shard-a') / 'shard' / 'manifest.json'
    row = json.loads(path.read_text())
    row[file][field] = 'b' * 64 if field == 'sha256' else 0
    path.write_text(json.dumps(row))
    bad = subject.check(control, files(control))
    assert bad and all(f'.{file}.{field}=' in item and 'manifest.json' in item for item in bad)


def test_registry_is_in_targets(control: Path) -> None:
    assert control / 'governance/verify-shards.toml' in subject.targets(control, files(control))


def test_finalizer_bucket_assertion_still_runs(control: Path) -> None:
    # tests.test_a. 前綴把模組當成資料夾；收尾不能繞過分籃唯一性守門。
    path = control / SETTINGS_PATH
    path.write_text(path.read_text().replace('engine_bucket = "tests."', 'engine_bucket = "tests.test_a."')
                    + '\n[[junit.bucket]]\nprefix = "tests.test_a."\ncollected_floor = 1\n')
    with pytest.raises(ToolBroken, match=str(Path('tests') / 'test_a.py')):
        subject.check(control, files(control))
