"""收尾模式四道守門的專屬考卷：設定路徑、正式收據懸空連結、下載目錄本身是連結、真根 HEAD 不抄 GITHUB_SHA。"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance.checks import green_finalizer_mode as finalizer
from governance.checks import green_must_be_real_green as subject
from governance.exit_codes import ToolBroken
from governance.loader import load_card
from tests.conftest import GitSandbox
from tests.test_green_finalizer_inputs import PARTS_PATH, SETTINGS_PATH, inputs, rewrite_setting
from tests.test_green_finalizer_mode import control as control
from tests.test_green_finalizer_mode import files


@pytest.mark.parametrize('key', ['fixture_finalizer_run_file', 'shard_registry', 'parts_dir'])
@pytest.mark.parametrize('value', ['"/abs/x"', '"../x"', '"a//b"'], ids=['absolute', 'parent', 'double-slash'])
def test_setting_path_must_stay_in_repo(control: Path, key: str, value: str) -> None:
    """設定裡的路徑只認 repo 內相對路徑；指到外面就判不出來，回 2。"""
    rewrite_setting(control, key, value)
    with pytest.raises(ToolBroken, match=f'settings.{key}: 預期 repo 內相對路徑'):
        subject.check(control, files(control))


def test_dangling_formal_receipt_link_counts_as_present(control: Path) -> None:
    """正式收據路徑放一個懸空連結也算帶入了單機收據；只看 exists 會被連結騙過。"""
    formal = control / 'governance' / 'receipts' / 'pytest.junit.xml'
    formal.symlink_to(control / 'nowhere.xml')
    assert any('收尾的正式收據在場' in item for item in subject.check(control, files(control)))


@pytest.mark.parametrize('target', ['empty', 'nowhere'])
def test_linked_parts_directory_is_broken(control: Path, tmp_path: Path, target: str) -> None:
    """下載目錄本身是連結：指到空目錄或懸空都回 2，不能因為底下沒有包、或連結跟不過去，就報成缺包。"""
    import shutil
    path = control / PARTS_PATH
    shutil.rmtree(path)
    other = tmp_path / 'download'
    if target == 'empty':
        other.mkdir()
    path.symlink_to(other, target_is_directory=True)
    with pytest.raises(ToolBroken, match='符號連結'):
        subject.check(control, files(control))


def test_real_root_tested_sha_is_git_head(git_sandbox: GitSandbox, control: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """真根的 tested_sha 取 git 真 HEAD；GITHUB_SHA 跟它不同時不能抄過去（兩者不一致由出處裁判報）。"""
    git_sandbox.git('commit', '--allow-empty', '-m', '收尾身分沙箱')
    head = git_sandbox.git('rev-parse', 'HEAD').stdout.strip()
    env, _head = inputs(control)
    env = dict(env, GITHUB_SHA='f' * 40)
    assert env['GITHUB_SHA'] != head
    monkeypatch.setattr(finalizer, 'process_env', lambda: env)
    settings = finalizer.read_settings(load_card(control / SETTINGS_PATH, control))
    assert settings is not None
    resolved = finalizer.resolve_run(git_sandbox.root, settings)
    assert resolved is not None
    assert resolved.own['tested_sha'] == head
