"""樣本用的假測試：自己再定義一支同名 fixture，把真樹當 sandbox 交出去。

壞在：``git_sandbox`` 在這支檔裡被重新定義成「回傳真樹」。測試看起來有經過 fixture，
實際上碰的是真的 repo。同名 fixture 只准有一支。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def git_sandbox():
    """假的那支：直接把真樹交出去。"""
    return REPO


def test_log_is_readable(git_sandbox):
    subprocess.run(["git", "log", "-1"], cwd=git_sandbox, check=True)
