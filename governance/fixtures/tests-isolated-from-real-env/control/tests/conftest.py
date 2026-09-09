"""樣本用的假 conftest：只放那支唯一的 sandbox fixture，讓合規的測試有東西可要。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的 conftest。
"""
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def git_sandbox(tmp_path):
    """暫存目錄裡的一個乾淨版控樹。樣本裡只要形狀對就好，不求跟主線那支一模一樣。"""
    root = tmp_path / "sandbox"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    yield root


def read_only_helper(path: Path) -> str:
    """讀檔不算違規，放在這裡當對照。"""
    return path.read_text(encoding="utf-8")
