"""控制樣本用的假測試：三條合規 ＋ 一條已知會咬。

已知會咬的最小輸入。用途不是測某個特定寫法，而是「檢查還活著」的活體證明——
這一份餵下去回 0，就代表 tests-isolated-from-real-env 自己死了。

同一棵樹裡刻意擺三條合規的寫法：唯一那支 fixture 自己 spawn、測試要了 fixture 才 spawn、
只讀不寫。它們必須都不被咬（報告行的 hits 要正好是 1），不然這支檢查就是在誤咬正常寫法。

外面再加一整條合規的函式：要了 fixture 之後，用 ``shutil.which``／``os.environ.get``
找出版控工具再 spawn（含別名鏈）。這些定位寫法經 fixture 之後一樣要乾淨——新的判準
擴到 which/env.get，這幾條當「擴了但沒誤咬」的活體對照。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import os
import shutil
import subprocess
from pathlib import Path

from conftest import read_only_helper

REPO = Path(__file__).resolve().parents[1]


def test_status_in_sandbox(git_sandbox):
    """合規：要了唯一那支 fixture，才在它給的暫存樹上 spawn。"""
    proc = subprocess.run(
        ["git", "status", "--porcelain"], cwd=git_sandbox, capture_output=True, text=True, check=True
    )
    assert proc.stdout == ""


def test_status_via_which_in_sandbox(git_sandbox):
    """合規：要了 fixture，又用 shutil.which 找出 git，才在暫存樹上 spawn——定位寫法也不咬。"""
    tool = shutil.which("git")
    alias = tool
    proc = subprocess.run([alias, "status", "--porcelain"], cwd=git_sandbox, capture_output=True, text=True, check=True)
    assert proc.stdout == ""


def test_status_via_env_get_in_sandbox(git_sandbox):
    """合規：要了 fixture，又用 os.environ.get 的 default 取 git，才 spawn——定位寫法也不咬。"""
    proc = subprocess.run(
        [os.environ.get("GIT", "git"), "status", "--porcelain"], cwd=git_sandbox, capture_output=True, text=True, check=True
    )
    assert proc.stdout == ""


def test_reads_a_file():
    """合規：只讀真樹裡的檔，沒有寫、沒有 spawn。"""
    assert read_only_helper(REPO / "WHY.txt").strip() != ""


def test_status_of_the_real_tree():
    """已知會咬的那一條：不經 fixture 直接問真樹。"""
    proc = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    assert proc.stdout == ""
