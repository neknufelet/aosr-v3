"""樣本用的假測試：不經 fixture 就在真樹上建 commit。

壞在：``subprocess.run(["git", "add", "."], cwd=REPO)`` 與後面那一顆 commit 都直接打在
``REPO``（``Path(__file__)`` 往上數出來的真樹）上，沒有經過唯一的 git_sandbox fixture。
v2 就是這樣讓 hook 把兩顆 fixture commit 寫回真 repo。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_prepush_hook_makes_a_fixture_commit():
    subprocess.run(["git", "add", "."], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=REPO, check=True)
    assert (REPO / ".git").is_dir()
