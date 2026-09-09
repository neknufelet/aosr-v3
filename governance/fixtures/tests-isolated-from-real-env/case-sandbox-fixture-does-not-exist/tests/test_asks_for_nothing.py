"""樣本用的假測試：要了一支這棵樹裡不存在的 fixture。

壞在：參數列寫了 ``git_sandbox``，但整棵 tests/ 底下沒有人定義它。
形式上「經過 fixture」，實際上沒有沙盒。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import subprocess


def test_status_in_sandbox(git_sandbox):
    subprocess.run(["git", "status"], cwd=git_sandbox, check=True)
