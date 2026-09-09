"""樣本用的假測試：參數陣列藏在名字後面。

壞在：``ARGV = ["git", "status", "--porcelain"]`` 之後 ``subprocess.run(ARGV)``——
跟直接寫在呼叫裡一模一樣，一樣沒經過 git_sandbox。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import subprocess

ARGV = ["git", "status", "--porcelain"]


def test_tree_is_clean():
    proc = subprocess.run(ARGV, capture_output=True, text=True, check=True)
    assert proc.stdout.strip() == ""
