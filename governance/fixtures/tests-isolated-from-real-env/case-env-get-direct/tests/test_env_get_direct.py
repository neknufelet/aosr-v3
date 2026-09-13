"""樣本用的假測試：第一格直接用 ``os.environ.get`` 的 default 取版控工具。

壞在：``subprocess.run([os.environ.get("GIT", "git"), "status"])`` 不經 git_sandbox 就
spawn。default 那一格是 ``"git"``，但舊判準只認第一格（也就是整個 ``os.environ.get(...)``
呼叫）是字面，呼叫不算，漏掉了。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import os
import subprocess


def test_status_of_the_real_tree():
    proc = subprocess.run([os.environ.get("GIT", "git"), "status"], capture_output=True, text=True, check=True)
    assert proc.stdout == ""
