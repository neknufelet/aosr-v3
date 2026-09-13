"""樣本用的假測試：``os.environ.get`` 的結果先指給名字，才拿去 spawn。

壞在：``tool = os.environ.get("GIT", "git")`` 之後 ``subprocess.run([tool, "status"])``
不經 git_sandbox 就 spawn。名字綁到的是定位呼叫而不是字面陣列，default 那一格是
``"git"``——舊判準到不了這一層。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import os
import subprocess


def test_status_of_the_real_tree():
    tool = os.environ.get("GIT", "git")
    proc = subprocess.run([tool, "status"], capture_output=True, text=True, check=True)
    assert proc.stdout == ""
