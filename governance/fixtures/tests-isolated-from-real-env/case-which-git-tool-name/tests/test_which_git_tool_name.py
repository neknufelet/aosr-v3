"""樣本用的假測試：``shutil.which`` 的 cmd 來自一個先綁好的名字。

壞在：``GIT_TOOL = "git"`` 之後 ``subprocess.run([shutil.which(GIT_TOOL), "status"])``
不經 git_sandbox 就 spawn。「工具名」藏在 ``shutil.which`` 的參數裡、而那個參數又是
先指給名字的字面——舊判準只看呼叫第一格，這裡第一格是 ``shutil.which(GIT_TOOL)``，
根本到不了 ``GIT_TOOL`` 那一層。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import shutil
import subprocess

GIT_TOOL = "git"


def test_status_of_the_real_tree():
    proc = subprocess.run([shutil.which(GIT_TOOL), "status"], capture_output=True, text=True, check=True)
    assert proc.stdout == ""
