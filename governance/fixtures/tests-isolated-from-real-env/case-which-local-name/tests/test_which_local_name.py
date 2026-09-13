"""樣本用的假測試：``shutil.which`` 的工具名先指給名字，其結果再指給名字，才拿去 spawn。

壞在：``GIT_TOOL = "git"``、``tool = shutil.which(GIT_TOOL)`` 之後
``subprocess.run([tool, "status"])`` 不經 git_sandbox 就 spawn。定位呼叫的結果藏在一條
名字鏈後面——舊判準只認「名字綁到字面陣列」，綁到定位呼叫（或再綁到另一支名字）的
一律當不存在。

這裡刻意讓工具名本身是一個名字（``GIT_TOOL``）而不是字面 ``"git"``：舊版對每個呼叫的
第一格都看，``shutil.which("git")`` 那種寫法單靠內層那句查找就會紅，這一跑在修改前
就不是綠的，證明不了「外層那次 spawn 被抓到」。把字面 ``"git"`` 換成名字之後，舊檢查
回 0、新檢查回 1。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import shutil
import subprocess

GIT_TOOL = "git"


def test_status_of_the_real_tree():
    tool = shutil.which(GIT_TOOL)
    proc = subprocess.run([tool, "status"], capture_output=True, text=True, check=True)
    assert proc.stdout == ""
