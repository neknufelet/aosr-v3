"""樣本用的假測試：第一格直接用 ``shutil.which`` 找出版控工具。

壞在：``subprocess.run([shutil.which(cmd="git"), "status"])`` 不經 git_sandbox 就 spawn。
「工具在哪」是 ``shutil.which`` 算出來的，所以要咬的是**外層那一次真的 spawn**，不是
內層 ``shutil.which(...)`` 那句查找——舊版只認第一格是字面 ``"git"``，這裡第一格是
``shutil.which(...)`` 呼叫，算都不要算就漏掉了。

這一份刻意用 ``cmd="git"`` 的關鍵字形、而不是 ``shutil.which("git")``：舊版對「每個呼叫
的第一格」都看，``shutil.which("git")`` 那種寫法單靠內層那句查找就會紅，於是這一跑在
修改前也是紅的，證明不了「外層那次 spawn 被抓到」。用關鍵字形把字面 ``"git"`` 從第一格
移走之後，舊檢查回 0、新檢查才回 1——這一跑才真的是在證明外層那筆。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import shutil
import subprocess


def test_status_of_the_real_tree():
    proc = subprocess.run([shutil.which(cmd="git"), "status"], capture_output=True, text=True, check=True)
    assert proc.stdout == ""
