"""樣本用的假測試：拿工作樹乾不乾淨當答案。

壞在：直接問真樹的狀態（``git status --porcelain``），再拿它當斷言的依據。
沒有經過 git_sandbox，所以問到的是「跑測試那台機器現在長什麼樣」。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import subprocess


def test_working_tree_is_clean():
    out = subprocess.check_output(["git", "status", "--porcelain"], text=True)
    assert out == ""
