"""樣本用的假測試：語法壞掉，AST 解不開。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import subprocess


def test_status(
    subprocess.run(["git", "status"])
