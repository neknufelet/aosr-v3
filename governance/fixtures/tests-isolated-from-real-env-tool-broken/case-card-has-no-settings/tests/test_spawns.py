"""樣本用的假測試：不經 fixture 直接 spawn。

這支檔本身是違規的，但這一份樣本要驗的是「卡上沒有門檻的時候檢查回 2 不回 1／0」。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
import subprocess


def test_status():
    subprocess.run(["git", "status"], check=True)
