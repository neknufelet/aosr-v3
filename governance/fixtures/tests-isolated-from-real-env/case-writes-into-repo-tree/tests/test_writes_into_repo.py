"""樣本用的假測試：往真樹寫檔。

壞在：``REPO`` 是 ``Path(__file__)`` 往上數出來的真樹根，``write_text`` 直接寫在那裡；
``mkdir`` 也一樣。要寫檔就得經過 git_sandbox 給的暫存樹。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的測試。
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_report_is_written():
    (REPO / "build").mkdir(exist_ok=True)
    (REPO / "leftover.txt").write_text("報告", encoding="utf-8")
    assert (REPO / "leftover.txt").is_file()
