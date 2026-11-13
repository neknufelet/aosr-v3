"""樣本用的假 conftest：准許的那個函式多寫了第三筆。

卡頂層的 [[by_design]] 寫死這個函式准兩筆寫入（建目錄、刪舊收據）。這裡多了一筆 write_text，
路徑一樣被 .gitignore 蓋住、函式名字也對——資料寫死筆數，多出來的那一筆不會跟著被准許。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的 conftest。
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "governance" / "receipts" / "pytest.junit.xml"


def pytest_sessionstart(session):
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.unlink(missing_ok=True)
    (TARGET.parent / "note.txt").write_text("多出來的一筆", encoding="utf-8")
