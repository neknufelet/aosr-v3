"""樣本用的假 conftest：准許的那個函式剛好寫兩筆，跟登記的筆數一樣。

這一份要壓的只有一件事：卡頂層的 [[by_design]] 把這個（檔、函式、路徑）登記了兩次。程式這一邊
是乾淨的，所以修補前的檢查回 0（重複的那一筆被後一筆蓋掉、沒人發現），修補後回 2。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的 conftest。
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "governance" / "receipts" / "pytest.junit.xml"


def pytest_sessionstart(session):
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.unlink(missing_ok=True)
