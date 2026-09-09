"""樣本用的假 conftest：往卡上放行的那個路徑寫檔。

壞不在這支檔，壞在卡：放行開給 docs/notes，而 docs/notes 進得了版控。
放行只准開給 .gitignore 蓋住的路徑。

只給 tests-isolated-from-real-env 的樣本當道具用，不是真的 conftest。
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTES = REPO / "docs" / "notes"


def pytest_sessionstart(session):
    NOTES.mkdir(parents=True, exist_ok=True)
    (NOTES / "seed.txt").write_text("種子", encoding="utf-8")
