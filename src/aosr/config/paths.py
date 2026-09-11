"""設定檔路徑的唯一住處。

**為什麼要有這一支。** 上一代有 5 支模組各自寫一次「從自己往上數兩層、再進設定檔目錄」
的式子，另外 4 支靠呼叫端把路徑傳進來——「同一件事有好幾份做法、沒有人是主人」。
而且那個往上數兩層的寫法**不准照抄**：上一代的檔案住得比較淺（數兩層剛好是 repo 根），
新家的檔案住 ``src/aosr/config``，同樣數兩層只到 ``src``——深度少一層，算出來的路徑
會指錯地方。錯法很安靜：它只會在開檔那一刻炸，不會在載入那一刻。收成一支之後，
改一處就是改一處。

**設定檔住哪裡。** 決策紙 ``docs/decisions/engine-first-block-config-shape.md`` 決定
「設定檔全部住第 2 層裡面」——也就是這個套件底下的 ``data`` 目錄，不是 repo 根層那個。

這一支只碰標準庫（``pathlib``），不開檔、不 import 上面的層。
"""

from __future__ import annotations

from pathlib import Path

# 設定檔目錄：這個套件底下的 `data`。用 `__file__` 推出來，跟行程的 cwd 無關
# （上一代有一支用的是相對路徑 ``Path("config")``，那綁 cwd，換一棵樹跑就找不到
# ——這一支刻意不那樣寫）。
CONFIG_DIR: Path = Path(__file__).resolve().parent / "data"


def config_path(name: str) -> Path:
    """回傳 ``data`` 目錄底下那個設定檔的路徑。

    只負責算路徑，不檢查它在不在：找不到檔要炸在開檔那一刻（大聲的
    ``FileNotFoundError``），不是在這裡安靜地回一個替代品。
    """
    return CONFIG_DIR / name

