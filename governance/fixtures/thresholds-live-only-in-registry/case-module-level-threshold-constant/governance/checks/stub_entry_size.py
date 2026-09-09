"""樣本道具：門檻寫死在模組層級。這一支要被咬。

v2 的形狀就是這樣——入口檔行數上限同時寫在檢查程式與規矩卡的人話裡，之後改了一邊。
"""
from pathlib import Path

MAX_LINES = 200


def check(scan_root: Path) -> list[str]:
    return [str(p) for p in scan_root.glob("*.py") if len(p.read_text().splitlines()) > MAX_LINES]
