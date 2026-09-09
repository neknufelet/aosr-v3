"""控制樣本的道具：一支檢查、一個模組層級的門檻常數，別的什麼都沒有。"""
from pathlib import Path

MAX_FILES = 500


def check(scan_root: Path) -> list[str]:
    found = list(scan_root.rglob("*"))
    return [f"檔案數 {len(found)} 超過上限"] if len(found) > MAX_FILES else []
