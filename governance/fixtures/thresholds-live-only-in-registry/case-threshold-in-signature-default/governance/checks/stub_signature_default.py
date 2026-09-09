"""樣本道具：門檻藏在函式簽章的預設值裡。這一支要被咬。

簽章的預設值在模組載入的當下就算好了，跟寫在模組層級是同一件事——只是名字換成參數名。
函式**內部**的字面值不算（那是演算法），所以這一份刻意把數字放在簽章上。
"""
from pathlib import Path


def check(scan_root: Path, max_lines: int = 200) -> list[str]:
    bad = []
    for path in scan_root.glob("*.py"):
        if len(path.read_text().splitlines()) > max_lines:
            bad.append(str(path))
    return bad
