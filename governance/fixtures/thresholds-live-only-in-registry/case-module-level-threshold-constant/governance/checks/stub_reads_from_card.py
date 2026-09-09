"""樣本道具：門檻從卡讀進來，程式裡一個數字都沒有。這一支**不准**被咬。

每份樣本都放一份，作用是證明檢查在分辨寫法，不是看到 .py 就報一筆——樣本的 hits
只該來自那份刻意寫壞的道具。
"""
import tomllib
from pathlib import Path

CARD = "governance/rules/sample-card.toml"
SETTING = "max_lines"


def limit(scan_root: Path) -> int:
    data = tomllib.loads((scan_root / CARD).read_text(encoding="utf-8"))
    return int(data["settings"][SETTING])
