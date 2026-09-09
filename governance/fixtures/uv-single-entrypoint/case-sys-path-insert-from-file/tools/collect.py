"""樣本道具：由 __file__ 推出來的自我定位，也是硬插模組搜尋路徑。

這個形狀以前放行，現在一律紅。留這份樣本就是為了證明那個放行真的收掉了。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    return 0
