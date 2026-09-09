"""樣本道具：從環境變數拿路徑，插進模組搜尋路徑。"""
import os
import sys

sys.path.append(os.environ["AOSR_LIBS"])


def main() -> int:
    return 0
