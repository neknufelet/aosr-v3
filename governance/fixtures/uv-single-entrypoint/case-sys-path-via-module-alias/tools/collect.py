"""樣本道具：`import sys as s` 之後用別名硬插模組搜尋路徑。"""
import sys as s

s.path.insert(0, "..")


def main() -> int:
    return 0
