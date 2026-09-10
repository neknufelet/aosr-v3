"""樣本道具：先把 sys.path 指給一個名字，再對那個名字動手。"""
import sys

p = sys.path

p.insert(0, "..")


def main() -> int:
    return 0
