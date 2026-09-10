"""樣本道具：`from sys import path` 之後直接對 path 動手。"""
from sys import path

path.insert(0, "..")


def main() -> int:
    return 0
