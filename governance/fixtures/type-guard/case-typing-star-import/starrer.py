"""樣本道具：`from typing import *`——看不見綁定。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
from typing import *  # noqa: F403  # expires=2026-12-09 reason=樣本道具：這一行就是要證明「看不見綁定」會被咬，不是可以整理掉的 import


def passthrough(value: str) -> int:
    """這一支本身完全合規。違規在檔頭那一行的星號 import。"""
    return len(value)
