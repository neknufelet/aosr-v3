"""樣本道具：改名 import 進來的 Any。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
from typing import Any as Loose


def passthrough(value: str) -> Loose:
    """回傳標註寫的是 `Loose`，那個名字其實就是 Any。

    這一份是協調席實測戳出來的洞：只比名字的最後一段（`Any` 這個字面）的話，
    這裡 hits=0，而第①層也不響——改個名字就整條繞過去。
    """
    return value
