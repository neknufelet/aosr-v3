"""樣本道具：`cast(Any, …)` 把型別檢查繞過去。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
from typing import Any, cast


def length(value: str) -> int:
    """`cast(Any, value)` 之後那個名字什麼都能做，mypy 不再檢查它。

    第②層咬 `cast(Any, …)` 這個呼叫本身；第①層咬不到——mypy 尊重 cast，整份回 0。
    """
    loosened = cast(Any, value)
    return len(loosened)
