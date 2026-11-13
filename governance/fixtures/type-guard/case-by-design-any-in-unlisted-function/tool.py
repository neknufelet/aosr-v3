"""樣本道具：讀進來的是動態資料，Any 是誠實的標註。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
from typing import Any


def load(text: str) -> dict[str, Any]:
    return {text: text}


def keep(row: Any) -> str:
    return str(row)
