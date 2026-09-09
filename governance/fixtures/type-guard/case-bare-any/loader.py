"""樣本道具：回傳標註裡藏一個 Any。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
from typing import Any


def load(text: str) -> dict[str, Any]:
    """回傳標註是 `dict[str, Any]`——第②層咬得到，第①層咬不到。

    這一份同時是「第②層必須獨立於 mypy」的證據：mypy 嚴格模式那一包不含
    disallow_any_explicit，所以整份餵給 mypy 是乾淨的。
    """
    return {text: text}
