"""控制樣本的第二支道具：卡上具名放行 Any 的那個檔。

它的 Any 是動態資料帶出來的（讀進來的東西型別本來就未知），卡的 [[settings.allow]]
具名放行了這個檔，所以第②層不咬它——這一份是「放行真的有效、不誤咬」的證據。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
import json
from typing import Any


def load(text: str) -> dict[str, Any]:
    """讀一段 json。回傳的值型別未知，所以標成 `dict[str, Any]`。"""
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("讀出來不是一張表")
    return data
