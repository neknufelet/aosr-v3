"""樣本道具：型別存根檔的回傳標註裡藏一個 Any。"""
from typing import Any

def parse(text: str) -> dict[str, Any]: ...
