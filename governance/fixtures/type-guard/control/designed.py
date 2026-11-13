"""控制樣本用的道具：Any 照卡頂層 [[by_design]] 登記的函式與處數，一處不多一處不少。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
from typing import Any

CACHE: dict[str, Any] = {}


class Reader:
    def read(self, text: str) -> Any:
        return CACHE.get(text)
