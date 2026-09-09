"""樣本道具：屬性寫法的 Any（`import typing` 之後寫 `typing.Any`）。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
import typing


def passthrough(value: str) -> typing.Any:
    """回傳標註是 `typing.Any`——沒有把 Any 這個名字 import 進來，一樣是 Any。"""
    return value
