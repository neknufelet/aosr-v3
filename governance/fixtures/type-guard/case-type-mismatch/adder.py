"""樣本道具：型別對不上——把 str 餵進吃 int 的參數。

只給 type-guard 的樣本當道具用，不是真的程式。
"""


def add(left: int, right: int) -> int:
    """全標了、也標對了。違規在下面那一支。"""
    return left + right


def total() -> int:
    """第①層咬這一行：`"1"` 是 str，`add` 的第一格吃 int。"""
    return add("1", 2)
