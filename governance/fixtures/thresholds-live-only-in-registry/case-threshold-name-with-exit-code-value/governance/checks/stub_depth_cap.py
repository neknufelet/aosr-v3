"""樣本道具：門檻的值剛好等於離開碼那三個數之一。這一支要被咬。

`MAX_DEPTH` 命中卡上登記的「名字像門檻」樣式，所以值就算落在離開碼白名單裡，一樣要具名
登記才准。沒有這一條，任何門檻只要把值寫成離開碼那三個數之一就整條繞過去了。
"""
import os

MAX_DEPTH = 1


def too_deep() -> bool:
    return int(os.environ.get("STUB_DEPTH", "0")) >= MAX_DEPTH
