"""樣本道具：一支標得好好的 .py，放在一棵沒有 mypy 設定檔的樹裡。

它本身完全合規（兩層都咬不到），所以這棵樹回 2 的唯一理由就是「尺不在」。
只給 type-guard 的樣本當道具用，不是真的程式。
"""


def add(left: int, right: int) -> int:
    """合規：全標了、也標對了。"""
    return left + right
