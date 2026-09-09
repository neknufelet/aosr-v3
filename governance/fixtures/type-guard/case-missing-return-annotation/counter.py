"""樣本道具：一個沒有回傳標註的公開函式。

只給 type-guard 的樣本當道具用，不是真的程式。
"""


def count_rows(rows: list[str]):
    """參數標了、回傳沒標——第①層（mypy 嚴格模式）咬這一條。

    第②層在這一份完全乾淨：沒有 Any、沒有 cast、沒有 ignore。一份樣本只證一件事。
    """
    return len(rows)
