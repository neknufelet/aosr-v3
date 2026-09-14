"""控制樣本道具：.py 裡帶錯誤碼的抑制註解仍然合法。"""


def pinned() -> int:
    """這一行只壓掉明寫的回傳型別錯誤。"""
    return "not an int"  # type: ignore[return-value]
