"""控制樣本的主道具：三條合規的寫法 ＋ 一條已知會咬的。

只給 type-guard 的樣本當道具用，不是真的程式。
"""
from pathlib import Path


def first_line(path: Path) -> str:
    """合規：參數與回傳都標了具體型別，型別也對得上。"""
    return path.read_text(encoding="utf-8").splitlines()[0]


def joined(rows: list[str], sep: str) -> str:
    """合規：容器的元素型別也寫明了，沒有 Any。"""
    return sep.join(rows)


def pinned() -> int:
    """合規（不誤咬的證據）：型別對不上，但那一行的抑制註解帶了錯誤碼。

    第②層只咬沒帶錯誤碼的抑制註解；帶了碼就只放過那一種錯，新錯誤照樣紅。
    """
    return "not an int"  # type: ignore[return-value]  # expires=2026-12-09 reason=控制樣本要證明「帶錯誤碼的抑制註解不會被咬」，這一行就是那個道具


def summarise(rows: list[str]):
    """違規（已知會咬）：沒有回傳標註，第①層咬得到。"""
    return len(rows)
