"""報告層的比例提醒；純函式，只用精確有理數判定。"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, Inexact, localcontext
from fractions import Fraction
import math
from typing import Literal

from aosr.config.room_proportions import RoomProportionRule, RoomProportions


PROPORTION_WARNING = "這個房間的比例可能產生模態重合，請仔細檢查低頻結果"
PROPORTION_EXPLANATION = "比例提醒照 ITU-R BS.1116-3 與 EBU Tech 3276 對參考聆聽室的建議，只提醒、不改名次與分數，也不拿來評比例的高下"
NO_VALUES = "無數值，不判"
Ratio = Literal["l/w", "l/h", "w/h"]
RATIO_LABELS: dict[Ratio, str] = {"l/w": "長寬比", "l/h": "長高比", "w/h": "寬高比"}
STANDARD_LABELS = {"ITU-R BS.1116-3 (02/2015)": "ITU-R BS.1116-3",
                   "EBU Tech 3276, 2nd edition (1998)": "EBU Tech 3276"}


@dataclass(frozen=True)
class RoomProportionHit:
    """命中的條文、比值與整數；保留原始有理數供其他報告共用。"""

    rule: RoomProportionRule
    ratio: Ratio
    value: Fraction
    near_integer: int | None = None


@dataclass(frozen=True)
class RoomProportionNotice:
    """空清單表示三類建議皆無提醒；沒有數值由呼叫端用 None 表示。"""

    hits: tuple[RoomProportionHit, ...]


def room_proportion_notice(length_m: float, width_m: float, height_m: float,
                           rules: RoomProportions) -> RoomProportionNotice:
    """長短邊不看朝向；尺寸取浮點本身的二進位值，所有比較皆在有理數中。"""
    if any(not math.isfinite(value) or value <= 0 for value in (length_m, width_m, height_m)):
        raise ValueError("房間尺寸必須為有限正數")
    x, y, h = map(Fraction, (length_m, width_m, height_m))
    l, w = max(x, y), min(x, y)
    ratios: dict[Ratio, Fraction] = {"l/w": l / w, "l/h": l / h, "w/h": w / h}
    hits: list[RoomProportionHit] = []
    for entry in rules.entries:
        limits = entry.thresholds
        if entry.kind == "band":
            assert limits.lower_factor is not None and limits.upper_factor is not None and limits.upper_offset is not None
            if not limits.lower_factor * ratios["w/h"] <= ratios["l/h"] <= limits.upper_factor * ratios["w/h"] - limits.upper_offset:
                hits.append(RoomProportionHit(entry, "l/h", ratios["l/h"]))
        elif entry.kind == "below":
            assert limits.upper is not None
            for name, value in ratios.items():
                if name != "l/w" and value >= limits.upper:
                    hits.append(RoomProportionHit(entry, name, value))
        else:
            width = limits.relative_width
            assert width is not None
            for ratio, value in ratios.items():
                first = max(1, math.ceil(value / (1 + width)))
                last = math.floor(value / (1 - width))
                hits.extend(RoomProportionHit(entry, ratio, value, n) for n in range(first, last + 1))
    return RoomProportionNotice(tuple(hits))


def _decimal(value: Fraction | None) -> str:
    """門檻照表上的十進位數值印，不印成分數、不用科學記號；尾端的零不留（"0.050" 印成 0.05）。

    表上的門檻都是有限小數，分母只有 2 與 5：商的位數不超過分子位數加三倍分母位數，
    精度給到這麼多、不精確就拋錯，所以不會像預設 28 位那樣把長門檻捨掉。
    """
    assert value is not None
    with localcontext(prec=len(str(value.numerator)) + 3 * len(str(value.denominator))) as context:
        context.traps[Inexact] = True
        return f"{Decimal(value.numerator) / Decimal(value.denominator):f}"


def _hit_text(hit: RoomProportionHit) -> str:
    label = f"{RATIO_LABELS[hit.ratio]} {float(hit.value):.6f}"
    limits = hit.rule.thresholds
    if hit.rule.kind == "near_integer":
        assert limits.relative_width is not None
        detail = f"在整數 {hit.near_integer} 的 ±{float(limits.relative_width * 100):g}% 內"
    elif hit.rule.kind == "band":
        detail = (f"在建議比例帶 {_decimal(limits.lower_factor)}·w/h ≤ l/h ≤ {_decimal(limits.upper_factor)}·w/h − "
                  f"{_decimal(limits.upper_offset)} 之外")
    else:
        detail = f"達到建議上限 {_decimal(limits.upper)}（建議小於此值）"
    return f"{label} {detail}（{'；'.join(STANDARD_LABELS[name] for name in hit.rule.standard)}）"


def room_proportion_text(notice: RoomProportionNotice | None) -> str:
    """共用報告字樣；沒有數值與有數值但無提醒分開呈現。"""
    if notice is None:
        return NO_VALUES
    if not notice.hits:
        return "無"
    return "；".join((PROPORTION_WARNING, *(_hit_text(hit) for hit in notice.hits)))
