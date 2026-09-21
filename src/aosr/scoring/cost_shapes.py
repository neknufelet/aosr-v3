"""排名代價共用的登記簿讀取與代價形狀。"""
from __future__ import annotations

from typing import Literal

from aosr.config.quality_targets import QualityPurpose, TargetEntry, Unit, WeightTable


ComponentRole = Literal["principal", "reference", "protection", "reported"]


def target(purpose: QualityPurpose, key: str, expected_unit: Unit) -> TargetEntry:
    """讀一條品質目標；鍵不在或型別不對就報錯，不猜相近鍵。"""
    entry = purpose.entry(key)
    if not isinstance(entry, TargetEntry):
        raise TypeError(f"{key} 不是品質目標")
    if entry.unit != expected_unit:
        raise ValueError(
            f"{key} 單位應為 {expected_unit}，登記簿寫 {entry.unit}"
        )
    return entry


def weight_table(purpose: QualityPurpose, key: str) -> WeightTable:
    """讀一張權重表。"""
    entry = purpose.entry(key)
    if not isinstance(entry, WeightTable):
        raise TypeError(f"{key} 不是權重表")
    return entry


def scalar(target_entry: TargetEntry) -> float:
    """目標值必須是單一數值；範圍型的值不能拿來當門檻或目標點。"""
    if isinstance(target_entry.value, tuple):
        raise TypeError(f"{target_entry.key} 必須是單一數值")
    return float(target_entry.value)


def shape_cost(target_entry: TargetEntry, values: tuple[float, ...]) -> float:
    """登記簿三種標準代價形狀的共用實作；型別由 ``cost_shape`` 決定。

    * ``less_is_better``：``x / worse_reference``（x 是一個不為負的量）。
    * ``in_range_best``：``max(0, |x − value| − tolerance) / worse_reference``，帶內零代價。
    * ``beyond_threshold_only``：每個特徵 ``max(0, |depth_db| − value)``（value 是門檻）加總後
      除 ``worse_reference``。只看深度：寬度未知（None）的特徵照深度算、標記原樣帶出去；
      太窄的特徵由呼叫端排除（最小寬度是工程篩選條件）。**這是第一版形狀，不是正式標準**；
      #345 第 5 格說「按深度與寬度給」，寬度怎麼進來等正式數字另拍。

    前兩型必須剛好收到一個值；第三型收到零個特徵時代價是零。若一個領域的最佳區間
    不是 ``value ± tolerance``（例如殘響逐帶各有上下限），呼叫端先求帶方向的超出量，
    再把該非負量交給 ``less_is_better``；這裡仍唯一負責正規化公式。
    """
    if target_entry.cost_shape == "beyond_threshold_only":
        threshold = scalar(target_entry)
        return sum(max(0.0, abs(value) - threshold) for value in values) / target_entry.worse_reference
    if len(values) != 1:
        raise ValueError(
            f"{target_entry.key} 的 {target_entry.cost_shape} 必須剛好收到一個值"
        )
    if target_entry.cost_shape == "less_is_better":
        return values[0] / target_entry.worse_reference
    if target_entry.tolerance is None:
        raise ValueError(f"{target_entry.key} 缺 tolerance")
    excess = max(
        0.0,
        abs(values[0] - scalar(target_entry)) - target_entry.tolerance,
    )
    return excess / target_entry.worse_reference
