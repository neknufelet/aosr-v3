"""能力表的模型缺項與人工確認原文；只彙整有內容的入口。"""
from __future__ import annotations

from aosr.config.capabilities import CapabilityTable


def capability_lists(capabilities: CapabilityTable) -> dict[str, tuple[str, ...]]:
    # 同一句寫在好幾節（例如「空氣吸收」）只列一次，兩頁一致（複查）。
    return {field: tuple(dict.fromkeys(item for entry in capabilities.entry for item in getattr(entry, field)))
            for field in ("not_modeled", "manual_checks")}
