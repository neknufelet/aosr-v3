"""能力表的模型缺項與人工確認原文；只彙整有內容的入口。"""
from __future__ import annotations

from aosr.config.capabilities import CapabilityTable


def capability_lists(capabilities: CapabilityTable) -> dict[str, tuple[str, ...]]:
    return {field: tuple(item for entry in capabilities.entry for item in getattr(entry, field))
            for field in ("not_modeled", "manual_checks")}
