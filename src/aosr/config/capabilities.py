"""能力與驗證範圍表的唯讀載入與欄位一致性驗證（票 #315）。

**這一支在做什麼。** 讀 `src/aosr/config/data/capabilities.toml`——哪個入口在什麼條件下真的被驗過——
把它變成凍結的 Pydantic 模型。表本身住設定層（第 2 層），物理層的入口往下拿；
路徑一樣由呼叫端必給（``aosr.config.paths.config_path("capabilities.toml")`` 或命令列參數），
這一支不替任何人決定它住哪裡。

**為什麼載入時就要咬。** 一張「誰驗過」的表最貴的錯法是它自己說謊：`validated` 卻指不到
測試、同一條組合寫兩次、頻率範圍頭尾顛倒。這些都不是要靠人讀出來的，載入那一刻就炸。
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


CapabilityStatus = Literal["validated", "experimental", "unsupported"]


class Capability(BaseModel):
    """一條「入口 × 房型 × 材料形式 × 頻率範圍 × 輸出欄位」的條件組合。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    room: str = Field(min_length=1)
    materials: str = Field(min_length=1)
    frequency_hz: tuple[float, float]
    outputs: tuple[str, ...] = Field(min_length=1)
    status: CapabilityStatus
    evidence: tuple[str, ...] = ()
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _frequency_range_ascends_between_positive_bounds(self) -> Self:
        """頻率範圍是兩個正數、而且遞增；否則這條組合沒有意義。"""
        lower, upper = self.frequency_hz
        if lower <= 0.0 or upper <= 0.0:
            raise ValueError("頻率範圍的兩端都必須是正數")
        if lower >= upper:
            raise ValueError("頻率範圍必須遞增")
        return self

    @model_validator(mode="after")
    def _validated_carries_evidence(self) -> Self:
        """標成 validated 就必須指名測試或答案檔；沒有收據的綠不是綠。"""
        if self.status == "validated" and not self.evidence:
            raise ValueError("validated 的條件組合必須有 evidence")
        return self


class CapabilityEntry(BaseModel):
    """一個入口一節；底下每一條是它的條件組合。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    module: str = Field(pattern=r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
    capability: tuple[Capability, ...] = Field(min_length=1)
    note: str = Field(min_length=1)

    @model_validator(mode="after")
    def _room_material_pairs_are_unique(self) -> Self:
        """同一個入口裡，同一個 (房型, 材料形式) 只准出現一次。"""
        seen: set[tuple[str, str]] = set()
        for item in self.capability:
            pair = (item.room, item.materials)
            if pair in seen:
                raise ValueError(f"同一入口重複的組合：{item.room} × {item.materials}")
            seen.add(pair)
        return self


class CapabilityTable(BaseModel):
    """整張能力表；凍結、拒收未登記欄位。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry: tuple[CapabilityEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _entry_names_are_unique(self) -> Self:
        """入口名不准重複——不然「這次輸入落在哪一節」會有兩個答案。"""
        seen: set[str] = set()
        for item in self.entry:
            if item.name in seen:
                raise ValueError(f"同名入口：{item.name}")
            seen.add(item.name)
        return self

    def for_entry(self, name: str) -> CapabilityEntry:
        """回傳指名入口的那一節；沒有就大聲炸，不回替代品。"""
        for item in self.entry:
            if item.name == name:
                return item
        raise KeyError(f"能力表沒有這個入口：{name}")


def load_capabilities(path: str | Path) -> CapabilityTable:
    """從呼叫端必給的 TOML 路徑讀出能力表。"""
    with Path(path).open("rb") as config_file:
        loaded = tomllib.load(config_file)
    return CapabilityTable.model_validate(loaded)


def status_for(
    table: CapabilityTable,
    entry_name: str,
    *,
    room: str,
    materials: str,
) -> CapabilityStatus:
    """回傳這次輸入落在哪一條組合的狀態；沒有這條組合就大聲炸。"""
    for item in table.for_entry(entry_name).capability:
        if item.room == room and item.materials == materials:
            return item.status
    raise KeyError(f"能力表沒有這個組合：{entry_name} × {room} × {materials}")


def evidence_for(
    table: CapabilityTable,
    entry_name: str,
    *,
    room: str,
    materials: str,
) -> tuple[str, ...]:
    """回傳這條組合指名的測試節點與答案檔；`experimental` 這一格是空的。"""
    for item in table.for_entry(entry_name).capability:
        if item.room == room and item.materials == materials:
            return item.evidence
    raise KeyError(f"能力表沒有這個組合：{entry_name} × {room} × {materials}")
