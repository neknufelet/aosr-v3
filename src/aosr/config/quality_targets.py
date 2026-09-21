"""品質目標登記簿的唯讀載入、驗證與內容指紋（票 #356）。

這一層只讀設定，不拿物理層或計分層的型別；用途、目標、較差參考、權重與資格的
唯一住處是呼叫端明示的 TOML（設定表）路徑。
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
SourceKind = Literal[
    "standard",
    "perceptual_literature",
    "engineering_recommendation",
    "product_choice",
]
EntryStatus = Literal["baseline", "calibrated"]
CostShape = Literal["in_range_best", "less_is_better", "beyond_threshold_only"]
Unit = Literal["dB", "dB/oct", "Hz", "oct", "s", "ms", "1"]
# 一律 Strict：TOML 的 true 在寬鬆模式會被當成 1，靜靜變成一個數字（找碴席實測）。
NumericValue = StrictInt | StrictFloat | tuple[StrictFloat, ...]
QualificationValue = StrictInt | StrictFloat | tuple[StrictFloat, ...] | tuple[str, ...]


class _SourceRecord(BaseModel):
    """每一條設定都帶的狀態與出處；calibrated（已校準）另有完整收據。"""

    model_config = FROZEN

    status: EntryStatus
    source_kind: SourceKind
    source: str = Field(min_length=1)
    source_id: str | None = None
    source_version: str | None = None
    locator: str | None = None
    conditions: str | None = None
    frequency_range_hz: tuple[float, float] | None = None
    context: str | None = None
    verification_digest: Annotated[
        str, Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ] | None = None

    @model_validator(mode="after")
    def _source_is_not_blank(self) -> Self:
        """精簡出處不准只有空白；baseline 與 calibrated 同一把尺。"""
        if not self.source.strip():
            raise ValueError("source 不可只有空白")
        return self

    @model_validator(mode="after")
    def _calibrated_has_structured_source(self) -> Self:
        """已校準條目缺任何可回查欄位都拒收，不讓一行散文冒充查證表。"""
        if self.status != "calibrated":
            return self
        required = (
            "source_id",
            "source_version",
            "locator",
            "conditions",
            "frequency_range_hz",
            "context",
            "verification_digest",
        )
        for name in required:
            value = getattr(self, name)
            if value is None or isinstance(value, str) and not value.strip():
                raise ValueError(f"calibrated 條目缺結構化出處 {name}")
        return self

    @model_validator(mode="after")
    def _source_frequency_range_ascends(self) -> Self:
        """來源適用頻段若存在，兩端必須有限、為正且遞增。"""
        if self.frequency_range_hz is None:
            return self
        lower, upper = self.frequency_range_hz
        if lower <= 0.0 or lower >= upper:
            raise ValueError("frequency_range_hz 必須是遞增的正頻率範圍")
        return self


class _SourceEntry(_SourceRecord):
    """可由用途鍵直接取得的一條來源記錄。"""

    key: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")


class SettingEntry(_SourceEntry):
    """不直接換算代價的量法設定。"""

    value: NumericValue
    unit: Unit


class TargetEntry(_SourceEntry):
    """帶代價形狀、較差參考與必要容許帶的品質目標或門檻。"""

    value: NumericValue
    unit: Unit
    cost_shape: CostShape
    tolerance: Annotated[StrictFloat, Field(ge=0.0)] | None = None
    worse_reference: Annotated[StrictFloat, Field(gt=0.0)]

    @model_validator(mode="after")
    def _tolerance_matches_cost_shape(self) -> Self:
        """只有範圍內最好需要容許帶；其他兩型多一格會讓代價語意不明。"""
        if self.cost_shape == "in_range_best" and self.tolerance is None:
            raise ValueError("in_range_best 必須帶 tolerance")
        if self.cost_shape != "in_range_best" and self.tolerance is not None:
            raise ValueError("只有 in_range_best 可以帶 tolerance")
        return self


class WeightItem(_SourceRecord):
    """權重表內的一類或一個主要分項；每列各自帶來源與狀態。"""

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    value: Annotated[StrictFloat, Field(ge=0.0)]
    note: str = Field(min_length=1)


class WeightTable(BaseModel):
    """一組共用用途鍵的類內或跨類權重。"""

    model_config = FROZEN

    key: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    item: tuple[WeightItem, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _item_names_are_unique(self) -> Self:
        """同一張權重表的一個名稱只准一列。"""
        seen: set[str] = set()
        for record in self.item:
            if record.name in seen:
                raise ValueError(f"權重表 {self.key} 重複名稱：{record.name}")
            seen.add(record.name)
        return self

    def _sorted_copy(self) -> WeightTable:
        """依名稱排序權重列，讓只換 TOML 排列不改設定指紋。"""
        return self.model_copy(
            update={"item": tuple(sorted(self.item, key=lambda record: record.name))}
        )


class QualificationEntry(_SourceEntry):
    """必評／選評清單或候選資料資格的一條規則。"""

    value: QualificationValue


QualityEntry = SettingEntry | TargetEntry | WeightTable | QualificationEntry


class QualityPurpose(BaseModel):
    """一種用途的全部目標、量法設定、權重與資格。"""

    model_config = FROZEN

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    setting: tuple[SettingEntry, ...]
    target: tuple[TargetEntry, ...]
    weight: tuple[WeightTable, ...]
    qualification: tuple[QualificationEntry, ...]

    @property
    def entries(self) -> tuple[QualityEntry, ...]:
        """回傳這個用途的全部條目，供狀態污染與來源盤點共用。"""
        return self.setting + self.target + self.weight + self.qualification

    @property
    def records(self) -> tuple[_SourceRecord, ...]:
        """攤平所有真正帶狀態與出處的列；權重表外殼本身不是一條品質尺。"""
        weighted = tuple(record for table in self.weight for record in table.item)
        return self.setting + self.target + weighted + self.qualification

    @model_validator(mode="after")
    def _entry_keys_are_unique(self) -> Self:
        """同一用途一個鍵只准一條；兩個答案等於沒有唯一住處。"""
        seen: set[str] = set()
        for entry in self.entries:
            if entry.key in seen:
                raise ValueError(f"同一用途重複 key：{entry.key}")
            seen.add(entry.key)
        return self

    def entry(self, key: str) -> QualityEntry:
        """回傳具名條目；沒有就報錯，不猜相近鍵名。"""
        for item in self.entries:
            if item.key == key:
                return item
        raise KeyError(f"用途 {self.name} 沒有這個品質鍵：{key}")

    def _sorted_copy(self) -> QualityPurpose:
        """只排序表內條目；數列值（例如頻段上下端）保留原本語意順序。"""
        return self.model_copy(
            update={
                "setting": tuple(sorted(self.setting, key=lambda item: item.key)),
                "target": tuple(sorted(self.target, key=lambda item: item.key)),
                "weight": tuple(
                    sorted(
                        (table._sorted_copy() for table in self.weight),
                        key=lambda table: table.key,
                    )
                ),
                "qualification": tuple(
                    sorted(self.qualification, key=lambda item: item.key)
                ),
            }
        )


class QualityTargets(BaseModel):
    """整份品質登記簿；驗過後凍結，並以正規化內容產生 SHA-256 指紋。"""

    model_config = FROZEN

    schema_version: Literal[1]
    purposes: tuple[QualityPurpose, ...] = Field(alias="purpose", min_length=1)

    @model_validator(mode="after")
    def _purpose_names_are_unique(self) -> Self:
        """用途名稱不可重複，否則 purpose（用途查詢）會有兩個答案。"""
        seen: set[str] = set()
        for item in self.purposes:
            if item.name in seen:
                raise ValueError(f"重複用途：{item.name}")
            seen.add(item.name)
        return self

    def purpose(self, name: str) -> QualityPurpose:
        """回傳指名用途；沒有預設用途或替代用途。"""
        for item in self.purposes:
            if item.name == name:
                return item
        raise KeyError(f"品質登記簿沒有這個用途：{name}")

    @property
    def fingerprint(self) -> str:
        """驗證後內容的正規化 JSON（資料交換格式）SHA-256 十六進位指紋。"""
        normalized = self.model_copy(
            update={
                "purposes": tuple(
                    sorted(
                        (item._sorted_copy() for item in self.purposes),
                        key=lambda item: item.name,
                    )
                )
            }
        )
        canonical = json.dumps(
            normalized.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_quality_targets(path: str | Path) -> QualityTargets:
    """從呼叫端必給的 TOML（設定表）路徑讀出並驗證品質登記簿。"""
    with Path(path).open("rb") as config_file:
        loaded = tomllib.load(config_file)
    return QualityTargets.model_validate(loaded)
