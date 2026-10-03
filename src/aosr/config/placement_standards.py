"""擺位條文登記簿的唯讀載入與驗證；門檻只來自呼叫端資料。"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
Measurement = Literal[
    "height_difference", "speaker_surfaces", "base_width", "listening_distance",
    "listening_angle", "area_radius", "speaker_height", "ear_height", "inclination",
    "speaker_walls", "listener_walls", "angle_distance",
]
Comparison = Literal[
    "equal", "minimum", "maximum", "closed_range", "open_range",
    "preferred_range", "scaled_range", "value_only",
]


class Thresholds(BaseModel):
    """判定用門檻與只列值條文的參考點；沒有隱含標準數字。"""

    model_config = FROZEN

    target: float | None = None
    lower: float | None = Field(default=None, ge=0.0)
    upper: float | None = Field(default=None, ge=0.0)
    acceptable_upper: float | None = Field(default=None, ge=0.0)
    upper_factor: float | None = Field(default=None, gt=0.0)
    reference_ratio: float | None = Field(default=None, ge=0.0)


class PlacementStandard(BaseModel):
    """一條可回查的原文、量法、判法與中文說明。"""

    model_config = FROZEN

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    standard: Literal["ITU-R BS.1116-3 (02/2015)", "EBU Tech 3276, 2nd edition (1998)"]
    clause: str = Field(min_length=1)
    pdf_page: int = Field(gt=0)
    quote: str = Field(min_length=1)
    measurement: Measurement
    comparison: Comparison
    thresholds: Thresholds
    description: str = Field(min_length=1)
    failure_note: str | None = None

    @model_validator(mode="after")
    def _quote_and_thresholds_match(self) -> Self:
        if not self.quote.strip():
            raise ValueError("原文句不可空白")
        required = {
            "equal": {"target"}, "minimum": {"lower"}, "maximum": {"upper"},
            "closed_range": {"lower", "upper"}, "open_range": {"lower", "upper"},
            "preferred_range": {"lower", "upper", "acceptable_upper"},
            "scaled_range": {"lower", "upper_factor"},
        }
        present = set(self.thresholds.model_dump(exclude_none=True))
        wanted = required.get(self.comparison, {"target", "reference_ratio"})
        if self.comparison == "value_only":
            if not present <= wanted:
                raise ValueError("只列值不准帶判定門檻")
        elif present != wanted:
            raise ValueError("判法與門檻欄位不相符")
        limits = self.thresholds
        if limits.lower is not None and limits.upper is not None and limits.lower > limits.upper:
            raise ValueError("下限不可超過上限")
        if (limits.acceptable_upper is not None and limits.upper is not None
                and limits.acceptable_upper <= limits.upper):
            raise ValueError("可接受上限必須超過偏好上限")
        return self


class PlacementStandards(BaseModel):
    """整份凍結條文清單；按資料檔順序輸出，不影響搜尋或評分。"""

    model_config = FROZEN

    entries: tuple[PlacementStandard, ...] = Field(alias="entry", min_length=1)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        identities = [entry.id for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("條文代號不可重複")
        return self


def load_placement_standards(path: str | Path) -> PlacementStandards:
    """從呼叫端指定的 TOML（設定表）路徑載入並驗證，不選預設檔。"""
    with Path(path).open("rb") as config_file:
        loaded = tomllib.load(config_file)
    return PlacementStandards.model_validate(loaded)
