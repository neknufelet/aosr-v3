"""房間比例建議的凍結條文表；十進位門檻精確載入，不選預設檔。"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import re
import tomllib
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


FROZEN = ConfigDict(frozen=True, extra="forbid")
Standard = Literal["ITU-R BS.1116-3 (02/2015)", "EBU Tech 3276, 2nd edition (1998)"]
Kind = Literal["band", "below", "near_integer"]


class ProportionThresholds(BaseModel):
    """只收十進位字串，不讓浮點門檻混進精確比較。"""

    model_config = FROZEN

    lower_factor: Fraction | None = None
    upper_factor: Fraction | None = None
    upper_offset: Fraction | None = None
    upper: Fraction | None = None
    relative_width: Fraction | None = None

    @field_validator("*", mode="before")
    @classmethod
    def _decimal_string(cls, value: object) -> Fraction:
        if not isinstance(value, str) or re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None:
            raise ValueError("門檻必須是非負十進位字串")
        return Fraction(value)


class RoomProportionRule(BaseModel):
    """一條判法；多份來源的標準、條次、頁碼與原文依位置配對。"""

    model_config = FROZEN

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    standard: tuple[Standard, ...] = Field(min_length=1)
    clause: tuple[str, ...] = Field(min_length=1)
    pdf_page: tuple[int, ...] = Field(min_length=1)
    quote: tuple[str, ...] = Field(min_length=1)
    kind: Kind
    thresholds: ProportionThresholds
    description: str = Field(min_length=1)

    @model_validator(mode="after")
    def _sources_and_thresholds_match(self) -> Self:
        if not all(len(values) == len(self.standard) for values in (self.clause, self.pdf_page, self.quote)):
            raise ValueError("每份標準的條次、頁碼與原文必須配對")
        if len(set(self.standard)) != len(self.standard):
            raise ValueError("同一條的來源標準不可重複")
        if any(not text.strip() for text in (*self.clause, *self.quote, self.description)):
            raise ValueError("條次、原文與中文說明不可空白")
        if any(page <= 0 for page in self.pdf_page):
            raise ValueError("PDF 頁碼必須是正整數")
        required = {"band": {"lower_factor", "upper_factor", "upper_offset"},
                    "below": {"upper"}, "near_integer": {"relative_width"}}
        present = self.thresholds.model_dump(exclude_none=True)
        if set(present) != required[self.kind]:
            raise ValueError("判法與門檻欄位不相符")
        limits = self.thresholds
        if any(value is not None and value <= 0 for value in (limits.lower_factor, limits.upper_factor, limits.upper)):
            raise ValueError("比例係數與上限必須為正")
        if limits.relative_width is not None and not 0 < limits.relative_width < 1:
            raise ValueError("整數相對帶寬必須介於零與一之間")
        return self


class RoomProportions(BaseModel):
    """只供提醒的條文表；不修改搜尋條件或評分。"""

    model_config = FROZEN

    entries: tuple[RoomProportionRule, ...] = Field(alias="entry", min_length=1)

    @model_validator(mode="after")
    def _ids_are_unique(self) -> Self:
        identities = [entry.id for entry in self.entries]
        if len(identities) != len(set(identities)):
            raise ValueError("條文代號不可重複")
        return self


def load_room_proportions(path: str | Path) -> RoomProportions:
    """載入呼叫端指定的 TOML（設定表），驗證來源與門檻。"""
    with Path(path).open("rb") as config_file:
        loaded = tomllib.load(config_file)
    return RoomProportions.model_validate(loaded)
