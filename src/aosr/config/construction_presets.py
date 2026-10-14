"""構造填表資料：只讀數字與出處，材料一律由完整方案的正式入口驗證。"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConstructionPreset(BaseModel):
    """名字只供顯示與選單使用，方案保存整份 material（材料）參數。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    material: dict[str, object]
    source: str = Field(min_length=1)
    links: tuple[str, ...] = Field(min_length=1)
    notes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def complete_sources(self) -> Self:
        if any(not link.startswith("https://") for link in self.links):
            raise ValueError("預設出處網址須以 https:// 開頭")
        if any(not note for note in self.notes):
            raise ValueError("預設顯示註記不可空白")
        return self


class _PresetFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preset: tuple[ConstructionPreset, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_names(self) -> Self:
        ids, labels = set(), set()
        for item in self.preset:
            if item.id in ids or item.label in labels:
                raise ValueError("預設代號與顯示名不可重複")
            ids.add(item.id)
            labels.add(item.label)
        return self


def load_construction_presets(path: str | Path) -> tuple[ConstructionPreset, ...]:
    """只讀明給的資料檔，不建材料模型、不推測預設出身。"""
    with Path(path).open("rb") as stream:
        return _PresetFile.model_validate(tomllib.load(stream)).preset
