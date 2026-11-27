"""文獻頻帶吸音率的硬面材料（門、窗、木地板、石膏板、磚、混凝土）：只讀數字與出處。

數字是型錄或手冊的無規入射頻帶吸音率；換成阻抗表的事在材料層（:mod:`aosr.materials.library`）做，
照決策紙 ``catalog-absorption-random-incidence-paris-inversion`` 的內插與反推。
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CatalogSurface(BaseModel):
    """一筆文獻硬面：顯示名、頻帶中心、每帶吸音率與出處。"""

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    band_center_hz: tuple[float, ...] = Field(min_length=2)
    absorption: tuple[float, ...] = Field(min_length=2)
    source: str = Field(min_length=1)
    links: tuple[str, ...] = Field(min_length=1)
    notes: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(self.band_center_hz) != len(self.absorption):
            raise ValueError(f"{self.id}：頻帶數與吸音率個數不同")
        if any(not link.startswith("https://") for link in self.links):
            raise ValueError(f"{self.id}：出處網址須以 https:// 開頭")
        if any(not note for note in self.notes):
            raise ValueError(f"{self.id}：顯示註記不可空白")
        return self


class CatalogTable(BaseModel):
    """換成阻抗表時取點的範圍與密度：表涵蓋整條產品頻率軸，表外不會被用到。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    lowest_hz: float = Field(gt=0.0)
    highest_hz: float = Field(gt=0.0)
    points_per_octave: int = Field(gt=0)
    reference_hz: float = Field(gt=0.0)


class _CatalogFile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    table: CatalogTable
    surface: tuple[CatalogSurface, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        if len({item.id for item in self.surface}) != len(self.surface) or \
                len({item.label for item in self.surface}) != len(self.surface):
            raise ValueError("文獻硬面的代號與顯示名不可重複")
        if self.table.highest_hz <= self.table.lowest_hz:
            raise ValueError("阻抗表的最高頻要大於最低頻")
        return self


def load_catalog_surfaces(path: str | Path) -> tuple[CatalogTable, tuple[CatalogSurface, ...]]:
    """只讀明給的資料檔，不建材料模型。"""
    with Path(path).open("rb") as stream:
        loaded = _CatalogFile.model_validate(tomllib.load(stream))
    return loaded.table, loaded.surface
