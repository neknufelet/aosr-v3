"""#559 代表模型：尺寸存毫米，明記聲學中心，提供 Cabinet（搜尋箱體）的公尺欄位。"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from aosr.config._furniture_records import (
    EstimatedProvenance, FrozenRecord, NonnegativeNumber, PositiveNumber,
)


class RepresentativeSpeaker(FrozenRecord):
    """箱高與底座分存；底座只增高，寬深維持箱體尺寸，不另建底座外形。

    聲學中心從整體箱底量：書架為箱底；落地含底座的整體箱底就是地面。
    acoustic_center_behind_front_mm 是從前面板往箱背的距離，0 明示在前面板。
    """

    width_mm: PositiveNumber
    depth_mm: PositiveNumber
    body_height_mm: PositiveNumber
    base_height_mm: NonnegativeNumber
    acoustic_center_above_bottom_mm: PositiveNumber
    acoustic_center_behind_front_mm: NonnegativeNumber
    acoustic_center_reference: Literal["cabinet_bottom", "floor"]
    label: str = Field(min_length=1)
    provenance: EstimatedProvenance

    @model_validator(mode="after")
    def center_within_box(self) -> Self:
        height = self.body_height_mm + self.base_height_mm
        if not math.isfinite(height):
            raise ValueError("含底座總高必須有限")
        if self.acoustic_center_above_bottom_mm > height:
            raise ValueError("聲學中心不得高於含底座箱頂")
        if self.acoustic_center_above_bottom_mm <= self.base_height_mm:
            raise ValueError("聲學中心必須高於底座，落在箱體上")
        if self.acoustic_center_behind_front_mm >= self.depth_mm:
            raise ValueError("聲學中心必須在箱背之前")
        return self

    def cabinet_fields_m(self) -> dict[str, float]:
        """Cabinet 五格的明確換算；呼叫端可用 Cabinet(**欄位) 驗證，不依賴箱高一半。"""
        return {"width_m": self.width_mm / 1000.0, "depth_m": self.depth_mm / 1000.0,
                "height_m": (self.body_height_mm + self.base_height_mm) / 1000.0,
                "acoustic_center_above_bottom_m": self.acoustic_center_above_bottom_mm / 1000.0,
                "acoustic_center_behind_front_m": self.acoustic_center_behind_front_mm / 1000.0}


class RepresentativeSpeakers(FrozenRecord):
    """書架與落地各自帶出處；量高基準與底座不能混用。"""

    bookshelf: RepresentativeSpeaker
    floorstanding: RepresentativeSpeaker

    @model_validator(mode="after")
    def explicit_references(self) -> Self:
        if self.bookshelf.acoustic_center_reference != "cabinet_bottom" or self.bookshelf.base_height_mm != 0.0:
            raise ValueError("書架聲學中心從箱底量，代表書架不含底座")
        if self.floorstanding.acoustic_center_reference != "floor":
            raise ValueError("落地聲學中心從含底座的地面量")
        return self


def load_representative_speakers(path: str | Path) -> RepresentativeSpeakers:
    """只讀呼叫端明給的路徑；聲學中心與四格出處必填。"""
    with Path(path).open("rb") as file:
        return RepresentativeSpeakers.model_validate(tomllib.load(file))
