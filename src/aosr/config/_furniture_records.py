"""家具資料共用的凍結模型與四格出處；不改既有物理閉包的出處型別。"""
from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictFloat


PositiveNumber: TypeAlias = Annotated[StrictFloat, Field(gt=0.0)]
NonnegativeNumber: TypeAlias = Annotated[StrictFloat, Field(ge=0.0)]


class FrozenRecord(BaseModel):
    """容器與內部序列皆不可變；拒收額外欄位、布林數字與非有限值。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class EstimatedProvenance(FrozenRecord):
    """estimated 是估計；三種來源分別為物件量測換算、手冊借值、廠商尺寸中位數。"""

    status: Literal["estimated"]
    source_kind: Literal["object_measurement_conversion", "handbook", "manufacturer_median"]
    source: str = Field(min_length=1)
    conditions: str = Field(min_length=1)
