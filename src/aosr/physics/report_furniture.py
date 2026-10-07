"""家具絕對輸入的凍結契約與說明事實；不收主位從屬關係，不做接觸幾何判斷。"""
from __future__ import annotations

import math
from typing import Annotated, Self, TypeAlias, TypeVar

from pydantic import BeforeValidator, Field, field_validator, model_validator

from aosr.config.furniture_materials import FURNITURE_MATERIAL_KINDS, FurnitureMaterialName
from aosr.geometry.furniture import FurnitureKind
from aosr.physics.report_facts import FactsModel, NO_BASIS_TEXT, NOT_MEASURED, facts, positive_facts


def finite_number(value: object) -> float:
    """欄位的有限數字契約：文字與布林不冒充座標。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("家具座標必須是有限數字")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError("家具座標必須是有限數字") from exc
    if not math.isfinite(number):
        raise ValueError("家具座標必須是有限數字")
    return number


def positive_dimension(value: object) -> float:
    """寬深高須能表示為有限正數。"""
    try:
        number = finite_number(value)
    except ValueError as exc:
        raise ValueError("家具尺寸必須是有限正數") from exc
    if number <= 0.0:
        raise ValueError("家具尺寸必須是有限正數")
    return number


def cardinal_yaw(value: object) -> float:
    """第一版的四個直角，不做三角函數或角度裁切。"""
    message = "家具角度只收數字 0／90／180／270 度"
    try:
        number = finite_number(value)
    except ValueError as exc:
        raise ValueError(message) from exc
    if number not in (0.0, 90.0, 180.0, 270.0):
        raise ValueError(message)
    return number


# 第五、六支接上鏡像法以前，任何算物理的入口收到家具都用這一句拒收，不准悄悄照沒家具算。
FURNITURE_UNSUPPORTED = "鏡像法尚未支援家具（#559 第五、六支施工中）"

FiniteCoordinate: TypeAlias = Annotated[float, BeforeValidator(finite_number)]
PositiveDimension: TypeAlias = Annotated[float, BeforeValidator(positive_dimension), Field(gt=0.0)]
CardinalYaw: TypeAlias = Annotated[float, BeforeValidator(cardinal_yaw)]


class FurnitureRecord(FactsModel):
    """相對方案與絕對輸入共用的種類、材質、尺寸契約；准用表只有 config 那一份。"""

    furniture_id: str = Field(min_length=1, json_schema_extra=facts("家具代號", "1", NO_BASIS_TEXT, NOT_MEASURED))
    kind: FurnitureKind = Field(json_schema_extra=facts("家具種類", "1", NO_BASIS_TEXT, NOT_MEASURED))
    material: FurnitureMaterialName = Field(json_schema_extra=facts("材質類型", "1", NO_BASIS_TEXT, NOT_MEASURED))
    width_m: PositiveDimension = Field(json_schema_extra=positive_facts("寬度", "m", "家具本地左右全長"))
    depth_m: PositiveDimension = Field(json_schema_extra=positive_facts("深度", "m", "家具本地前後全長"))
    height_m: PositiveDimension = Field(json_schema_extra=positive_facts("高度", "m", "底面到頂面全長"))

    @field_validator("furniture_id")
    @classmethod
    def nonblank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("furniture_id 不可為空白")
        return value

    @field_validator("material", mode="before")
    @classmethod
    def known_material(cls, value: object) -> object:
        if not isinstance(value, str) or value not in FURNITURE_MATERIAL_KINDS:
            raise ValueError("家具材質必須是五種材質之一：fabric／leather／wood／glass／absorptive_cloud")
        return value

    @model_validator(mode="after")
    def allowed_material(self) -> Self:
        if self.kind not in FURNITURE_MATERIAL_KINDS[self.material]:
            raise ValueError(f"家具 {self.furniture_id} 的材質 {self.material} 不適用 {self.kind.value}")
        return self


class AbsoluteFurniture(FurnitureRecord):
    """底面中心房間座標與絕對角度；0 度時寬沿房間 x、深沿房間 y。"""

    bottom_center_m: tuple[FiniteCoordinate, FiniteCoordinate, FiniteCoordinate] = Field(
        description="底面中心的房間 x、y、z 座標；不是耳朵高度或家具體心",
        json_schema_extra=facts("底面中心座標", "m", "房間角落為原點", NOT_MEASURED),
    )
    yaw_deg: CardinalYaw = Field(
        description="絕對直角轉向：0 度寬沿 x、深沿 y；只收 0／90／180／270",
        json_schema_extra=facts("轉角", "deg", "房間 x 軸為寬度方向，繞垂直軸逆時針", NOT_MEASURED)
        | {"enum": [0.0, 90.0, 180.0, 270.0]},
    )


FurnitureT = TypeVar("FurnitureT", bound=FurnitureRecord)


def normalized_furniture(items: tuple[FurnitureT, ...] | None) -> tuple[FurnitureT, ...] | None:
    """空清單等同省略；代號唯一並排序，宣告順序不成為場景身分。"""
    if not items:
        return None
    seen = set()
    for item in items:
        if item.furniture_id in seen:
            raise ValueError(f"furniture_id 不可重複：{item.furniture_id}")
        seen.add(item.furniture_id)
    return tuple(sorted(items, key=lambda item: item.furniture_id))
