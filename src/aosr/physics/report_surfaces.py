"""整面表面分區的輸入格式與共用正規化；不載入材料求值或 JAX。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from pydantic_core import PydanticCustomError
from pydantic.config import JsonDict

from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import ConstantImpedance, SurfaceMaterial


MEASURED_IMPEDANCE_UNSUPPORTED = "實測阻抗表這一版還不收：表沒涵蓋的頻點怎麼處理還沒拍板（#309）；請改填構造或固定阻抗"
WHOLE_WALL_ONLY = "這一版每面只收整面一區；矩形分區是 #309 第二步"
OLD_WALL_FIELD = "impedance_pa_s_per_m_by_wall"
ZONE_WALL_FIELD = "surface_zones_by_wall"


class WholeWallZone(BaseModel):
    """這一版每面恰好一區；材料直接採用既有構造型別。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    area: Literal["whole_wall"] = Field(description=WHOLE_WALL_ONLY)
    material: SurfaceMaterial = Field(description="牆面材料：構造或固定阻抗")

    @field_validator("area", mode="before")
    @classmethod
    def _whole_wall(cls, value: object) -> object:
        if value != "whole_wall":
            raise ValueError(WHOLE_WALL_ONLY)
        return value

    @field_validator("material", mode="before")
    @classmethod
    def _reject_measured(cls, value: object) -> object:
        if isinstance(value, Mapping) and value.get("kind") == "measured_impedance":
            raise ValueError(MEASURED_IMPEDANCE_UNSUPPORTED)
        return value


def _single_zone(value: object) -> object:
    if not isinstance(value, (list, tuple)) or len(value) != 1:
        raise ValueError(WHOLE_WALL_ONLY)
    return value


ZonesByWall: TypeAlias = dict[str, Annotated[
    tuple[WholeWallZone, ...], BeforeValidator(_single_zone), Field(min_length=1, max_length=1)]]
_ZONES = TypeAdapter(ZonesByWall)


def _located(path: tuple[str, ...], message: str, value: object) -> ValidationError:
    return ValidationError.from_exception_data("牆面輸入", [{
        "type": PydanticCustomError("wall_input", message), "loc": path, "input": value}])


def normalized_surfaces(value: object) -> object:
    """先解析成已驗型別，再把全固定寫法折回六個浮點數；兩個模型共用且冪等。"""
    if not isinstance(value, Mapping):
        return value
    document = dict(value)
    old, new = document.get(OLD_WALL_FIELD), document.get(ZONE_WALL_FIELD)
    if (OLD_WALL_FIELD in document) == (ZONE_WALL_FIELD in document) or (old is None and new is None):
        raise _located((ZONE_WALL_FIELD,), "新舊牆面欄位必須恰好給一個：固定阻抗或表面分區清單", document)
    if new is None:
        document.pop(ZONE_WALL_FIELD, None)
        return document
    if not isinstance(new, Mapping):
        raise _located((ZONE_WALL_FIELD,), "表面分區清單必須是 JSON 物件", new)
    names = {wall.wall_name() for wall in Wall.all()}
    for name in sorted(names - new.keys()):
        raise _located((ZONE_WALL_FIELD, name), f"缺少牆面 {name}", new)
    for name in sorted(new.keys() - names):
        raise _located((ZONE_WALL_FIELD, str(name)), f"不認識的牆名 {name}", new)
    try:
        zones = _ZONES.validate_python(new)
    except ValidationError as exc:
        # 保留 tagged union（依種類選型別）的完整位置，外加正式欄位名。
        raise ValidationError.from_exception_data("牆面輸入", [{
            "type": PydanticCustomError("wall_input", _zone_error_message(error)),
            "loc": (ZONE_WALL_FIELD, *error["loc"]), "input": error["input"]}
            for error in exc.errors()]) from exc
    zones = {wall.wall_name(): zones[wall.wall_name()] for wall in Wall.all()}
    constants = {name: zone[0].material for name, zone in zones.items()
                 if isinstance(zone[0].material, ConstantImpedance)}
    if constants.keys() == names:
        document[OLD_WALL_FIELD] = {name: material.impedance_pa_s_m for name, material in constants.items()}
        document.pop(ZONE_WALL_FIELD, None)
    else:
        document[ZONE_WALL_FIELD] = zones
        document.pop(OLD_WALL_FIELD, None)
    return document


def zone_object_schema(schema: JsonDict) -> None:
    """將自動產生的每面陣列綱要保留，再展開六面封閉物件。"""
    variants = schema["anyOf"]
    assert isinstance(variants, list)
    owner = variants[0]
    assert isinstance(owner, dict)
    cell = owner["additionalProperties"]
    schema.update({"type": "object", "additionalProperties": False,
        "required": [wall.wall_name() for wall in Wall.all()],
        "properties": {wall.wall_name(): cell for wall in Wall.all()}})


def exclusive_wall_schema(schema: JsonDict) -> None:
    """綱要宣告新舊入口恰好一個，省略欄位不影響固定場景的存檔。"""
    properties = schema["properties"]
    assert isinstance(properties, dict)
    field = properties[ZONE_WALL_FIELD]
    assert isinstance(field, dict)
    zone_field = JsonDict(field)
    zone_object_schema(zone_field)
    properties[ZONE_WALL_FIELD] = zone_field
    schema["oneOf"] = [{"required": [OLD_WALL_FIELD], "not": {"required": [ZONE_WALL_FIELD]}},
                       {"required": [ZONE_WALL_FIELD], "not": {"required": [OLD_WALL_FIELD]}}]


def report_materials_code(materials_by_wall: Mapping[Wall, SurfaceMaterial]) -> str:
    """查表與未查表的報表標籤使用同一個材料分類。"""
    return ("real_frequency_independent_impedance" if all(isinstance(material, ConstantImpedance)
            for material in materials_by_wall.values()) else "assembly_by_wall")


def _zone_error_message(error: Mapping[str, object]) -> str:
    """材料的數值界限帶位置與白話，其他模型訊息保持原文。"""
    if error["type"] == "greater_than":
        context = error.get("ctx")
        if isinstance(context, dict):
            return f"要大於 {context['gt']}（層參數或阻抗必須為正）"
    return str(error["msg"]).removeprefix("Value error, ")
