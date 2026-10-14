"""整面表面分區的輸入格式與共用正規化；不載入材料求值或 JAX。"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import Annotated, Literal, TypeAlias, cast

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter, ValidationError, field_validator
from pydantic_core import ErrorDetails
from pydantic_core import PydanticCustomError
from pydantic.config import JsonDict

from aosr.geometry.shoebox import Wall
from aosr.materials.construction_models import ConstantImpedance, MeasuredImpedance, SurfaceMaterial


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


def normalized_surfaces(value: object, *, json_shape: bool = False) -> object:
    """先解析成已驗型別，再把全固定寫法折回六個浮點數；兩個模型共用且冪等。"""
    value = validation_content(value, json_shape=json_shape)
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
            for error in _primary_zone_errors(exc)]) from exc
    constants = {name: zone[0].material for name, zone in zones.items()
                 if isinstance(zone[0].material, ConstantImpedance)}
    if constants.keys() == names:
        document[OLD_WALL_FIELD] = {name: material.impedance_pa_s_m for name, material in constants.items()}
        document.pop(ZONE_WALL_FIELD, None)
    else:
        document[ZONE_WALL_FIELD] = zones
        document.pop(OLD_WALL_FIELD, None)
    return document


def validation_content(value: object, *, json_shape: bool = False) -> object:
    """物件回到它的欄位內容再驗：省略而且仍是預設空值的欄位照舊省略、明給空值照舊明給。

    其餘欄位照現值帶上：直接建構時用預設的種類代號 kind（聯合型別靠它分辨），與驗證器補出的值，
    都是物件的現值，省略了會掉回預設或認不出材料。
    ``json_shape``（JSON 形狀）給報表輸入用：那份契約只收 JSON 物件與陣列，資料類別與 tuple 都攤平。
    其餘（方案、場景）裡面沒有模型的資料類別與 tuple（固定序列）保留原型別，嚴格模式照收。
    """
    if isinstance(value, BaseModel):
        return {name: validation_content(getattr(value, name), json_shape=json_shape)
                for name, field in type(value).model_fields.items()
                if name in value.model_fields_set or getattr(value, name) is not None or field.default is not None}
    if is_dataclass(value) and not isinstance(value, type):
        given = {item.name: getattr(value, item.name) for item in fields(value) if item.init}
        content = {name: validation_content(item, json_shape=json_shape) for name, item in given.items()}
        return content if json_shape or any(content[name] is not item for name, item in given.items()) else value
    if isinstance(value, Mapping):
        return {name: validation_content(item, json_shape=json_shape) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        items = [validation_content(item, json_shape=json_shape) for item in value]
        if isinstance(value, tuple) and not json_shape:
            return value if all(new is old for new, old in zip(items, value)) else tuple(items)
        return items
    return value


def _primary_zone_errors(exc: ValidationError) -> list[ErrorDetails]:
    """壞元素被驗證器移除後也會觸發最少長度；保留元素原錯，不重報衍生短列。"""
    errors = exc.errors()
    return [error for error in errors if not (error["type"] == "too_short" and
        any(other["loc"][:len(error["loc"])] == error["loc"] and other["loc"] != error["loc"]
            for other in errors))]


def zone_object_schema(schema: JsonDict) -> None:
    """將自動產生的每面陣列綱要保留，再展開六面封閉物件。"""
    variants = schema["anyOf"]
    assert isinstance(variants, list)
    owner = variants[0]
    assert isinstance(owner, dict)
    cell = owner["additionalProperties"]
    assert isinstance(cell, dict)
    # BeforeValidator（先驗清單）讓自動產生器把長度放錯關鍵字，這裡恢復陣列語意。
    cell["minItems"] = cell.pop("minLength", cell.get("minItems"))
    cell["maxItems"] = cell.pop("maxLength", cell.get("maxItems"))
    schema.pop("anyOf")
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
    old_field = properties[OLD_WALL_FIELD]
    assert isinstance(old_field, dict)
    old_field.pop("anyOf", None)
    old_field["type"] = "object"
    schema["oneOf"] = [{"required": [OLD_WALL_FIELD], "not": {"required": [ZONE_WALL_FIELD]}},
                       {"required": [ZONE_WALL_FIELD], "not": {"required": [OLD_WALL_FIELD]}}]


def report_materials_code(materials_by_wall: Mapping[Wall, SurfaceMaterial]) -> str:
    """查表與未查表的報表標籤使用同一個材料分類。"""
    if any(isinstance(material, MeasuredImpedance) for material in materials_by_wall.values()):
        return "measured_impedance_by_wall"
    return ("real_frequency_independent_impedance" if all(isinstance(material, ConstantImpedance)
            for material in materials_by_wall.values()) else "assembly_by_wall")


def _zone_error_message(error: Mapping[str, object]) -> str:
    """正式材料錯誤共用白話；位置由呼叫端保留，不把英文錯誤句傳出去。"""
    kind = error["type"]
    # 明給 null 多半是沒填；但多給的欄位、選項欄位填 null 要照原錯說，不能叫人去填值。
    if kind == "missing" or error.get("input") is None and kind not in ("value_error", "extra_forbidden", "literal_error"):
        return "必填：這一格不能空著"
    messages = {"float_type": "要填數字", "float_parsing": "要填數字",
        "finite_number": "要填有限數字（不收無限大或非數值）",
        "string_type": "要填文字", "string_too_short": "必填：這一格不能空著",
        "union_tag_not_found": "必填材料或層的種類代號 kind（種類）",
        "union_tag_invalid": "不認識的材料或層種類代號 kind（種類）；請選合法種類",
        "literal_error": "只能填這個材料或層的合法選項",
        "extra_forbidden": "多了這個材料或層不收的欄位",
        "tuple_type": "要填層的清單", "model_type": "材料或層要填一組欄位",
        "model_attributes_type": "材料或層要填一組欄位", "dict_type": "材料或層要填一組欄位"}
    if kind in messages:
        return messages[kind]
    context = cast(dict[str, object], error.get("ctx") or {})
    bounds = {"greater_than": ("要大於", "gt"), "greater_than_equal": ("不可小於", "ge"),
              "less_than": ("要小於", "lt"), "less_than_equal": ("不可大於", "le")}
    if kind in bounds:
        words, key = bounds[kind]
        leaf = "孔隙率" if "porosity" in cast(tuple[object, ...], error.get("loc", ())) else "數值"
        return f"{leaf}{words} {context[key]}"
    if kind in ("too_short", "too_long"):
        words, key = ("至少要", "min_length") if kind == "too_short" else ("最多只收", "max_length")
        return f"清單{words} {context[key]} 項"
    return str(error["msg"]).removeprefix("Value error, ")
