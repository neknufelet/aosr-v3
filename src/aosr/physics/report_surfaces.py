"""表面分區的輸入格式與共用正規化；不載入材料求值或 JAX。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Annotated, Literal, Self, TypeAlias, cast

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictFloat, TypeAdapter, ValidationError, model_validator
from pydantic_core import ErrorDetails
from pydantic_core import PydanticCustomError
from pydantic.config import JsonDict

from aosr.geometry.shoebox import Room, Wall
from aosr.geometry.shoebox_mesh import MeshRectangle
from aosr.materials.construction_models import ConstantImpedance, MeasuredImpedance, MountedOnWall, SurfaceMaterial


WHOLE_WALL_ONLY = "每面只收整面一區，或一組蓋滿整面、互不重疊的矩形分區"
OLD_WALL_FIELD = "impedance_pa_s_per_m_by_wall"
ZONE_WALL_FIELD = "surface_zones_by_wall"


class WholeWallZone(BaseModel):
    """整面一區的舊寫法；材料直接採用既有構造型別。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    area: Literal["whole_wall"] = Field(description=WHOLE_WALL_ONLY)
    material: SurfaceMaterial = Field(description="牆面材料：構造或固定阻抗")


class RectangleZone(BaseModel):
    """人指定的牆面矩形，公尺座標；u 是欄、v 是列，軸向同 wall_grid_cell。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    area: Literal["rectangle"]
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$", description="同面唯一代號：ASCII 英數、底線、連字號，不收句點")
    u_m: tuple[StrictFloat, StrictFloat] = Field(description="u 軸起、迄公尺；floor/ceiling 為 x，x0/xL 為 y，y0/yL 為 x")
    v_m: tuple[StrictFloat, StrictFloat] = Field(description="v 軸起、迄公尺；floor/ceiling 為 y，其餘為 z")
    material: SurfaceMaterial = Field(description="分區材料：構造或固定阻抗")

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not self.u_m[0] < self.u_m[1] or not self.v_m[0] < self.v_m[1]:
            raise ValueError(f"分區 {self.id}：起點必須小於迄點")
        return self


SurfaceZone: TypeAlias = Annotated[WholeWallZone | RectangleZone, Field(discriminator="area")]


def _zone_list(value: object) -> object:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(WHOLE_WALL_ONLY)
    if len(value) == 1 and isinstance(value[0], Mapping) and "area" not in value[0]:
        return value
    areas = [item.get("area") if isinstance(item, Mapping) else getattr(item, "area", None) for item in value]
    if areas != ["whole_wall"] and any(area != "rectangle" for area in areas):
        ids = [str(item.get("id", "未填代號")) if isinstance(item, Mapping) else str(getattr(item, "id", "未填代號"))
               for item in value]
        raise ValueError(f"{WHOLE_WALL_ONLY}；分區 {'、'.join(ids)}")
    return value


ZonesByWall: TypeAlias = dict[str, Annotated[
    tuple[SurfaceZone, ...], BeforeValidator(_zone_list), Field(min_length=1)]]
_ZONES = TypeAdapter(ZonesByWall)


class RectangleZonesUnsupported(ValueError):
    """格式已收下矩形，這條計算或顯示路徑尚未支援。"""

    def __init__(self, wall: str, *, reason: str = "分區還不支援這一頁／這一路") -> None:
        super().__init__(f"牆面 {wall}：{reason}")


def whole_wall_materials(zones_by_wall: Mapping[str, Sequence[SurfaceZone]]) -> dict[str, SurfaceMaterial]:
    """先核每面形狀再讀材料；任何矩形（含單一滿牆矩形）一律拒收。"""
    materials = {}
    for name, zones in zones_by_wall.items():
        if len(zones) != 1 or not isinstance(zones[0], WholeWallZone) or zones[0].area != "whole_wall":
            raise RectangleZonesUnsupported(name)
        materials[name] = zones[0].material
    return materials


def rectangle_zone_at(
    wall: str, point: tuple[float, float, float], zones: Sequence[RectangleZone],
) -> RectangleZone:
    """已驗證分區的共用查點：沿用 shoebox.wall_grid_cell 的軸向與右上歸屬。

    內界用左閉右開；只有牆的最外緣收右端點，所以同一面的一次反彈只取一區。
    不加容差或挪動反彈點；牆與牆的交線由呼叫端逐面各查一次。
    """
    axes = [axis for axis in range(3) if axis != Wall.from_name(wall).axis()]
    u, v = point[axes[0]], point[axes[1]]
    width, height = max(zone.u_m[1] for zone in zones), max(zone.v_m[1] for zone in zones)
    for zone in zones:
        if (zone.u_m[0] <= u < zone.u_m[1] or u == zone.u_m[1] == width) and (
                zone.v_m[0] <= v < zone.v_m[1] or v == zone.v_m[1] == height):
            return zone
    raise ValueError(f"牆面 {wall} 的反彈點不在分區內：{point!r}")


def validate_surface_partition(room: Room, zones_by_wall: ZonesByWall | None) -> None:
    """文件／場景共用的精確覆蓋檢查；不以面積和或容差判斷。"""
    sizes = {"floor": (room.Lx, room.Ly), "ceiling": (room.Lx, room.Ly),
             "x0": (room.Ly, room.Lz), "xL": (room.Ly, room.Lz),
             "y0": (room.Lx, room.Lz), "yL": (room.Lx, room.Lz)}
    for name, zones in (zones_by_wall or {}).items():
        rectangles = [zone for zone in zones if isinstance(zone, RectangleZone)]
        if rectangles:
            _rectangle_partition(name, rectangles, *sizes[name])


def mesh_partitions(
    room: Room, zones_by_wall: Mapping[str, Sequence[SurfaceZone]],
) -> dict[Wall, tuple[MeshRectangle, ...]]:
    """驗完方案矩形後只交座標與代號給網格層；整面材料不送入 geometry。"""
    result = {}
    for wall in Wall.all():
        rectangles = [zone for zone in zones_by_wall.get(wall.wall_name(), ()) if isinstance(zone, RectangleZone)]
        if rectangles:
            axes = [axis for axis in range(3) if axis != wall.axis()]
            _rectangle_partition(wall.wall_name(), rectangles, *(room.length(axis) for axis in axes))
            result[wall] = tuple((zone.id, zone.u_m, zone.v_m) for zone in rectangles)
    return result


def _rectangle_partition(name: str, rectangles: Sequence[RectangleZone], width: float, height: float) -> None:
    ids: set[str] = set()
    for zone in rectangles:
        if zone.id in ids:
            raise _located((ZONE_WALL_FIELD, name), f"牆面 {name} 分區 {zone.id}：代號重複", zone)
        ids.add(zone.id)
        if not (0 <= zone.u_m[0] < zone.u_m[1] <= width and 0 <= zone.v_m[0] < zone.v_m[1] <= height):
            raise _located((ZONE_WALL_FIELD, name), f"牆面 {name} 分區 {zone.id}：座標超出牆面範圍", zone)
    u_edges = sorted({edge for zone in rectangles for edge in zone.u_m})
    v_edges = sorted({edge for zone in rectangles for edge in zone.v_m})
    label = f"牆面 {name} 分區 {'、'.join(zone.id for zone in rectangles)}"
    if u_edges[0] != 0 or u_edges[-1] != width or v_edges[0] != 0 or v_edges[-1] != height:
        raise _located((ZONE_WALL_FIELD, name), f"{label}：外框必須正好蓋滿整面", rectangles)
    for u0, u1 in zip(u_edges, u_edges[1:]):
        for v0, v1 in zip(v_edges, v_edges[1:]):
            owners = [zone.id for zone in rectangles
                      if zone.u_m[0] <= u0 and u1 <= zone.u_m[1] and zone.v_m[0] <= v0 and v1 <= zone.v_m[1]]
            if len(owners) > 1:
                raise _located((ZONE_WALL_FIELD, name), f"牆面 {name} 分區 {'、'.join(owners)}：矩形重疊", rectangles)
            if not owners:
                raise _located((ZONE_WALL_FIELD, name), f"{label}：有留縫，必須蓋滿整面", rectangles)


def _located(path: tuple[str, ...], message: str, value: object) -> ValidationError:
    return ValidationError.from_exception_data("牆面輸入", [{
        "type": PydanticCustomError("wall_input", message), "loc": path, "input": value}])


def normalized_surfaces(value: object, *, json_shape: bool = False) -> object:
    """先解析成已驗型別，再把六面整面且全固定的寫法折回六個浮點數；兩個模型共用且冪等。"""
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
            "type": PydanticCustomError("wall_input", _located_zone_message(error, new)),
            "loc": (ZONE_WALL_FIELD, *_zone_location(error)), "input": error["input"]}
            for error in _primary_zone_errors(exc)]) from exc
    constants = {name: zone[0].material for name, zone in zones.items()
                 if len(zone) == 1 and isinstance(zone[0], WholeWallZone)
                 and isinstance(zone[0].material, ConstantImpedance)}
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


def _zone_location(error: ErrorDetails) -> tuple[int | str, ...]:
    location = error["loc"]
    if error["type"] == "union_tag_not_found" and len(location) == 2:
        return (*location, "area")
    return location[:2] + location[3:] if len(location) > 2 and location[2] in ("whole_wall", "rectangle") else location


def _located_zone_message(error: ErrorDetails, walls: Mapping[str, object]) -> str:
    location = error["loc"]
    entries = walls.get(str(location[0]))
    zone = (entries[location[1]] if len(location) > 1 and isinstance(location[1], int)
            and isinstance(entries, (list, tuple)) else None)
    label = f"牆面 {location[0]}"
    if isinstance(zone, Mapping) and zone.get("area") == "rectangle":
        label += f" 分區 {zone.get('id', '未填代號')}"
    if error["type"] == "union_tag_not_found" and len(location) == 2:
        return f"{label}：必填分區種類 area"
    return f"{label}：{_zone_error_message(error)}"


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
    cell.pop("maxLength", None)
    items = cell["items"]
    assert isinstance(items, dict)
    variants = items["oneOf"]
    assert isinstance(variants, list)
    whole, rectangle = variants
    cell.update({"description": f"{WHOLE_WALL_ONLY}；蓋滿與不重疊由程式以房間尺寸驗證",
                 "anyOf": [{"items": whole, "maxItems": 1}, {"items": rectangle}]})
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


def report_materials_code(materials_by_wall: Mapping[Wall, SurfaceMaterial],
                          surface_zones_by_wall: ZonesByWall | None = None) -> str:
    """查表與未查表的報表標籤使用同一個材料分類。"""
    materials = (tuple(z.material for cells in surface_zones_by_wall.values() for z in cells)
                 if surface_zones_by_wall else tuple(materials_by_wall.values()))
    # 貼在原牆上的構造（#990）照它的原牆分類：原牆是實測表就算實測表那一類，其餘算整面構造。
    if any(isinstance(material, MeasuredImpedance) or isinstance(material, MountedOnWall) and isinstance(material.wall, MeasuredImpedance)
           for material in materials):
        return "measured_impedance_by_wall"
    return ("real_frequency_independent_impedance" if all(isinstance(material, ConstantImpedance)
            for material in materials) else "assembly_by_wall")


def _zone_error_message(error: Mapping[str, object]) -> str:
    """正式材料錯誤共用白話；位置由呼叫端保留，不把英文錯誤句傳出去。"""
    kind = error["type"]
    # 明給 null 多半是沒填；但多給的欄位、選項欄位填 null 要照原錯說，不能叫人去填值。
    if kind == "missing" or error.get("input") is None and kind not in ("value_error", "extra_forbidden", "literal_error"):
        return "必填：這一格不能空著"
    messages = {"float_type": "要填數字", "float_parsing": "要填數字",
        "string_pattern_mismatch": "分區代號只准 ASCII 英數、底線、連字號，不收句點",
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
