"""牆面構造的凍結資料格式；所有連續欄位用 SI（國際單位制）。

層順序是房間側 → 牆體側，背後負載必填。實測阻抗表在表內對數頻率內插，表外交必填底構造。
孔隙率與曲折度沿用傳遞矩陣的合法域：皆正；不額外收緊曲折度為大於等於一。
Mechel 1976（低頻延伸多孔模型）的孔隙率另須不大於一。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Literal, Self, TypeAlias, TypeVar

if TYPE_CHECKING:
    import jax

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictFloat, ValidationError, field_validator, model_validator
from pydantic.config import JsonDict


_PositiveNumber: TypeAlias = Annotated[StrictFloat, Field(gt=0)]
_NonnegativeNumber: TypeAlias = Annotated[StrictFloat, Field(ge=0)]


def _required_kind(schema: JsonDict) -> None:
    """聯合分支必須明給 kind（種類）；保留直接建構物件的既有預設值。"""
    properties = schema.get("properties")
    if isinstance(properties, dict) and "kind" in properties:
        required = schema.setdefault("required", [])
        assert isinstance(required, list)
        if "kind" not in required:
            required.append("kind")


def _porous_schema(schema: JsonDict) -> None:
    """將多孔模型的參數契約表達為條件式；未給模型等於 komatsu（纖維經驗式）。"""
    _required_kind(schema)
    schema["allOf"] = [
        {"if": {"properties": {"model": {"enum": ["komatsu", "miki"]}}},
         "then": {"not": {"anyOf": [{"required": ["porosity"]}, {"required": ["tortuosity"]}]}}},
        {"if": {"properties": {"model": {"const": "wilson_2015"}}, "required": ["model"]},
         "then": {"required": ["porosity", "tortuosity"],
                  "properties": {"porosity": {"not": {"type": "null"}}, "tortuosity": {"not": {"type": "null"}}}}},
        {"if": {"properties": {"model": {"const": "mechel_1976"}}, "required": ["model"]},
         "then": {"required": ["porosity"], "properties": {"porosity": {"not": {"type": "null"}, "maximum": 1}},
                  "not": {"required": ["tortuosity"]}}},
    ]


class _ConstructionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_required_kind)


class PorousLayer(_ConstructionRecord):
    """PorousLayer（多孔層）：模型決定必填參數，不接受會被忽略的欄位。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_porous_schema)

    kind: Literal["porous"] = Field(default="porous", description="kind（層種類代號，porous 表示多孔層）")
    thickness_m: _PositiveNumber = Field(description="thickness_m（層厚度，m）")
    flow_resistivity_pa_s_m2: _PositiveNumber = Field(description="flow_resistivity_pa_s_m2（流阻，Pa·s/m²）")
    flow_resistivity_source: Literal["measured", "density_table", "catalog", "unspecified"] = Field(default="unspecified", exclude_if=lambda value: value == "unspecified", description="flow_resistivity_source（流阻從哪裡來：measured 表示 實測，density_table 表示 照密度查論文表，catalog 表示 廠商型錄，unspecified 表示 未註明；只記出處，不影響計算）")
    model: Literal["komatsu", "miki", "wilson_2015", "mechel_1976"] = Field(default="komatsu", description="model（多孔模型：komatsu 為小松纖維經驗式，miki 為三木纖維經驗式，wilson_2015 為威爾森鬆弛模型，mechel_1976 為梅歇爾低頻延伸式）")
    material_class: Literal["fibrous", "foam"] = Field(description="material_class（材料類別：fibrous 是纖維類，foam 是泡棉）")
    porosity: _PositiveNumber | None = Field(default=None, exclude_if=lambda value: value is None, description="porosity（孔隙率，無量綱；模型要求時必填）")
    tortuosity: _PositiveNumber | None = Field(default=None, exclude_if=lambda value: value is None, description="tortuosity（曲折度，無量綱；模型要求時必填）")

    @model_validator(mode="after")
    def parameters_used_by_model(self) -> Self:
        """拒收未用參數，包含顯式填空值；也不替模型默填必要物性。"""
        given = self.model_fields_set & {"porosity", "tortuosity"}
        if self.model in ("komatsu", "miki") and given:
            raise ValueError(f"{self.model} 多孔模型不讀孔隙率或曲折度；請刪除欄位 {sorted(given)}，或明確指定讀取它們的模型")
        if self.model == "wilson_2015" and (self.porosity is None or self.tortuosity is None):
            raise ValueError("wilson_2015 鬆弛模型必須明給正孔隙率與正曲折度")
        if self.model == "mechel_1976":
            if "tortuosity" in given:
                raise ValueError("mechel_1976 低頻延伸模型不讀曲折度；請刪除該欄位")
            if self.porosity is None or self.porosity > 1:
                raise ValueError("mechel_1976 低頻延伸模型必須明給孔隙率，且 0 < 孔隙率 ≤ 1")
        return self


class AirLayer(_ConstructionRecord):
    """AirLayer（空氣層）：以正厚度記錄構造內空腔。"""

    kind: Literal["air"] = Field(default="air", description="kind（層種類代號，air 表示空氣層）")
    thickness_m: _PositiveNumber = Field(description="thickness_m（空腔厚度，m）")


class BoardLayer(_ConstructionRecord):
    """BoardLayer（薄板層）：壓力跳躍阻抗 r+jωm，沿用正時間指數慣例。"""

    kind: Literal["board"] = Field(default="board", description="kind（層種類代號，board 表示薄板層）")
    surface_mass_kg_m2: _PositiveNumber = Field(description="surface_mass_kg_m2（面密度，kg/m²）")
    resistance_pa_s_m: _NonnegativeNumber = Field(description="resistance_pa_s_m（面阻力，Pa·s/m）")


_OpeningNumber = TypeVar("_OpeningNumber", float, "jax.Array")


def layer_open_area(size: _OpeningNumber, spacing: _OpeningNumber, pattern: str, spacing_2: _OpeningNumber | None = None) -> _OpeningNumber:
    """唯一孔距換開孔率公式；純算術亦接受陣列，核心先轉陣列再呼叫。

    交錯 45° 的 spacing 量同一排相鄰兩孔中心距；斜向最近距的型錄請選方格。
    長縫用 slotted；開孔率嚴格 0 < φ < 1 由格式驗證，不沿用原套件允許等於一。
    """
    if pattern == "slotted":
        return size / spacing
    if pattern == "square":
        return math.pi / 4 * (size / spacing) ** 2
    if pattern == "staggered_60":
        return math.pi / (2 * math.sqrt(3)) * (size / spacing) ** 2
    if pattern == "staggered_45":
        return math.pi / 2 * (size / spacing) ** 2
    if pattern == "rectangular" and spacing_2 is not None:
        return math.pi / 4 * size ** 2 / (spacing * spacing_2)
    raise ValueError("請選方格、交錯 60°、交錯 45°、長方格或長縫；長方格還要填排距")


def _geometry_schema(schema: JsonDict) -> None:
    """二擇一與禁填同驗證器；跨欄數值大小和幾何運算無法用標準綱要表達。"""
    _required_kind(schema)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    circular = "hole_pattern" in properties
    geometry = ["hole_spacing_m", "hole_pattern", "hole_spacing_2_m"] if circular else ["slot_spacing_m"]
    nonnull: JsonDict = {name: {"not": {"type": "null"}} for name in ["open_area", "surface_mass_kg_m2", *geometry]}
    direct: JsonDict = {"required": ["open_area"], "not": {"anyOf": [{"required": [name]} for name in geometry]}}
    spacing: JsonDict = {"required": [*geometry[:2]] if circular else [*geometry], "not": {"required": ["open_area"]}}
    if circular:
        spacing["if"] = {"properties": {"hole_pattern": {"const": "rectangular"}}}
        spacing["then"] = {"required": ["hole_spacing_2_m"]}
        spacing["else"] = {"not": {"required": ["hole_spacing_2_m"]}}
    conditions: list[JsonValue] = [{"properties": nonnull}, {"oneOf": [direct, spacing]}]
    schema["allOf"] = conditions
    if "backing_fabric_surface_mass_kg_m2" in properties:
        names = ["backing_fabric_surface_mass_kg_m2", "backing_fabric_flow_resistance_pa_s_m"]
        conditions.append({"oneOf": [
            {"not": {"anyOf": [{"required": [name]} for name in names]}},
            {"required": [*names], "properties": {name: {"not": {"type": "null"}} for name in names}}]})


class FabricLayer(_ConstructionRecord):
    """布層：質量與面流阻並聯；只收兩格，不收厚度。

    核心傳正厚度常數給傳遞矩陣的必填位置；明給兩個物性後厚度不影響結果。
    """

    kind: Literal["fabric"] = Field(default="fabric", description="kind（層種類代號，fabric 表示 布層）")
    surface_mass_kg_m2: _PositiveNumber = Field(description="surface_mass_kg_m2（面密度，kg/m²）")
    flow_resistance_pa_s_m: _PositiveNumber = Field(description="flow_resistance_pa_s_m（面流阻，Pa·s/m；整片布擋氣流的值，不是每公尺的流阻率）")


class _FacingGeometry(_ConstructionRecord):
    """面層幾何：開孔率嚴格 0 < φ < 1，原套件收 ≤ 1，產品拒收全開面。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_geometry_schema)
    thickness_m: _PositiveNumber = Field(description="thickness_m（層厚度，m）")
    open_area: Annotated[StrictFloat, Field(gt=0, lt=1)] | None = Field(default=None, exclude_if=lambda value: value is None, description="open_area（開孔率，比例；嚴格大於零小於一，不是百分比，與孔距或縫距二擇一）")
    surface_mass_kg_m2: _PositiveNumber | None = Field(default=None, exclude_if=lambda value: value is None, description="surface_mass_kg_m2（面密度，kg/m²：指整片板）")

    @model_validator(mode="after")
    def geometry_used(self) -> Self:
        """拒收空值、填一半與兩路都填；孔不接觸，縫寬小於縫距。"""
        circular = isinstance(self, _CircularGeometry)
        names = {"hole_spacing_m", "hole_pattern", "hole_spacing_2_m"} if circular else {"slot_spacing_m"}
        given = self.model_fields_set
        if any(getattr(self, name) is None for name in given & (names | {"open_area", "surface_mass_kg_m2"})):
            raise ValueError("已填的開孔率、孔距、排法、縫距或面密度不可空白；不用的欄位請刪除")
        if "open_area" in given:
            if given & names:
                raise ValueError("開孔率與孔距加排法或縫距只能選一條填；請刪掉另一條")
        elif isinstance(self, _CircularGeometry):
            self._check_holes()
        else:
            assert isinstance(self, SlottedLayer)
            if self.slot_spacing_m is None:
                raise ValueError("請填開孔率，或填縫距，兩種填法選一種")
            if self.slot_width_m >= self.slot_spacing_m:
                raise ValueError("縫寬必須小於縫距，縫不可重疊")
        return self


class _CircularGeometry(_FacingGeometry):
    """圓孔：只設文獻 1／2 mm 分界與孔不重疊；不設未核原文的 0.1 mm 下限或一般孔上限。"""

    hole_diameter_m: _PositiveNumber = Field(description="hole_diameter_m（孔徑，m；只設微孔與一般孔的 1／2 mm 分界及孔不重疊，不用未核原文的 0.1 mm 下限，一般圓孔板沒有另設上限）")
    hole_spacing_m: _PositiveNumber | None = Field(default=None, exclude_if=lambda value: value is None, description="hole_spacing_m（孔距，m；同一排相鄰兩孔中心距，交錯 60° 為正三角形邊長；交錯 45° 若型錄量斜向最近兩孔請選方格）")
    hole_pattern: Literal["square", "staggered_60", "staggered_45", "rectangular"] | None = Field(default=None, exclude_if=lambda value: value is None, description="hole_pattern（孔排法：square 表示 方格，staggered_60 表示 交錯 60°，staggered_45 表示 交錯 45°，rectangular 表示 長方格）")
    hole_spacing_2_m: _PositiveNumber | None = Field(default=None, exclude_if=lambda value: value is None, description="hole_spacing_2_m（排距，m；排與排的距離，只限長方格必填）")

    def _check_holes(self) -> None:
        if self.hole_spacing_m is None or self.hole_pattern is None:
            raise ValueError("請填開孔率，或一起填孔距與孔排法；長方格還要填排距")
        rectangular = self.hole_pattern == "rectangular"
        if rectangular != ("hole_spacing_2_m" in self.model_fields_set):
            raise ValueError("長方格必須多填排距；其他三種排法請刪掉排距")
        nearest = self.hole_spacing_m
        if self.hole_pattern == "staggered_45":
            nearest /= math.sqrt(2)
        if self.hole_spacing_2_m is not None:
            nearest = min(nearest, self.hole_spacing_2_m)
        if self.hole_diameter_m >= nearest:
            raise ValueError("孔不可重疊或接觸，孔徑必須小於最近兩孔中心的孔距")

    @model_validator(mode="after")
    def diameter_range(self) -> Self:
        if isinstance(self, MicroPerforatedLayer) and self.hole_diameter_m > .002:
            raise ValueError("孔徑較大請改用一般圓孔板")
        if isinstance(self, PerforatedLayer) and self.hole_diameter_m < .001:
            raise ValueError("孔徑較小請改用微孔板")
        return self


class _BackingFabric(_ConstructionRecord):
    """背面貼布工程近似：板阻抗加面流阻除板自身開孔率，沒有一手出處；布當不會動。"""

    backing_fabric_surface_mass_kg_m2: _PositiveNumber | None = Field(default=None, exclude_if=lambda value: value is None, description="backing_fabric_surface_mass_kg_m2（背面貼布面密度，kg/m²；面密度先記錄、不進計算，與背面貼布面流阻同給同不給）")
    backing_fabric_flow_resistance_pa_s_m: _PositiveNumber | None = Field(default=None, exclude_if=lambda value: value is None, description="backing_fabric_flow_resistance_pa_s_m（背面貼布面流阻，Pa·s/m；工程近似：板阻抗加布面流阻除板開孔率，布當不會動，沒有一手出處，長縫板同圓孔板）")

    @model_validator(mode="after")
    def paired_backing(self) -> Self:
        names = {"backing_fabric_surface_mass_kg_m2", "backing_fabric_flow_resistance_pa_s_m"}
        given = self.model_fields_set & names
        if given and (given != names or any(getattr(self, name) is None for name in names)):
            raise ValueError("背面貼布的面密度與面流阻必須一起填正數，或兩格都刪除")
        return self


class MicroPerforatedLayer(_CircularGeometry):
    """微孔板：微穿孔薄片，孔徑 ≤ 2 mm，1–2 mm 亦可選一般圓孔板；不收背面貼布。"""

    kind: Literal["micro_perforated"] = Field(default="micro_perforated", description="kind（層種類代號，micro_perforated 表示 微孔板）")
    hole_diameter_m: Annotated[StrictFloat, Field(gt=0, le=.002)] = Field(description=_CircularGeometry.model_fields["hole_diameter_m"].description)

    @field_validator("hole_diameter_m", mode="before")
    @classmethod
    def diameter_hint(cls, value: object) -> object:
        if isinstance(value, int | float) and value > .002:
            raise ValueError("孔徑較大請改用一般圓孔板")
        return value


class PerforatedLayer(_CircularGeometry, _BackingFabric):
    """一般圓孔板：集中孔道薄片加末端修正，孔徑 ≥ 1 mm，不另設上限；背面貼布是工程近似。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_geometry_schema)
    kind: Literal["perforated"] = Field(default="perforated", description="kind（層種類代號，perforated 表示 一般圓孔板）")
    hole_diameter_m: Annotated[StrictFloat, Field(ge=.001)] = Field(description=_CircularGeometry.model_fields["hole_diameter_m"].description)

    @field_validator("hole_diameter_m", mode="before")
    @classmethod
    def diameter_hint(cls, value: object) -> object:
        if isinstance(value, int | float) and value < .001:
            raise ValueError("孔徑較小請改用微孔板")
        return value


class SlottedLayer(_FacingGeometry, _BackingFabric):
    """長縫板：假設縫無限長、週期排列，有限長木條未驗；也收背面貼布，式子同圓孔板的工程近似。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_geometry_schema)
    kind: Literal["slotted"] = Field(default="slotted", description="kind（層種類代號，slotted 表示 長縫板）")
    slot_width_m: _PositiveNumber = Field(description="slot_width_m（縫寬，m；假設縫無限長、週期排列）")
    slot_spacing_m: _PositiveNumber | None = Field(default=None, exclude_if=lambda value: value is None, description="slot_spacing_m（縫距，m；縫寬必須小於縫距，與開孔率二擇一）")


Layer: TypeAlias = Annotated[PorousLayer | AirLayer | BoardLayer | FabricLayer | MicroPerforatedLayer | PerforatedLayer | SlottedLayer, Field(discriminator="kind")]
"""Layer（構造層型別）依 kind（層種類代號）分辨多孔、空氣、板、布、微孔板、一般圓孔板與長縫板。"""


class Assembly(_ConstructionRecord):
    """Assembly（多層構造）：房間側 → 牆體側排列，明給背後負載。"""

    kind: Literal["assembly"] = Field(default="assembly", description="kind（表面種類代號，assembly 表示多層構造）")
    layers: tuple[Layer, ...] = Field(min_length=1, description="layers（至少一層，依房間側到牆體側排序）")
    backing: Literal["rigid", "air"] = Field(description="backing（背後負載：rigid 是剛背，air 是空氣負載）")


class ConstantImpedance(_ConstructionRecord):
    """ConstantImpedance（固定實數阻抗）：整面局部反應，不能當成矩陣層。"""

    kind: Literal["constant_impedance"] = Field(default="constant_impedance", description="kind（表面種類代號，constant_impedance 表示固定阻抗）")
    impedance_pa_s_m: _PositiveNumber = Field(description="impedance_pa_s_m（阻抗，Pa·s/m；整面一個正有限實數）")


class MeasuredImpedancePoint(_ConstructionRecord):
    """MeasuredImpedancePoint（實測列）：SI 阻抗，正時間指數。"""

    frequency_hz: _PositiveNumber = Field(description="frequency_hz（量測頻率，Hz；有限正數、逐列嚴格遞增）")
    real_pa_s_m: _PositiveNumber = Field(description="real_pa_s_m（阻抗實部，Pa·s/m；有限且大於零，不收正規化值）")
    imag_pa_s_m: StrictFloat = Field(description="imag_pa_s_m（阻抗虛部，Pa·s/m；有限，時間慣例 e^{+jωt}；e^{−iωt} 的表須將虛部變號）")


class MeasuredImpedance(_ConstructionRecord):
    """MeasuredImpedance（實測阻抗表）：表外用描述同一面牆的底構造補值。"""

    kind: Literal["measured_impedance"] = Field(default="measured_impedance", description="kind（表面種類代號，measured_impedance 表示實測阻抗表）")
    points: tuple[MeasuredImpedancePoint, ...] = Field(min_length=2, description="points（實測阻抗列；至少兩列，頻率嚴格遞增）")
    base_assembly: Assembly = Field(description="base_assembly（同一面牆的底構造；表外模型補值，只收多層構造）")
    source_and_conditions: str = Field(min_length=1, json_schema_extra={"pattern": r"\S"},  # 全空白由驗證器拒收，綱要同一條
                                       description="source_and_conditions（量測出處與條件，包含阻抗管、入射方式與樣品厚度）")
    fill_rule: Literal["log_linear_table_base_assembly_outside_v1"] = Field(
        default="log_linear_table_base_assembly_outside_v1",
        description="fill_rule（補值規則版本：log_linear_table_base_assembly_outside_v1 表示表內對數頻率線性內插、表外底構造補值第一版）")

    def coverage_text(self) -> str:
        """拒收訊息與表頭共用的表範圍字樣。"""
        return f"實測阻抗表 {len(self.points)} 列，涵蓋 {self.points[0].frequency_hz:g}–{self.points[-1].frequency_hz:g} Hz"

    @field_validator("base_assembly", mode="before")
    @classmethod
    def _base_kind_given(cls, value: object) -> object:
        """輸出的綱要要求底構造明給 kind；原始文件省略就拒收，直接建構的物件照舊帶預設。"""
        if isinstance(value, Mapping) and "kind" not in value:
            raise ValueError("必填底構造的種類代號 kind（assembly）")
        return value

    @field_validator("source_and_conditions")
    @classmethod
    def _nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("量測出處與條件不可空白")
        return value

    @field_validator("points")
    @classmethod
    def _increasing(cls, points: tuple[MeasuredImpedancePoint, ...]) -> tuple[MeasuredImpedancePoint, ...]:
        for index in range(1, len(points)):
            if points[index].frequency_hz <= points[index - 1].frequency_hz:
                raise ValidationError.from_exception_data("實測阻抗表", [{"type": "value_error",
                    "loc": (index, "frequency_hz"), "input": points[index].frequency_hz,
                    "ctx": {"error": ValueError("頻率必須嚴格遞增，不可重複")}}])
        return points


WallMaterial: TypeAlias = Assembly | ConstantImpedance | MeasuredImpedance
"""WallMaterial（原牆材料型別）：貼上去之前那面牆的材料——多層構造、固定阻抗或實測阻抗表（不含貼在原牆上的構造，只有一層巢狀）。"""


class MountedOnWall(_ConstructionRecord):
    """MountedOnWall（貼在原牆上的構造，#990）：新增層房間側 → 牆體側排列，最後一層背後接原牆的表面阻抗，不是剛背。

    原牆是貼上去之前那面牆的材料：固定阻抗、實測阻抗表或多層構造；聯合型別不含本類別，只有一層巢狀。
    沒有背後負載欄：負載就是原牆。多層構造的意思一個位元都不變。
    """

    kind: Literal["mounted_on_wall"] = Field(default="mounted_on_wall", description="kind（表面種類代號，mounted_on_wall 表示貼在原牆上的構造）")
    layers: tuple[Layer, ...] = Field(min_length=1, description="layers（至少一層，依房間側到牆體側排序）")
    wall: Annotated[WallMaterial, Field(discriminator="kind")] = Field(
        description="wall（原牆：貼上去之前那面牆的材料；最後一層背後接它的表面阻抗，收固定阻抗、實測阻抗表或多層構造）")


SurfaceMaterial: TypeAlias = Annotated[Assembly | ConstantImpedance | MeasuredImpedance | MountedOnWall, Field(discriminator="kind")]
"""SurfaceMaterial（牆面材料型別）依 kind（表面種類代號）分辨構造、固定阻抗、實測表與貼在原牆上的構造。"""


def _note_schema(schema: JsonDict) -> None:
    """層與材料類別只屬於 Komatsu 頻率、類別與低流阻註記；補值註記不允許提供這兩格。"""
    schema["allOf"] = [{"if": {"properties": {"code": {"const": "measured_table_model_fill"}}},
        "then": {"not": {"anyOf": [{"required": ["layer_index"]}, {"required": ["material_class"]}]}},
        "else": {"required": ["layer_index", "material_class"], "properties": {
            "layer_index": {"type": "integer"}, "material_class": {"type": "string"}}}}]


class ConstructionNote(_ConstructionRecord):
    """ConstructionNote（結構化材料註記）：層編號從房間側的一開始，不因角度重複。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_note_schema)

    code: Literal["komatsu_frequency_extrapolation", "komatsu_material_class_outside_fit", "komatsu_flow_below_samples", "measured_table_model_fill"] = Field(description="code（註記代號：頻率外推、材料類別不在擬合範圍、流阻低於原研究樣品範圍或實測表模型補值）")
    layer_index: int | None = Field(default=None, exclude_if=lambda value: value is None, ge=1, description="layer_index（從房間側起算的層編號）")
    frequencies_hz: tuple[_PositiveNumber, ...] = Field(
        default=(), json_schema_extra={"uniqueItems": True},  # 不重複由報表輸出驗證器守，綱要同一條；遞增寫不進綱要
        description="frequencies_hz（外推頻點清單，Hz；報表輸出裡遞增不重複；類別與低流阻註記沒有頻點）")
    material_class: Literal["fibrous", "foam"] | None = Field(default=None, exclude_if=lambda value: value is None, description="material_class（材料類別：fibrous 是纖維類，foam 是泡棉）")
    message: str = Field(min_length=1, description="message（中文說明）")

    @model_validator(mode="after")
    def _code_fields(self) -> Self:
        if self.code == "measured_table_model_fill":
            if self.model_fields_set & {"layer_index", "material_class"}:
                raise ValueError("模型補值註記不給層編號或材料類別")
        elif self.layer_index is None or self.material_class is None:
            raise ValueError("Komatsu 頻率、類別與低流阻註記必填層編號與材料類別")
        return self
