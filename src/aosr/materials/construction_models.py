"""牆面構造的凍結資料格式；所有連續欄位用 SI（國際單位制）。

層順序是房間側 → 牆體側，背後負載必填。實測阻抗表在表內對數頻率內插，表外交必填底構造。
孔隙率與曲折度沿用傳遞矩陣的合法域：皆正；不額外收緊曲折度為大於等於一。
Mechel 1976（低頻延伸多孔模型）的孔隙率另須不大於一。
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, ValidationError, field_validator, model_validator
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


Layer: TypeAlias = Annotated[PorousLayer | AirLayer | BoardLayer, Field(discriminator="kind")]
"""Layer（構造層型別）依 kind（層種類代號）分辨多孔、空氣與板。"""


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


SurfaceMaterial: TypeAlias = Annotated[Assembly | ConstantImpedance | MeasuredImpedance, Field(discriminator="kind")]
"""SurfaceMaterial（牆面材料型別）依 kind（表面種類代號）分辨構造、固定阻抗與實測表。"""


def _note_schema(schema: JsonDict) -> None:
    """層與材料類別只屬於 Komatsu 註記；補值註記不允許提供這兩格。"""
    schema["allOf"] = [{"if": {"properties": {"code": {"const": "measured_table_model_fill"}}},
        "then": {"not": {"anyOf": [{"required": ["layer_index"]}, {"required": ["material_class"]}]}},
        "else": {"required": ["layer_index", "material_class"], "properties": {
            "layer_index": {"type": "integer"}, "material_class": {"type": "string"}}}}]


class ConstructionNote(_ConstructionRecord):
    """ConstructionNote（結構化材料註記）：層編號從房間側的一開始，不因角度重複。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, json_schema_extra=_note_schema)

    code: Literal["komatsu_frequency_extrapolation", "komatsu_material_class_outside_fit", "measured_table_model_fill"] = Field(description="code（註記代號：頻率外推或材料類別不在擬合範圍）")
    layer_index: int | None = Field(default=None, exclude_if=lambda value: value is None, ge=1, description="layer_index（從房間側起算的層編號）")
    frequencies_hz: tuple[_PositiveNumber, ...] = Field(
        default=(), json_schema_extra={"uniqueItems": True},  # 不重複由報表輸出驗證器守，綱要同一條；遞增寫不進綱要
        description="frequencies_hz（外推頻點清單，Hz；報表輸出裡遞增不重複；類別註記沒有頻點）")
    material_class: Literal["fibrous", "foam"] | None = Field(default=None, exclude_if=lambda value: value is None, description="material_class（材料類別：fibrous 是纖維類，foam 是泡棉）")
    message: str = Field(min_length=1, description="message（中文說明）")

    @model_validator(mode="after")
    def _code_fields(self) -> Self:
        if self.code == "measured_table_model_fill":
            if self.model_fields_set & {"layer_index", "material_class"}:
                raise ValueError("模型補值註記不給層編號或材料類別")
        elif self.layer_index is None or self.material_class is None:
            raise ValueError("Komatsu 註記必填層編號與材料類別")
        return self
