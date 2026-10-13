"""牆面構造的凍結資料格式；所有連續欄位用 SI（國際單位制）。

層順序是房間側 → 牆體側，背後負載必填。實測阻抗表的插值尚未拍板，不在此型別內。
孔隙率與曲折度沿用傳遞矩陣的合法域：皆正；不額外收緊曲折度為大於等於一。
Mechel 1976（低頻延伸多孔模型）的孔隙率另須不大於一。
"""
from __future__ import annotations

from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator
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
    model: Literal["komatsu", "miki", "wilson_2015", "mechel_1976"] = Field(default="komatsu", description="model（多孔模型：komatsu、miki 為纖維經驗式，wilson_2015 為鬆弛模型，mechel_1976 為低頻延伸式）")
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

    kind: Literal["constant_impedance"] = Field(default="constant_impedance", description="kind（表面種類代號，constant_impedance 表示固定實數阻抗）")
    impedance_pa_s_m: _PositiveNumber = Field(description="impedance_pa_s_m（整面正有限實數阻抗，Pa·s/m）")


SurfaceMaterial: TypeAlias = Annotated[Assembly | ConstantImpedance, Field(discriminator="kind")]
"""SurfaceMaterial（牆面材料型別）依 kind（表面種類代號）分辨構造與固定阻抗。"""


class ConstructionNote(_ConstructionRecord):
    """ConstructionNote（結構化材料註記）：層編號從房間側的一開始，不因角度重複。"""

    code: Literal["komatsu_frequency_extrapolation", "komatsu_material_class_outside_fit"] = Field(description="code（註記代號：頻率外推或材料類別不在擬合範圍）")
    layer_index: int = Field(ge=1, description="layer_index（從房間側起算的層編號）")
    frequencies_hz: tuple[_PositiveNumber, ...] = Field(default=(), description="frequencies_hz（外推頻點清單，Hz；類別註記沒有頻點）")
    material_class: Literal["fibrous", "foam"] = Field(description="material_class（材料類別：fibrous 是纖維類，foam 是泡棉）")
    message: str = Field(min_length=1, description="message（中文說明）")
