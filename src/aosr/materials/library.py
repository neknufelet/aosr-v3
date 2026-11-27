"""材料庫（#1082 段 2-1）：一筆材料的格式與檢查；內建材料由產品設定檔生成、唯讀。

四種材料，``kind`` 只說搜尋可以拿它做什麼：

- ``optimizable``（可最佳化）：一層多孔加可有可無的空腔。多孔層厚度與空腔深度就是搜尋的兩個變數
  （一面一個多孔厚度、一個空腔，跟第三階段梯度搜尋的變數一樣多，存得進來的搜尋就收得下）；流阻是產品身分、不調。
  每一格寫得出能不能調、範圍、產品厚度（取規格用）與初值；另可帶材料自己的總深度上限與建議總深。
  背後接剛背還是貼在原牆上是個案的事，不存在材料裡。
- ``fixed``（固定）：存一份完整牆面材料，搜尋只能整塊選或不選。實測阻抗表可以整張沒有虛部：
  轉成物理用的實測阻抗表時虛部補 0、標「近似」；物理格式與物理身分一個位元都不動。
- ``construction``（構造）：依序存層，明寫背後、成品旗標與放置位置。剛背與空氣背可整塊使用，接原牆的需由分區供應背後。
- ``tunable_layers``（可調多層構造，#1200 第 1 支）：存一份初值排法的多層構造，另列「哪一層的哪一格可調、範圍、產品規格、初值」。
  格式是通用的，現在只收 ``structure = "vpr"``（可調 VPR）：可有可無的固定前層＋鋼板＋整面黏住的背襯（填了骨架兩格的多孔層，
  #1171），背後是牆、沒有空腔；能調的是板重（kg/m²，毫米只是顯示，用材料自己帶的鋼板密度換算）與背襯厚度。
  現在只有格式：存得進、讀得回、轉得成家族；搜尋那一層還沒接（第 2 支），交給第三階段時計畫那一步拒收。

畫面標語照「阻抗怎麼得到」推，不照 ``kind``：層構造是模擬、有虛部的實測表是實測、
沒虛部的實測表與固定實數阻抗是近似。超出算法可信範圍的警告在編輯頁那一段（2-2b），這裡不做。

使用者自己的材料存成一份 JSON（網頁資料夾裡，由編輯頁那一段寫），代號與名稱不准跟內建的撞：
內建的改不到，要改就複製一份新代號。
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Final, Literal, Self, TypeAlias, overload

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator, model_validator

from aosr.config.catalog_surfaces import CatalogSurface, CatalogTable, load_catalog_surfaces
from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.material_gradient import NONE_TARGET, FamilySettings, MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.product_air import product_air_environment
from aosr.materials.catalog_absorption import MAX_RANDOM_INCIDENCE_ABSORPTION, CatalogAbsorption, impedance_on_axis
from aosr.materials.construction_models import (
    AirLayer,
    Assembly,
    BoardLayer,
    ConstantImpedance,
    Layer,
    MeasuredImpedance,
    MeasuredImpedancePoint,
    PorousLayer,
)

SCHEMA_VERSION: Final = "aosr.material_library.v1"
Provenance: TypeAlias = Literal["measured", "literature", "model"]
PROVENANCE_LABELS: Final[dict[str, str]] = {"measured": "實測", "literature": "文獻", "model": "模型"}
Basis: TypeAlias = Literal["simulated", "measured", "approximate"]
BASIS_LABELS: Final[dict[str, str]] = {"simulated": "模擬，之後要拿實測對照", "measured": "實測", "approximate": "近似"}

_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)


class ThicknessSpec(BaseModel):
    """一格厚度（公尺）：能不能調、範圍、產品厚度（取規格用，可空）與初值（空腔可以是「一開始沒有這一層」）。"""

    model_config = _FROZEN

    adjustable: StrictBool = Field(description="adjustable（搜尋能不能調這一格）")
    range_m: tuple[StrictFloat, StrictFloat] = Field(description="range_m（下限、上限，公尺；不能調的兩個相等）")
    choices_m: tuple[StrictFloat, ...] | None = Field(default=None, description="choices_m（產品厚度，公尺，由薄到厚；取規格用）")
    initial_m: StrictFloat | None = Field(description="initial_m（初值，公尺；空腔寫 null 表示一開始沒有空氣層）")

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        low, high = self.range_m
        if not 0.0 <= low <= high:
            raise ValueError(f"厚度範圍要 0 ≤ 下限 ≤ 上限，收到 {self.range_m}")
        if not self.adjustable and low != high:
            raise ValueError(f"不能調的厚度，範圍上下限要相等，收到 {self.range_m}")
        choices = self.choices_m
        if choices is not None:
            if not choices or list(choices) != sorted(set(choices)) or choices[0] <= 0.0:
                raise ValueError(f"產品厚度要是正數、由薄到厚、不重複，收到 {choices}")
            if (choices[0], choices[-1]) != self.range_m:
                raise ValueError(f"有產品厚度時，範圍就是最薄到最厚 {(choices[0], choices[-1])}，收到 {self.range_m}")
        initial = self.initial_m
        if initial is None and low > 0.0:
            raise ValueError(f"初值空白＝一開始沒有這一層（0），範圍要從 0 開始，收到 {self.range_m}")
        if initial is not None:
            if not low <= initial <= high:
                raise ValueError(f"初值 {initial} 不在範圍 {self.range_m} 內")
            if choices is not None and initial not in choices:
                raise ValueError(f"初值 {initial} 不是產品厚度 {choices} 之一")
        return self


class _EntryBase(BaseModel):
    model_config = _FROZEN

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$", description="id（代號，小寫英數與底線）")
    label: str = Field(min_length=1, description="label（顯示名，不准跟別筆重複）")
    provenance: Provenance = Field(description="provenance（數字從哪來：measured 實測、literature 文獻、model 模型）")
    source: str = Field(min_length=1, description="source（出處與條件的文字）")
    links: tuple[StrictStr, ...] = Field(default=(), description="links（出處網址，https:// 開頭）")
    notes: tuple[StrictStr, ...] = Field(default=(), description="notes（顯示用註記）")

    @model_validator(mode="after")
    def _common(self) -> Self:
        if self.id == NONE_TARGET:
            raise ValueError(f"材料代號不能用 {NONE_TARGET}：那是「不貼」的保留代號")
        if any(not link.startswith("https://") for link in self.links):
            raise ValueError("出處網址要 https:// 開頭")
        if any(not note for note in self.notes):
            raise ValueError("註記不可空白")
        return self


class OptimizableEntry(_EntryBase):
    """可最佳化材料：一層多孔（``porous`` 的厚度＝多孔厚度初值）＋可有可無的空腔（在牆那一側）。"""

    kind: Literal["optimizable"] = Field(description="kind（optimizable：搜尋可以調厚度與空腔）")
    porous: PorousLayer = Field(description="porous（多孔層參數；流阻不調，厚度要等於多孔厚度初值）")
    porous_thickness: ThicknessSpec = Field(description="porous_thickness（多孔層厚度；能調就要給產品厚度）")
    cavity: ThicknessSpec | None = Field(default=None, description="cavity（空腔深度；null 表示沒有空腔這一格）")
    max_total_depth_m: StrictFloat | None = Field(default=None, gt=0.0, description="max_total_depth_m（材料自己的總深度上限，公尺；硬上限，可空）")
    suggested_total_depth_m: StrictFloat | None = Field(default=None, gt=0.0, description="suggested_total_depth_m（建議總深，公尺；只當預設值）")
    preset_members: tuple[StrictStr, ...] = Field(default=(), description="preset_members（屬於這組的構造預設代號；舊家族對應用）")
    template_preset: StrictStr | None = Field(default=None, description="template_preset（換樣板時用的構造預設代號；舊家族對應用）")

    def depth_bounds(self) -> tuple[float, float]:
        """層自己排得出的總深（公尺）：最小、最大；不含材料自己的上限。"""
        low, high = self.porous_thickness.range_m
        if self.cavity is not None:
            low, high = low + self.cavity.range_m[0], high + self.cavity.range_m[1]
        return low, high

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        porous, cavity = self.porous_thickness, self.cavity
        if self.porous.bonded():
            raise ValueError("可最佳化材料的多孔層不能填骨架壓縮模數與骨架損耗：那兩格表示整面黏住前面薄板的 VPR 背襯，"
                             "可最佳化材料是多孔層＋空腔、前面沒有薄板；VPR 請存成固定材料的多層構造（薄板層＋填了兩格的多孔層）")
        if porous.initial_m is None or porous.initial_m != self.porous.thickness_m:
            raise ValueError(f"多孔層厚度 {self.porous.thickness_m} 要等於多孔厚度初值 {porous.initial_m}")
        if porous.adjustable and porous.choices_m is None:
            raise ValueError("能調的多孔層厚度要給產品厚度（搜尋最後取規格）")
        if cavity is not None and cavity.choices_m is not None:
            raise ValueError("空腔是連續量，不給產品厚度")
        if not porous.adjustable and (cavity is None or not cavity.adjustable):
            raise ValueError("可最佳化材料至少要有一格能調；都不能調請存成固定材料")
        self._depths()
        if len(set(self.preset_members)) != len(self.preset_members):
            raise ValueError("preset_members 重複")
        if self.template_preset is not None and self.template_preset not in self.preset_members:
            raise ValueError(f"換樣板構造 {self.template_preset} 不在 preset_members 裡")
        return self

    def _depths(self) -> None:
        low, high = self.depth_bounds()
        cap = self.max_total_depth_m
        if cap is not None:
            if low > cap:
                raise ValueError(f"最薄排法的總深 {low} 已超過材料自己的總深度上限 {cap}")
            initial = self.porous.thickness_m + (0.0 if self.cavity is None or self.cavity.initial_m is None else self.cavity.initial_m)
            if initial > cap:
                raise ValueError(f"初值總深 {initial} 超過材料自己的總深度上限 {cap}")
        suggested = self.suggested_total_depth_m
        if suggested is not None and not low <= suggested <= min(high, math.inf if cap is None else cap):
            raise ValueError(f"建議總深 {suggested} 排不出來：層能排的範圍 {low}～{high}、材料上限 {cap}")


class TunableVariable(BaseModel):
    """可調多層構造的一個變數：第幾層（從房間側數、0 起，同可微樣板 ``construction_diff`` 的數法）的哪一格、範圍、
    產品規格（買得到的那幾個值，可空＝連續）與初值。單位就是那一格自己的單位（板重 kg/m²、厚度公尺…）。
    沒列成變數的格子不調。"""

    model_config = _FROZEN

    layer: StrictInt = Field(description="layer（第幾層，從房間側數、0 起）")
    field: str = Field(description="field（那一層的哪一格，欄位名）")
    range: tuple[StrictFloat, StrictFloat] = Field(description="range（下限、上限，那一格自己的單位）")
    choices: tuple[StrictFloat, ...] | None = Field(default=None, description="choices（產品規格，由小到大；可空＝連續）")
    initial: StrictFloat = Field(description="initial（初值；要等於那一層那一格現在寫的值）")

    def words(self) -> str:
        return f"第 {self.layer + 1} 層的 {self.field}"

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        low, high = self.range
        if not 0.0 < low < high:
            raise ValueError(f"{self.words()}：可調範圍要 0 < 下限 < 上限，收到 {self.range}")
        choices = self.choices
        if choices is not None:
            if not choices or list(choices) != sorted(set(choices)) or choices[0] <= 0.0:
                raise ValueError(f"{self.words()}：產品規格要是正數、由小到大、不重複，收到 {choices}")
            if (choices[0], choices[-1]) != self.range:
                raise ValueError(f"{self.words()}：有產品規格時，範圍就是最小到最大 {(choices[0], choices[-1])}，收到 {self.range}")
        if not low <= self.initial <= high:
            raise ValueError(f"{self.words()}：初值 {self.initial} 不在範圍 {self.range} 內")
        if choices is not None and self.initial not in choices:
            raise ValueError(f"{self.words()}：初值 {self.initial} 不是產品規格 {choices} 之一")
        return self


Structure: TypeAlias = Literal["vpr"]
STRUCTURE_LABELS: Final[dict[str, str]] = {"vpr": "可調 VPR"}
_PLATE_MASS: Final = "surface_mass_kg_m2"
_THICKNESS: Final = "thickness_m"


class TunableLayersEntry(_EntryBase):
    """可調多層構造（#1200 第 1 支）：``layers`` 是初值排法（房間側 → 牆、剛背），``variables`` 列出哪一層的哪一格可調。

    寫法是通用的（哪一層、哪一格、範圍、產品規格），之後開縫板、圓孔板、微孔板加棉照同一個格式加 ``structure``；
    現在只收 ``vpr``：最後一層是整面黏住鋼板的背襯（填了骨架兩格的多孔層，#1171）、前一層是鋼板、更前面是不調的固定層，
    能調的只有鋼板的板重與背襯厚度；板重是不是一定要給規格還沒定（選填），背襯能調就要給產品厚度。
    """

    kind: Literal["tunable_layers"] = Field(description="kind（tunable_layers：搜尋可以調 variables 列的那幾格；搜尋那一層還沒接）")
    structure: Structure = Field(description="structure（哪一種構造：vpr 表示複合板共振器，鋼板＋黏住的背襯）")
    layers: tuple[Layer, ...] = Field(min_length=1, description="layers（初值排法，房間側到牆體側；背後是牆）")
    variables: tuple[TunableVariable, ...] = Field(min_length=1, description="variables（可調的格子；至少一格）")
    plate_density_kg_m3: StrictFloat | None = Field(default=None, gt=0.0, description="plate_density_kg_m3（鋼板密度，kg/m³；只用來把板重換成板厚：顯示與總深；vpr 必填）")
    max_total_depth_m: StrictFloat | None = Field(default=None, gt=0.0, description="max_total_depth_m（材料自己的總深度上限，公尺；硬上限，可空）")

    def variable(self, layer: int, field: str) -> TunableVariable | None:
        """那一層那一格的變數；沒列（不調）回 None。"""
        return next((item for item in self.variables if item.layer == layer and item.field == field), None)

    def structure_label(self) -> str:
        return STRUCTURE_LABELS[self.structure]

    def plate_thickness_m(self, mass_kg_m2: float) -> float:
        """板重換板厚（公尺）：除以材料自己帶的鋼板密度。"""
        if self.plate_density_kg_m3 is None:
            raise ValueError("沒有鋼板密度，板重換不成板厚")
        return mass_kg_m2 / self.plate_density_kg_m3

    def _layer_depths(self, index: int) -> tuple[float, float, float]:
        """一層佔的深度（公尺）：最小、最大、初值。有厚度的層照厚度；鋼板（vpr 倒數第二層）照板重換板厚；
        其他沒有厚度的層（布、前面的薄板）算 0，同貼上去總深的算法。"""
        layer = self.layers[index]
        if self.structure == "vpr" and index == len(self.layers) - 2 and isinstance(layer, BoardLayer):
            tuned = self.variable(index, _PLATE_MASS)
            initial = layer.surface_mass_kg_m2
            low, high = (initial, initial) if tuned is None else tuned.range
            return self.plate_thickness_m(low), self.plate_thickness_m(high), self.plate_thickness_m(initial)
        initial = float(getattr(layer, _THICKNESS, 0.0))
        tuned = self.variable(index, _THICKNESS)
        low, high = (initial, initial) if tuned is None else tuned.range
        return low, high, initial

    def depth_bounds(self) -> tuple[float, float]:
        """排得出的總深（公尺）：最小、最大；不含材料自己的上限。"""
        depths = [self._layer_depths(index) for index in range(len(self.layers))]
        return math.fsum(item[0] for item in depths), math.fsum(item[1] for item in depths)

    def initial_depth_m(self) -> float:
        """初值排法的總深（公尺）。"""
        return math.fsum(self._layer_depths(index)[2] for index in range(len(self.layers)))

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        seen: set[tuple[int, str]] = set()
        for item in self.variables:
            if not 0 <= item.layer < len(self.layers):
                raise ValueError(f"變數指的{item.words()}不存在：這份構造只有 {len(self.layers)} 層")
            if (item.layer, item.field) in seen:
                raise ValueError(f"{item.words()}列了兩次")
            seen.add((item.layer, item.field))
        self._vpr()
        try:  # 先核構造再讀變數指的那一格：過了這一關，倒數第二層一定是薄板、最後一層是多孔層，那兩格都在
            initial_assembly(self)
        except ValueError as error:
            raise ValueError(f"初值排法不是合法的多層構造：{error}") from error
        for item in self.variables:
            written = getattr(self.layers[item.layer], item.field)
            if written != item.initial:
                raise ValueError(f"{item.words()}寫的是 {written}，要等於這個變數的初值 {item.initial}")
        cap = self.max_total_depth_m
        if cap is not None:
            low = self.depth_bounds()[0]
            if low > cap:
                raise ValueError(f"最薄排法的總深 {low} 已超過材料自己的總深度上限 {cap}")
            if self.initial_depth_m() > cap:
                raise ValueError(f"初值總深 {self.initial_depth_m()} 超過材料自己的總深度上限 {cap}")
        return self

    def _vpr(self) -> None:
        backing_index, plate_index = len(self.layers) - 1, len(self.layers) - 2
        backing = self.layers[backing_index]
        if not isinstance(backing, PorousLayer) or not backing.bonded():
            raise ValueError("可調 VPR 的最後一層（最靠牆）要是填了骨架壓縮模數與骨架損耗的多孔層：那兩格表示整面黏住前面的鋼板；"
                             "沒黏住的多孔層＋空腔請存成可最佳化材料")
        allowed = {(plate_index, _PLATE_MASS), (backing_index, _THICKNESS)}
        for item in self.variables:
            if (item.layer, item.field) not in allowed:
                raise ValueError(f"可調 VPR 能調的只有鋼板（倒數第二層）的板重 {_PLATE_MASS} 與背襯（最後一層）的厚度 {_THICKNESS}，"
                                 f"收到{item.words()}")
        if self.plate_density_kg_m3 is None:
            raise ValueError("可調 VPR 要填鋼板密度 plate_density_kg_m3：板重靠它換成板厚（顯示與總深）")
        thickness = self.variable(backing_index, _THICKNESS)
        if thickness is not None and thickness.choices is None:
            raise ValueError("能調的背襯厚度要給產品規格（搜尋最後取規格）")


class MeasuredPoint(BaseModel):
    """實測列：頻率（Hz）、阻抗實部與虛部（Pa·s/m，時間慣例 e^{+jωt}）；虛部可以整張表都沒有。"""

    model_config = _FROZEN

    frequency_hz: StrictFloat = Field(gt=0.0)
    real_pa_s_m: StrictFloat = Field(gt=0.0)
    imag_pa_s_m: StrictFloat | None = None


class MeasuredTable(BaseModel):
    """實測阻抗表（固定材料用）：跟物理用的實測阻抗表同一份契約，只多一條「整張可以沒有虛部」。

    字串不去頭尾空白（物理那邊原樣存）；原始文件的底構造要明寫 kind（物理那邊對原始文件的同一條）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    kind: Literal["measured_impedance"]
    points: tuple[MeasuredPoint, ...] = Field(min_length=2)
    base_assembly: Assembly = Field(description="base_assembly（表外補值用的同一面牆底構造；物理契約必填）")
    source_and_conditions: str = Field(min_length=1)
    fill_rule: Literal["log_linear_table_base_assembly_outside_v1"] = Field(
        default="log_linear_table_base_assembly_outside_v1", description="fill_rule（補值規則版本；原樣交給物理）")

    @field_validator("base_assembly", mode="before")
    @classmethod
    def _base_kind_given(cls, value: object) -> object:
        if isinstance(value, Mapping) and "kind" not in value:
            raise ValueError("必填底構造的種類代號 kind（assembly）")
        return value

    @property
    def imaginary_missing(self) -> bool:
        """整張表沒有虛部（轉物理時補 0、標「近似」）。"""
        return self.points[0].imag_pa_s_m is None

    def to_physics(self) -> MeasuredImpedance:
        """物理用的實測阻抗表；沒有虛部的補 0.0，其餘欄原樣。"""
        rows = tuple(MeasuredImpedancePoint(frequency_hz=row.frequency_hz, real_pa_s_m=row.real_pa_s_m,
                                            imag_pa_s_m=0.0 if row.imag_pa_s_m is None else row.imag_pa_s_m)
                     for row in self.points)
        return MeasuredImpedance(points=rows, base_assembly=self.base_assembly,
                                 source_and_conditions=self.source_and_conditions, fill_rule=self.fill_rule)

    @model_validator(mode="after")
    def _all_or_none(self) -> Self:
        given = {row.imag_pa_s_m is not None for row in self.points}
        if len(given) > 1:
            raise ValueError("虛部要整張表都有、或整張都沒有；只缺幾列的表不收")
        self.to_physics()
        return self


FixedMaterial: TypeAlias = Annotated[Assembly | ConstantImpedance | MeasuredTable, Field(discriminator="kind")]


StandaloneSurface: TypeAlias = Assembly | ConstantImpedance | MeasuredImpedance


class FixedEntry(_EntryBase):
    """固定材料：一份完整牆面材料（多層構造、固定實數阻抗或實測阻抗表），搜尋只能整塊選或不選。"""

    kind: Literal["fixed"] = Field(description="kind（fixed：整塊選或不選，不能微調）")
    material: FixedMaterial

    @property
    def imaginary_missing(self) -> bool:
        return isinstance(self.material, MeasuredTable) and self.material.imaginary_missing

    def surface_material(self) -> StandaloneSurface:
        """物理用的牆面材料。"""
        material = self.material
        return material.to_physics() if isinstance(material, MeasuredTable) else material


class ConstructionEntry(_EntryBase):
    """構造的各層從房間側到牆側；成品也保留整份構造。"""

    kind: Literal["construction"] = Field(description="kind（構造：各層與背後負載）")
    layers: tuple[Layer, ...] = Field(min_length=1, description="layers（從房間側到牆側的各層）")
    backing: Literal["rigid", "air", "zone_wall"] = Field(description="backing（背後負載）")
    product: StrictBool = Field(description="product（是否為成品）")
    placement: Literal["wall", "floor", "ceiling_build", "door_window"] = Field(description="placement（能貼在哪）")

    def preview_assembly(self) -> Assembly:
        """預覽缺原牆時採剛背；其他構造的預覽就是實際的背後。"""
        return Assembly(layers=self.layers, backing="rigid" if self.backing == "zone_wall" else self.backing)

    @model_validator(mode="after")
    def _validate_stack(self) -> Self:
        self.preview_assembly()  # 共用物理構造對黏住背襯的位置與負載檢查。
        return self


LibraryEntry: TypeAlias = Annotated[OptimizableEntry | FixedEntry | TunableLayersEntry | ConstructionEntry, Field(discriminator="kind")]


@overload
def fixed_surface(entry: FixedEntry) -> StandaloneSurface: ...


@overload
def fixed_surface(entry: LibraryEntry) -> StandaloneSurface | None: ...


def fixed_surface(entry: LibraryEntry) -> StandaloneSurface | None:
    """可獨立使用的固定牆面材料；需要其他輸入才決定的條目回 None。"""
    if isinstance(entry, ConstructionEntry) and entry.backing != "zone_wall":
        return entry.preview_assembly()
    return entry.surface_material() if isinstance(entry, FixedEntry) else None


class _LibraryFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["aosr.material_library.v1"]
    materials: tuple[LibraryEntry, ...]


def _check_distinct(entries: tuple[LibraryEntry, ...]) -> None:
    ids = [entry.id for entry in entries]
    labels = [entry.label for entry in entries]
    if len(set(ids)) != len(ids) or len(set(labels)) != len(labels):
        raise ValueError("材料代號與顯示名都不可重複")


def impedance_basis(entry: LibraryEntry) -> Basis:
    """阻抗怎麼得到：層構造＝模擬；有虛部的實測表＝實測；沒虛部的實測表與固定實數阻抗＝近似。"""
    material = fixed_surface(entry)
    if material is None or isinstance(material, Assembly):
        return "simulated"
    if isinstance(material, MeasuredImpedance) and isinstance(entry, FixedEntry) and not entry.imaginary_missing:
        return "measured"
    return "approximate"


def basis_label(entry: LibraryEntry) -> str:
    """畫面標語。"""
    return BASIS_LABELS[impedance_basis(entry)]


def _family_entry(family: FamilySettings, presets: dict[str, ConstructionPreset]) -> OptimizableEntry:
    template = presets[family.template_preset]
    layers = Assembly.model_validate(template.material).layers
    porous = layers[0]
    if not isinstance(porous, PorousLayer):
        raise ValueError(f"家族 {family.id} 的換樣板構造第一層不是多孔層")
    words = template.label.split(" ")
    return OptimizableEntry(
        kind="optimizable", id=family.id, label=f"{' '.join(words[:2])}（可調厚度與空腔）", provenance="literature",
        source=template.source, links=template.links, notes=template.notes, porous=porous,
        porous_thickness=ThicknessSpec(adjustable=True, range_m=(family.thicknesses_m[0], family.thicknesses_m[-1]),
                                       choices_m=family.thicknesses_m, initial_m=family.initial_thickness_m),
        cavity=ThicknessSpec(adjustable=True, range_m=family.cavity_range_m, initial_m=family.initial_cavity_m),
        max_total_depth_m=family.max_total_depth_m, preset_members=family.presets, template_preset=family.template_preset)


def _catalog_axis(table: CatalogTable) -> tuple[float, ...]:
    """阻抗表取點：以參考頻率對齊、每八度固定點數，含最低與最高頻之間的每一點。"""
    low = math.ceil(table.points_per_octave * math.log2(table.lowest_hz / table.reference_hz))
    high = math.floor(table.points_per_octave * math.log2(table.highest_hz / table.reference_hz))
    return tuple(table.reference_hz * 2.0 ** (step / table.points_per_octave) for step in range(low, high + 1))


def _catalog_entry(surface: CatalogSurface, axis: tuple[float, ...], rho_c: float) -> FixedEntry:
    """文獻頻帶吸音率換成沒有虛部的阻抗表（決策紙 catalog-absorption-random-incidence-paris-inversion）。

    表的點夠密、涵蓋整條產品軸，表內對數內插跟逐點反推只差內插誤差；表外（用不到）當硬面：
    底構造是 1 mm 空氣層加剛背，低頻阻抗遠大於 ρc。
    """
    # 超過 Paris 頂點的手冊值會被夾掉、不再是那一格；硬面不該走到這裡，寧可整筆拒收也不默默換掉。
    # 直接看手冊值本身：帶心不一定落在取點上，只看取到的點會漏。
    if any(alpha > MAX_RANDOM_INCIDENCE_ABSORPTION for alpha in surface.absorption):
        raise ValueError(f"catalog surface {surface.id!r} has absorption above the Paris peak; it would be clamped")
    # 頻帶外是平的（延伸），只留最外那一點；頻帶內照常密取（內建頻帶都是標準八度帶，落在取點上）。
    low, high = surface.band_center_hz[0], surface.band_center_hz[-1]
    kept = (axis[0], *(f for f in axis if low <= f <= high), axis[-1])
    converted = impedance_on_axis(CatalogAbsorption(surface.id, surface.band_center_hz, surface.absorption), kept, rho_c)
    if any(converted.clamped):
        # 保險：取點上的內插值也不准被夾。
        raise ValueError(f"catalog surface {surface.id!r} has absorption above the Paris peak; it would be clamped")
    # 存檔與材料頁清單要讀得動：頻率留到 0.001 Hz、阻抗留到 0.1 Pa·s/m，遠小於表內內插的差。
    points = [{"frequency_hz": round(frequency, 3), "real_pa_s_m": round(value, 1)}
              for frequency, value in zip(converted.frequencies_hz, converted.impedance_pa_s_per_m, strict=True)]
    bands = "、".join(f"{band:g} Hz {alpha:g}" for band, alpha in zip(surface.band_center_hz, surface.absorption, strict=True))
    conditions = (f"手冊無規入射頻帶吸音率（{bands}），以 Paris 公式反推的實數阻抗；不是阻抗管量測，沒有虛部。"
                  f"{low:g} Hz 以下、{high:g} Hz 以上平延伸最近一帶的值（估計，不是量的）；"
                  f"頻帶之間在表點上反推、表點之間阻抗內插，跟逐點反推最多差約 2%。表外當硬面。出處：{surface.source}")
    return FixedEntry.model_validate({
        "kind": "fixed", "id": surface.id, "label": surface.label, "provenance": "literature",
        "source": surface.source, "links": surface.links, "notes": surface.notes,
        "material": {"kind": "measured_impedance", "points": points, "source_and_conditions": conditions,
                     "base_assembly": {"kind": "assembly", "backing": "rigid",
                                       "layers": [{"kind": "air", "thickness_m": _CATALOG_BASE_AIR_M}]}}})


_CATALOG_BASE_AIR_M: Final = 0.001


def catalog_entries() -> tuple[FixedEntry, ...]:
    """文獻硬面（門、窗、木地板、石膏板、磚、混凝土），每筆一張阻抗表；ρc 用產品標準空氣。"""
    table, surfaces = load_catalog_surfaces(config_path("catalog_surfaces.toml"))
    air, _ = product_air_environment()
    axis = _catalog_axis(table)
    return tuple(_catalog_entry(surface, axis, air.rho_c) for surface in surfaces)


def builtin_entries() -> tuple[LibraryEntry, ...]:
    """內建材料：構造預設每筆一個固定材料、文獻硬面每筆一個固定材料、第三階段梯度設定的每個家族一個可最佳化材料；唯讀起點。"""
    presets = load_construction_presets(config_path("construction_presets.toml"))
    settings: MaterialGradientSettings = load_material_gradient_settings(config_path("material_gradient.toml"))
    fixed: list[LibraryEntry] = [
        FixedEntry.model_validate({"kind": "fixed", "id": preset.id, "label": preset.label, "provenance": "literature",
                                   "source": preset.source, "links": preset.links, "notes": preset.notes,
                                   "material": preset.material})
        for preset in presets]
    table = {preset.id: preset for preset in presets}
    return (*fixed, *catalog_entries(), *(_family_entry(family, table) for family in settings.family))


def family_settings(entry: OptimizableEntry) -> FamilySettings:
    """可最佳化材料 → 第三階段搜尋的家族設定。只收有舊家族對應（構造預設與換樣板構造）的材料，其餘留給搜尋接材料庫那一段。"""
    porous, cavity = entry.porous_thickness, entry.cavity
    if entry.template_preset is None or porous.choices_m is None or entry.max_total_depth_m is None:
        raise ValueError(f"材料 {entry.id} 沒有對應的構造預設、產品厚度或總深度上限，第三階段搜尋現在還收不到")
    return FamilySettings(id=entry.id, presets=entry.preset_members, thicknesses_m=porous.choices_m,
                          cavity_range_m=(0.0, 0.0) if cavity is None else cavity.range_m,
                          max_total_depth_m=entry.max_total_depth_m, template_preset=entry.template_preset,
                          initial_thickness_m=entry.porous.thickness_m,
                          initial_cavity_m=None if cavity is None else cavity.initial_m)


def template_assembly(entry: OptimizableEntry) -> Assembly:
    """初值排出來的剛背多層構造（空腔初值是 null 或 0 就沒有空氣層）；給換樣板構造對照用。"""
    cavity = None if entry.cavity is None else entry.cavity.initial_m
    layers = (entry.porous,) if not cavity else (entry.porous, AirLayer(thickness_m=cavity))
    return Assembly(layers=layers, backing="rigid")


def initial_assembly(entry: TunableLayersEntry) -> Assembly:
    """可調多層構造初值排出來的剛背多層構造（位置規則由多層構造自己核）。"""
    return Assembly(layers=entry.layers, backing="rigid")


def preview_material(entry: LibraryEntry) -> StandaloneSurface:
    """拿一筆材料去算時用的牆面材料：可最佳化與可調多層取初值排法，背後要原牆的構造當剛背。"""
    if isinstance(entry, TunableLayersEntry):
        return initial_assembly(entry)
    if isinstance(entry, OptimizableEntry):
        return template_assembly(entry)
    if isinstance(entry, ConstructionEntry):
        return entry.preview_assembly()
    return entry.surface_material()


def load_user_library(path: Path) -> tuple[LibraryEntry, ...]:
    """使用者材料庫；檔案不在就是空的。代號或顯示名跟內建撞到就拒收（內建唯讀）。"""
    if not path.exists():
        return ()
    entries = _LibraryFile.model_validate(json.loads(path.read_text(encoding="utf-8"))).materials
    _check_distinct((*builtin_entries(), *entries))
    return entries


def dump_user_library(entries: tuple[LibraryEntry, ...]) -> str:
    """使用者材料庫寫成 JSON 文字（鍵排序、兩格縮排）；寫之前照讀檔同一套檢查。"""
    document = _LibraryFile(schema_version=SCHEMA_VERSION, materials=entries)
    _check_distinct((*builtin_entries(), *entries))
    return json.dumps(document.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def full_library(user_path: Path | None) -> tuple[LibraryEntry, ...]:
    """內建在前、使用者在後；沒給路徑就只有內建。"""
    return (*builtin_entries(), *(() if user_path is None else load_user_library(user_path)))
