"""材料庫（#1082 段 2-1）：一筆材料的格式與檢查；內建材料由產品設定檔生成、唯讀。

兩種材料，``kind`` 只說搜尋可以拿它做什麼：

- ``optimizable``（可最佳化）：一層多孔加可有可無的空腔。多孔層厚度與空腔深度就是搜尋的兩個變數
  （一面一個多孔厚度、一個空腔，跟第三階段梯度搜尋的變數一樣多，存得進來的搜尋就收得下）；流阻是產品身分、不調。
  每一格寫得出能不能調、範圍、產品厚度（取規格用）與初值；另可帶材料自己的總深度上限與建議總深。
  背後接剛背還是貼在原牆上是個案的事，不存在材料裡。
- ``fixed``（固定）：存一份完整牆面材料，搜尋只能整塊選或不選。實測阻抗表可以整張沒有虛部：
  轉成物理用的實測阻抗表時虛部補 0、標「近似」；物理格式與物理身分一個位元都不動。

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
from typing import Annotated, Final, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictStr, field_validator, model_validator

from aosr.config.construction_presets import ConstructionPreset, load_construction_presets
from aosr.config.material_gradient import NONE_TARGET, FamilySettings, MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.materials.construction_models import (
    AirLayer,
    Assembly,
    ConstantImpedance,
    MeasuredImpedance,
    MeasuredImpedancePoint,
    PorousLayer,
    SurfaceMaterial,
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


class FixedEntry(_EntryBase):
    """固定材料：一份完整牆面材料（多層構造、固定實數阻抗或實測阻抗表），搜尋只能整塊選或不選。"""

    kind: Literal["fixed"] = Field(description="kind（fixed：整塊選或不選，不能微調）")
    material: FixedMaterial

    def surface_material(self) -> SurfaceMaterial:
        """物理用的牆面材料。"""
        material = self.material
        return material.to_physics() if isinstance(material, MeasuredTable) else material


LibraryEntry: TypeAlias = Annotated[OptimizableEntry | FixedEntry, Field(discriminator="kind")]


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
    if isinstance(entry, OptimizableEntry) or isinstance(entry.material, Assembly):
        return "simulated"
    if isinstance(entry.material, MeasuredTable) and not entry.material.imaginary_missing:
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


def builtin_entries() -> tuple[LibraryEntry, ...]:
    """內建材料：構造預設每筆一個固定材料、第三階段梯度設定的每個家族一個可最佳化材料；唯讀起點。"""
    presets = load_construction_presets(config_path("construction_presets.toml"))
    settings: MaterialGradientSettings = load_material_gradient_settings(config_path("material_gradient.toml"))
    fixed: list[LibraryEntry] = [
        FixedEntry.model_validate({"kind": "fixed", "id": preset.id, "label": preset.label, "provenance": "literature",
                                   "source": preset.source, "links": preset.links, "notes": preset.notes,
                                   "material": preset.material})
        for preset in presets]
    table = {preset.id: preset for preset in presets}
    return (*fixed, *(_family_entry(family, table) for family in settings.family))


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
