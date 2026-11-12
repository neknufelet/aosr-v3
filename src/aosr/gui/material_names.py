"""材料給人看的名字（#1119，#1082 段 1-8）：結果頁逐區表先寫名字、工程參數收進細節。

名字只從現有的登記來源取，不另抄一份：

- 材料庫的固定材料（內建構造預設＋使用者材料庫，``library.full_library``）逐位相等就用它的顯示名；
  貼在原牆上的構造比貼上去的那幾層（同立面圖 ``zone_modes.js`` 的 ``libraryLabel``）。
  收位後才像的不冠登記名：搜尋選出來的厚度常帶浮點尾數，冠了預設名就是說謊。
- 對不到的由層的參數組名，照實標「依參數組名」：多孔層的名字取材料庫可最佳化材料那一筆
  （流阻、模型、類別、出處都一樣、只差厚度），其他層用模型描述裡的種類中文；厚度一律收位成公釐。
- 固定阻抗寫成「固定阻抗 123,480 Pa·s/m」；「原牆」兩個字只由呼叫的人在真的知道是原牆的地方加。

這一支只排字，不建材料、不算物理、不改存檔（住在 gui 底下，不在計算指紋與物理身分範圍裡）。
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from aosr.gui.problem_text import kind_word, rounded_number
from aosr.materials.construction_models import (
    AirLayer, Assembly, BoardLayer, ConstantImpedance, Layer, MeasuredImpedance, MountedOnWall, PorousLayer,
    SurfaceMaterial)
from aosr.materials.library import FixedEntry, LibraryEntry, OptimizableEntry

COMPOSED_NOTE = "（依參數組名）"
MOUNTED_NOTE = "貼在原牆上"
# 內建家族（有換樣板構造的那幾筆）的顯示名尾巴是一段括號說明「（可調厚度與空腔）」，拿來當多孔層的名字時去掉；
# 使用者材料的括號可能正是分辨兩筆的字（廠牌、批號），照原名不動。
_TRAILING_NOTE = re.compile(r"\s*（[^（）]*）\Z")


@dataclass(frozen=True)
class MaterialName:
    """名字與它怎麼來：``composed`` 是真的表示對不到材料庫、依層的參數組的。"""

    text: str
    composed: bool = False

    def shown(self) -> str:
        return self.text + COMPOSED_NOTE if self.composed else self.text


def _mm(value_m: float) -> str:
    return f"{rounded_number(value_m * 1000)} mm"


def impedance_words(impedance_pa_s_m: float) -> str:
    """固定阻抗的一句（收位）：固定阻抗 123,480 Pa·s/m。"""
    return f"{kind_word('constant_impedance')} {rounded_number(impedance_pa_s_m)} Pa·s/m"


def _porous_name(layer: PorousLayer, library: Sequence[LibraryEntry]) -> str | None:
    """多孔層除了厚度、其餘參數跟材料庫哪一筆可最佳化材料一樣，就借那一筆的名字。"""
    for entry in library:
        if isinstance(entry, OptimizableEntry) and entry.porous.model_copy(
                update={"thickness_m": layer.thickness_m}) == layer:
            if entry.template_preset is None:
                return entry.label
            return _TRAILING_NOTE.sub("", entry.label) or entry.label
    return None


def _layer_words(layer: Layer, library: Sequence[LibraryEntry]) -> str:
    if isinstance(layer, PorousLayer):
        name = _porous_name(layer, library)
        if name is None:
            return (f"{kind_word(layer.kind)} {_mm(layer.thickness_m)}"
                    f"（流阻 {rounded_number(layer.flow_resistivity_pa_s_m2)} Pa·s/m²）")
        return f"{name} {_mm(layer.thickness_m)}"
    if isinstance(layer, AirLayer):
        return f"{_mm(layer.thickness_m)} 空腔"
    if isinstance(layer, BoardLayer):
        return f"{kind_word(layer.kind)}（{rounded_number(layer.surface_mass_kg_m2)} kg/m²）"
    return kind_word(layer.kind)


def registered_entry(material: SurfaceMaterial, library: Sequence[LibraryEntry]) -> FixedEntry | None:
    """材料庫固定材料逐位相等的那一筆；貼在原牆上的比貼上去的那幾層。"""
    for entry in library:
        if not isinstance(entry, FixedEntry):
            continue
        chosen = entry.surface_material()
        if isinstance(material, MountedOnWall):
            if isinstance(chosen, Assembly) and chosen.layers == material.layers:
                return entry
        elif chosen == material:
            return entry
    return None


def _registered(material: SurfaceMaterial, library: Sequence[LibraryEntry]) -> str | None:
    """材料庫固定材料逐位相等的那一筆的顯示名。"""
    entry = registered_entry(material, library)
    return None if entry is None else entry.label


def layers_name(layers: Sequence[Layer], library: Sequence[LibraryEntry]) -> MaterialName:
    """一疊層的名字：從房間側到牆那一側，用「＋」接（同構造預設的寫法）。"""
    return MaterialName("＋".join(_layer_words(layer, library) for layer in layers), composed=True)


def material_name(material: SurfaceMaterial, library: Sequence[LibraryEntry]) -> MaterialName:
    """一份牆面材料的名字。貼在原牆上的只講貼上去的那幾層；原牆另外問 ``material.wall``。"""
    label = _registered(material, library)
    if label is not None:
        return MaterialName(label)
    if isinstance(material, ConstantImpedance):
        return MaterialName(impedance_words(material.impedance_pa_s_m))
    if isinstance(material, MeasuredImpedance):
        points = material.points
        return MaterialName(f"{kind_word(material.kind)}（{points[0].frequency_hz:g}–{points[-1].frequency_hz:g} Hz，"
                            f"{len(points)} 列）")
    return layers_name(material.layers, library)


def zone_material_words(material: SurfaceMaterial, library: Sequence[LibraryEntry]) -> str:
    """逐區表一格的名字：貼在原牆上的寫「名字，貼在原牆上（原牆：…）」，其餘只寫名字。"""
    name = material_name(material, library).shown()
    if isinstance(material, MountedOnWall):
        return f"{name}，{MOUNTED_NOTE}（原牆：{material_name(material.wall, library).shown()}）"
    return name


def original_wall_words(material: SurfaceMaterial, library: Sequence[LibraryEntry]) -> str:
    """確定是原牆的地方（原牆基準欄、保留原牆的區）：原牆（固定阻抗 123,480 Pa·s/m）。"""
    return f"原牆（{zone_material_words(material, library)}）"
