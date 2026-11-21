"""空腔範圍的下限介於 0 和匯出最小空腔之間（#1234）：收下的範圍，匯出的構造一定守得住它。

匯出規則（辛二，``endpoint.kept_cavity``，這一支不改）：空腔小於設定檔 ``[export] min_cavity_m`` 就整層拿掉。範圍的下限要是 0
（可以沒有空腔）或至少那個數，不然搜尋落在下限附近時，交出去的構造沒有空氣層，違反使用者寫的範圍。兩個來源：分區自己的範圍
（``ZoneRange``，起跑設計那一步收窄）與材料頁那一份材料自己的空腔範圍（使用者材料庫接成搜尋家族那一步）。

數字全是考卷自己編的：分區那一組用內建 48K（設定檔：產品厚度 0.05／0.1／0.2、空腔 0～0.25、總深 0.3；樣板 50 mm 棉＋50 mm 空腔），
x0 那一面收窄成棉 0.05～0.1、只換空腔範圍；材料頁那一組是這裡編的一筆可最佳化材料（棉厚產品 0.05／0.1、總深 0.2，只換空腔範圍）。
最小空腔從設定檔讀；另用一份自訂設定（最小空腔加倍）核判法跟著這一場的設定走，不是寫死一個數。
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import TypeAdapter

from aosr.config.material_gradient import ExportSettings, MaterialGradientSettings, load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.gradient import endpoint
from aosr.gradient.depth_fit import ZoneRange
from aosr.gradient.start_config import designs_for, presets_table
from aosr.gradient.surface_design import SurfaceDesign
from aosr.materials import library
from aosr.materials.construction_models import AirLayer, Assembly, ConstantImpedance, SurfaceMaterial
from aosr.materials.search_library import read_user_entries, with_user_materials

SETTINGS = load_material_gradient_settings(config_path("material_gradient.toml"))
WOOL = "glass_wool_48k"
PRESET = "glass_wool_48k_50mm_gap50"
PAGE = "demo_thin_gap_wool"
FACE = "x0"
# 拒收句裡一定有的那幾個字（講匯出時整層拿掉）；別的原因拒收不算數。
REFUSED_WORDS = "整層拿掉"
_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
Designs = Callable[[MaterialGradientSettings, tuple[float, float], Path], tuple[SurfaceDesign, ...]]


def _doubled() -> MaterialGradientSettings:
    """自訂設定：最小空腔是設定檔那個數的兩倍，其餘照設定檔。"""
    return SETTINGS.model_copy(update={"export": ExportSettings(min_cavity_m=SETTINGS.export.min_cavity_m * 2.0)})


def _cases(minimum: float) -> tuple[tuple[str, float, float], ...]:
    """（名字, 下限, 上限），全照這一場的最小空腔換算：下限 0 與恰好最小空腔照收，介於中間的拒收（固定深度是下限＝上限）。"""
    return (("兩端都比最小空腔薄", minimum / 2.0, minimum * 0.8),
            ("下限 0、上限比最小空腔薄", 0.0, minimum * 0.8),
            ("下限恰是最小空腔", minimum, minimum * 2.0),
            ("下限比最小空腔薄、上限寬", minimum / 2.0, 0.05),
            ("固定深度比最小空腔薄", minimum / 2.0, minimum / 2.0))


def _kept() -> dict[Wall, SurfaceMaterial]:
    return {wall: ConstantImpedance(impedance_pa_s_m=12348.0) for wall in Wall.all()}


def _zone_designs(settings: MaterialGradientSettings, cavity: tuple[float, float], _folder: Path) -> tuple[SurfaceDesign, ...]:
    """分區自己的範圍：六面都起跑 48K，x0 那一面寫範圍（命令列 ``--range`` 與網頁開跑都走這一支起跑設計）。"""
    ranges = {FACE: {WOOL: ZoneRange(porous_m=(0.05, 0.1), cavity_m=cavity)}}
    return designs_for((PRESET,) * 6, settings, kept=_kept(), ranges=ranges)


def _page_entry(cavity: tuple[float, float]) -> library.LibraryEntry:
    """材料頁那一份：自編一筆可最佳化材料，空腔範圍照給的；兩端相等就是固定深度（不能調），否則能調、初值取上限。"""
    low, high = cavity
    return _ENTRY.validate_python({
        "kind": "optimizable", "id": PAGE, "label": "示範薄空腔棉（自編）", "provenance": "model", "source": "考卷自編",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 17500.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": low < high, "range_m": [low, high], "initial_m": high}, "max_total_depth_m": 0.2})


def _page_designs(settings: MaterialGradientSettings, cavity: tuple[float, float], folder: Path) -> tuple[SurfaceDesign, ...]:
    """材料頁那一份材料自己的範圍：材料庫寫成檔、開跑那一步讀回來接成家族（``read_user_entries`` → ``with_user_materials``），
    六面都起跑這一種、不寫分區範圍。"""
    path = folder / f"{cavity[0]!r}-{cavity[1]!r}-{settings.export.min_cavity_m!r}.json"
    path.write_text(library.dump_user_library((_page_entry(cavity),)), encoding="utf-8")
    materials = with_user_materials(settings, presets_table(), read_user_entries(path))
    return designs_for((PAGE,) * 6, materials.settings, kept=_kept(), table=materials.presets)


def _theta(designs: tuple[SurfaceDesign, ...], cavity: float) -> np.ndarray:
    """每面棉取最薄的合法厚度；x0 的空腔放 ``cavity``，其他面放自己的空腔下界。"""
    values: list[float] = []
    for design in designs:
        if design.material is None:
            continue
        values.append(design.legal_thicknesses_m[0])
        if design.cavity_variable is not None:
            values.append(cavity if design.surface_id == FACE else design.lower[1])
    return np.asarray(values, dtype=np.float64)


def _exported_cavity(designs: tuple[SurfaceDesign, ...], cavity: float, settings: MaterialGradientSettings) -> float:
    """真的走終點匯出（``materials_from_surfaces``），回 x0 那一面匯出構造裡的空腔（沒有空氣層是 0）。"""
    material = endpoint.materials_from_surfaces(designs, _theta(designs, cavity), settings)[Wall.from_name(FACE)]
    assert isinstance(material, Assembly)
    return sum(layer.thickness_m for layer in material.layers if isinstance(layer, AirLayer))


def test_accepted_cavity_ranges_are_kept_by_the_exported_material(tmp_path: Path) -> None:
    """抓 #1234：空腔範圍的下限介於 0 和最小空腔之間照樣收下，匯出時整層拿掉、交出去的構造違反範圍。

    兩個來源（分區範圍、材料頁那一份）× 兩份設定（設定檔、最小空腔加倍）× 五種範圍：收下的每一個，θ 的空腔放在範圍兩端走
    ``materials_from_surfaces``，x0 匯出構造的空腔要在範圍裡（沒有空氣層算 0，只有下限是 0 才合）；開跑前拒收的要是
    下限介於 0 和那一場最小空腔之間的那幾個，而且拒收句講的是匯出整層拿掉，其餘一個都不准拒收。"""
    sources: tuple[tuple[str, Designs], ...] = (("分區範圍", _zone_designs), ("材料頁", _page_designs))
    violations: list[str] = []
    refused: set[tuple[str, float, str]] = set()
    expected: set[tuple[str, float, str]] = set()
    for settings in (SETTINGS, _doubled()):
        minimum = settings.export.min_cavity_m
        for name, low, high in _cases(minimum):
            for source, build in sources:
                key = (source, minimum, name)
                if 0.0 < low < minimum:
                    expected.add(key)
                try:
                    designs = build(settings, (low, high), tmp_path)
                except ValueError as error:
                    if REFUSED_WORDS not in str(error):
                        violations.append(f"{source}（最小空腔 {minimum}）{name} {low}～{high}：別的原因拒收 {error}")
                    refused.add(key)
                    continue
                for end in dict.fromkeys((low, high)):
                    exported = _exported_cavity(designs, end, settings)
                    if not low <= exported <= high:
                        violations.append(f"{source}（最小空腔 {minimum}）{name} {low}～{high} 收下了，"
                                          f"空腔 {end} 匯出成 {exported}")
    assert violations == []
    assert refused == expected
