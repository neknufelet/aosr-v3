"""逐區表白話一句（#1134）：貼在原牆上的一層吸音材講成「幾公分厚、直接貼牆／離牆幾公分」，其他排法照原名。"""
from __future__ import annotations

from typing import cast

from aosr.gui.material_names import plain_words
from aosr.materials.construction_models import Assembly, ConstantImpedance, MountedOnWall, PorousLayer
from aosr.materials.library import FixedEntry, OptimizableEntry, builtin_entries

WALL = {"kind": "constant_impedance", "impedance_pa_s_m": 123480.0}


def _mounted(layers: list[dict[str, object]]) -> MountedOnWall:
    return MountedOnWall.model_validate({"kind": "mounted_on_wall", "layers": layers, "wall": WALL})


def test_library_porous_with_a_gap_reads_as_thickness_and_distance_from_the_wall() -> None:
    library = builtin_entries()
    entry = next(item for item in library if isinstance(item, OptimizableEntry))
    # 搜尋選出來的厚度不是預設那幾種（預設的 50、100 mm 貼牆對得到材料庫固定材料，照它的名字）。
    porous = entry.porous.model_copy(update={"thickness_m": 0.075}).model_dump(mode="json")
    # 搜尋選出來的空腔常帶浮點尾數：0.01999999999999999 公尺照兩公分講。
    words = plain_words(_mounted([porous, {"kind": "air", "thickness_m": 0.01999999999999999}]), library)
    assert words.endswith(" 7.5 公分厚，離牆 2 公分（中間是空氣）"), words
    assert "依參數組名" not in words and "mm" not in words
    assert plain_words(_mounted([porous]), library).endswith(" 7.5 公分厚，直接貼牆")


def test_other_arrangements_keep_their_original_names() -> None:
    library = builtin_entries()
    board = {"kind": "board", "surface_mass_kg_m2": 5.8, "resistance_pa_s_m": 40.0}
    assert plain_words(_mounted([board, {"kind": "air", "thickness_m": 0.2}]), library) == ""
    assert plain_words(ConstantImpedance(impedance_pa_s_m=123480.0), library) == ""


def test_library_fixed_materials_and_unnamed_porous_keep_their_names() -> None:
    """審查實測：內建地毯貼牆時第一行被講成「多孔層 0.7 公分厚」；對得到材料庫固定材料就照它的名字。"""
    library = builtin_entries()
    fixed = next(item for item in library if isinstance(item, FixedEntry)
                 and isinstance(item.surface_material(), Assembly)
                 and len(cast(Assembly, item.surface_material()).layers) == 1
                 and isinstance(cast(Assembly, item.surface_material()).layers[0], PorousLayer))
    layers = [layer.model_dump(mode="json") for layer in cast(Assembly, fixed.surface_material()).layers]
    assert plain_words(_mounted(layers), library) == ""
    # 借不到名字的多孔層（流阻改過）照原名；空氣在吸音材前面也照原名。
    entry = next(item for item in library if isinstance(item, OptimizableEntry))
    odd = entry.porous.model_copy(update={"thickness_m": 0.05,
                                          "flow_resistivity_pa_s_m2": entry.porous.flow_resistivity_pa_s_m2 * 3})
    assert plain_words(_mounted([odd.model_dump(mode="json")]), library) == ""
    porous = entry.porous.model_copy(update={"thickness_m": 0.05}).model_dump(mode="json")
    assert plain_words(_mounted([{"kind": "air", "thickness_m": 0.02}, porous]), library) == ""
