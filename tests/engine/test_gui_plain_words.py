"""逐區表白話一句（#1134）：貼在原牆上的一層吸音材講成「幾公分厚、直接貼牆／離牆幾公分」，其他排法照原名。"""
from __future__ import annotations

from aosr.gui.material_names import plain_words
from aosr.materials.construction_models import ConstantImpedance, MountedOnWall
from aosr.materials.library import OptimizableEntry, builtin_entries

WALL = {"kind": "constant_impedance", "impedance_pa_s_m": 123480.0}


def _mounted(layers: list[dict[str, object]]) -> MountedOnWall:
    return MountedOnWall.model_validate({"kind": "mounted_on_wall", "layers": layers, "wall": WALL})


def test_library_porous_with_a_gap_reads_as_thickness_and_distance_from_the_wall() -> None:
    library = builtin_entries()
    entry = next(item for item in library if isinstance(item, OptimizableEntry))
    porous = entry.porous.model_copy(update={"thickness_m": 0.1}).model_dump(mode="json")
    # 搜尋選出來的空腔常帶浮點尾數：0.01999999999999999 公尺照兩公分講。
    words = plain_words(_mounted([porous, {"kind": "air", "thickness_m": 0.01999999999999999}]), library)
    assert words.endswith(" 10 公分厚，離牆 2 公分（中間是空氣）"), words
    assert "依參數組名" not in words and "mm" not in words
    assert plain_words(_mounted([porous]), library).endswith(" 10 公分厚，直接貼牆")


def test_other_arrangements_keep_their_original_names() -> None:
    library = builtin_entries()
    board = {"kind": "board", "surface_mass_kg_m2": 5.8, "resistance_pa_s_m": 40.0}
    assert plain_words(_mounted([board, {"kind": "air", "thickness_m": 0.2}]), library) == ""
    assert plain_words(ConstantImpedance(impedance_pa_s_m=123480.0), library) == ""
