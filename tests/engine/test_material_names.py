"""#1119（#1082 段 1-8）結果頁的材料名字：先寫人看得懂的名字、數字收位；名字只從材料庫的登記取，對不到照實依參數組名。

數字直接寫 109 示範搜尋結果裡的原值（0.04999999999999999、123479.99999999999）：那正是示範時畫面印出來的醜數字。
"""
from __future__ import annotations

import pytest

from aosr.gui.material_names import material_name, original_wall_words, zone_material_words
from aosr.gui.problem_text import material_summary, rounded_number
from aosr.materials.construction_models import Assembly, ConstantImpedance, MountedOnWall, SurfaceMaterial
from aosr.materials.library import FixedEntry, LibraryEntry, OptimizableEntry, builtin_entries

UGLY_WALL = 123479.99999999999
DENSITY_48K = {"kind": "porous", "flow_resistivity_pa_s_m2": 19100.0, "flow_resistivity_source": "density_table",
               "model": "komatsu", "material_class": "fibrous"}


def _mounted(*layers: dict[str, object]) -> MountedOnWall:
    return MountedOnWall.model_validate({"kind": "mounted_on_wall", "layers": list(layers),
                                         "wall": {"kind": "constant_impedance", "impedance_pa_s_m": UGLY_WALL}})


def _preset(entry_id: str) -> FixedEntry:
    entry = next(item for item in builtin_entries() if item.id == entry_id)
    assert isinstance(entry, FixedEntry)
    return entry


@pytest.mark.parametrize(("value", "text"), [
    (0.04999999999999999, "0.05"), (UGLY_WALL, "123,480"), (40.4681860230972, "40.4682"), (19100.0, "19,100"),
    (0.15, "0.15"), (1646.4, "1,646.4")])
def test_rounded_number_is_six_significant_digits_with_thousands(value: float, text: str) -> None:
    assert rounded_number(value) == text


def test_search_pick_without_a_preset_is_named_from_its_layers_and_says_so() -> None:
    """109 示範三區：48K 棉 100 mm＋空腔 0.04999999999999999 m，內建預設沒有這一組 → 依參數組名，棉的名字借 48K 家族。"""
    picked = _mounted({**DENSITY_48K, "thickness_m": 0.1}, {"kind": "air", "thickness_m": 0.04999999999999999})
    library = builtin_entries()
    assert zone_material_words(picked, library) == (
        "48K 玻璃棉 100 mm＋50 mm 空腔（依參數組名），貼在原牆上（原牆：固定阻抗 123,480 Pa·s/m）")
    detail = material_summary(picked, rounded_number)
    assert "0.05 m" in detail and "123,480 Pa·s/m" in detail
    assert "0.0499" not in detail and "123479" not in detail


def test_layers_equal_to_a_preset_take_its_label_but_near_misses_do_not() -> None:
    """逐位相等才冠登記名（109 示範 y 起點牆 z1 就是「48K 玻璃棉 100 mm 貼牆」那幾層）；收位後才像的照實依參數組名。"""
    preset = _preset("glass_wool_48k_100mm")
    assert isinstance(preset.material, Assembly)
    exact = MountedOnWall(layers=preset.material.layers, wall=ConstantImpedance(impedance_pa_s_m=UGLY_WALL))
    name = material_name(exact, builtin_entries())
    assert (name.text, name.composed) == (preset.label, False)
    assert zone_material_words(exact, builtin_entries()) == f"{preset.label}，貼在原牆上（原牆：固定阻抗 123,480 Pa·s/m）"
    near = _mounted({**DENSITY_48K, "thickness_m": 0.10000000000000002})
    assert material_name(near, builtin_entries()).shown() == "48K 玻璃棉 100 mm（依參數組名）"
    assert material_name(preset.surface_material(), builtin_entries()).text == preset.label


def test_constant_impedance_and_original_wall_words() -> None:
    wall = ConstantImpedance(impedance_pa_s_m=UGLY_WALL)
    assert zone_material_words(wall, builtin_entries()) == "固定阻抗 123,480 Pa·s/m"
    assert original_wall_words(wall, builtin_entries()) == "原牆（固定阻抗 123,480 Pa·s/m）"


def test_user_library_names_win_for_fixed_materials_and_porous_layers() -> None:
    """使用者材料庫的固定材料逐位相等就用它的名字；沒有登記的層用模型描述的種類中文。"""
    concrete = FixedEntry.model_validate({"kind": "fixed", "id": "my_concrete", "label": "我家的混凝土牆",
                                          "provenance": "measured", "source": "自己量", "material": {
                                              "kind": "constant_impedance", "impedance_pa_s_m": UGLY_WALL}})
    library: tuple[LibraryEntry, ...] = (*builtin_entries(), concrete)
    assert original_wall_words(ConstantImpedance(impedance_pa_s_m=UGLY_WALL), library) == "原牆（我家的混凝土牆）"
    ceiling: SurfaceMaterial = Assembly.model_validate({"kind": "assembly", "backing": "rigid", "layers": [
        {"kind": "board", "surface_mass_kg_m2": 5.8, "resistance_pa_s_m": 40.4681860230972},
        {"kind": "air", "thickness_m": 0.2}]})
    assert zone_material_words(ceiling, library) == "薄板層（5.8 kg/m²）＋200 mm 空腔（依參數組名）"


def test_default_summary_still_prints_saved_values_for_the_compare_page() -> None:
    """比較頁要分得出兩份不同的材料：沒指定寫法時照存檔原值印，不收位。"""
    assert "123479.99999999999" in material_summary(ConstantImpedance(impedance_pa_s_m=UGLY_WALL))


def test_user_material_brackets_stay_only_builtin_family_notes_are_dropped() -> None:
    """審查抓到（#1119）：使用者材料名的括號可能正是分辨兩筆的字（廠牌），不准去掉；內建家族的「（可調厚度與空腔）」才去。"""
    family = next(entry for entry in builtin_entries() if isinstance(entry, OptimizableEntry))
    names = []
    for vendor, factor in (("甲廠", 1.0), ("乙廠", 1.1)):
        porous = family.porous.model_copy(update={
            "flow_resistivity_pa_s_m2": family.porous.flow_resistivity_pa_s_m2 * factor, "thickness_m": 0.123})
        entry = family.model_copy(update={"label": f"棉（{vendor}）", "porous": porous, "template_preset": None})
        material = Assembly.model_validate({"kind": "assembly", "backing": "rigid", "layers": [porous.model_dump()]})
        names.append(material_name(material, (entry,)).shown())
    assert names == ["棉（甲廠） 123 mm（依參數組名）", "棉（乙廠） 123 mm（依參數組名）"]
