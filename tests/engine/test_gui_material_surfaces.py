"""材料能用在哪幾面（#1176）：現在的內建材料＋合成材料庫 12 筆（自己編的，見 ``_sample_material_library``）每一筆分到哪一組，
釘死；材料清單介面 ``/api/materials`` 每一筆帶同一份分組。抓門、窗、地板、整片吊頂、另一間示範房的材料又被列成牆上能貼的
（老闆誤勾過「木門」），或內建多一筆沒人看過分到哪一組。"""
from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.material_editor import user_library_path
from aosr.gui.material_surfaces import GROUP_SHORT_WORDS, GROUP_WORDS, WALLS, placement
from aosr.materials.library import LibraryEntry, builtin_entries, dump_user_library
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES

_ENTRIES: TypeAdapter[tuple[LibraryEntry, ...]] = TypeAdapter(tuple[LibraryEntry, ...])
WALL_GROUP = (*WALLS, "ceiling")

# 內建：玻璃棉、織布、開孔板、開縫板都是貼在牆上的；穿孔石膏板那一筆描述整片吊頂；地毯只鋪地板。
BUILTIN_GROUPS = {
    "glass_wool_48k_50mm": "wall", "glass_wool_48k_50mm_gap50": "wall", "glass_wool_48k_50mm_gap100": "wall",
    "glass_wool_48k_100mm": "wall", "glass_wool_24k_50mm": "wall", "glass_wool_96k_50mm": "wall",
    "fabric_pet_a_gap100": "wall", "fabric_pet_b_gap100": "wall",
    "perforated_gypsum_8_18r_backed_gw30_ceiling": "ceiling_build",
    "perforated_0p8mm_d10_gw24k_50mm": "wall", "mpp_t1mm_d0p5_gap50": "wall", "mpp_t0p5mm_d0p5_gap50": "wall",
    "slotted_gypsum_b6_backed_gap65": "wall", "slotted_wood_typ9_absorber40_gap98": "wall",
    "carpet_on_concrete": "floor",
    # 文獻硬面是整面的表面阻抗（阻抗表），只能整面換掉，歸「牆面、地面本身」。
    "door_wood_solid": "surface_itself", "glass_single_3mm": "surface_itself", "glass_double_gap30": "surface_itself",
    "wood_floor_on_joists": "surface_itself", "wood_floor_on_sleepers": "surface_itself",
    "gypsum_double_13mm_stud": "surface_itself", "plasterboard_ceiling": "surface_itself",
    "brick_rendered": "surface_itself", "brick_painted": "surface_itself", "concrete_smooth": "surface_itself",
    "concrete_floor": "surface_itself", "hard_surface_average": "surface_itself",
    "glass_wool_48k": "wall", "glass_wool_24k": "wall", "glass_wool_96k": "wall",
}
# 合成材料庫：玻璃窗表列上限是實測表，照規則先後歸「牆面、地面本身」（樣稿把它畫在門窗那一組，兩組都不列）。
SAMPLE_GROUPS = {
    "demo109_rockwool60k_capped": "demo109", "demo109_measured_pet35_synthetic_b7": "demo109",
    "hall_door_hollow_core": "door_window", "joist_floor_plank_fixed": "surface_itself",
    "side_window_glass_12mm_single": "door_window", "side_window_glass_table_upper": "surface_itself",
    "ceiling_gypsum9mm_gap300": "ceiling_build",
    "ceiling_fibrecement8mm_wool30_on_board_gap180": "ceiling_build",
    "ceiling_fibrecement8mm_gap180_wool30_on_slab": "ceiling_build",
    "vpr_alu2mm_foam85_sample": "wall", "pet60_vpr_steel1p2mm_foam90_sample": "wall",
    "slotted18_pet45_vpr_steel2p25mm_foam70_sample": "wall",
}
SURFACES = {"wall": WALL_GROUP, "floor": ("floor",), "demo109": (), "surface_itself": (), "door_window": (),
            "ceiling_build": ()}


def _sample() -> tuple[LibraryEntry, ...]:
    return _ENTRIES.validate_python(SAMPLE_USER_ENTRIES)


def test_every_current_material_lands_in_its_pinned_group() -> None:
    found = {entry.id: placement(entry) for entry in (*builtin_entries(), *_sample())}
    assert {key: item.group for key, item in found.items()} == BUILTIN_GROUPS | SAMPLE_GROUPS
    for key, item in found.items():
        assert item.surfaces == SURFACES[item.group], key
        assert item.reason and item.group in GROUP_WORDS, key
    # 牆上看得到的（樣稿那一面）：19 筆；不列的 23 筆（含 12 筆文獻硬面），門窗、地板、天花構造、示範房、硬面一筆都不混進來。
    wall = sorted(key for key, item in found.items() if "x0" in item.surfaces)
    assert len(wall) == 19 and len([key for key, item in found.items() if "x0" not in item.surfaces]) == 23
    assert not any(word in key for key in wall
                   for word in ("door", "window", "floor", "carpet", "ceiling", "demo109", "glass_single", "brick", "concrete"))
    # 地板只有地毯；天花跟牆一樣（貼在吊頂下面），整片吊頂不列。
    assert [key for key, item in found.items() if "floor" in item.surfaces] == ["carpet_on_concrete"]
    assert sorted(key for key, item in found.items() if "ceiling" in item.surfaces) == wall


def test_the_ceiling_label_rule_and_the_floor_name_rule_each_hold_a_material() -> None:
    """審查 11-15：「顯示名以天花開頭」與「代號含 floor」兩條規則，現在的 30 筆沒有一筆只靠它們分組（拿掉照綠）。
    各釘一筆假材料（剛背的固定多層構造，照內建 48K 50 mm 改代號與顯示名）：拿掉那一條，它就掉進牆面那一組。"""
    template = next(entry for entry in builtin_entries() if entry.id == "glass_wool_48k_50mm").model_dump(mode="json")
    adapter: TypeAdapter[LibraryEntry] = TypeAdapter(LibraryEntry)
    ceiling = adapter.validate_python({**template, "id": "suspended_gypsum_gap200", "label": "天花 石膏板吊頂＋20 cm 空腔"})
    floor = adapter.validate_python({**template, "id": "raised_wood_floor_on_joists", "label": "架高木地板"})
    assert (placement(ceiling).group, placement(ceiling).surfaces) == ("ceiling_build", ())
    assert (placement(floor).group, placement(floor).surfaces) == ("floor", ("floor",))


def test_material_listing_carries_the_placement_of_each_row(tmp_path: Path) -> None:
    path = user_library_path(tmp_path)
    path.write_text(dump_user_library(_sample()), encoding="utf-8")
    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path))
    with TestClient(app, base_url="http://localhost") as client:
        listing = client.get("/api/materials").json()
    assert listing["user_file_problem"] is None
    rows = {row["entry"]["id"]: row for row in listing["materials"]}
    assert {key: row["placement"]["group"] for key, row in rows.items()} == BUILTIN_GROUPS | SAMPLE_GROUPS
    assert rows["hall_door_hollow_core"]["placement"] == {
        "group": "door_window", "surfaces": [], "reason": placement(_sample()[2]).reason}
    assert rows["glass_wool_48k"]["placement"]["surfaces"] == list(WALL_GROUP)
    assert listing["surface_groups"] == GROUP_WORDS
    # 短名（「還有 N 筆不列出來：…」那一句用）：每一組都有、順序同標題，短名裡不准再有括號。
    assert listing["surface_group_short"] == GROUP_SHORT_WORDS and list(GROUP_SHORT_WORDS) == list(GROUP_WORDS)
    assert not any("（" in words for words in GROUP_SHORT_WORDS.values())
