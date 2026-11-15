"""材料能用在哪幾面（#1176）：現在的內建材料＋老闆材料庫 12 筆（11-15 照抄，見 ``_boss_material_library``）每一筆分到哪一組，
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
from tests.engine._boss_material_library import BOSS_USER_ENTRIES

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
    "glass_wool_48k": "wall", "glass_wool_24k": "wall", "glass_wool_96k": "wall",
}
# 老闆材料庫：玻璃窗型錄上限是實測表，照規則先後歸「牆面、地面本身」（樣稿把它畫在門窗那一組，兩組都不列）。
BOSS_GROUPS = {
    "demo109_gw48k_limited": "demo109", "demo109_measured_gw49_gaborit_s1": "demo109",
    "own_room_door_wood_solid": "door_window", "own_room_floor_wood_on_concrete": "surface_itself",
    "own_room_window_glass_10mm_single": "door_window", "own_room_window_glass_catalog_upper": "surface_itself",
    "own_room_ceiling_calsil_6mm_gap250": "ceiling_build",
    "own_room_ceiling_calsil_melamine40_on_board_gap210": "ceiling_build",
    "own_room_ceiling_calsil_gap210_melamine40_on_concrete": "ceiling_build",
    "vpr_steel1mm_softfoam100_draft": "wall", "melamine100_vpr_steel1mm_softfoam100_draft": "wall",
    "slotted16_mel50_vpr_steel3_5mm_mel100_draft": "wall",
}
SURFACES = {"wall": WALL_GROUP, "floor": ("floor",), "demo109": (), "surface_itself": (), "door_window": (),
            "ceiling_build": ()}


def _boss() -> tuple[LibraryEntry, ...]:
    return _ENTRIES.validate_python(BOSS_USER_ENTRIES)


def test_every_current_material_lands_in_its_pinned_group() -> None:
    found = {entry.id: placement(entry) for entry in (*builtin_entries(), *_boss())}
    assert {key: item.group for key, item in found.items()} == BUILTIN_GROUPS | BOSS_GROUPS
    for key, item in found.items():
        assert item.surfaces == SURFACES[item.group], key
        assert item.reason and item.group in GROUP_WORDS, key
    # 牆上看得到的（樣稿那一面）：19 筆；不列的 11 筆，門窗、地板、天花構造、示範房一筆都不混進來。
    wall = sorted(key for key, item in found.items() if "x0" in item.surfaces)
    assert len(wall) == 19 and len([key for key, item in found.items() if "x0" not in item.surfaces]) == 11
    assert not any(word in key for key in wall for word in ("door", "window", "floor", "carpet", "ceiling", "demo109"))
    # 地板只有地毯；天花跟牆一樣（貼在吊頂下面），整片吊頂不列。
    assert [key for key, item in found.items() if "floor" in item.surfaces] == ["carpet_on_concrete"]
    assert sorted(key for key, item in found.items() if "ceiling" in item.surfaces) == wall


def test_material_listing_carries_the_placement_of_each_row(tmp_path: Path) -> None:
    path = user_library_path(tmp_path)
    path.write_text(dump_user_library(_boss()), encoding="utf-8")
    app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path))
    with TestClient(app, base_url="http://localhost") as client:
        listing = client.get("/api/materials").json()
    assert listing["user_file_problem"] is None
    rows = {row["entry"]["id"]: row for row in listing["materials"]}
    assert {key: row["placement"]["group"] for key, row in rows.items()} == BUILTIN_GROUPS | BOSS_GROUPS
    assert rows["own_room_door_wood_solid"]["placement"] == {
        "group": "door_window", "surfaces": [], "reason": placement(_boss()[2]).reason}
    assert rows["glass_wool_48k"]["placement"]["surfaces"] == list(WALL_GROUP)
    assert listing["surface_groups"] == GROUP_WORDS
    # 短名（「還有 N 筆不列出來：…」那一句用）：每一組都有、順序同標題，短名裡不准再有括號。
    assert listing["surface_group_short"] == GROUP_SHORT_WORDS and list(GROUP_SHORT_WORDS) == list(GROUP_WORDS)
    assert not any("（" in words for words in GROUP_SHORT_WORDS.values())
