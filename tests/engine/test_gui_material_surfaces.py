"""釘住材料位置與四種家族，驗證材料清單回覆使用相同分組。"""
from __future__ import annotations

from pathlib import Path

import pytest
from starlette.applications import Starlette
from pydantic import TypeAdapter
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.material_editor import user_library_path
from aosr.gui.material_surfaces import GROUP_SHORT_WORDS, GROUP_WORDS, WALLS, placement
from aosr.materials.library import LibraryEntry, builtin_entries, dump_user_library
from tests.engine._construction_preset_answers import FITTED_VPR_IDS
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine.test_material_construction_baseline import legacy_entries

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
    **dict.fromkeys(FITTED_VPR_IDS, "wall"),
    "carpet_on_concrete": "floor",
    # 文獻硬面是整面的表面阻抗（阻抗表），只能整面換掉，歸「牆面、地面本身」。
    "door_wood_solid": "surface_itself", "glass_single_3mm": "surface_itself", "glass_double_gap30": "surface_itself",
    "wood_floor_on_joists": "surface_itself", "wood_floor_on_sleepers": "surface_itself",
    "gypsum_double_13mm_stud": "surface_itself", "plasterboard_ceiling": "surface_itself",
    "brick_rendered": "surface_itself", "brick_painted": "surface_itself", "concrete_smooth": "surface_itself",
    "concrete_floor": "surface_itself", "hard_surface_average": "surface_itself",
    "glass_wool_48k": "wall", "glass_wool_24k": "wall", "glass_wool_96k": "wall",
}
# 文獻硬面裡講地板的只鋪地板；名字同時講牆與地板的六面都能用；其餘在牆與天花。
CATALOG_SURFACES = {"wood_floor_on_joists": ("floor",), "wood_floor_on_sleepers": ("floor",), "concrete_floor": ("floor",),
                    "hard_surface_average": ("x0", "xL", "y0", "yL", "ceiling", "floor")}
# 合成材料庫：玻璃窗表列上限是實測表，照規則先後歸「牆面、地面本身」（在牆面與天花可選）。
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
SURFACES = {"wall": WALL_GROUP, "floor": ("floor",), "demo109": (), "surface_itself": WALL_GROUP, "door_window": WALLS,
            "ceiling_build": ("ceiling",)}


def _sample() -> tuple[LibraryEntry, ...]:
    return _ENTRIES.validate_python(SAMPLE_USER_ENTRIES)


def test_every_current_material_lands_in_its_pinned_group() -> None:
    found = {entry.id: placement(entry) for entry in (*builtin_entries(), *_sample())}
    assert {key: item.group for key, item in found.items()} == BUILTIN_GROUPS | SAMPLE_GROUPS
    for key, item in found.items():
        assert item.surfaces == (CATALOG_SURFACES.get(key) or (("floor",) if key == "joist_floor_plank_fixed" else SURFACES[item.group])), key
        assert item.reason and item.group in GROUP_WORDS, key
    walls = {key for key, item in found.items() if "x0" in item.surfaces}
    expected = {key for key, group in (BUILTIN_GROUPS | SAMPLE_GROUPS).items() if group in ("wall", "door_window")}
    hard_walls = {"door_wood_solid", "glass_single_3mm", "glass_double_gap30", "gypsum_double_13mm_stud", "plasterboard_ceiling",
                  "brick_rendered", "brick_painted", "concrete_smooth", "hard_surface_average"}
    assert walls == expected | {"side_window_glass_table_upper"} | hard_walls
    assert {key for key, item in found.items() if "floor" in item.surfaces} == {
        "carpet_on_concrete", "joist_floor_plank_fixed", "wood_floor_on_joists", "wood_floor_on_sleepers", "concrete_floor", "hard_surface_average"}
    assert {key for key, item in found.items() if "ceiling" in item.surfaces} == {
        key for key, group in (BUILTIN_GROUPS | SAMPLE_GROUPS).items() if group in ("wall", "ceiling_build")} | {"side_window_glass_table_upper"} | hard_walls


def test_the_ceiling_label_rule_and_the_floor_name_rule_each_hold_a_material() -> None:
    """用考卷自編的剛背構造分別考天花標籤與地板代號。"""
    template = next(entry for entry in legacy_entries() if entry.id == "pin_rigid").model_dump(mode="json")
    adapter: TypeAdapter[LibraryEntry] = TypeAdapter(LibraryEntry)
    ceiling = adapter.validate_python({**template, "id": "suspended_gypsum_gap200", "label": "天花 石膏板吊頂＋20 cm 空腔"})
    floor = adapter.validate_python({**template, "id": "raised_wood_floor_on_joists", "label": "架高木地板"})
    assert (placement(ceiling).group, placement(ceiling).surfaces) == ("ceiling_build", ("ceiling",))
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
        "group": "door_window", "family": "surface", "surfaces": list(WALLS), "reason": placement(_sample()[2]).reason}
    assert rows["glass_wool_48k"]["placement"]["surfaces"] == list(WALL_GROUP)
    assert listing["surface_groups"] == GROUP_WORDS
    # 短名（「還有 N 筆不列出來：…」那一句用）：每一組都有、順序同標題，短名裡不准再有括號。
    assert listing["surface_group_short"] == GROUP_SHORT_WORDS and list(GROUP_SHORT_WORDS) == list(GROUP_WORDS)
    assert not any("（" in words for words in GROUP_SHORT_WORDS.values())

def test_builtin_and_designer_constructions_use_catalog_origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.gui import material_editor
    from tests.engine._construction_cases import construction

    docs = [construction(placement="door_window", id="sample_opening", label="自編甲"),
            construction(placement="floor", id="sample_finish", label="自編乙")]
    builtin, mine = _ENTRIES.validate_python(docs)
    monkeypatch.setattr(material_editor, "builtin_entries", lambda: (builtin,))
    user_library_path(tmp_path).write_text(dump_user_library((mine,)), encoding="utf-8")
    with TestClient(Starlette(routes=material_editor.material_routes(tmp_path))) as client:
        rows = client.get("/api/materials").json()["materials"]
    assert [(row["builtin"], row["placement"]["family"]) for row in rows] == [(True, "surface"), (False, "custom")]


def test_face_and_family_follow_the_material_fields() -> None:
    entries = {entry.id: entry for entry in legacy_entries()}
    floor = entries["pin_impedance"].model_copy(update={"label": "考卷自編木地板"})
    carpet = floor.model_copy(update={"id": "exam_carpet", "label": "試片乙"})
    for entry in (floor, carpet):
        assert (placement(entry).surfaces, placement(entry).family) == (("floor",), "surface")
    assert placement(entries["pin_air"]).surfaces == WALLS
    assert placement(entries["pin_impedance"]).surfaces == WALL_GROUP
    assert placement(entries["pin_family_false"]).family == placement(entries["pin_rigid"]).family == "porous"
    samples = {entry.id: placement(entry) for entry in _sample()}
    assert samples["vpr_alu2mm_foam85_sample"].family == "resonant"
    assert samples["ceiling_gypsum9mm_gap300"].family == "surface"
    assert samples["demo109_measured_pet35_synthetic_b7"].surfaces == ()


@pytest.mark.parametrize("plate", [
    {"kind": "board", "surface_mass_kg_m2": 7.3, "resistance_pa_s_m": 0.0},
    {"kind": "perforated", "hole_diameter_m": 0.004, "thickness_m": 0.008, "open_area": 0.12},
    {"kind": "micro_perforated", "hole_diameter_m": 0.0007, "thickness_m": 0.001, "open_area": 0.07},
    {"kind": "slotted", "slot_width_m": 0.003, "thickness_m": 0.007, "open_area": 0.11},
])
def test_any_plate_layer_makes_a_rigid_product_resonant(plate: dict[str, object]) -> None:
    own = next(entry for entry in legacy_entries() if entry.id == "pin_rigid").model_dump(mode="json")
    own["material"]["layers"].insert(1, plate)
    (entry,) = _ENTRIES.validate_python([own])
    assert placement(entry).family == "resonant"


def test_floor_only_label_mentioning_wall_stays_on_floor() -> None:
    entry = next(item for item in legacy_entries() if item.id == "pin_impedance")
    floor = entry.model_copy(update={"id": "exam_finish", "label": "地板專用（不可上牆）"})
    assert placement(floor).surfaces == ("floor",)
