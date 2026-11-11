"""超過限制的區塊標紅（#1111，#1082 第 1-4 段）：深度上限三層取最嚴並記是哪一層擋的、固定區貼太深是紅、
直接取代牆面的材料照實說比不了、位置（出界、太小、重疊、沒蓋滿）跟切牆面同一把判、可信範圍走材料頁同一條路
（物理拒算紅、醒目提醒只警告）；設計設定新欄位沒給時存檔逐位不變；開搜尋的參數照三層帶深度上限、請求只能更嚴、
有紅就不開。房間一律用 109 號房的長寬高。"""
import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError
from starlette.testclient import TestClient
import httpx

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.config.product_air import product_air_environment
from aosr.gradient.fixed_surfaces import designs_with_max_depths
from aosr.gradient.run import designs_for, scene_wall_materials
from aosr.gui.app import GuiSettings, create_app
from aosr.config.construction_presets import load_construction_presets
from aosr.gui.gradient_plan import GradientRequest, PlanRefused, builtin_depth_targets, library_depth_targets, plan_search, start_arguments
from aosr.gui.zone_design import DesignSettings, ZoneDesign
from aosr.gui.zone_edit import MIN_SIDE_M, CutBox, cut_wall
from aosr.gui.zone_limits import DepthCap, DepthTargets, added_depth, depth_cap, range_target, zone_limits
from aosr.materials import library
from aosr.materials.construction_models import Assembly, MountedOnWall, SurfaceMaterial
from aosr.materials.library import FixedEntry, builtin_entries
from aosr.materials.search_library import with_user_materials
from aosr.reporting.scheme import Scheme
from tests.engine._construction_gui_server import real_server
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._surface_zone_cases import NEW, ROOM_109, room109_document

PRESET = "glass_wool_48k_50mm"          # 50 mm 玻璃棉，貼在原牆上總深 0.05 m
DEEP = "glass_wool_48k_50mm_gap100"     # 50 mm 玻璃棉＋100 mm 空腔，總深 0.15 m
SETTINGS = load_material_gradient_settings(config_path("material_gradient.toml"))
TARGETS = builtin_depth_targets(SETTINGS)
_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
Doc = dict[str, object]


def _zones(document: Doc) -> dict[str, list[dict[str, object]]]:
    return cast(dict[str, list[dict[str, object]]], cast(Doc, document["scene"])[NEW])


def _layers(material_id: str) -> list[object]:
    entry = next(item for item in builtin_entries() if item.id == material_id)
    assert isinstance(entry, FixedEntry) and isinstance(entry.material, Assembly)
    return cast(list[object], entry.material.model_dump(mode="json")["layers"])


def _cut(fixed: str | None = DEEP) -> Doc:
    """x0 牆左下畫 z1（1.0×1.2 m），其餘自動補滿；z1 貼一組固定材料（貼在原牆上）。"""
    document = room109_document()
    zones = cut_wall(document, "x0", [CutBox(id="z1", u_m=(0.0, 1.0), v_m=(0.0, 1.2))])["zones"]
    _zones(document)["x0"] = cast(list[dict[str, object]], zones)
    if fixed is not None:
        z1 = next(zone for zone in _zones(document)["x0"] if zone["id"] == "z1")
        z1["material"] = {"kind": "mounted_on_wall", "layers": _layers(fixed), "wall": z1["material"]}
    return document


def _design(**fields: object) -> DesignSettings:
    return DesignSettings.model_validate({"scheme_id": "room109", **fields})


def _lines(result: dict[str, object], wall: str, zone: str) -> list[dict[str, str]]:
    return cast(dict[str, dict[str, list[dict[str, str]]]], result["zones"]).get(wall, {}).get(zone, [])


def _codes(result: dict[str, object], wall: str, zone: str) -> list[str]:
    return [line["code"] for line in _lines(result, wall, zone)]


# ---- 深度上限三層 ----

def test_depth_cap_takes_the_smallest_of_three_and_names_the_level() -> None:
    zone = {"x0": {"z1": {"mode": "fixed", "max_depth_m": 0.12}}}
    assert depth_cap(None, "x0", "z1") is None
    assert depth_cap(_design(), "x0", "z1") is None
    assert depth_cap(_design(room_max_depth_m=0.2), "x0", "z1") == DepthCap(0.2, "room")
    assert depth_cap(_design(room_max_depth_m=0.2, wall_max_depth_m={"x0": 0.1}), "x0", "z1") == DepthCap(0.1, "wall")
    assert depth_cap(_design(room_max_depth_m=0.2, wall_max_depth_m={"x0": 0.1}, zones=zone), "x0", "z1") == DepthCap(0.1, "wall")
    assert depth_cap(_design(room_max_depth_m=0.2, wall_max_depth_m={"x0": 0.15}, zones=zone), "x0", "z1") == DepthCap(0.12, "zone")
    # 別面、別區不受這面這一塊的值影響；一樣小時取範圍最小的那一層
    assert depth_cap(_design(room_max_depth_m=0.2, wall_max_depth_m={"x0": 0.1}, zones=zone), "y0", "z1") == DepthCap(0.2, "room")
    assert depth_cap(_design(room_max_depth_m=0.12, wall_max_depth_m={"x0": 0.12}, zones=zone), "x0", "z1") == DepthCap(0.12, "zone")
    assert depth_cap(_design(room_max_depth_m=0.12, wall_max_depth_m={"x0": 0.12}), "x0", "z1") == DepthCap(0.12, "wall")


@pytest.mark.parametrize("fields", [{"room_max_depth_m": 0.0}, {"room_max_depth_m": -0.1}, {"wall_max_depth_m": {"x0": 0.0}},
                                    {"wall_max_depth_m": {"wall9": 0.1}}, {"room_max_depth_m": float("inf")},
                                    {"zones": {"x0": {"z1": {"mode": "fixed", "max_depth_m": 0.0}}}}])
def test_depth_limits_must_be_positive_finite_and_on_a_real_wall(fields: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _design(**fields)


def test_designs_without_depth_limits_save_byte_for_byte_as_before() -> None:
    """新欄位沒給時存檔不多一格：跟加欄位前那一版寫出來的字串一樣（逐字釘死）。"""
    design = _design(zones={"x0": {"z1": {"mode": "search", "allowed": ["glass_wool_48k", "none"]}, "auto1": {"mode": "keep_original"}}},
                     original_scheme_id="room109-original")
    assert design.model_dump_json() == ('{"schema_version":"aosr.zone_design.v1","scheme_id":"room109","zones":{"x0":{"z1":{"mode":"search",'
                                        '"allowed":["glass_wool_48k","none"]},"auto1":{"mode":"keep_original"}}},'
                                        '"original_scheme_id":"room109-original"}')
    limited = _design(room_max_depth_m=0.2, wall_max_depth_m={"x0": 0.1}, zones={"x0": {"z1": {"mode": "fixed", "max_depth_m": 0.05}}})
    assert DesignSettings.model_validate_json(limited.model_dump_json()) == limited
    assert json.loads(limited.model_dump_json())["wall_max_depth_m"] == {"x0": 0.1}


# ---- 固定區的深度 ----

def test_added_depth_counts_only_layers_mounted_on_the_wall() -> None:
    document = _cut()
    z1 = next(zone for zone in _zones(document)["x0"] if zone["id"] == "z1")
    mounted = MountedOnWall.model_validate(z1["material"])
    assert added_depth(mounted) == pytest.approx(0.15)
    assert added_depth(MountedOnWall.model_validate({**cast(dict[str, object], z1["material"]), "layers": _layers(PRESET)})) == pytest.approx(0.05)
    # 直接取代牆面的（原牆本身、固定阻抗）看不出貼了多深
    assert added_depth(mounted.wall) is None


@pytest.mark.parametrize(("fields", "level", "word"), [
    ({"wall_max_depth_m": {"x0": 0.1}}, "wall", "這面牆"),
    ({"zones": {"x0": {"z1": {"mode": "fixed", "max_depth_m": 0.12}}}, "room_max_depth_m": 0.3}, "zone", "這一塊"),
    ({"room_max_depth_m": 0.1, "wall_max_depth_m": {"y0": 0.05}}, "room", "房間預設"),
])
def test_fixed_zone_mounted_deeper_than_its_cap_is_red_and_names_the_level(fields: dict[str, object], level: str, word: str) -> None:
    result = zone_limits(_cut(), _design(**fields).model_dump(mode="json"))
    assert _codes(result, "x0", "z1") == ["depth_over"]
    line = _lines(result, "x0", "z1")[0]
    assert line["level"] == "red" and "0.15 公尺" in line["text"] and f"{word}的上限" in line["text"]
    assert result["red"] == 1 and "x 起點牆「z1」" in cast(list[str], result["refusals"])[0]
    assert cast(dict[str, dict[str, dict[str, object]]], result["caps"])["x0"]["z1"]["level"] == level


def test_an_uncut_wall_fixed_too_deep_is_red_on_the_page_and_at_the_search() -> None:
    """整面沒切（方案裡那一面只有一筆「整面」）：一樣比、一樣擋；示範最常見的就是這一種。"""
    document = room109_document()
    whole = _zones(document)["y0"][0]
    whole["material"] = {"kind": "mounted_on_wall", "layers": _layers(DEEP), "wall": whole["material"]}
    design = _design(room_max_depth_m=0.1, zones={"x0": {"whole_wall": {"mode": "search"}}})
    result = zone_limits(document, design.model_dump(mode="json"))
    assert [(line["code"], line["level"]) for line in _lines(result, "y0", "whole_wall")] == [("depth_over", "red")]
    assert result["refusals"] == ["y 起點牆「整面」：" + _lines(result, "y0", "whole_wall")[0]["text"]]
    with pytest.raises(PlanRefused, match="y 起點牆「整面」"):
        plan_search(_scheme(document), design, GradientRequest(), SETTINGS)


def test_fixed_zone_within_its_cap_and_zones_without_caps_pass() -> None:
    assert zone_limits(_cut(PRESET), _design(wall_max_depth_m={"x0": 0.1}).model_dump(mode="json"))["red"] == 0
    assert zone_limits(_cut(), _design().model_dump(mode="json"))["red"] == 0
    assert zone_limits(_cut(), None)["red"] == 0
    # 剛好等於上限不算超過（0.05+0.1 在浮點不是正好 0.15，也要過）
    assert zone_limits(_cut(), _design(wall_max_depth_m={"x0": 0.15}).model_dump(mode="json"))["red"] == 0


def test_replacing_materials_say_the_cap_cannot_be_compared_and_do_not_go_red() -> None:
    """自動補的兩塊是原牆（固定阻抗，直接取代牆面）：看不出貼了多深，照實寫比不了，不標紅、也不拿原牆厚度比。"""
    result = zone_limits(_cut(None), _design(room_max_depth_m=0.01).model_dump(mode="json"))
    assert result["red"] == 0
    for zone in ("z1", "auto1", "auto2"):
        assert [(line["code"], line["level"]) for line in _lines(result, "x0", zone)] == [("depth_not_comparable", "note")]
    assert "比不了" in _lines(result, "x0", "z1")[0]["text"]


def test_keep_original_zones_have_no_cap_and_search_zones_report_theirs() -> None:
    design = _design(room_max_depth_m=0.01, zones={"x0": {"auto1": {"mode": "search"}, "auto2": {"mode": "keep_original"}}})
    result = zone_limits(_cut(None), design.model_dump(mode="json"))
    caps = cast(dict[str, dict[str, dict[str, object]]], result["caps"])["x0"]
    assert caps["auto1"] == {"value_m": 0.01, "level": "room", "words": "0.01 公尺（房間預設的上限）"}
    assert "auto2" not in caps and _lines(result, "x0", "auto1") == [] and _lines(result, "x0", "auto2") == []


# ---- 位置：跟切牆面同一把 ----

def test_zones_outside_a_shrunk_wall_are_red() -> None:
    document = _cut(PRESET)
    cast(dict[str, float], cast(Doc, document["scene"])["room_m"])["Ly"] = 0.9   # x0 牆寬 4.42 → 0.9 公尺
    result = zone_limits(document)
    assert "outside" in _codes(result, "x0", "z1") and "outside" in _codes(result, "x0", "auto1")
    assert "現在寬 0.9" in _lines(result, "x0", "z1")[0]["text"]
    assert cast(int, result["red"]) >= 2


def test_too_small_and_overlapping_zones_are_red_and_gaps_red_the_wall() -> None:
    document = _cut(PRESET)
    zones = _zones(document)["x0"]
    z1 = next(zone for zone in zones if zone["id"] == "z1")
    z1["u_m"] = [0.0, MIN_SIDE_M - 0.01]                  # 太小（也留出一條縫）
    result = zone_limits(document)
    assert _codes(result, "x0", "z1") == ["too_small"]
    z1["u_m"] = [0.0, 1.0]
    overlapping = deepcopy(z1) | {"id": "z2", "u_m": [0.5, 1.5]}
    zones.append(overlapping)
    result = zone_limits(document)
    assert "overlap" in _codes(result, "x0", "z1") and "overlap" in _codes(result, "x0", "z2")
    assert "跟 z1 重疊" in [line["text"] for line in _lines(result, "x0", "z2")]
    zones.remove(overlapping)
    z1["v_m"] = [0.0, 1.0]                                  # 上緣往下：z1 跟自動補的那一塊之間留空
    result = zone_limits(document)
    assert [line["code"] for line in cast(dict[str, list[dict[str, str]]], result["walls"])["x0"]] == ["not_covered"]
    assert _lines(result, "x0", "z1") == []


def test_a_clean_cut_has_no_place_problems_and_bad_rooms_are_reported_not_raised() -> None:
    assert zone_limits(_cut(PRESET))["red"] == 0
    broken = room109_document()
    cast(dict[str, object], cast(Doc, broken["scene"])["room_m"])["Lx"] = None
    result = zone_limits(broken)
    assert result["status"] == "room" and result["red"] == 0


def test_a_malformed_design_is_red_not_an_error() -> None:
    result = zone_limits(_cut(), {"scheme_id": "room109", "room_max_depth_m": -1})
    assert result["red"] == 1 and "深度上限要填大於 0" in cast(list[str], result["refusals"])[0]


# ---- 可信範圍：材料頁同一條路 ----

def _fake(status: str, calls: list[SurfaceMaterial], **extra: object) -> Callable[[SurfaceMaterial], dict[str, object]]:
    def check(material: SurfaceMaterial) -> dict[str, object]:
        calls.append(material)
        return {"status": status, **extra}
    return check


def test_range_check_sees_only_the_mounted_layers_once_per_material() -> None:
    document = _cut(PRESET)
    for zone in _zones(document)["x0"]:            # 三塊都貼同一組：只查一次
        if zone["id"] != "z1":
            zone["material"] = deepcopy(next(item for item in _zones(document)["x0"] if item["id"] == "z1")["material"])
    calls: list[SurfaceMaterial] = []
    result = zone_limits(document, None, _fake("drawn", calls, warnings=[]))
    assert result["red"] == 0 and len(calls) == 1
    assert calls[0] == Assembly(layers=MountedOnWall.model_validate(_zones(document)["x0"][0]["material"]).layers, backing="rigid")
    # 搜尋與保留原牆的區不查（那一區在方案裡放的是原牆）
    calls.clear()
    design = _design(zones={"x0": {"z1": {"mode": "search"}, "auto1": {"mode": "keep_original"}, "auto2": {"mode": "search"}}})
    zone_limits(document, design.model_dump(mode="json"), _fake("drawn", calls, warnings=[]))
    assert calls == []


def test_uncomputable_is_red_strong_warnings_only_warn_and_no_precision_is_said() -> None:
    document = _cut(PRESET)
    result = zone_limits(document, None, _fake("uncomputable", [], message="這組參數算不出吸音曲線：流阻超出手動試算範圍"))
    assert [(line["code"], line["level"]) for line in _lines(result, "x0", "z1")] == [("range_uncomputable", "red")]
    assert result["red"] == 1
    warnings = [{"level": "strong", "code": "komatsu_flow_below_samples", "text": "第 1 層：流阻低於原研究樣品範圍"},
                {"level": "note", "code": "komatsu_frequency_extrapolation", "text": "外推"}]
    result = zone_limits(document, None, _fake("drawn", [], warnings=warnings))
    assert [(line["code"], line["level"]) for line in _lines(result, "x0", "z1")] == [("komatsu_flow_below_samples", "warn")]
    assert result["red"] == 0 and "不擋" in _lines(result, "x0", "z1")[0]["text"]
    result = zone_limits(document, None, _fake("no_precision", []))
    assert [(line["code"], line["level"]) for line in _lines(result, "x0", "z1")] == [("range_unchecked", "note")]


def test_range_target_skips_materials_without_registered_ranges() -> None:
    wall = MountedOnWall.model_validate(next(zone for zone in _zones(_cut(PRESET))["x0"] if zone["id"] == "z1")["material"])
    assert range_target(wall) == Assembly(layers=wall.layers, backing="rigid")
    assert range_target(wall.wall) is None                   # 固定阻抗：沒有範圍可查
    fabric = MountedOnWall.model_validate({**wall.model_dump(mode="json"), "layers": _layers("fabric_pet_a_gap100")})
    assert range_target(fabric) is None                      # 布＋空腔：沒有小松層、沒登記範圍
    carpet = next(item for item in builtin_entries() if item.id == "carpet_on_concrete")
    assert isinstance(carpet, FixedEntry) and isinstance(carpet.material, Assembly)
    assert range_target(carpet.material) == carpet.material   # 直接取代的組合照原樣查


# ---- 開搜尋：三層帶進參數、請求只能更嚴、有紅不開 ----

def _scheme(document: Doc) -> Scheme:
    return Scheme.model_validate(document)


def test_plan_carries_design_caps_for_search_zones_and_requests_can_only_tighten() -> None:
    scheme = _scheme(_cut(PRESET))
    design = _design(room_max_depth_m=0.2, wall_max_depth_m={"x0": 0.1},
                     zones={"x0": {"auto1": {"mode": "search"}, "auto2": {"mode": "search", "max_depth_m": 0.08}}})
    plan = plan_search(scheme, design, GradientRequest(), SETTINGS)
    assert plan.max_depths == {"x0@auto1": 0.1, "x0@auto2": 0.08}
    assert plan.depth_sources == {"x0@auto1": "wall", "x0@auto2": "zone"}
    looser = plan_search(scheme, design, GradientRequest(max_depths={"x0": {"auto1": 0.3, "auto2": 0.05}}), SETTINGS)
    assert looser.max_depths == {"x0@auto1": 0.1, "x0@auto2": 0.05}
    assert looser.depth_sources == {"x0@auto1": "wall", "x0@auto2": "request"}


def test_plans_without_design_caps_are_written_as_before() -> None:
    scheme = _scheme(_cut(PRESET))
    design = _design(zones={"x0": {"auto1": {"mode": "search"}}})
    plan = plan_search(scheme, design, GradientRequest(max_depths={"x0": {"auto1": 0.1}}), SETTINGS)
    assert plan.max_depths == {"x0@auto1": 0.1} and plan.depth_sources == {}
    assert "depth_sources" not in json.loads(plan.model_dump_json())


def test_a_fixed_zone_over_its_cap_refuses_the_search() -> None:
    scheme = _scheme(_cut())
    design = _design(wall_max_depth_m={"x0": 0.1}, zones={"x0": {"auto1": {"mode": "search"}}})
    with pytest.raises(PlanRefused, match=r"有超過限制的區.*x 起點牆「z1」.*0\.15 公尺"):
        plan_search(scheme, design, GradientRequest(), SETTINGS)


def test_caps_deeper_than_the_family_take_the_stricter_and_too_shallow_caps_are_red() -> None:
    """#1082 段 2-4（取代 1-4 那一題現況釘）：個案上限比家族自己的上限大，引擎取較嚴的那一個、不拒收；
    比每一組最淺排法都淺，頁面標紅、開跑擋，同一句講是哪一條擋住。"""
    scheme = _scheme(_cut(PRESET))
    kept = scene_wall_materials(scheme, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    designs = designs_for((None,) * len(kept), SETTINGS, kept=kept)
    upper = max(family.max_total_depth_m for family in SETTINGS.family)
    assert designs_with_max_depths(designs, SETTINGS, {"x0@auto1": upper + 0.1})
    thinnest = min(family.thicknesses_m[0] for family in SETTINGS.family)
    design = _design(zones={"x0": {"auto1": {"mode": "search", "max_depth_m": thinnest / 2}}})
    assert zone_limits(_cut(PRESET), design.model_dump(mode="json"))["red"] == 0  # 沒給材料組：照 1-4 不判這一項
    result = zone_limits(_cut(PRESET), design.model_dump(mode="json"), depth_targets=lambda: TARGETS)
    assert result["red"] == 1 and _codes(result, "x0", "auto1") == ["depth_no_group"]
    refusal = cast(list[str], result["refusals"])[0]
    assert refusal.startswith("x 起點牆「自動 auto1」：准用的材料組一組都放不進深度上限")
    assert "0.025 公尺（這一塊的上限）比這一組最淺排法 0.05 公尺還淺（個案上限擋住）" in refusal
    with pytest.raises(PlanRefused, match="一組都放不進深度上限"):
        plan_search(scheme, design, GradientRequest(), SETTINGS)


def _deep_user() -> library.OptimizableEntry:
    """最淺排法 0.08 m（最薄 0.05 m＋空腔下界 0.03 m）、自己的上限 0.15 m 的使用者可最佳化材料。"""
    return cast(library.OptimizableEntry, _ENTRY.validate_python({
        "kind": "optimizable", "id": "deep_rock_wool", "label": "深背岩棉（自填）", "provenance": "model", "source": "自填",
        "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 20000.0, "material_class": "fibrous"},
        "porous_thickness": {"adjustable": True, "range_m": [0.05, 0.1], "choices_m": [0.05, 0.1], "initial_m": 0.05},
        "cavity": {"adjustable": True, "range_m": [0.03, 0.1], "initial_m": 0.04}, "max_total_depth_m": 0.15}))


def _user_targets() -> DepthTargets:
    chosen = with_user_materials(SETTINGS, load_construction_presets(config_path("construction_presets.toml")), (_deep_user(),))
    return library_depth_targets(chosen.settings, (_deep_user(),))


def test_each_group_gets_its_own_effective_cap_and_blocker() -> None:
    """抓生效上限沒有逐組取較嚴、沒寫哪一條擋住、部分放不下被當成紅。"""
    targets = _user_targets()
    design = _design(room_max_depth_m=0.2, zones={"x0": {"auto1": {"mode": "search", "max_depth_m": 0.06},
                                                         "auto2": {"mode": "search"}}})
    result = zone_limits(_cut(PRESET), design.model_dump(mode="json"), depth_targets=lambda: targets)
    assert result["red"] == 0
    caps = cast(dict[str, dict[str, dict[str, object]]], result["caps"])["x0"]
    shallow = cast(list[dict[str, object]], caps["auto1"]["targets"])
    assert [(line["target"], line["usable"]) for line in shallow] == [
        ("glass_wool_48k", True), ("glass_wool_24k", True), ("glass_wool_96k", True), ("deep_rock_wool", False)]
    assert shallow[0]["text"] == "48K 玻璃棉（可調厚度與空腔）：生效 0.06 公尺（這一塊的上限擋住；材料組自己 0.3 公尺比較寬）"
    assert shallow[3]["text"] == ("深背岩棉（自填）：不可用——深度上限 0.06 公尺（這一塊的上限）比這一組最淺排法 0.08 公尺還淺"
                                  "（個案上限擋住）")
    room = cast(list[dict[str, object]], caps["auto2"]["targets"])
    assert (room[3]["effective_m"], room[3]["blocker"]) == (0.15, "material")
    assert room[3]["text"] == "深背岩棉（自填）：生效 0.15 公尺（材料組自己的上限擋住；0.2 公尺（房間預設的上限）比較寬）"
    assert (room[0]["effective_m"], room[0]["blocker"]) == (0.2, "case")


def test_plan_drops_groups_that_do_not_fit_and_skips_them_as_start() -> None:
    """抓放不下的組照樣進 --allowed、起點挑到放不下的那一組（命令列開跑前才拒收）、清單只剩不貼時照樣開。"""
    targets = _user_targets()
    scheme = _scheme(_cut(PRESET))
    fixed_ids = frozenset(targets.fixed)
    design = _design(zones={"x0": {"auto1": {"mode": "search", "max_depth_m": 0.06,
                                             "allowed": ["deep_rock_wool", "glass_wool_24k", "glass_wool_48k_50mm_gap100", "none"]}}})
    plan = plan_search(scheme, design, GradientRequest(), targets.settings, fixed_ids, depth_targets=targets)
    assert plan.allowed == {"x0@auto1": ("glass_wool_24k", "none")}
    assert plan.presets[plan.surfaces.index("x0@auto1")] == "glass_wool_24k_50mm"
    lines = plan.depth_targets["x0@auto1"]
    assert [(line.target, line.usable) for line in lines] == [
        ("deep_rock_wool", False), ("glass_wool_24k", True), ("fixed:glass_wool_48k_50mm_gap100", False)]
    assert lines[2].text == "48K 玻璃棉 50 mm＋100 mm 空腔：不可用——總深 0.15 公尺超過深度上限 0.06 公尺（這一塊的上限）（個案上限擋住）"
    assert "--allowed" in start_arguments(plan) and "x0@auto1=glass_wool_24k,none" in start_arguments(plan)
    # 沒寫清單：選定家族放不下就挑設定檔順序第一個放得下的；清單照舊不寫（命令列換樣板時自己濾）
    free = _design(zones={"x0": {"auto1": {"mode": "search", "max_depth_m": 0.06}}})
    plan = plan_search(scheme, free, GradientRequest(family="deep_rock_wool"), targets.settings, fixed_ids, depth_targets=targets)
    assert plan.presets[plan.surfaces.index("x0@auto1")] == "glass_wool_48k_50mm_gap50" and plan.allowed == {}
    # 清單只剩不貼：不貼不算救，紅
    only = _design(zones={"x0": {"auto1": {"mode": "search", "max_depth_m": 0.06, "allowed": ["deep_rock_wool", "none"]}}})
    with pytest.raises(PlanRefused, match="x 起點牆「自動 auto1」：准用的材料組一組都放不進"):
        plan_search(scheme, only, GradientRequest(), targets.settings, fixed_ids, depth_targets=targets)


def test_a_tighter_request_cap_is_judged_too() -> None:
    """抓開跑頁另填的上限更嚴時，沒重判材料組放不放得下（命令列開跑前才拒收）。"""
    design = _design(zones={"x0": {"auto1": {"mode": "search", "max_depth_m": 0.2}}})
    with pytest.raises(PlanRefused, match="0.01 公尺（這一跑另填的上限）比這一組最淺排法"):
        plan_search(_scheme(_cut(PRESET)), design, GradientRequest(max_depths={"x0": {"auto1": 0.01}}), SETTINGS)


def test_route_judges_groups_from_the_library_and_says_when_it_cannot(tmp_path: Path) -> None:
    """抓網址沒接材料庫（使用者材料放不下不標）、材料庫壞掉時回 500 或當成放得下。"""
    (tmp_path / "materials.json").write_text(library.dump_user_library((_deep_user(),)), encoding="utf-8")
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")
    design = _design(zones={"x0": {"auto1": {"mode": "search", "max_depth_m": 0.06}}}).model_dump(mode="json")
    body = client.post("/api/zone-limits", json={"scheme": _cut(PRESET), "design": design}).json()
    lines = body["caps"]["x0"]["auto1"]["targets"]
    assert body["red"] == 0 and lines[-1]["label"] == "深背岩棉（自填）" and lines[-1]["usable"] is False
    (tmp_path / "materials.json").write_text("{壞掉", encoding="utf-8")
    response = client.post("/api/zone-limits", json={"scheme": _cut(PRESET), "design": design})
    assert response.status_code == 200 and response.json()["red"] == 1
    assert "查不了交給搜尋的區放不放得下材料組" in response.json()["refusals"][0]
    fixed_only = _design(zones={"x0": {"auto1": {"mode": "fixed"}}}).model_dump(mode="json")
    assert client.post("/api/zone-limits", json={"scheme": _cut(PRESET), "design": fixed_only}).json()["red"] == 0


# ---- 網址：內容有問題一律 200 照講 ----

def test_route_answers_drafts_with_200_and_refuses_non_json(tmp_path: Path) -> None:
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")
    response = client.post("/api/zone-limits", json={"scheme": _cut(), "design": _design(wall_max_depth_m={"x0": 0.1}).model_dump(mode="json")})
    assert response.status_code == 200 and response.json()["red"] == 1
    half = room109_document()
    cast(dict[str, object], cast(Doc, half["scene"])["room_m"])["Lz"] = "abc"
    assert client.post("/api/zone-limits", json={"scheme": half}).status_code == 200
    assert client.post("/api/zone-limits", content=b"nope", headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/api/zone-limits", json={"design": {}}).status_code == 400
    assert ROOM_109["Ly"] > 1.0


def test_the_search_page_preview_refuses_a_saved_design_with_red_zones(tmp_path: Path) -> None:
    """存好的方案裡固定區貼太深：搜尋頁要開跑參數那一步就被擋（400、原文照講），不必等按開跑。"""
    (tmp_path / "schemes").mkdir()
    (tmp_path / "designs").mkdir()
    (tmp_path / "schemes" / "room109.json").write_text(_scheme(_cut()).model_dump_json(), encoding="utf-8")
    design = _design(wall_max_depth_m={"x0": 0.1}, zones={"x0": {"auto1": {"mode": "search"}}})
    (tmp_path / "designs" / "room109.json").write_text(design.model_dump_json(), encoding="utf-8")
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")
    response = client.post("/api/gradient/schemes/room109/plan", json={})
    assert response.status_code == 400 and "有超過限制的區" in response.json()["error"]
    (tmp_path / "designs" / "room109.json").write_text(_design(wall_max_depth_m={"x0": 0.2}, zones={"x0": {"auto1": {"mode": "search"}}})
                                                     .model_dump_json(), encoding="utf-8")
    response = client.post("/api/gradient/schemes/room109/plan", json={})
    assert response.status_code == 200, response.text
    assert response.json()["plan"]["max_depths"] == {"x0@auto1": 0.2}
    assert response.json()["plan"]["depth_sources"] == {"x0@auto1": "wall"}


def test_the_real_entry_checks_the_trust_range_the_same_way_as_the_material_page(tmp_path: Path) -> None:
    """正式網頁入口開機就開雙精度：可信範圍真的算（材料頁那三個流阻值）。內建玻璃棉在範圍內、什麼都不說；
    流阻 5000 低於原研究樣品，警告不擋；3000 低於手動試算下限，物理拒算，紅。"""
    def flow(value: float) -> Doc:
        document = _cut(PRESET)
        z1 = next(zone for zone in _zones(document)["x0"] if zone["id"] == "z1")
        layers = cast(list[dict[str, object]], cast(Doc, z1["material"])["layers"])
        layers[0]["flow_resistivity_pa_s_m2"] = value
        return document

    with real_server(tmp_path) as base, httpx.Client(trust_env=False, timeout=60) as client:
        def ask(document: Doc) -> list[tuple[str, str]]:
            response = client.post(f"{base}/api/zone-limits", json={"scheme": document})
            assert response.status_code == 200, response.text
            return [(line["code"], line["level"]) for line in _lines(response.json(), "x0", "z1")]

        assert ask(_cut(PRESET)) == []
        assert ask(flow(5000.0)) == [("komatsu_flow_below_samples", "warn")]
        assert ask(flow(3000.0)) == [("range_uncomputable", "red")]


def test_building_a_surrogate_refuses_red_designs_too(tmp_path: Path) -> None:
    """建替身也走開跑同一份計畫：有紅照講、400，不是 500。"""
    (tmp_path / "schemes").mkdir()
    (tmp_path / "designs").mkdir()
    scheme = _scheme(_cut())
    scheme = scheme.model_copy(update={"scene": scheme.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.SEARCH})})
    (tmp_path / "schemes" / "room109.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    design = _design(wall_max_depth_m={"x0": 0.1}, zones={"x0": {"auto1": {"mode": "search"}}})
    (tmp_path / "designs" / "room109.json").write_text(design.model_dump_json(), encoding="utf-8")
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")
    response = client.post("/api/gradient/schemes/room109/surrogate/plan", json={})
    assert response.status_code == 400 and "有超過限制的區" in response.json()["error"]


def test_the_real_entry_refuses_to_plan_a_search_over_an_uncomputable_fixed_material(tmp_path: Path) -> None:
    """頁面上物理拒算是紅：伺服器開跑那一步也擋（同一把，帶可信範圍）；在範圍內的照常回計畫。"""
    document = _cut(PRESET)
    z1 = next(zone for zone in _zones(document)["x0"] if zone["id"] == "z1")
    cast(list[dict[str, object]], cast(Doc, z1["material"])["layers"])[0]["flow_resistivity_pa_s_m2"] = 3000.0
    (tmp_path / "schemes").mkdir()
    (tmp_path / "designs").mkdir()
    scheme = _scheme(document)    # 建替身要方案有低頻軸
    scheme = scheme.model_copy(update={"scene": scheme.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.SEARCH})})
    (tmp_path / "schemes" / "room109.json").write_text(scheme.model_dump_json(), encoding="utf-8")
    design = _design(zones={"x0": {"auto1": {"mode": "search"}}})
    (tmp_path / "designs" / "room109.json").write_text(design.model_dump_json(), encoding="utf-8")
    with real_server(tmp_path) as base, httpx.Client(trust_env=False, timeout=60) as client:
        for path in ("plan", "surrogate/plan"):
            response = client.post(f"{base}/api/gradient/schemes/room109/{path}", json={})
            assert response.status_code == 400, (path, response.text)
            assert "x 起點牆「z1」" in response.json()["error"] and "算不出吸音曲線" in response.json()["error"], path
        (tmp_path / "schemes" / "room109.json").write_text(_scheme(_cut(PRESET)).model_dump_json(), encoding="utf-8")
        # 在範圍內就過了限制這一關（考卷行程連不上使用者的服務匯流排，下一步查「現在有沒有一跑在算」會回 503，那不是這一段的事）
        again = client.post(f"{base}/api/gradient/schemes/room109/plan", json={})
        assert "有超過限制的區" not in again.text and "算不出吸音曲線" not in again.text, again.text


def test_sources_stay_when_requests_tighten_every_design_cap() -> None:
    """設計設定有給上限、請求把每一區都收得更嚴：照記「請求」，不當成沒給過（審查 #1111 抓到的）。"""
    plan = plan_search(_scheme(_cut(PRESET)), _design(room_max_depth_m=0.1, zones={"x0": {"auto1": {"mode": "search"}}}),
                       GradientRequest(max_depths={"x0": {"auto1": 0.05}}), SETTINGS)
    assert plan.max_depths == {"x0@auto1": 0.05} and plan.depth_sources == {"x0@auto1": "request"}


def test_a_zone_list_with_a_non_object_is_reported_not_raised() -> None:
    document = _cut(PRESET)
    _zones(document)["x0"].append(cast(dict[str, object], "z9"))
    result = zone_limits(document)
    assert [line["code"] for line in cast(dict[str, list[dict[str, str]]], result["walls"])["x0"]] == ["unreadable"]
    assert result["red"] == 1


def test_refusals_name_walls_and_zones_as_the_page_shows_them() -> None:
    """擋下的原文（表上方那行總述、開跑被擋的回覆）用畫面上的牆名與區名，不露 x0、whole_wall 這類內部代號。"""
    import re
    document = _cut()                                                  # x0：z1 貼太深
    whole = _zones(document)["y0"][0]                                  # y0 沒切：整面貼太深
    whole["material"] = {"kind": "mounted_on_wall", "layers": _layers(DEEP), "wall": whole["material"]}
    cast(dict[str, float], cast(Doc, document["scene"])["room_m"])["Ly"] = 0.9   # x 起點牆變窄：自動補的那塊出界
    result = zone_limits(document, _design(room_max_depth_m=0.1).model_dump(mode="json"))
    refusals = cast(list[str], result["refusals"])
    assert any(line.startswith("x 起點牆「z1」：") for line in refusals)
    assert any(line.startswith("y 起點牆「整面」：") for line in refusals)
    assert any(line.startswith("x 起點牆「自動 auto1」：超出牆面") for line in refusals)
    assert not [line for line in refusals if re.search(r"\b(x0|xL|y0|yL|floor|ceiling|whole_wall)\b", line)], refusals
    with pytest.raises(PlanRefused) as refused:
        plan_search(_scheme(_cut()), _design(wall_max_depth_m={"x0": 0.1}, zones={"x0": {"auto1": {"mode": "search"}}}),
                    GradientRequest(), SETTINGS)
    assert "x 起點牆「z1」" in str(refused.value) and "x0" not in str(refused.value)
