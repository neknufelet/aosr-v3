"""可調 VPR 的材料格式（#1200 第 1 支）考卷：新的一種材料存得進、讀得回、轉得成家族；舊的東西一個位元都不變。

格式是通用的「可調多層構造」（``tunable_layers``：初值排法的層＋哪一層的哪一格可調），這一支只收、只考 ``structure = "vpr"``。

兩半：
- 逐位不變：內建材料、一份自編的使用者材料庫、家族快照（v1、v2）、接上材料庫的搜尋設定與指紋、舊材料的搜尋計畫、
  材料頁清單（``/api/materials``）、分區頁限制檢查（``/api/zone-limits``）——每一樣的 SHA-256 與長度釘在 ``PINNED``
  （在主線 aff9e22e 的產品程式上用同一支 :func:`artifacts` 存的）。材料庫裡多一筆可調 VPR 時同樣這幾樣也要等於釘的值
  （使用者材料庫與材料頁清單先拿掉那一筆再比）。
- 新格式：每一條拒收理由一題；轉家族；交給第三階段時計畫那一步用白話拒收。

數字全部自編（鋼板 1.5 mm、背襯 8 cm、流阻 9000），不是任何真房間或文獻的值。
"""
from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from pydantic import TypeAdapter, ValidationError
from starlette.testclient import TestClient

from aosr.config import material_gradient
from aosr.config.construction_presets import load_construction_presets
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import GradientJobs
from aosr.gui.gradient_plan import GradientRequest, PlanRefused
from aosr.gui.material_editor import user_library_path
from aosr.gui.zone_design import DesignSettings, ZoneDesign
from aosr.materials import library, search_library
from aosr.materials.construction_models import Assembly
from tests.engine.test_gui_gradient_jobs import FakeLauncher, _scheme
from tests.engine.test_gui_zone_limits import PRESET, _cut
from tests.engine.test_gui_zone_limits import _design as _limits_design

Doc = dict[str, object]
_ENTRY: TypeAdapter[library.LibraryEntry] = TypeAdapter(library.LibraryEntry)
VPR = "steel_vpr"
WOOL = "user_wool_32k"
PANEL = "user_panel"
REFUSAL = "材料「鋼板 VPR（自編）」（steel_vpr）是可調 VPR：還不能交給電腦挑，搜尋那一層還沒接（#1200 第 2 支）"

# 主線 aff9e22e 的產品程式上 artifacts(…, with_vpr=False) 每一樣的（SHA-256, 位元組數）；樹外
# baseline/ 存的是同一批位元組。
PINNED: dict[str, tuple[str, int]] = {
    "builtin.json": ("2886008af20f124a749debf5c3eb16381d7e45817e9cb707c2f0530c607f5a3e", 32852),
    "user_library.json": ("dfdbb6d69d52d79dd1a35fba1fe62c7644e5554c713c4ab27ed5f7f7cc99f9d5", 3181),
    "snapshot_v1.json": ("cbb87d20d9335d04fdb1ca08f78cc054b44109ad28faaa5210fcb955e8ed9477", 1014),
    "snapshot_v2.json": ("a8711bcf8dc5580f4eef70b9bcf86b776077d82d7a689bc8235359c6306595a2", 2461),
    "search_materials.json": ("e779070cfddb3dd095b23eccd93655df06388c0126cbdd516be17038fd76916c", 10428),
    "plan.json": ("6a743d091e87f5d333f25931f2f7cf106ee90a99dc49ccc1e4d490b253adb831", 2442),
    "api_materials_rows.json": ("5966007c1f8495b136dd94b813d3a00f5413acea163a6b634edba91fa3610931", 44352),
    "api_materials_old_kind_labels.json": ("3bcd2f32eaaaca4e02ca9633ab80d1fefabdabb428e044fb335d78b9c32bdf8c", 69),
    "api_materials_rest.json": ("f39da21a44d281354d0469e793177a77a8bbfab3f07550becfa307983d10391d", 14634),
    "api_zone_limits.json": ("c481a05b047016e09044d100b8b184b0eade65002be412f7b7c596cefb0081ff", 3446),
}


def _wool() -> Doc:
    return {"kind": "optimizable", "id": WOOL, "label": "32K 棉（自編）", "provenance": "model", "source": "考卷自編",
            "porous": {"kind": "porous", "thickness_m": 0.05, "flow_resistivity_pa_s_m2": 11000.0, "material_class": "fibrous"},
            "porous_thickness": {"adjustable": True, "range_m": [0.025, 0.1], "choices_m": [0.025, 0.05, 0.1], "initial_m": 0.05},
            "cavity": {"adjustable": True, "range_m": [0.0, 0.1], "initial_m": 0.02},
            "max_total_depth_m": 0.2, "suggested_total_depth_m": 0.1}


def _panel() -> Doc:
    return {"kind": "fixed", "id": PANEL, "label": "薄吸音板（自編）", "provenance": "model", "source": "考卷自編",
            "material": {"kind": "assembly", "backing": "rigid", "layers": [
                {"kind": "porous", "thickness_m": 0.03, "flow_resistivity_pa_s_m2": 15000.0, "material_class": "fibrous"}]}}


def _table() -> Doc:
    rows = [{"frequency_hz": 100.0, "real_pa_s_m": 950.0}, {"frequency_hz": 400.0, "real_pa_s_m": 620.0}]
    return {"kind": "fixed", "id": "user_table", "label": "實測板（自編，只有實部）", "provenance": "measured", "source": "考卷自編",
            "material": {"kind": "measured_impedance", "points": rows, "source_and_conditions": "考卷自編",
                         "base_assembly": {"kind": "assembly", "backing": "rigid", "layers": [{"kind": "air", "thickness_m": 0.04}]}}}


def _fixed_vpr() -> Doc:
    """今天就存得進來的固定 VPR（薄板層＋填了骨架兩格的多孔層）：新的一種材料不准動到它。"""
    return {"kind": "fixed", "id": "user_fixed_vpr", "label": "固定 VPR（自編）", "provenance": "model", "source": "考卷自編",
            "material": {"kind": "assembly", "backing": "rigid", "layers": [
                {"kind": "board", "surface_mass_kg_m2": 11.775, "resistance_pa_s_m": 0.0}, _backing()]}}


def _backing(thickness_m: float = 0.08) -> Doc:
    return {"kind": "porous", "thickness_m": thickness_m, "flow_resistivity_pa_s_m2": 9000.0, "material_class": "foam",
            "frame_compression_modulus_pa": 60000.0, "frame_loss_factor": 0.1}


def _plate(mass_kg_m2: float = 11.775) -> Doc:
    return {"kind": "board", "surface_mass_kg_m2": mass_kg_m2, "resistance_pa_s_m": 0.0}


def _mass(**changes: object) -> Doc:
    """板重這一格：鋼板 1.5 mm（密度 7850 → 11.775 kg/m²），可調 1～3 mm 的板重、沒給規格（連續）。"""
    return {"layer": 0, "field": "surface_mass_kg_m2", "range": [7.85, 23.55], "initial": 11.775} | changes


def _thickness(**changes: object) -> Doc:
    """背襯厚度這一格：8 cm，可調 6／8／10 cm。"""
    return {"layer": 1, "field": "thickness_m", "range": [0.06, 0.1], "choices": [0.06, 0.08, 0.1], "initial": 0.08} | changes


def _vpr(**changes: object) -> Doc:
    """可調 VPR：鋼板＋黏住的背襯，兩格都能調；總深上限 12 cm。"""
    entry: Doc = {
        "kind": "tunable_layers", "id": VPR, "label": "鋼板 VPR（自編）", "provenance": "model", "source": "考卷自編",
        "structure": "vpr", "layers": [_plate(), _backing()], "variables": [_mass(), _thickness()],
        "plate_density_kg_m3": 7850.0, "max_total_depth_m": 0.12}
    entry.update(changes)
    return entry


def _old_user() -> list[Doc]:
    return [_wool(), _panel(), _table(), _fixed_vpr()]


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_library(path: Path, with_vpr: bool) -> None:
    """直接寫檔（不走 ``dump_user_library``）：讀回、寫出才是被考的那一半。可調 VPR 夾在舊材料中間。"""
    old = _old_user()
    materials = [*old[:2], _vpr(), *old[2:]] if with_vpr else old
    path.write_text(json.dumps({"schema_version": library.SCHEMA_VERSION, "materials": materials}, ensure_ascii=False),
                    encoding="utf-8")


def _without_vpr(document: Doc) -> Doc:
    rows = cast(list[Doc], document["materials"])
    return document | {"materials": [row for row in rows if cast(Doc, row.get("entry", row)).get("kind") != "tunable_layers"]}


def _plan_design() -> DesignSettings:
    """兩個搜尋區：一區准用清單點名內建家族、使用者可最佳化材料、使用者固定材料與不貼；另一區沒寫清單（全部家族）。"""
    return DesignSettings(scheme_id="zoned", zones={
        "x0": {"r0_c0": ZoneDesign(mode="search", allowed=("glass_wool_48k", WOOL, PANEL, "none")),
               "r1_c1": ZoneDesign(mode="search")},
        "ceiling": {"r0_c1": ZoneDesign(mode="keep_original")}})


def _jobs(data_dir: Path) -> GradientJobs:
    return GradientJobs(data_dir, launcher=FakeLauncher(), engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())


def artifacts(data_dir: Path, *, with_vpr: bool) -> dict[str, bytes]:
    """要逐位守住的每一樣（名字 → 位元組）；``with_vpr`` 時材料庫多一筆可調 VPR，使用者材料庫與材料頁清單拿掉那一筆再比。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = user_library_path(data_dir)
    _write_library(path, with_vpr)
    found: dict[str, bytes] = {}
    # 內建材料：dump_user_library 不收內建（代號會撞自己），照它同一套 JSON 寫法直接寫。
    found["builtin.json"] = _canonical([entry.model_dump(mode="json") for entry in library.builtin_entries()])
    entries = library.load_user_library(path)
    found["user_library.json"] = _canonical(_without_vpr(json.loads(library.dump_user_library(entries))))
    presets = load_construction_presets(config_path("construction_presets.toml"))
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    materials = search_library.with_user_materials(settings, presets, entries)
    fixed = search_library.user_fixed_entries([PANEL, "user_table"], entries)
    found["snapshot_v1.json"] = search_library.snapshot_bytes(materials.entries)
    found["snapshot_v2.json"] = search_library.snapshot_bytes(materials.entries, fixed)
    found["search_materials.json"] = _canonical({
        "settings": materials.settings.model_dump(mode="json"), "fingerprint": materials.settings.fingerprint,
        "full_fingerprint": materials.settings.full_fingerprint,
        "presets": [{"id": preset.id, "material": preset.material} for preset in materials.presets],
        "entries": [entry.id for entry in materials.entries]})
    plan, planned, _ = _jobs(data_dir).plan(_scheme(), _plan_design(), GradientRequest())
    found["plan.json"] = _canonical({"plan": plan.model_dump(mode="json"), "families": [f.id for f in planned.settings.family],
                                     "fingerprint": planned.settings.fingerprint})
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir)), base_url="http://127.0.0.1")
    listing = client.get("/api/materials")
    assert listing.status_code == 200, listing.text
    body = _without_vpr(cast(Doc, listing.json()))
    found["api_materials_rows.json"] = _canonical(body.pop("materials"))
    labels = cast(dict[str, str], body.pop("entry_kind_labels"))
    found["api_materials_old_kind_labels.json"] = _canonical({kind: labels[kind] for kind in ("optimizable", "fixed")})
    found["api_materials_rest.json"] = _canonical(body)
    design = _limits_design(room_max_depth_m=0.2, zones={
        "x0": {"auto1": {"mode": "search", "max_depth_m": 0.06}, "auto2": {"mode": "search", "allowed": [WOOL, PANEL, "none"]}}})
    limits = client.post("/api/zone-limits", json={"scheme": _cut(PRESET), "design": design.model_dump(mode="json")})
    assert limits.status_code == 200, limits.text
    found["api_zone_limits.json"] = limits.content
    return found


def _digests(found: dict[str, bytes]) -> dict[str, tuple[str, int]]:
    return {name: (hashlib.sha256(content).hexdigest(), len(content)) for name, content in found.items()}


# ---- 逐位不變 ----

def test_old_library_snapshot_plan_and_pages_are_byte_identical_to_main(tmp_path: Path) -> None:
    """沒有可調 VPR 的材料庫：每一樣都等於主線存的那一份。"""
    assert _digests(artifacts(tmp_path / "data", with_vpr=False)) == PINNED


def test_adding_a_tunable_vpr_changes_no_old_byte(tmp_path: Path) -> None:
    """材料庫多一筆可調 VPR：舊材料的讀寫、快照、搜尋設定與指紋、計畫、材料頁的舊列、分區頁限制檢查都等於主線那一份。"""
    assert _digests(artifacts(tmp_path / "data", with_vpr=True)) == PINNED


# ---- 新格式：讀、寫、一筆長什麼樣 ----

def test_a_tunable_vpr_loads_dumps_and_reloads_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "materials.json"
    _write_library(path, with_vpr=True)
    entries = library.load_user_library(path)
    entry = next(item for item in entries if item.id == VPR)
    assert entry.kind == "tunable_layers" and entry.model_dump(mode="json") == _vpr() | {
        "links": [], "notes": [], "layers": [_plate(), _backing() | {"model": "komatsu"}],
        "variables": [_mass(choices=None), _thickness()]}
    text = library.dump_user_library(entries)
    path.write_text(text, encoding="utf-8")
    assert library.load_user_library(path) == entries and library.dump_user_library(library.load_user_library(path)) == text
    assert library.basis_label(entry) == "模擬，之後要拿實測對照"


_FRONT: list[Doc] = [{"kind": "slotted", "thickness_m": 0.012, "slot_width_m": 0.004, "open_area": 0.2},
                     {"kind": "porous", "thickness_m": 0.01, "flow_resistivity_pa_s_m2": 8000.0, "material_class": "fibrous"}]


def _faced(**changes: object) -> Doc:
    """前面多兩層固定層（長縫板＋薄棉）：鋼板與背襯變成第 3、4 層，變數的層號跟著走。"""
    return _vpr(layers=[*_FRONT, _plate(), _backing()], variables=[_mass(layer=2), _thickness(layer=3)],
                max_total_depth_m=0.2) | changes


def test_depth_counts_plate_backing_and_fixed_front_layers() -> None:
    """總深＝固定前層＋板厚（板重除以鋼板密度）＋背襯；範圍照兩格各自的上下限。"""
    bare = cast("library.TunableLayersEntry", _ENTRY.validate_python(_vpr()))
    assert bare.depth_bounds() == pytest.approx((0.001 + 0.06, 0.003 + 0.1), rel=1e-12)
    assert bare.initial_depth_m() == pytest.approx(0.0015 + 0.08, rel=1e-12)
    faced = cast("library.TunableLayersEntry", _ENTRY.validate_python(_faced()))
    assert faced.depth_bounds() == pytest.approx((0.022 + 0.001 + 0.06, 0.022 + 0.003 + 0.1), rel=1e-12)
    assembly = library.initial_assembly(faced)
    assert isinstance(assembly, Assembly) and assembly.backing == "rigid" and assembly.layers == faced.layers
    assert [layer.kind for layer in assembly.layers] == ["slotted", "porous", "board", "porous"]


def test_only_the_listed_cells_move_and_one_is_enough() -> None:
    """沒列成變數的格子不調：只調板重、只調背襯、板重給規格三種都收；不調的那一格總深照初值算。"""
    stocked = _vpr(variables=[_mass(range=[7.85, 15.7], choices=[7.85, 11.775, 15.7]), _thickness()])
    for document in (_vpr(variables=[_mass()]), _vpr(variables=[_thickness()]), stocked):
        assert _ENTRY.validate_python(document).kind == "tunable_layers"
    plate_only = cast("library.TunableLayersEntry", _ENTRY.validate_python(_vpr(variables=[_mass()])))
    assert plate_only.depth_bounds() == pytest.approx((0.001 + 0.08, 0.003 + 0.08), rel=1e-12)


_PLAIN_BACKING = {key: value for key, value in _backing().items() if not key.startswith("frame_")}

REFUSALS: list[tuple[str, Doc, str]] = [
    ("range_not_positive", {"variables": [_mass(range=[0.0, 23.55]), _thickness()]}, "第 1 層的 surface_mass_kg_m2：可調範圍要 0 < 下限 < 上限"),
    ("range_reversed", {"variables": [_mass(range=[23.55, 7.85]), _thickness()]}, "第 1 層的 surface_mass_kg_m2：可調範圍要 0 < 下限 < 上限"),
    ("range_is_one_point", {"variables": [_mass(range=[11.775, 11.775]), _thickness()]}, "可調範圍要 0 < 下限 < 上限"),
    ("choices_unsorted", {"variables": [_mass(choices=[23.55, 7.85]), _thickness()]}, "產品規格要是正數、由小到大、不重複"),
    ("choices_off_range", {"variables": [_mass(choices=[7.85, 11.775, 15.7]), _thickness()]}, "有產品規格時，範圍就是最小到最大"),
    ("initial_off_range", {"layers": [_plate(31.4), _backing()], "variables": [_mass(initial=31.4), _thickness()]},
     "初值 31.4 不在範圍"),
    ("initial_off_choices", {"variables": [_mass(choices=[7.85, 15.7, 23.55]), _thickness()]}, "初值 11.775 不是產品規格"),
    ("no_variables", {"variables": []}, "tunable_layers.variables\n  Tuple should have at least 1 item"),
    ("variable_layer_missing", {"variables": [_mass(), _thickness(layer=2)]}, "變數指的第 3 層的 thickness_m不存在：這份構造只有 2 層"),
    ("variable_layer_negative", {"variables": [_mass(layer=-1), _thickness()]}, "變數指的第 0 層的 surface_mass_kg_m2不存在"),
    ("variable_listed_twice", {"variables": [_mass(), _thickness(), _thickness()]}, "第 2 層的 thickness_m列了兩次"),
    ("backing_not_bonded", {"layers": [_plate(), _PLAIN_BACKING]}, "可調 VPR 的最後一層（最靠牆）要是填了骨架壓縮模數與骨架損耗的多孔層"),
    ("cell_not_tunable_in_a_vpr", {"variables": [_mass(), {"layer": 1, "field": "flow_resistivity_pa_s_m2",
                                                           "range": [5000.0, 12000.0], "initial": 9000.0}]},
     "可調 VPR 能調的只有鋼板（倒數第二層）的板重"),
    ("plate_density_missing", {"plate_density_kg_m3": None}, "可調 VPR 要填鋼板密度 plate_density_kg_m3"),
    ("plate_density_not_positive", {"plate_density_kg_m3": 0.0}, "tunable_layers.plate_density_kg_m3\n  Input should be greater than 0"),
    ("backing_without_choices", {"variables": [_mass(), _thickness(choices=None)]}, "能調的背襯厚度要給產品規格"),
    ("layer_value_not_initial", {"layers": [_plate(15.7), _backing()]}, "第 1 層的 surface_mass_kg_m2寫的是 15.7，要等於這個變數的初值 11.775"),
    ("backing_value_not_initial", {"layers": [_plate(), _backing(0.1)]}, "第 2 層的 thickness_m寫的是 0.1，要等於這個變數的初值 0.08"),
    ("layer_before_backing_is_not_a_plate", {"layers": [_plate(), _FRONT[1], _backing()],
                                             "variables": [_thickness(layer=2)]}, "初值排法不是合法的多層構造"),
    ("plate_variable_on_a_layer_that_is_not_a_plate", {"layers": [_plate(), _FRONT[1], _backing()],
                                                       "variables": [_mass(layer=1), _thickness(layer=2)]},
     "初值排法不是合法的多層構造"),
    ("no_layers", {"layers": []}, "tunable_layers.layers\n  Tuple should have at least 1 item"),
    ("thinnest_over_cap", {"max_total_depth_m": 0.05}, "最薄排法的總深"),
    ("initial_over_cap", {"max_total_depth_m": 0.07}, "初值總深"),
    ("cap_not_positive", {"max_total_depth_m": 0.0}, "tunable_layers.max_total_depth_m\n  Input should be greater than 0"),
    ("structure_unknown", {"structure": "slotted_panel"}, "tunable_layers.structure\n  Input should be 'vpr'"),
]


@pytest.mark.parametrize(("changes", "message"), [pytest.param(changes, message, id=name) for name, changes, message in REFUSALS])
def test_every_refusal_of_the_tunable_vpr_format(changes: Doc, message: str) -> None:
    """每一條拒收理由一題：訊息要是那一條自己的（別條剛好也擋下不算）。"""
    with pytest.raises(ValidationError) as caught:
        _ENTRY.validate_python(_vpr(**changes))
    assert message in str(caught.value), str(caught.value)


def test_the_optimizable_kind_still_refuses_a_bonded_backing() -> None:
    """不放寬可最佳化材料：多孔層填了骨架兩格照舊拒收，句子一個字不變。"""
    with pytest.raises(ValidationError, match="可最佳化材料的多孔層不能填骨架壓縮模數與骨架損耗"):
        _ENTRY.validate_python(_wool() | {"porous": _backing(0.05)})


# ---- 轉成家族 ----

def test_a_tunable_vpr_becomes_its_own_family_outside_the_search_settings() -> None:
    """可調 VPR 的家族是另一種（板重範圍＋背襯合法厚度＋總深上限），不進搜尋設定的家族清單：舊家族與指紋不動。"""
    entry = cast("library.TunableLayersEntry", _ENTRY.validate_python(_vpr()))
    family, preset = search_library.vpr_family(entry)
    assert isinstance(family, material_gradient.VprFamilySettings)
    assert family.model_dump(mode="json") == {
        "id": VPR, "plate_density_kg_m3": 7850.0, "plate_mass_range_kg_m2": [7.85, 23.55], "plate_mass_choices_kg_m2": None,
        "initial_plate_mass_kg_m2": 11.775, "backing_thicknesses_m": [0.06, 0.08, 0.1], "initial_backing_thickness_m": 0.08,
        "front_depth_m": 0.0, "max_total_depth_m": 0.12, "template_preset": VPR}
    assert preset.id == VPR and preset.material == library.initial_assembly(entry).model_dump(mode="json")
    plate_only = cast("library.TunableLayersEntry", _ENTRY.validate_python(_faced(variables=[_mass(layer=2)], max_total_depth_m=None)))
    loose, _ = search_library.vpr_family(plate_only)
    assert loose.backing_thicknesses_m == (0.08,) and loose.front_depth_m == pytest.approx(0.022, rel=1e-12)
    assert loose.max_total_depth_m == pytest.approx(0.022 + 0.003 + 0.08, rel=1e-12)
    backing_only = cast("library.TunableLayersEntry", _ENTRY.validate_python(_vpr(variables=[_thickness()])))
    assert search_library.vpr_family(backing_only)[0].plate_mass_range_kg_m2 == (11.775, 11.775)
    presets = load_construction_presets(config_path("construction_presets.toml"))
    settings = load_material_gradient_settings(config_path("material_gradient.toml"))
    merged = search_library.with_user_materials(settings, presets, (entry,))
    assert merged.settings == settings and merged.entries == () and list(merged.presets) == list(presets)


# ---- 交給第三階段：計畫那一步白話拒收 ----

def _plan(tmp_path: Path, design: DesignSettings, request: GradientRequest) -> Callable[[], object]:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_library(user_library_path(data_dir), with_vpr=True)
    return lambda: _jobs(data_dir).plan(_scheme(), design, request)


def test_plan_refuses_a_tunable_vpr_in_an_allowed_list_in_plain_words(tmp_path: Path) -> None:
    design = DesignSettings(scheme_id="zoned", zones={
        "x0": {"r0_c0": ZoneDesign(mode="search", allowed=("glass_wool_48k", VPR, "none")), "r1_c1": ZoneDesign(mode="search")}})
    with pytest.raises(PlanRefused) as caught:
        _plan(tmp_path, design, GradientRequest())()
    assert str(caught.value) == REFUSAL + "；先從准用清單或選的材料組拿掉它，其他材料照舊可以交給電腦挑"


def test_plan_refuses_a_tunable_vpr_picked_as_the_family(tmp_path: Path) -> None:
    with pytest.raises(PlanRefused, match=REFUSAL.replace("（", r"\（").replace("）", r"\）")):
        _plan(tmp_path, _plan_design(), GradientRequest(family=VPR))()


def test_plan_of_other_materials_goes_through_with_a_tunable_vpr_in_the_library(tmp_path: Path) -> None:
    plan, materials, entries = cast(tuple[object, search_library.SearchMaterials, tuple[library.LibraryEntry, ...]],
                                    _plan(tmp_path, _plan_design(), GradientRequest())())
    assert VPR in {entry.id for entry in entries} and VPR not in {family.id for family in materials.settings.family}
    assert plan is not None


# ---- 網頁：材料頁列得出來、分區不能指定它 ----

def test_material_page_lists_the_tunable_vpr_with_its_kind_label_and_curve_inputs(tmp_path: Path) -> None:
    _write_library(user_library_path(tmp_path), with_vpr=True)
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")
    listing = client.get("/api/materials").json()
    assert listing["user_file_problem"] is None and listing["entry_kind_labels"]["tunable_layers"] == "可調多層構造"
    row = next(row for row in listing["materials"] if row["entry"]["id"] == VPR)
    assert row["builtin"] is False and row["basis"] == "模擬，之後要拿實測對照" and row["placement"]["group"] == "wall"


def test_a_zone_cannot_be_pinned_to_a_tunable_vpr(tmp_path: Path) -> None:
    """分區「指定貼一種」點到可調 VPR：照講還不能用，不是 500、也不說它是可最佳化材料。"""
    _write_library(user_library_path(tmp_path), with_vpr=True)
    client = TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)), base_url="http://127.0.0.1")
    reply = client.post("/api/zone-material", json={"scheme": _cut(PRESET), "wall": "x0", "zone": "z1", "choice": "fixed",
                                                    "material_id": VPR})
    assert reply.status_code == 400 and "是可調 VPR" in reply.json()["error"], reply.text
