"""由程式最佳化的區背後換成材料庫一筆（#1330，VPR 固定當背牆、程式只調前面的層）：伺服器那半邊。

- 設計設定兩格（背後哪一筆、換之前的原牆）：同有同無、只給由程式最佳化的區、原牆不准是貼在牆上的構造；舊檔逐位不變。
- 換分區材料一批（``/api/zone-material/batch``）收 ``backing``：方案那一區放那一筆整塊材料、原牆另回；之後換回原牆、
  改指定貼都照記的那一份原牆，不把背後那一筆當原牆。
- 深度：最深從原牆往房間量、含背後那一筆，開跑計畫與分區檢查的前層上限都扣掉它；最深比背後還淺就擋、只講那一句。
- 起跑配置：搜尋那一面的背材逐欄等於選的那一筆。
- 左右對稱：兩區背後不同不綁。

房間是考卷自己編的（``_surface_zone_cases.zone_document`` 改尺寸與牆面阻抗；``_zone_symmetry_room``）。
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.material_gradient import load_material_gradient_settings
from aosr.config.paths import config_path
from aosr.gradient.cli import start_designs
from aosr.gradient.start_config import presets_table
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_plan import SearchPlan
from aosr.gui.zone_design import DesignSettings, ZoneDesign
from aosr.gui.zone_symmetry import pair_zones
from aosr.materials.library import builtin_entries, fixed_surface
from aosr.reporting.scheme import Scheme
from tests.engine._construction_cases import construction
from tests.engine._construction_preset_answers import FITTED_VPR_IDS
from tests.engine._directivity import DIRECTIVITY
from tests.engine._furniture_cases import CAPABILITIES
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import zone_document
from tests.engine._zone_symmetry_room import SIDE, searching, symmetry_document
from tests.engine.test_gui_gradient_jobs import FakeLauncher

VPR, = FITTED_VPR_IDS
WOOL = "glass_wool_48k"
SETTINGS = load_material_gradient_settings(config_path("material_gradient.toml"))
Doc = dict[str, object]


def _vpr() -> Doc:
    entry, = [item for item in builtin_entries() if item.id == VPR]
    material = fixed_surface(entry)
    assert material is not None
    return cast(Doc, material.model_dump(mode="json"))


def _room() -> Doc:
    document = zone_document()
    document["scheme_id"] = "room"
    scene = cast(Doc, document["scene"])
    scene.update(room_m={"Lx": 6.4, "Ly": 4.3, "Lz": 2.85}, low_frequency_axis=LowFrequencyAxis.SEARCH.value)
    for index, zones in enumerate(cast(dict[str, list[Doc]], scene["surface_zones_by_wall"]).values()):
        zones[0]["material"] = {"kind": "constant_impedance", "impedance_pa_s_m": 1700.0 + 35 * index}
    return document


def _front(document: Doc) -> Doc:
    return cast(dict[str, list[Doc]], cast(Doc, document["scene"])["surface_zones_by_wall"])["x0"][0]


class _Gui:
    def __init__(self, folder: Path) -> None:
        settings = GuiSettings(engine_commit="a49d00f7b8083bb62fe9e3a50dc04fe631538ffa", data_dir=folder, gradient_launcher=FakeLauncher())
        self.client = TestClient(create_app(settings), base_url="http://127.0.0.1")

    def call(self, method: str, url: str, body: object = None, code: int = 200) -> Doc:
        response = self.client.request(method, url, json=body)
        assert response.status_code == code, (url, response.status_code, response.json())
        return cast(Doc, response.json())

    def change(self, document: Doc, **choice: object) -> Doc:
        body = {"scheme": document, "changes": [{"wall": "x0", "zone": "whole_wall", **choice}]}
        return cast(list[Doc], self.call("POST", "/api/zone-material/batch", body)["changes"])[0]

    def design(self, cap: float, backed: Doc | None) -> None:
        item: Doc = {"mode": "search", "allowed": [WOOL, "none"], "max_depth_m": cap}
        if backed is not None:
            item.update(backing_entry=backed["backing_entry"], backing_wall=backed["backing_wall"])
        self.call("PUT", "/api/designs/room", {"scheme_id": "room", "zones": {"x0": {"whole_wall": item}}})


Backed = tuple[_Gui, Doc, Doc, Doc]


@pytest.fixture
def backed(tmp_path: Path) -> Backed:
    """前牆整面換上 VPR 背後的方案（已存）、那一批回的那一區、換之前的原牆。"""
    gui = _Gui(tmp_path)
    document = _room()
    wall = cast(Doc, deepcopy(_front(document)["material"]))
    gui.call("PUT", "/api/schemes/room", document)
    got = gui.change(document, choice="backing", material_id=VPR)
    _front(document)["material"] = got["material"]
    gui.call("PUT", "/api/schemes/room", document)
    return gui, document, got, wall


# ---- 設計設定兩格 ----

def test_backing_fields_come_together_only_for_search_and_never_hold_a_mounted_wall() -> None:
    wall = {"kind": "constant_impedance", "impedance_pa_s_m": 1700.0}
    ZoneDesign.model_validate({"mode": "search", "backing_entry": VPR, "backing_wall": wall})
    with pytest.raises(ValidationError, match="一起給"):
        ZoneDesign.model_validate({"mode": "search", "backing_entry": VPR})
    with pytest.raises(ValidationError, match="只給由程式最佳化的區"):
        ZoneDesign.model_validate({"mode": "fixed", "backing_entry": VPR, "backing_wall": wall})
    mounted = {"kind": "mounted_on_wall", "layers": _vpr()["layers"], "wall": wall}
    with pytest.raises(ValidationError, match="不能是貼在牆上的構造"):
        ZoneDesign.model_validate({"mode": "search", "backing_entry": VPR, "backing_wall": mounted})


def test_old_design_without_backing_round_trips_byte_identical() -> None:
    old = ('{"schema_version":"aosr.zone_design.v1","scheme_id":"room","zones":{"x0":{"whole_wall":{"mode":"search",'
           '"allowed":["glass_wool_48k","none"],"max_depth_m":0.25}}},"original_scheme_id":null}')
    assert DesignSettings.model_validate_json(old).model_dump_json() == old


# ---- 換背後、換回原牆 ----

def test_backing_puts_the_entry_into_the_zone_and_returns_the_original_wall(backed: Backed) -> None:
    _, _, got, wall = backed
    assert got["material"] == _vpr()
    assert got["backing_entry"] == VPR
    assert got["backing_wall"] == wall
    assert "程式只調前面的層" in cast(str, got["note"])


def test_switching_back_restores_the_remembered_wall_not_the_backing(backed: Backed) -> None:
    gui, document, got, wall = backed
    back = gui.change(document, choice="original", backing_wall=got["backing_wall"])
    assert back["material"] == wall
    assert back["origin"] == "unwrapped"
    assert "backing_entry" not in back


def test_fixed_after_a_backing_mounts_on_the_remembered_wall(backed: Backed) -> None:
    gui, document, got, wall = backed
    fixed = cast(Doc, gui.change(document, choice="fixed", material_id="glass_wool_48k_50mm", backing_wall=got["backing_wall"])["material"])
    assert fixed["kind"] == "mounted_on_wall"
    assert fixed["wall"] == wall


def test_unchanged_backing_keeps_the_saved_material_even_if_the_library_changed(backed: Backed) -> None:
    gui, document, got, wall = backed
    saved = deepcopy(_vpr())
    cast(list[Doc], saved["layers"])[1]["thickness_m"] = 0.12  # 存檔後材料庫同代號改過：方案裡那一份跟材料庫的不一樣
    _front(document)["material"] = saved
    kept = gui.change(document, choice="backing", material_id=VPR, backing_wall=got["backing_wall"], keep_backing=True)
    assert kept["material"] == saved and kept["backing_wall"] == wall
    renewed = gui.change(document, choice="backing", material_id=VPR, backing_wall=got["backing_wall"])
    assert renewed["material"] == _vpr()


def test_own_rigid_construction_works_as_a_backing_end_to_end(tmp_path: Path) -> None:
    """自己組的剛背構造（材料庫「我組的」，走 construction_pick 那條路）當背後：放進方案、扣最深、限制檢查不說看不出、換回原牆。"""
    gui = _Gui(tmp_path)
    own = construction(id="lab_rigid", label="自編剛背共振板")
    gui.call("PUT", "/api/materials/lab_rigid", own)
    document = _room()
    wall = cast(Doc, deepcopy(_front(document)["material"]))
    gui.call("PUT", "/api/schemes/room", document)
    got = gui.change(document, choice="backing", material_id="lab_rigid")
    assert got["material"] == {"kind": "assembly", "layers": own["layers"], "backing": "rigid"}
    _front(document)["material"] = got["material"]
    gui.call("PUT", "/api/schemes/room", document)
    gui.design(0.25, got)
    plan = SearchPlan.model_validate(gui.call("POST", "/api/gradient/schemes/room/plan", {})["plan"])
    assert plan.max_depths["x0"] == 0.187  # 0.25－布（沒有厚度欄）－0.063 空氣層
    limits = gui.call("POST", "/api/zone-limits", {"scheme": document, "design": gui.call("GET", "/api/designs/room")["design"]})
    lines = cast(dict[str, dict[str, list[Doc]]], limits.get("zones", {})).get("x0", {}).get("whole_wall", [])
    assert not [line for line in lines if line["level"] == "red" or "看不出" in cast(str, line["text"])], lines
    back = gui.change(document, choice="original", backing_wall=got["backing_wall"])
    assert back["material"] == wall


def test_original_scheme_wins_over_the_remembered_wall(backed: Backed) -> None:
    gui, document, got, _ = backed
    other = _room()
    other["scheme_id"] = "before"
    _front(other)["material"] = {"kind": "constant_impedance", "impedance_pa_s_m": 2600.0}
    gui.call("PUT", "/api/schemes/before", other)
    body = {"scheme": document, "original_scheme_id": "before",
            "changes": [{"wall": "x0", "zone": "whole_wall", "choice": "original", "backing_wall": got["backing_wall"]}]}
    back = cast(list[Doc], gui.call("POST", "/api/zone-material/batch", body)["changes"])[0]
    assert back["material"] == {"kind": "constant_impedance", "impedance_pa_s_m": 2600.0}
    assert back["origin"] == "original_scheme"


def test_a_wall_mounted_construction_cannot_be_a_backing(tmp_path: Path) -> None:
    gui = _Gui(tmp_path)
    entry = {"kind": "construction", "id": "lab_mounted", "label": "自編貼牆鋼板", "provenance": "literature", "source": "本次自編",
             "layers": _vpr()["layers"], "backing": "zone_wall", "placement": "wall", "product": False}
    gui.call("PUT", "/api/materials/lab_mounted", entry)
    body = {"scheme": _room(), "changes": [{"wall": "x0", "zone": "whole_wall", "choice": "backing", "material_id": "lab_mounted"}]}
    refused = gui.call("POST", "/api/zone-material/batch", body, 400)
    assert "不能整塊當背後的牆" in cast(str, refused["error"])


# ---- 深度與起跑配置 ----

def test_plan_and_limits_subtract_the_backing_and_the_start_uses_it(backed: Backed) -> None:
    gui, document, got, _ = backed
    gui.design(0.25, got)
    plan = SearchPlan.model_validate(gui.call("POST", "/api/gradient/schemes/room/plan", {})["plan"])
    assert plan.max_depths["x0"] == pytest.approx(0.15, abs=1e-12)
    designs = start_designs(Scheme.model_validate(document), SETTINGS, presets_table(), plan.presets, mount=plan.request.mount,
                            fixed=plan.fixed_surfaces, caps=plan.max_depths, allowed=plan.allowed,
                            capabilities=CAPABILITIES, directivity=DIRECTIVITY)[0]
    front, = [item for item in designs if item.surface_id == "x0"]
    assert front.backing_material is not None and front.backing_material.model_dump(mode="json") == _vpr()
    limits = gui.call("POST", "/api/zone-limits", {"scheme": document, "design": gui.call("GET", "/api/designs/room")["design"]})
    cap = cast(dict[str, dict[str, Doc]], limits["caps"])["x0"]["whole_wall"]
    assert cap["value_m"] == pytest.approx(0.15, abs=1e-12)
    assert cap["backing_m"] == pytest.approx(0.1, abs=1e-12)


def test_depth_shallower_than_the_backing_is_refused_with_one_sentence(backed: Backed) -> None:
    gui, _, got, _ = backed
    gui.design(0.08, got)
    error = cast(str, gui.call("POST", "/api/gradient/schemes/room/plan", {}, 400)["error"])
    assert "沒有比背後的「VPR" in error and "前面放不了東西" in error
    assert "沒有一種放得進深度上限" not in error


def test_depth_equal_to_the_backing_is_refused_not_sent_as_a_zero_cap(backed: Backed) -> None:
    gui, _, got, _ = backed
    gui.design(0.1, got)
    error = cast(str, gui.call("POST", "/api/gradient/schemes/room/plan", {}, 400)["error"])
    assert "沒有比背後的「VPR" in error


def test_a_front_layer_that_fits_exactly_is_kept(backed: Backed) -> None:
    gui, _, got, _ = backed
    gui.design(0.15, got)  # 0.15－0.1 照十進位是 0.05，剛好放得下 5 公分的 48K 玻璃棉
    plan = SearchPlan.model_validate(gui.call("POST", "/api/gradient/schemes/room/plan", {})["plan"])
    assert plan.max_depths["x0"] == 0.05
    assert WOOL in plan.allowed["x0"]


def test_this_run_depth_shallower_than_the_backing_is_refused(backed: Backed) -> None:
    gui, _, got, _ = backed
    gui.design(0.25, got)
    body = {"max_depths": {"x0": {"whole_wall": 0.08}}}
    error = cast(str, gui.call("POST", "/api/gradient/schemes/room/plan", body, 400)["error"])
    assert "這一跑另填的最深 0.08 公尺沒有比背後那一筆 0.1 公尺深" in error
    body = {"max_depths": {"x0": {"whole_wall": 0.22}}}
    plan = SearchPlan.model_validate(gui.call("POST", "/api/gradient/schemes/room/plan", body)["plan"])
    assert plan.max_depths["x0"] == 0.12


def test_rigid_mount_with_a_backing_is_refused(backed: Backed) -> None:
    gui, _, got, _ = backed
    gui.design(0.25, got)
    error = cast(str, gui.call("POST", "/api/gradient/schemes/room/plan", {"mount": "rigid"}, 400)["error"])
    assert "裝法請選「貼在原牆上」" in error


def _listed(gui: _Gui, got: Doc, allowed: list[str]) -> None:
    item: Doc = {"mode": "search", "allowed": allowed, "max_depth_m": 0.25, "backing_entry": got["backing_entry"], "backing_wall": got["backing_wall"]}
    gui.call("PUT", "/api/designs/room", {"scheme_id": "room", "zones": {"x0": {"whole_wall": item}}})


def test_fixed_products_the_engine_could_pick_are_refused_in_front_of_a_backing(backed: Backed) -> None:
    gui, _, got, _ = backed
    _listed(gui, got, ["none", "glass_wool_48k_50mm"])
    error = cast(str, gui.call("POST", "/api/gradient/schemes/room/plan", {}, 400)["error"])
    assert "只能讓程式調可調材料或不貼" in error and "48K 玻璃棉 50 mm 貼牆" in error


def test_a_fixed_product_covered_by_a_ticked_family_is_not_refused(backed: Backed) -> None:
    gui, _, got, _ = backed
    _listed(gui, got, [WOOL, "none", "glass_wool_48k_50mm"])  # 48K 玻璃棉那一族已包含這一款，程式不會另外挑它
    plan = SearchPlan.model_validate(gui.call("POST", "/api/gradient/schemes/room/plan", {})["plan"])
    assert set(plan.allowed["x0"]) == {WOOL, "none"}


def test_no_backing_leaves_the_plan_cap_alone(tmp_path: Path) -> None:
    gui = _Gui(tmp_path)
    gui.call("PUT", "/api/schemes/room", _room())
    gui.design(0.25, None)
    plan = SearchPlan.model_validate(gui.call("POST", "/api/gradient/schemes/room/plan", {})["plan"])
    assert plan.max_depths["x0"] == 0.25


# ---- 左右對稱 ----

def _agree(first: str, second: str) -> str | None:
    """假的引擎檢查：每一對都收（只看背後那一條擋不擋）。"""
    return None


def _side_design(left: str | None, right: str | None) -> DesignSettings:
    design = searching({"y0": ["s"], "yL": ["s"]})
    wall = {"kind": "constant_impedance", "impedance_pa_s_m": 71500.0}
    for side, entry in (("y0", left), ("yL", right)):
        if entry is not None:
            cast(dict[str, dict[str, Doc]], design["zones"])[side]["s"].update(backing_entry=entry, backing_wall=wall)
    return DesignSettings.model_validate(design)


def test_same_backing_code_with_different_saved_material_does_not_pair() -> None:
    document = symmetry_document({"y0": SIDE, "yL": SIDE})
    zones = cast(dict[str, list[Doc]], cast(Doc, document["scene"])["surface_zones_by_wall"])
    for side, thickness in (("y0", 0.1), ("yL", 0.12)):
        material = deepcopy(_vpr())
        cast(list[Doc], material["layers"])[1]["thickness_m"] = thickness
        next(zone for zone in zones[side] if zone.get("id") == "s")["material"] = material
    found = pair_zones(Scheme.model_validate(document), _side_design(VPR, VPR), _agree)
    assert not found.pairs
    assert [item.reason for item in found.missed] == ["兩區背後不一樣"]


@pytest.mark.parametrize(("left", "right", "paired"), [(VPR, None, False), (VPR, VPR, True), (None, None, True)])
def test_mirrored_zones_with_different_backings_do_not_pair(left: str | None, right: str | None, paired: bool) -> None:
    scheme = Scheme.model_validate(symmetry_document({"y0": SIDE, "yL": SIDE}))
    found = pair_zones(scheme, _side_design(left, right), _agree)
    assert bool(found.pairs) is paired
    if not paired:
        assert [item.reason for item in found.missed] == ["兩區背後不一樣"]
