"""第 2 段牆那一行、右欄清單、清掉鈕那幾條走得到、之前沒考卷的分支（#1265 單元 3 審查「保險行」，單元 3 之二）：

- 原本的牆是自訂構造（跟哪一個預設都不完全一樣）：寫「自訂構造（參數與預設不完全相同）」。
- 分區的牆每一區原本的牆不一樣：寫「每一區不一樣（看右邊「每一區的位置與大小」）」。
- 「換原本的牆」打開著，這一面切了區：那一塊收起來、鈕也收起來（分區的牆不從表單改）。
- 有一區的構造跟材料庫哪一筆一樣、看不出是原牆還是貼的：清單分到「看不出現在是哪一種」那一組，照那一句寫。
- 材料庫讀不全、由程式最佳化的區又沒寫清單：寫從全部可調材料與「不貼」裡挑、列不出是哪幾種。
- 指定貼一種的區有總厚度上限：那一句接「；總厚度最多 N 公分（哪一層）」。
- 清掉之後把房間高度清空：清掉鈕與復原都不擺（畫不出這一面）。清掉鈕那一格開頭「畫不出就不擺」那一行拿掉了
  （實測走不到：立面圖重畫自己先清空那一格），這一題釘住拿掉之後照樣不擺。

方案與材料庫都是考卷自己編的（109 號房、區名考卷那間假房間）。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, expect

from aosr.gui.zone_edit import CutBox, cut_wall
from tests.engine._zone_material_steps import pick_fixed_row
from tests.engine._bonded_backing_cases import MELAMINE_100
from tests.engine._construction_cases import construction
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._input_page_steps import open_fact_editor
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _drag, _open_scheme, _write
from tests.engine.test_gui_zone_modes_browser import _write_room109_design
from tests.engine.test_gui_zone_overview_tasks_browser import _prepare

LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]
SAMPLE = cast(str, ZONE_NAMING_SCHEME["scheme_id"])
Doc = dict[str, object]
Zones = dict[str, list[dict[str, object]]]


def _zones(document: Doc) -> Zones:
    return cast(Zones, cast(Doc, document["scene"])["surface_zones_by_wall"])


def _settled(page: Page) -> None:
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def test_a_custom_construction_wall_says_it_matches_no_preset(tmp_path: Path, browser: Browser) -> None:
    document = room109_document()
    _zones(document)["x0"][0]["material"] = {"kind": "assembly", "backing": "rigid", "layers": [MELAMINE_100]}
    _write(tmp_path, document)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#elevation-original")).to_have_text("原本的牆：自訂構造（參數與預設不完全相同）")
        _assert_quiet(watched)


def test_a_cut_wall_whose_zones_started_different_says_so(tmp_path: Path, browser: Browser) -> None:
    document = room109_document()
    zones = cast(list[dict[str, object]], cut_wall(document, "x0", (CutBox(id="z1", u_m=(0.3, 1.3), v_m=(0.3, 1.3)),))["zones"])
    first = next(zone for zone in zones if zone["id"] == "z1")
    first["material"] = {"kind": "constant_impedance", "impedance_pa_s_m": 2 * cast(float, cast(Doc, first["material"])["impedance_pa_s_m"])}
    _zones(document)["x0"] = zones
    _write(tmp_path, document)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#elevation-original")).to_have_text("原本的牆：每一區不一樣（看右邊「每一區的位置與大小」）")
        _assert_quiet(watched)


def test_cutting_the_wall_closes_the_open_original_wall_editor(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        open_fact_editor(page, "edit-wall-original")
        expect(page.locator("#wall-original-editor")).to_be_visible()
        _drag(page, LY, LZ, (0.3, 0.3), (1.3, 1.3))
        _settled(page)
        expect(page.locator("#wall-original-editor")).to_be_hidden()
        expect(page.locator("#edit-wall-original")).to_be_hidden()
        expect(page.locator("#edit-wall-original")).to_have_attribute("aria-expanded", "false")
        _assert_quiet(watched)


def test_a_zone_that_cannot_be_told_apart_from_a_library_entry_is_listed_as_unknown(tmp_path: Path, browser: Browser) -> None:
    """天花 auto1 是一份裸構造，材料庫有一筆層一模一樣的（布置用面板），又沒存原方案：看不出它是原牆還是貼上去的。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        saved = page.request.get(f"{base}/api/schemes/{SAMPLE}").json()["scheme"]
        ceiling = next(zone for zone in saved["scene"]["surface_zones_by_wall"]["ceiling"] if zone["id"] == "auto1")
        doc = construction("rigid", id="ceiling_copy", label="布置用面板", layers=ceiling["material"]["layers"])
        assert page.request.put(f"{base}/api/materials/ceiling_copy", data=doc).ok
        _open_scheme(page, SAMPLE)
        page.locator("#elevation-wall-ceiling").click()
        group = page.locator("#elevation-list #wall-zones-unknown")
        expect(group.locator("#wall-zone-ceiling-auto1 .tag")).to_have_text("看不出現在是哪一種")
        expect(page.locator("#wall-zone-what-ceiling-auto1")).to_contain_text("看不出：跟材料庫「布置用面板」一樣。")
        _assert_quiet(watched)


def test_a_search_zone_without_a_list_on_a_broken_library_says_the_list_is_unknown(tmp_path: Path, browser: Browser) -> None:
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "zones": {"x0": {"whole_wall": {"mode": "search"}}}})
    (tmp_path / "materials.json").write_text("{not json", encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#wall-zone-what-x0-whole_wall")).to_contain_text(
            "已交給程式挑：從全部可調材料與「不貼」裡挑（材料庫沒讀全，列不出是哪幾種），")
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_fixed_zone_with_a_cap_names_the_cap(tmp_path: Path, browser: Browser) -> None:
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": 0.15})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        page.locator("#zone-goto-x0-whole_wall").click()
        page.locator("#zone-panel-mode-fixed").check()
        pick_fixed_row(page, "glass_wool_48k_50mm")
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        page.locator("#zone-page-close").click()
        page.wait_for_function("zoneLimitsGeneration > 0 && zoneLimitsShown === zoneLimitsGeneration")
        expect(page.locator("#elevation-list #wall-zones-fixed #wall-zone-what-x0-whole_wall")).to_contain_text("；總厚度最多 15 公分（房間預設）。")
        _assert_quiet(watched)


def test_blanking_the_room_height_after_a_clear_shows_neither_clear_nor_undo(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.3, 0.3), (1.3, 1.3))
        _settled(page)
        _cut_by(page, "#elevation-clear")
        _settled(page)
        expect(page.locator("#elevation-undo-clear")).to_have_count(1)
        page.locator("#room-Lz").fill("")
        expect(page.locator("#elevation-size")).to_have_text("先把房間的長、寬、高都填好正數，才畫得出這一面。")
        expect(page.locator("#elevation-clear-slot")).to_be_empty()
        assert watched.page_errors == []
