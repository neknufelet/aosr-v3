"""「顯示全部材料」搬到「這間房貼什麼」小框（#1265 單元 3 之二第 B 條；#1176 舊做法表那一個照留到拆表那一趟）。

小框「指定貼一種」下拉平常只列選中的每一面都能貼的；旁邊的開關打開才分組列出不列的（門、窗、整片天花構造、另一間示範房的），
分組與組標題跟舊做法表同一支。選了不是這一面用的那一筆：關掉開關下拉照樣是它，套用後小框寫「X 不是○牆用的：理由」。
選到跟原牆一模一樣的那一筆：做法變成不貼，回話照舊做法表那一句講（第 E 條）。

三題各照 test_gui_zone_choices_browser.py 考「顯示全部」的那一題（同名去掉舊表那一半），斷言的字照原樣。
方案與材料庫都是考卷自己編的（109 號房、_sample_material_library）。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._surface_zone_cases import room109_document
from tests.engine._zone_panel_steps import open_panel
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_choices_browser import DOOR, FLOOR_WOOD, HIDDEN_ON_WALLS, _sample_library, _values
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write
from tests.engine.test_gui_zone_modes_browser import _modes

SHOW_ALL = "顯示全部材料（門、窗、整片天花構造、另一間示範房的這些平常不列；要選它們才打開）"
CEILING = "ceiling_fibrecement8mm_wool30_on_board_gap180"
Zones = dict[str, dict[str, list[dict[str, object]]]]


def _fixed_panel(page: Page, wall: str) -> None:
    """輸入頁點那一面、按那一區（整面）的「改這一區 ▸」，小框選「指定貼一種」。"""
    page.locator(f"#elevation-wall-{wall}").click()
    open_panel(page, wall, "whole_wall")
    page.locator("#zone-panel-mode-fixed").check()
    expect(page.locator("#zone-panel-material")).to_be_visible()


def _apply(page: Page) -> None:
    page.locator("#zone-panel-apply").click()
    expect(page.locator("#zone-panel")).to_have_count(0)
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy && !document.getElementById('zone-page').inert")


def test_the_panel_lists_only_what_this_face_can_take_and_show_all_reveals_the_rest(tmp_path: Path, browser: Browser) -> None:
    """前牆整面：指定貼一種的下拉列牆上能貼的固定成品（含合成的 VPR），門、窗、天花構造、地毯、示範房一筆都沒有；
    打開「顯示全部材料」才列出來，關掉又收起；開著時加候選的挑選框照樣不列它們。地板只列地毯那一筆。"""
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _fixed_panel(page, "x0")
        fixed = _values(page, "#zone-panel-material option")
        assert "glass_wool_48k_50mm" in fixed and "pet60_vpr_steel1p2mm_foam90_sample" in fixed
        assert not set(HIDDEN_ON_WALLS) & set(fixed)
        expect(page.locator("#zone-page label.zv-show-all")).to_have_text(SHOW_ALL)
        page.locator("#zone-panel-show-all").check()
        assert set(HIDDEN_ON_WALLS) - {"demo109_rockwool60k_capped"} <= set(_values(page, "#zone-panel-material option"))
        page.locator("#zone-panel-show-all").uncheck()
        assert not set(HIDDEN_ON_WALLS) & set(_values(page, "#zone-panel-material option"))
        # 開關開著換成由程式最佳化：加候選的挑選框不跟著列不列的那幾筆。
        page.locator("#zone-panel-show-all").check()
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-add").click()
        for choice in HIDDEN_ON_WALLS:
            expect(page.locator(f"#zone-panel-pick-{choice}")).to_have_count(0)
        page.locator("#zone-panel-cancel").click()
        page.locator("#zone-page-close").click()
        _fixed_panel(page, "floor")
        page.locator("#zone-panel-show-all").uncheck()
        assert _values(page, "#zone-panel-material option") == ["", "carpet_on_concrete"]
        # 小框一次選好幾面時（zone_overview.js 下拉那一段傳的是選中的每一面）：有任何一面不能貼就算不列的。
        hidden = "walls => hiddenGroups(walls, assignableEntries()).flatMap(([, entries]) => entries.map(entry => entry.id))"
        assert "glass_wool_48k_50mm" not in page.evaluate(hidden, ["x0"])
        assert {"glass_wool_48k_50mm", "carpet_on_concrete"} <= set(page.evaluate(hidden, ["x0", "floor"]))
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_zone_holding_a_material_from_another_group_still_shows_it(tmp_path: Path, browser: Browser) -> None:
    """打開「顯示全部材料」選門：門列在「不列的：門、窗」那一組；關掉之後下拉照樣是門；套用後小框寫門不是前牆用的與理由。
    直接取代整面的（地板本身就是木地板那一筆）是地板本身，不寫不是地板用的。"""
    document = room109_document()
    floor = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == FLOOR_WOOD)
    cast(Zones, document["scene"])["surface_zones_by_wall"]["floor"][0]["material"] = cast(dict[str, object], floor)
    _write(tmp_path, document)
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _fixed_panel(page, "x0")
        page.locator("#zone-panel-show-all").check()
        expect(page.locator(f"#zone-panel-material optgroup[label='不列的：門、窗'] option[value={DOOR}]")).to_have_count(1)
        page.locator("#zone-panel-material").select_option(DOOR)
        page.locator("#zone-panel-show-all").uncheck()
        expect(page.locator("#zone-panel-material")).to_have_value(DOOR)
        _apply(page)
        assert page.evaluate("scheme.scene.surface_zones_by_wall.x0[0].material.kind") == "mounted_on_wall"
        page.locator("#zone-map [data-wall='x0'][data-zone='whole_wall'] .zv-fill").click()
        expect(page.locator("#zone-panel-away-x0-whole_wall")).to_contain_text("不是前牆用的")
        expect(page.locator("#zone-panel-away-x0-whole_wall")).to_contain_text("背後是空氣")
        page.locator("#zone-panel-cancel").click()
        page.locator("#zone-page-close").click()
        _fixed_panel(page, "floor")
        expect(page.locator("#zone-panel-away-floor-whole_wall")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_picking_what_the_ceiling_already_is_turns_into_keep_original(tmp_path: Path, browser: Browser) -> None:
    """#1175：天花整面本來就是那一筆吊頂，小框打開「顯示全部材料」選同一筆、套用：做法變成「不貼，保留原樣」、
    回話說「選的跟原本一樣，當成不貼」，那一區放的還是原本那一份（第 E 條：回話跟存下來的對得上）。"""
    built = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == CEILING)
    document = room109_document()
    cast(Zones, document["scene"])["surface_zones_by_wall"]["ceiling"][0]["material"] = cast(dict[str, object], built)
    _write(tmp_path, document)
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _fixed_panel(page, "ceiling")
        # 整片天花構造平常不列成貼上去的處理。
        assert CEILING not in _values(page, "#zone-panel-material option")
        page.locator("#zone-panel-show-all").check()
        page.locator("#zone-panel-material").select_option(CEILING)
        _apply(page)
        expect(page.locator("#zone-page-note")).to_contain_text("選的跟原本一樣，當成不貼")
        expect(page.locator("#zone-page-note")).not_to_contain_text("改成「指定貼一種")
        assert page.evaluate("structuredClone(scheme.scene.surface_zones_by_wall.ceiling[0].material)") == built
        assert _modes(page, "ceiling") == {"whole_wall": {"mode": "keep_original"}}
        assert watched.page_errors == [] and watched.console_errors == []


def test_an_unreadable_depth_stays_refused_and_says_so_after_show_all_redraws_the_panel(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二第二趟追加設計洞：小框「最深」打了讀不出來的字（「--1」，數字格讀成空字串、另標不成數字），再勾「顯示全部材料」
    （整個小框重畫、換一個新格子）：新格子放不回那幾個字，但那一次照樣不收、套用照擋，格子旁照講是怎麼回事，不讓空白的格子
    看起來像沒填卻報錯；重打一個數字，那一句收掉、套用按得下去。"""
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _fixed_panel(page, "x0")
        page.locator("#zone-panel-material").select_option("glass_wool_48k_50mm")
        page.locator("#zone-panel-depth").press_sequentially("--1")
        problem = page.locator("#zone-panel-problem")
        expect(problem).to_contain_text("「最深」要填大於 0 的數字（公分，最少 0.1）")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        expect(page.locator("#zone-panel-depth-lost")).to_have_count(0)  # 還是同一格、看得到打的字：不另講
        page.locator("#zone-panel-show-all").check()
        expect(page.locator("#zone-panel-depth")).to_have_value("")
        lost = page.locator("#zone-panel-depth-lost")
        expect(lost).to_have_text("剛才打的還不成數字，小框重畫後格子顯示不出來：這一次不收；重打一次，或按退格鍵、刪除鍵清成空白（空白＝不限）")
        expect(page.locator("#zone-panel-depth")).to_have_attribute("aria-invalid", "true")
        expect(problem).to_contain_text("「最深」要填大於 0 的數字（公分，最少 0.1）")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        page.locator("#zone-panel-depth").press_sequentially("12")
        expect(lost).to_have_count(0)
        expect(page.locator("#zone-panel-depth")).not_to_have_attribute("aria-invalid", "true")
        expect(problem).to_be_hidden()
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        assert watched.page_errors == [] and watched.console_errors == []


@pytest.mark.parametrize("key", ["Backspace", "Delete"])
def test_an_unreadable_depth_lost_in_a_redraw_can_be_cleared_back_to_blank_with_one_key(tmp_path: Path, browser: Browser, key: str) -> None:
    """同一個情況的出口（#1265 單元 3 之二收尾追加第 6 條）：重畫後格子是空的，對它按退格或刪除就是人要的空白——不必亂打一個字，
    那一句收掉、套用按得下去，套用之後那一區沒有最深（空白＝不限）。沒按之前照樣不收（不默默換成空白）。"""
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _fixed_panel(page, "x0")
        page.locator("#zone-panel-material").select_option("glass_wool_48k_50mm")
        depth = page.locator("#zone-panel-depth")
        depth.press_sequentially("--1")
        page.locator("#zone-panel-show-all").check()
        lost = page.locator("#zone-panel-depth-lost")
        expect(lost).to_be_visible()  # 那一句的字（講出口）上一題逐字考
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        depth.click()
        page.keyboard.press(key)
        expect(depth).to_have_value("")
        expect(lost).to_have_count(0)
        expect(depth).not_to_have_attribute("aria-invalid", "true")
        expect(page.locator("#zone-panel-problem")).to_be_hidden()
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        _apply(page)
        stored = cast(dict[str, object], page.evaluate("structuredClone(zoneDesign.zones.x0.whole_wall)"))
        assert stored["mode"] == "fixed" and "max_depth_m" not in stored, stored
        assert watched.page_errors == [] and watched.console_errors == []
