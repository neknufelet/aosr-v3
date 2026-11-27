"""分區右欄的材料分組、相符面、選料後的草稿與套用。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.engine._zone_material_steps import pick_fixed_row
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._surface_zone_cases import room109_document
from tests.engine._zone_panel_steps import open_panel
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_choices_browser import DOOR, FLOOR_CHOICES, FLOOR_WOOD, HIDDEN_ON_WALLS, _sample_library, _values
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write
from tests.engine.test_gui_zone_modes_browser import _modes

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


def test_both_pickers_group_rows_and_remember_folds(tmp_path: Path, browser: Browser) -> None:
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _fixed_panel(page, "x0")
        fixed = _values(page, "#zone-panel-material [role=radio]")
        assert "glass_wool_48k_50mm" in fixed and "pet60_vpr_steel1p2mm_foam90_sample" in fixed
        assert not set(HIDDEN_ON_WALLS) & set(fixed)
        for mode, container in (("fixed", "material"), ("search", "picker")):
            page.locator(f"#zone-panel-mode-{mode}").check()
            groups = page.locator(f"#zone-panel-{container} details[data-material-group]")
            expect(groups.locator(":scope > summary")).to_have_count(4)
            assert groups.evaluate_all("els => els.map(e => e.getAttribute('aria-label'))") == ["多孔吸音", "共振吸音", "門、窗、一般牆面", "我組的"]
            expect(groups.last.locator(":scope > summary")).to_have_text("我組的0")
            assert not groups.last.evaluate("e => e.open")
            for group in groups.all():
                assert int(group.locator(".zv-family-count").inner_text()) == group.locator(".zv-material").count()
            if mode == "search":
                wool = groups.first.locator("#zone-panel-pick-glass_wool_48k")
                expect(wool).to_be_checked()
                groups.first.locator(":scope > summary").click()
                expect(wool).to_be_hidden()
                groups.first.locator(":scope > summary").click()
                expect(wool).to_be_visible()
            expect(groups.nth(2).locator(f"[value={DOOR}]")).to_be_visible()
            groups.nth(2).locator(":scope > summary").click()
            expect(groups.nth(2).locator(f"[value={DOOR}]")).to_be_hidden()
            page.locator("#zone-panel-mode-keep_original").check()
            page.locator(f"#zone-panel-mode-{mode}").check()
            expect(groups.nth(2).locator(f"[value={DOOR}]")).to_be_hidden()
            if mode == "fixed":
                groups.first.locator(":scope > summary").click()
                visible = groups.nth(1).get_by_role("radio")
                expect(visible.first).to_have_attribute("tabindex", "0")
                visible.last.focus()
                page.keyboard.press("ArrowRight")
                expect(visible.first).to_be_checked()
                expect(visible.first).to_be_focused()
        expect(page.locator("#zone-panel-show-all")).to_have_count(0)
        page.locator("#zone-panel-cancel").click()
        page.locator("#zone-page-close").click()
        _fixed_panel(page, "floor")
        groups = page.locator("#zone-panel-material details")
        assert groups.evaluate_all("els => els.map(e => e.getAttribute('aria-label'))") == ["地板", "多孔吸音", "共振吸音", "我組的"]
        assert _values(page, "#zone-panel-material [role=radio]") == FLOOR_CHOICES
        assert _values(page, "#zone-panel-material [data-material-group=surface] [role=radio]") == FLOOR_CHOICES
        expect(groups.first.locator(f"[value={FLOOR_WOOD}]")).to_be_visible()
        assert watched.page_errors == [] and watched.console_errors == []


def test_picked_door_reopens_with_its_own_backing(tmp_path: Path, browser: Browser) -> None:
    """牆上直接選門，顯示自己的背後；套用、重開後仍保有分區指定的門。"""
    document = room109_document()
    floor = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == FLOOR_WOOD)
    cast(Zones, document["scene"])["surface_zones_by_wall"]["floor"][0]["material"] = cast(dict[str, object], floor)
    _write(tmp_path, document)
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _fixed_panel(page, "x0")
        expect(page.get_by_role("group", name="門、窗、一般牆面").locator(f"[value={DOOR}]")).to_be_visible()
        pick_fixed_row(page, DOOR)
        expect(page.locator("#zone-panel-material [aria-checked=true]")).to_have_attribute("value", DOOR)
        page.locator("#zone-panel-preview > summary").click()
        expect(page.locator("#zone-panel-show h4").first).to_contain_text("背後是空氣")
        _apply(page)
        assert page.evaluate("structuredClone(scheme.scene.surface_zones_by_wall.x0[0].material)") == next(
            entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == DOOR)
        page.locator("#zone-map [data-wall='x0'][data-zone='whole_wall'] .zv-fill").click()
        expect(page.locator("#zone-panel-away-x0-whole_wall")).to_have_count(0)
        expect(page.locator("#zone-panel-material [aria-checked=true]")).to_have_attribute("value", DOOR)
        page.locator("#zone-panel-cancel").click()
        page.locator("#zone-page-close").click()
        _fixed_panel(page, "floor")
        expect(page.locator("#zone-panel-away-floor-whole_wall")).to_have_count(0)
        assert watched.page_errors == [] and watched.console_errors == []


def test_picking_what_the_ceiling_already_is_turns_into_keep_original(tmp_path: Path, browser: Browser) -> None:
    """#1175：天花整面本來就是那一筆吊頂，小框直接選同一筆、套用：做法變成「不貼，保留原樣」、
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
        # 天花可直接選整片吊頂。
        assert CEILING in _values(page, "#zone-panel-material [role=radio]")
        pick_fixed_row(page, CEILING)
        _apply(page)
        expect(page.locator("#zone-page-note")).to_contain_text("選的跟原本一樣，當成不貼")
        expect(page.locator("#zone-page-note")).not_to_contain_text("改成「指定貼一種")
        assert page.evaluate("structuredClone(scheme.scene.surface_zones_by_wall.ceiling[0].material)") == built
        assert _modes(page, "ceiling") == {"whole_wall": {"mode": "keep_original"}}
        assert watched.page_errors == [] and watched.console_errors == []


def test_an_unreadable_depth_stays_refused_and_says_so_after_show_all_redraws_the_panel(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二第二趟追加設計洞：小框「最深」打了讀不出來的字（「--1」，數字格讀成空字串、另標不成數字），再選另一筆材料
    （整個小框重畫、換一個新格子）：新格子放不回那幾個字，但那一次照樣不收、套用照擋，格子旁照講是怎麼回事，不讓空白的格子
    看起來像沒填卻報錯；重打一個數字，那一句收掉、套用按得下去。"""
    _write(tmp_path, room109_document())
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _fixed_panel(page, "x0")
        pick_fixed_row(page, "glass_wool_48k_50mm")
        page.locator("#zone-panel-depth").press_sequentially("--1")
        problem = page.locator("#zone-panel-problem")
        expect(problem).to_contain_text("「最深」要填大於 0 的數字（公分，最少 0.1）")
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        expect(page.locator("#zone-panel-depth-lost")).to_have_count(0)  # 還是同一格、看得到打的字：不另講
        pick_fixed_row(page, "fabric_pet_a_gap100")
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
        pick_fixed_row(page, "glass_wool_48k_50mm")
        depth = page.locator("#zone-panel-depth")
        depth.press_sequentially("--1")
        pick_fixed_row(page, "fabric_pet_a_gap100")
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
