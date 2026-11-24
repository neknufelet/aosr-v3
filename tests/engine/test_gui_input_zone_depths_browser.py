"""第 2 段右欄「前牆這 N 區」清單（#1265 單元 3 之二第 C、D 條）：

- 交給程式挑的區底下接逐組生效上限（`depth-cap-<牆>-<區>` 那半句「個案上限…每一組跟自己的上限取較嚴」、
  `depth-target-<牆>-<區>-<i>` 逐組一句、放不下的那一組橘字 `limit-warn`）；字由伺服器逐組寫好
  （zoneLimits.caps[牆][區].targets），id 不變、整頁一份，舊做法表那一格不再有。
- 判決回來重畫清單時，正在打的字不被吃掉（打完離開那一格，清單才照新的判決畫）。
- 設計設定讀不到／對不上、材料庫有問題那兩句：寫在清單上方（選不選牆都看得到）；舊做法表那一份照留到拆表。

方案與材料庫都是考卷自己編的（109 號房、深背岩棉那一筆使用者材料）。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from tests.engine._input_page_steps import open_fact_editor
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_zone_elevation_browser import _open_scheme
from tests.engine.test_gui_zone_limits_browser import _deep_user_library
from tests.engine.test_gui_zone_modes_browser import _write_room109_design

ALLOWED = ["glass_wool_48k", "glass_wool_24k", "glass_wool_96k", "none", "deep_rock_wool"]
DUPLICATE_IDS = "() => { const seen = new Map(); for (const node of document.querySelectorAll('[id]')) seen.set(node.id, (seen.get(node.id) || 0) + 1); return [...seen].filter(([, count]) => count > 1).map(([id]) => id); }"


def _search_room(tmp_path: Path, room_cm: float) -> None:
    """前牆整面交給程式挑（三種玻璃棉、不貼、深背岩棉），房間預設總厚度上限 room_cm 公分。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": room_cm / 100,
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ALLOWED}}}})
    (tmp_path / "materials.json").write_text(_deep_user_library(), encoding="utf-8")


def _settled(page: Page) -> None:
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
    page.wait_for_function("zoneLimitsGeneration > 0 && zoneLimitsShown === zoneLimitsGeneration")


def _targets(page: Page) -> list[dict[str, object]]:
    return cast(list[dict[str, object]], page.evaluate("structuredClone(zoneLimits.caps.x0.whole_wall.targets)"))


def test_each_group_cap_sits_under_its_zone_in_the_list_once(tmp_path: Path, browser: Browser) -> None:
    """房間預設 6 公分：五組逐組一句排在清單前牆整面那一區底下，字照伺服器那一份；深背岩棉最淺排法 8 公分放不下，
    那一句橘字。整頁每一個 depth-target- 只有一份、舊做法表裡沒有；整頁沒有重複的 id。"""
    _search_room(tmp_path, 6)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _settled(page)
        row = page.locator("#elevation-list #wall-zone-x0-whole_wall")
        expect(row.locator("#depth-cap-x0-whole_wall")).to_have_text("個案上限 0.06 公尺（房間預設）；每一組跟自己的上限取較嚴：")
        targets = _targets(page)
        assert len(targets) == len(ALLOWED) - 1  # 「不貼」不算一組
        for index, target in enumerate(targets):
            line = row.locator(f"#depth-target-x0-whole_wall-{index}")
            expect(line).to_be_visible()
            expect(line).to_have_text(str(target["text"]))
            expect(line).to_have_class("note" if target["usable"] else "limit-warn")
        warn = row.locator(".limit-warn")
        expect(warn).to_have_count(1)
        expect(warn).to_contain_text("深背岩棉（自填）：不可用")
        assert warn.evaluate("node => getComputedStyle(node).color") == "rgb(154, 91, 0)"
        expect(page.locator("[id^=depth-target-]")).to_have_count(len(targets))
        expect(page.locator("#zone-table-fold [id^=depth-target-], #zone-table-fold #depth-cap-x0-whole_wall")).to_have_count(0)
        expect(page.locator("#depth-zone-x0-whole_wall")).to_have_text("這一區沒給")  # 舊做法表那一格只剩這一區自己給的
        assert page.evaluate(DUPLICATE_IDS) == []
        shots = Path(os.environ.get("AOSR_SHOTS") or tmp_path)  # 給了資料夾就把截圖存去那裡（證據），沒給存暫存
        page.locator("#elevation-list").screenshot(path=str(shots / "u3b-list-depth-targets.png"))
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_verdict_redrawing_the_list_keeps_what_is_being_typed(tmp_path: Path, browser: Browser) -> None:
    """改房間預設那一次的限制檢查攔著不放；這時在「這一面另設」打 8、還沒離開那一格，判決回來：格子裡的 8 照留、焦點照留，
    清單還是舊的；離開那一格之後清單照新的那一次（這面牆 8 公分）畫。"""
    _search_room(tmp_path, 20)
    held: list[Route] = []
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _settled(page)
        cap = page.locator("#elevation-list #depth-cap-x0-whole_wall")
        expect(cap).to_have_text("個案上限 0.2 公尺（房間預設）；每一組跟自己的上限取較嚴：")
        open_fact_editor(page, "edit-depth")
        page.route("**/api/zone-limits", lambda route: held.append(route))
        page.locator("#depth-room").fill("10")
        page.locator("#depth-room").blur()
        page.wait_for_function("() => zoneLimitsReceived !== zoneLimitsGeneration")
        open_fact_editor(page, "edit-wall-depth")
        box = page.locator("#depth-wall-x0")
        box.click()
        page.keyboard.type("8")
        assert held, "限制檢查應該還在路上"
        page.unroute("**/api/zone-limits")
        _release(held)
        page.wait_for_function("() => zoneLimitsReceived === zoneLimitsGeneration")
        expect(box).to_have_value("8")
        expect(box).to_be_focused()
        expect(cap).to_have_text("個案上限 0.2 公尺（房間預設）；每一組跟自己的上限取較嚴：")
        page.keyboard.press("Tab")
        _settled(page)
        expect(page.locator("#depth-wall-x0")).to_have_value("8")
        expect(cap).to_have_text("個案上限 0.08 公尺（這面牆）；每一組跟自己的上限取較嚴：")
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_design_that_does_not_match_the_scheme_is_said_above_the_list(tmp_path: Path, browser: Browser) -> None:
    """設計設定有一區方案裡沒有：那一句寫在第 2 段右欄清單上方，沒選牆就看得到，字跟 zoneDesignNote 一樣。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "zones": {"x0": {"ghost": {"mode": "keep_original"}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        said = cast(str, page.evaluate("zoneDesignNote"))
        assert said.startswith("設計設定有幾條跟方案對不上")
        line = page.locator("#elevation-list-notes .notice")
        expect(line).to_have_count(1)
        expect(line).to_be_visible()
        expect(line).to_have_text(said)
        page.locator("#elevation-wall-x0").click()
        expect(line).to_have_text(said)
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_broken_material_library_is_said_above_the_list(tmp_path: Path, browser: Browser) -> None:
    """使用者材料庫壞了：那一句寫在清單上方，字跟 zoneLibraryProblem 一樣。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    (tmp_path / "materials.json").write_text("{not json", encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        said = cast(str, page.evaluate("zoneLibraryProblem"))
        assert said.startswith("使用者材料庫讀不進來")
        line = page.locator("#elevation-list-notes .notice")
        expect(line).to_have_count(1)
        expect(line).to_be_visible()
        expect(line).to_have_text(said)
        page.locator("#elevation-wall-x0").click()
        expect(line).to_have_text(said)
        assert watched.page_errors == [] and watched.console_errors == []
