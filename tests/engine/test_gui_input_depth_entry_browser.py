"""總厚度上限那幾格（#1265 單元 3 審查真錯 1、2、4）：第 1 段右卡的房間預設 `#depth-room`、第 2 段「這一面另設」
`#depth-wall-<牆>`、「這間房貼什麼」小框的「最深」`#zone-panel-depth`。

- 1000 公分以上：格子裡的值不帶千分位逗號（數字格收不下逗號，會變空白、被當成一直在打字）。
- 打數字格讀不出來的字（`--1`）：拒收、退回上一次確認的值、寫提示句，不把已存的上限當成空白清掉；
  輸入頁兩格與小框的「最深」判法與提示字是同一支（都寫「最少 0.1」）。
- 填錯的提示句只屬於那一次打開的方案（範本）：換範本、重開同一份、換方案都不留著。

方案資料都是考卷自己編的 109 號房（_surface_zone_cases）。"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, Page, Route, expect

from tests.engine.test_gui_input_clear_undo_browser import _release

from tests.engine._input_page_steps import new_from_template, open_fact_editor, open_scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import room109_document
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write
from tests.engine.test_gui_zone_modes_browser import _write_room109_design

ROOM_REFUSED = "總厚度上限要填大於 0 的數字（公分，最少 0.1）；空白表示房間不設上限"
WALL_REFUSED = "總厚度上限要填大於 0 的數字（公分，最少 0.1）；空白表示照房間預設"
PANEL_REFUSED = "「最深」要填大於 0 的數字（公分，最少 0.1）；空白＝不限"


def _settled(page: Page) -> None:
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def _confirm(page: Page, selector: str, text: str) -> None:
    """填一格、離開那一格（change），等那一次限制檢查畫上去。"""
    generation = page.evaluate("zoneLimitsGeneration")
    page.locator(selector).fill(text)
    page.locator(selector).blur()
    page.wait_for_function("g => zoneLimitsGeneration > g && zoneLimitsShown === zoneLimitsGeneration", arg=generation)


def _type_bad(page: Page, selector: str, text: str) -> None:
    """數字格 fill 不收讀不出來的字，照人打字那樣全選再打。"""
    page.locator(selector).click()
    page.keyboard.press("Control+A")
    page.keyboard.type(text)
    page.locator(selector).blur()


def _both_refused(page: Page) -> None:
    """房間預設、前牆這一面各填一次錯的，兩句提示句都在。"""
    open_fact_editor(page, "edit-depth")
    page.locator("#depth-room").fill("-1")
    page.locator("#depth-room").blur()
    page.locator("#elevation-wall-x0").click()
    open_fact_editor(page, "edit-wall-depth")
    page.locator("#depth-wall-x0").fill("-1")
    page.locator("#depth-wall-x0").blur()
    # 前置：兩句都寫出來了（這裡只認共同那一段，字的細節由另一題考）。
    expect(page.locator("#depth-room-note")).to_contain_text("總厚度上限要填大於 0 的數字（公分")
    expect(page.locator("#depth-wall-note")).to_contain_text("總厚度上限要填大於 0 的數字（公分")


def test_a_thousand_centimetres_stays_in_the_box_without_a_comma(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 1：房間預設、這一面另設、小框「最深」各填 1000 公分，格子裡是 1000（不是空白、基準值不帶逗號），
    不被當成正在打字；存下來的是 10 公尺。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        open_fact_editor(page, "edit-depth")
        _confirm(page, "#depth-room", "1000")
        assert (page.locator("#depth-room").input_value(), page.locator("#depth-room").evaluate("e => e.defaultValue")) == ("1000", "1000")
        assert page.evaluate("zoneDesign.room_max_depth_m") == 10
        assert page.evaluate("Boolean(typingInElevation())") is False
        page.locator("#elevation-wall-x0").click()
        open_fact_editor(page, "edit-wall-depth")
        _confirm(page, "#depth-wall-x0", "1000")
        assert (page.locator("#depth-wall-x0").input_value(), page.locator("#depth-wall-x0").evaluate("e => e.defaultValue")) == ("1000", "1000")
        assert page.evaluate("zoneDesign.wall_max_depth_m.x0") == 10
        assert page.evaluate("Boolean(typingInElevation())") is False
        # 小框「最深」：指定貼一種、填 1000、套用；重開同一區，格子帶出 1000。
        page.locator("#zone-goto-x0-whole_wall").click()
        expect(page.locator("#zone-panel")).to_have_attribute("data-zones", "x0/whole_wall")
        page.locator("#zone-panel-mode-fixed").check()
        page.locator("#zone-panel-material").select_option("glass_wool_48k_50mm")
        page.locator("#zone-panel-depth").fill("1000")
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        _settled(page)
        assert page.evaluate("zoneSetting('x0', 'whole_wall').max_depth_m") == 10
        page.locator("#zone-map [data-wall='x0'][data-zone='whole_wall'] .zv-fill").click()
        expect(page.locator("#zone-panel-depth")).to_have_value("1000")
        _assert_quiet(watched)


def test_a_box_the_browser_cannot_read_is_refused_and_keeps_the_saved_cap(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 2：確認過 15 公分，再打 `--1`（數字格讀不出來，value 是空字串）：拒收、格子退回 15、寫提示句，
    存的 0.15 公尺不被當成空白清掉；這一面另設那一格同樣。提示句跟小框「最深」同一支寫的（都寫「最少 0.1」）。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        open_fact_editor(page, "edit-depth")
        _confirm(page, "#depth-room", "15")
        _type_bad(page, "#depth-room", "--1")
        expect(page.locator("#depth-room-note")).to_have_text(ROOM_REFUSED)
        expect(page.locator("#depth-room")).to_have_value("15")
        assert page.evaluate("zoneDesign.room_max_depth_m") == 0.15
        page.locator("#elevation-wall-x0").click()
        open_fact_editor(page, "edit-wall-depth")
        _confirm(page, "#depth-wall-x0", "12")
        _type_bad(page, "#depth-wall-x0", "--1")
        expect(page.locator("#depth-wall-note")).to_have_text(WALL_REFUSED)
        expect(page.locator("#depth-wall-x0")).to_have_value("12")
        assert page.evaluate("zoneDesign.wall_max_depth_m.x0") == 0.12
        # 小框「最深」打同樣讀不出來的字：同一句的小框版（主詞與空白的意思各自的，中間那一段同一支）。
        page.locator("#zone-goto-x0-whole_wall").click()
        page.locator("#zone-panel-mode-fixed").check()
        page.locator("#zone-panel-material").select_option("glass_wool_48k_50mm")
        page.locator("#zone-panel-depth").click()
        page.keyboard.type("--1")
        expect(page.locator("#zone-panel-problem")).to_have_text(PANEL_REFUSED)
        expect(page.locator("#zone-panel-apply")).to_be_disabled()
        _assert_quiet(watched)


def test_a_refused_note_does_not_follow_into_another_template(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 4 之一：從範本開的新方案上兩格各填錯一次，再從另一個範本開一份新的：兩句提示句都不在
    （#1265 單元 5b 起範本一定先取名存好才打開；擁有者換成新的那一份，填錯的那一句不跟過去）。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputStart === 'empty'")
        new_from_template(page, "first-room")
        _both_refused(page)
        new_from_template(page, "second-room", "stage_one")
        page.wait_for_function("!inputBusy && scheme.source_model === 'omnidirectional'")
        expect(page.locator("#depth-room-note")).to_have_text("")
        expect(page.locator("#depth-wall-note")).to_have_text("")
        _assert_quiet(watched)


def test_a_refused_note_does_not_survive_reopening_the_same_scheme(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 4 之二：同一份方案填錯兩格，再按一次打開同一份：兩句提示句都不在（重開同一份方案代號不變）。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _both_refused(page)
        _open_scheme(page, "room109")
        expect(page.locator("#depth-room-note")).to_have_text("")
        expect(page.locator("#depth-wall-note")).to_have_text("")
        _assert_quiet(watched)


def test_a_refused_note_is_gone_as_soon_as_another_scheme_opens(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 4 之三：換另一份方案，限制檢查還在路上（攔著不放）：兩句提示句已經不在，不等判決回來才清。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off"})
    other = room109_document()
    other["scheme_id"] = "room109-other"
    _write(tmp_path, other)
    held: list[Route] = []
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _both_refused(page)
        page.route("**/api/zone-limits", lambda route: held.append(route))
        open_scheme(page, "room109-other")
        page.wait_for_function("document.getElementById('save-id').value === 'room109-other' && !inputBusy")
        assert held, "限制檢查應該還在路上"
        expect(page.locator("#depth-room-note")).to_have_text("")
        expect(page.locator("#depth-wall-note")).to_have_text("")
        _release(held)
        page.unroute("**/api/zone-limits")
        _assert_quiet(watched)
