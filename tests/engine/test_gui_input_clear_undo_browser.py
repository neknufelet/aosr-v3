"""第 2 段「清掉這一面的切法」的復原（#1265 單元 3 審查真錯 3、設計洞 1）：復原只回一步、只在下一個改動之前有效。

- 存檔、另存新名字之後復原就不在（另存後按復原，會把舊方案的區放進新方案）。
- 清掉之後等圖面檢查（/api/plan）的空檔頁面已經解鎖：這時改了別格，復原就不該出現（復原的快照在解鎖前就記好）；
  這時換到別一面，那一面的那一格不被清掉的那一面蓋過去（保險行：「還看著這一面才重畫」走得到）。

方案資料都是考卷自己編的 109 號房（_surface_zone_cases）。"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, Error, Page, Route, expect

from tests.engine._input_page_steps import open_fact_editor, open_save_as
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _drag, _open_scheme, _write

LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]


def _settled(page: Page) -> None:
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def _release(held: list[Route]) -> None:
    """放行扣住的請求；頁面自己已經放棄的（被新一次蓋掉而中止）那幾個不用再放。"""
    for route in held:
        try:
            route.continue_()
        except Error as error:
            assert "already handled" in str(error), error


def _cut_and_clear(page: Page) -> None:
    """前牆畫一個框、存檔，再清掉這一面的切法：復原出現。"""
    page.locator("#elevation-wall-x0").click()
    _drag(page, LY, LZ, (0.3, 0.3), (1.3, 1.3))
    page.locator("#save").click()
    _settled(page)
    _cut_by(page, "#elevation-clear")
    _settled(page)
    expect(page.locator("#elevation-undo-clear")).to_have_count(1)


def test_saving_after_a_clear_takes_the_undo_away(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 3 之一：清掉之後按「儲存」：存好了，復原就不在、回到「清掉這一面的切法」。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _cut_and_clear(page)
        page.locator("#save").click()
        _settled(page)
        expect(page.locator("#messages")).to_have_class("ok")
        expect(page.locator("#elevation-undo-clear")).to_have_count(0)
        expect(page.locator("#elevation-clear")).to_have_count(1)
        _assert_quiet(watched)


def test_saving_as_a_new_name_after_a_clear_takes_the_undo_away(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 3 之二：清掉之後另存新名字：新方案上沒有復原（按下去會把舊方案清掉前的區放進新方案）。"""
    _write(tmp_path, room109_document())
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _cut_and_clear(page)
        open_save_as(page)
        page.locator("#save-as-id").fill("room109-new")
        page.locator("#save-as").click()
        page.wait_for_function("openedId === 'room109-new' && !inputBusy")
        expect(page.locator("#elevation-undo-clear")).to_have_count(0)
        expect(page.locator("#elevation-clear")).to_have_count(1)
        _assert_quiet(watched)


def test_an_edit_made_while_the_plan_check_is_on_its_way_takes_the_undo_away(tmp_path: Path, browser: Browser) -> None:
    """設計洞 1：清掉之後圖面檢查還在路上（攔著不放），頁面已解鎖；這時改房間預設總厚度：檢查回來後復原不出現
    （快照要在解鎖前記好，不能等檢查回來才記、把改完別格的樣子當成清掉後那一刻）。"""
    _write(tmp_path, room109_document())
    held: list[Route] = []
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        open_fact_editor(page, "edit-depth")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.3, 0.3), (1.3, 1.3))
        _settled(page)
        page.route("**/api/plan", lambda route: held.append(route))
        _cut_by(page, "#elevation-clear")
        assert held, "圖面檢查應該還在路上"
        page.locator("#depth-room").fill("20")
        page.locator("#depth-room").blur()
        page.unroute("**/api/plan")
        _release(held)
        _settled(page)
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        expect(page.locator("#elevation-undo-clear")).to_have_count(0)
        expect(page.locator("#elevation-clear")).to_have_count(1)
        _assert_quiet(watched)


def test_switching_walls_while_the_plan_check_is_on_its_way_keeps_the_slot_of_the_wall_shown(tmp_path: Path, browser: Browser) -> None:
    """保險行：清掉前牆之後圖面檢查攔著不放（頁面已解鎖），點後牆；檢查回來後那一格是後牆的（清掉鈕、沒有復原），
    不是前牆的復原；點回前牆，復原還在（換牆不算改動）。"""
    _write(tmp_path, room109_document())
    held: list[Route] = []
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, LY, LZ, (0.3, 0.3), (1.3, 1.3))
        _settled(page)
        page.route("**/api/plan", lambda route: held.append(route))
        _cut_by(page, "#elevation-clear")
        assert held, "圖面檢查應該還在路上"
        page.locator("#elevation-wall-xL").click()
        expect(page.locator("#elevation-wall-xL")).to_have_attribute("aria-pressed", "true")
        page.unroute("**/api/plan")
        _release(held)
        _settled(page)
        expect(page.locator("#elevation-undo-clear")).to_have_count(0)
        expect(page.locator("#elevation-clear")).to_have_count(1)
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#elevation-undo-clear")).to_have_count(1)
        _assert_quiet(watched)
