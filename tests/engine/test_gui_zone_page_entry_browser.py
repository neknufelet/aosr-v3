"""從首頁、結果頁、打開方案之後，找得到「這間房貼什麼」那一頁（#1198）。

資料全是合成的：區名考卷那間假房間（``_zone_naming_room`` 的 zone-naming-sample，5.2 × 5.5 × 3.15 m，牆面切了區）、
109 號房沒切區的那一份（存成 a-whole-room：名字排在清單第一個，沒指定開哪一份時開到的會是它）。
視窗一律 1440 × 900；「在第一屏」＝按之前、沒捲過，整顆落在視窗裡（``getBoundingClientRect``）。

- 首頁第一屏「挑材料」一下就進最近算的那一份的「這間房貼什麼」（不是清單第一份、不是第三階段頁）；沒切區的方案一樣進得去。
- 「打開來改」或「打開」之後，方案那一節的入口就在視窗裡，再一下進那一頁。
- 帶著名字進來但那一份打不開（存著的檢查不過、清單裡沒有）：原因寫在「打開」旁邊，不開那一頁。
- 結果頁「這一間挑材料」連到同一條路。
- 帶著名字進來、輪到打開那一份時輸入頁正忙（開頁還沒完就改了表單，那一筆還在路上）：不開那一頁，也不把它開在範本上。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from tests.engine._input_page_steps import open_scheme
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME, write_zone_naming_library
from tests.engine.test_gui_browser import RUN_ID, _assert_quiet, _open, _save, _serve, browser as browser, result as result
from tests.engine.test_gui_wall_words import _part2d_layout
from tests.engine.test_gui_zone_elevation_browser import _write

SAMPLE, WHOLE = "zone-naming-sample", "a-whole-room"
SAMPLE_ROOM, WHOLE_ROOM = "深 5.5 × 寬 5.2 × 高 3.15 公尺", "深 6.99 × 寬 4.42 × 高 3.16 公尺"
_INSIDE = """selector => { const node = document.querySelector(selector);
    if (!node) return false;
    const box = node.getBoundingClientRect();
    return box.width > 0 && box.height > 0 && box.top >= 0 && box.left >= 0
        && box.bottom <= window.innerHeight && box.right <= window.innerWidth; }"""


def _prepare(tmp_path: Path, latest: str | None) -> None:
    """存兩份方案；``latest`` 那一份有一筆算好的結果（首頁「最近算的房間」只要結果檔裡的方案快照）。"""
    documents = {WHOLE: _part2d_layout() | {"scheme_id": WHOLE}, SAMPLE: ZONE_NAMING_SCHEME}
    for document in documents.values():
        _write(tmp_path, document)
    write_zone_naming_library(tmp_path)
    if latest is not None:
        (tmp_path / "results").mkdir()
        snapshot = Scheme.model_validate(documents[latest]).model_dump(mode="json")
        (tmp_path / "results" / f"{'d' * 32}.json").write_text(json.dumps({"scheme": snapshot}), encoding="utf-8")


def _first_screen(page: Page) -> None:
    page.set_viewport_size({"width": 1440, "height": 900})
    page.wait_for_function("document.body.dataset.inputReady === 'true'")
    assert page.evaluate("window.scrollY") == 0


def _inside(page: Page, selector: str) -> bool:
    return cast(bool, page.evaluate(_INSIDE, selector))


def _opened(page: Page, name: str) -> None:
    page.wait_for_function("name => document.getElementById('save-id').value === name", arg=name)
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def _on_zone_page(page: Page, name: str, room: str) -> None:
    """「這間房貼什麼」開著，開的是那一份：方案代號是它、頁首那句房間尺寸是它的。"""
    expect(page.locator("#zone-page")).to_be_visible()
    expect(page.locator("#zone-map")).to_be_visible()
    expect(page.locator("#zone-page h1")).to_have_text("這間房貼什麼")
    expect(page.locator("#save-id")).to_have_value(name)
    expect(page.locator("#zone-page .zv-head")).to_contain_text(room)


@pytest.mark.parametrize(("latest", "room"), [(SAMPLE, SAMPLE_ROOM), (WHOLE, WHOLE_ROOM)], ids=["zoned", "whole"])
def test_home_first_screen_goes_straight_to_the_zone_page_of_the_latest_scheme(
        tmp_path: Path, browser: Browser, latest: str, room: str) -> None:
    _prepare(tmp_path, latest)
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        _first_screen(page)
        link = page.locator("#home-materials")
        expect(link.locator("b")).to_have_text("挑材料：每一面牆貼什麼")
        assert "第三階段" not in link.inner_text() and "/gradient" not in (link.get_attribute("href") or "")
        expect(link).to_have_attribute("href", f"/?zones={latest}")
        assert _inside(page, "#home-materials")
        link.click()  # 第一下，也是唯一的一下
        page.wait_for_url(f"{base}/?zones={latest}")
        _opened(page, latest)  # 整頁重載後打開方案：等的時間照打開那一條（不擠在下面斷言的幾秒裡）
        _on_zone_page(page, latest, room)
        assert page.evaluate("window.scrollY") == 0
        _assert_quiet(watched)


@pytest.mark.parametrize("way", ["home-row", "open-button"])
def test_after_opening_a_scheme_the_way_in_is_on_the_first_screen(tmp_path: Path, browser: Browser, way: str) -> None:
    """首頁那一列的「打開來改」（第一下）→ 方案那一節的入口（第二下）：兩下進任何一份的那一頁；自己用「打開」也一樣看得到。
    「已存的方案」那幾列在不在第一屏這裡不量：它排在俯視圖底下，份數多了本來就要捲（最近算的那一份走上一題那條路，不用捲）。"""
    _prepare(tmp_path, WHOLE)
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        _first_screen(page)
        if way == "home-row":
            page.locator("#scheme-menu-button").click()
            page.locator(f'#home-schemes [data-scheme="{SAMPLE}"] button.home-open').click()
        else:
            open_scheme(page, SAMPLE)
        _opened(page, SAMPLE)
        # 碰那顆入口之前量：Playwright 的 click 會自己捲，量在後面就量不出「不用捲」。
        assert _inside(page, "#scheme-zone-open")
        expect(page.locator("#scheme-zone-open")).to_have_text("設定每一區貼什麼 ▸")
        page.locator("#scheme-zone-open").click()
        _on_zone_page(page, SAMPLE, SAMPLE_ROOM)
        _assert_quiet(watched)


@pytest.mark.parametrize("name", ["outside", "gone"])
def test_a_link_to_a_scheme_that_cannot_open_says_why_and_stays_off_the_zone_page(
        tmp_path: Path, browser: Browser, name: str) -> None:
    """outside：存著、但喇叭在房間外，打開時檢查不過；gone：清單裡沒有（封存了、舊書籤）。"""
    _prepare(tmp_path, SAMPLE)
    broken = _part2d_layout() | {"scheme_id": "outside"}
    speakers = cast(dict[str, dict[str, float]], broken["speakers"])
    speakers[next(iter(speakers))]["x"] = 99.0
    (tmp_path / "schemes" / "outside.json").write_text(json.dumps(broken), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, f"{base}/?zones={name}") as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        note = page.locator("#open-note")
        expect(note).to_contain_text(f"沒有打開「{name}」：")
        page.wait_for_function("document.body.dataset.inputReady === 'true' && !document.querySelector('main').inert && !inputBusy")
        expect(note).to_have_class("notice")
        if name == "gone":
            expect(note).to_contain_text("已存的方案裡沒有這一份")
        else:
            assert note.inner_text() == f"沒有打開「{name}」：{page.locator('#messages').inner_text()}"
        expect(page.locator("#zone-page")).to_be_hidden()
        expect(page.locator("main")).to_be_visible()
        assert page.locator("#save-id").input_value() != name
        assert _inside(page, "#open-note")
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


def test_a_link_arriving_while_the_form_is_busy_does_not_open_the_zone_page(tmp_path: Path, browser: Browser) -> None:
    """帶著 ?zones=<名> 開頁，清單還沒載完就勾了沙發，那一筆改動還沒回來：輪到打開那一份時「打開」被擋下
    （上一個動作還沒做完）。這時不准開「這間房貼什麼」——表單上還是範本，開了就是開在範本上。那一筆回來之後也不開。"""
    _prepare(tmp_path, SAMPLE)
    held: dict[str, Route] = {}
    errors: list[str] = []
    with _serve(tmp_path) as base:
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        try:
            page.on("pageerror", lambda error: errors.append(str(error)))
            # 結果清單那一支卡住：開頁停在「清單還沒載完」；表單改動那一支也卡住：頁面一直鎖著。
            page.route("**/api/results", lambda route: held.__setitem__("results", route))
            page.route("**/api/input-edit", lambda route: held.__setitem__("edit", route))
            page.goto(f"{base}/?zones={SAMPLE}", wait_until="commit")
            page.wait_for_function("document.querySelectorAll('#home-schemes .home-open').length > 0")
            template = page.locator("#save-id").input_value()
            assert template and template not in (SAMPLE, WHOLE)
            page.locator("#furniture-sofa").click()
            page.wait_for_function("inputBusy")
            while not {"results", "edit"} <= held.keys():
                page.wait_for_timeout(20)
            held["results"].continue_()
            page.wait_for_function("document.body.dataset.inputReady === 'true'")
            # 開頁做完的同一步就輪到那條連結：被擋下的那一句寫在「打開」旁邊，那一頁沒開。
            assert page.evaluate("inputBusy") is True
            expect(page.locator("#open-note")).to_have_text(
                f"上一個動作還沒做完，沒有打開「{SAMPLE}」；等它做完再按一次「打開」")
            expect(page.locator("#zone-page")).to_be_hidden()
            held["edit"].continue_()
            page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
            expect(page.locator("#zone-page")).to_be_hidden()
            expect(page.locator("main")).to_be_visible()
            expect(page.locator("#save-id")).to_have_value(template)
            assert errors == []
        finally:
            page.close()


def test_result_page_link_goes_to_the_zone_page_of_that_scheme(tmp_path: Path, browser: Browser,
                                                               result: SchemeResult) -> None:
    """結果頁「這一間挑材料」：字不再寫第三階段搜尋，連到首頁帶名字的那條路，按下去進那一份的那一頁。
    那個連結只在分區附表畫出來時才給；這裡的結果是整面牆的，附表那一支回覆用考卷自己編的一份（只為了讓它畫）。"""
    name = result.scheme.scheme_id
    _prepare(tmp_path, None)
    _write(tmp_path, result.scheme.model_dump(mode="json"))
    _save(tmp_path, result)
    zones = {"zoned": True, "notes": [], "rows": [], "state_text": "", "running": False, "can_start": False,
             "baseline_result_url": "", "improvement": None}

    def answer(route: Route) -> None:
        route.fulfill(status=200, content_type="application/json", body=json.dumps(zones))

    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        page.route(f"**/api/results/{RUN_ID}/zones", answer)
        page.goto(f"{base}/results/{RUN_ID}", wait_until="networkidle")
        link = page.locator("#next-materials")
        expect(link).to_be_visible()
        expect(link).to_have_text("這一間挑材料：每一面牆貼什麼")
        expect(link).to_have_attribute("href", f"/?zones={name}")
        assert _inside(page, "#next-materials")
        link.click()
        page.wait_for_url(f"{base}/?zones={name}")
        _opened(page, name)
        expect(page.locator("#zone-page")).to_be_visible()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(page.locator("#save-id")).to_have_value(name)
        _assert_quiet(watched)
