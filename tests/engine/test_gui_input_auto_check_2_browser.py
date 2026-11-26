"""#1265 單元 6 之二：頁底兩顆按之前也檢查，不過就停在問題上；鎖住的方案、空畫面下頂條那一句；整頁高度。

「照現在牆上貼的算一次」平面圖不過、「交給程式最佳化 ▸」超過限制：不送（沒有存檔、沒有開算請求），頁面捲到問題那裡，
焦點放在頂條那一顆字（名字念得出原因）。哪一顆擋哪一種照原本的規則：平面圖問題兩顆都擋、限制只擋搜尋。
方案、設計設定、結果都是考卷自己編的。"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Request, expect

from aosr.reporting.result import SchemeResult
from tests.engine.test_gui_archive import _bundle, _saved
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser, result as result
from tests.engine.test_gui_input_auto_check_browser import (
    ROOM, _commit, _hold, _let_go, _opened, _quiet_but_refused, _red_ceiling, _room, _seen)
from tests.engine.test_scheme_pipeline import _run_control

FOCUSED = "name => document.activeElement?.id === name"


def _sent(page: Page) -> list[str]:
    """記下頁面送出去的存檔（PUT 方案、做法）與開算（POST /api/runs）請求。"""
    sent: list[str] = []

    def note(request: Request) -> None:
        if (request.method == "PUT" and "/api/schemes/" in request.url) or (
                request.method == "POST" and request.url.endswith("/api/runs")):
            sent.append(f"{request.method} {request.url}")

    page.on("request", note)
    return sent


def _bottom(page: Page) -> None:
    page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")


def test_calculate_with_a_plan_problem_sends_nothing_and_stops_at_the_problem(tmp_path: Path, browser: Browser) -> None:
    """平面圖不過時按「照現在牆上貼的算一次」：不存、不開算；捲回第 1 段平面圖，焦點在頂條「平面圖幾條問題」那一顆。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _opened(page)
        _commit(page, "#speaker-left-x", "20")
        expect(page.locator("#check-state")).to_have_text(re.compile(r"^檢查沒過：平面圖 \d+ 條問題$"))
        _bottom(page)
        expect(page.locator("#plan-xy")).not_to_be_in_viewport()
        sent = _sent(page)
        page.locator("#calculate").click()
        # 先等按的那一下整個收尾（鎖放開），再看有沒有送出東西：沒送是這一題頭一件要考的。
        page.wait_for_function("!inputBusy && document.getElementById('check-state').dataset.kind !== 'pending'")
        assert sent == []
        page.wait_for_function(FOCUSED, arg="check-plan")
        expect(page.locator("#plan-xy")).to_be_in_viewport()
        expect(page.locator("#check-plan")).to_have_attribute("aria-label", re.compile("座標超出房間"))
        _quiet_but_refused(watched)


# 那一塊整個在頂條底下、頁底那條上面（兩條都黏在視窗上）。
PLACES = """selector => [document.querySelector(selector).getBoundingClientRect(),
  document.getElementById("input-bars").getBoundingClientRect(), document.getElementById("input-footer").getBoundingClientRect()]
  .map(box => [Math.round(box.top), Math.round(box.bottom)])"""


def _clear_of_bars(page: Page, selector: str) -> None:
    (top, bottom), (_, bar), (footer, _) = cast(list[list[int]], page.evaluate(PLACES, selector))
    assert bar <= top and bottom <= footer, f"{selector} 在 {top}～{bottom}，頂條底 {bar}、頁底那條頂 {footer}"


def test_a_jump_from_the_top_bar_lands_below_the_top_bar_even_when_the_message_line_makes_it_taller(
        tmp_path: Path, browser: Browser) -> None:
    """兩種問題都有、訊息列在頂條折成好幾行（頂條變高）：點「平面圖幾條問題」，第 1 段標題整個露在頂條下面；
    點「幾條超過限制」，標紅的那一塊整個露在頂條與頁底那條中間。"""
    _room(tmp_path, deep_ceiling=True)
    # 900 寬：頂條那一排折行、訊息列那一句折成好幾行，頂條高過第 1 段原本固定的捲動邊距（9rem）。
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=900) as watched:
        page = watched.page
        page.set_viewport_size({"width": 900, "height": 900})
        _opened(page)
        _red_ceiling(page)
        _commit(page, "#speaker-left-x", "20")
        expect(page.locator("#check-state button")).to_have_count(2)
        expect(page.locator("#messages")).to_be_visible()
        assert cast(float, page.evaluate("document.getElementById('input-bars').getBoundingClientRect().height")) > 160
        _bottom(page)
        page.locator("#check-plan").click()
        _clear_of_bars(page, "#scheme-section h2")
        page.evaluate("window.scrollTo(0, 0)")
        page.locator("#check-limits").click()
        expect(page.locator("#elevation-wall-ceiling")).to_have_attribute("aria-pressed", "true")
        _clear_of_bars(page, "#elevation [data-zone-id=u1v1]")
        _quiet_but_refused(watched)


def _search_room(tmp_path: Path) -> None:
    """天花右上那塊很深、指定貼一種；左下那塊交給程式最佳化（頁底那一顆按得下）。房間預設不設上限：開頁時沒有超過。"""
    _room(tmp_path, deep_ceiling=True)
    path = tmp_path / "designs" / f"{ROOM}.json"
    design = json.loads(path.read_text(encoding="utf-8"))
    design["zones"]["ceiling"]["u0v0"] = {"mode": "search", "allowed": ["none", "glass_wool_48k"]}
    path.write_text(json.dumps(design), encoding="utf-8")


def test_search_with_limits_over_sends_nothing_and_stops_at_the_red_zone(tmp_path: Path, browser: Browser) -> None:
    """改了上限、限制判決還在路上就按「交給程式最佳化 ▸」（那時鈕還按得下）：判決回來是超過，不存、不跳；
    選天花那一面、捲到標紅的那一塊，焦點在頂條「幾條超過限制」那一顆。"""
    _search_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _opened(page)
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        expect(page.locator("#zone-to-search")).to_be_enabled()
        held = _hold(page, "**/api/zone-limits")
        _commit(page, "#depth-room", "20")
        _seen(page, held, "限制檢查")
        expect(page.locator("#check-state")).to_have_text("檢查中…")
        page.evaluate("window.scrollTo(0, 0)")
        sent = _sent(page)
        page.locator("#zone-to-search").click()
        _let_go(page, "**/api/zone-limits", held)
        page.wait_for_function(FOCUSED, arg="check-limits")
        red = cast(int, page.evaluate("zoneLimits.red"))
        expect(page.locator("#check-limits")).to_have_text(f"{red} 條超過限制")
        expect(page.locator("#elevation-wall-ceiling")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#elevation [data-zone-id=u1v1]")).to_be_in_viewport()
        expect(page.locator("#zone-to-search-note")).to_contain_text("超過限制")
        assert sent == [] and page.url.startswith(base) and "/gradient" not in page.url
        _assert_quiet(watched)


def test_calculate_is_not_held_back_by_limits_and_search_is_held_back_by_a_plan_problem(tmp_path: Path, browser: Browser) -> None:
    """規則照舊：超過限制時「照現在牆上貼的算一次」照樣存、照樣開算；平面圖不過時「交給程式最佳化 ▸」也不送、停在平面圖。"""
    _search_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _opened(page)
        _commit(page, "#speaker-left-x", "20")
        expect(page.locator("#check-state")).to_have_text(re.compile("平面圖"))
        _bottom(page)
        sent = _sent(page)
        page.locator("#zone-to-search").click()
        page.wait_for_function(FOCUSED, arg="check-plan")
        expect(page.locator("#plan-xy")).to_be_in_viewport()
        assert sent == [] and "/gradient" not in page.url
        _commit(page, "#speaker-left-x", "1.2")
        _red_ceiling(page)
        expect(page.locator("#zone-to-search")).to_be_disabled()
        # 開算請求送到就算數（不真的算）：考卷自己回 409，頁面照講。
        page.route("**/api/runs", lambda route: route.fulfill(status=409, content_type="application/json",
                   body=json.dumps({"error": "考卷不開算"})) if route.request.method == "POST" else route.continue_())
        sent = _sent(page)
        page.locator("#calculate").click()
        expect(page.locator("#messages")).to_have_text("考卷不開算")
        assert [line.split(" ")[0] for line in sent] == ["PUT", "POST"], sent
        assert page.evaluate("document.activeElement?.id") != "check-limits"
        assert watched.page_errors == []
        assert all("422" in text or "409" in text for text in watched.console_errors), watched.console_errors


def test_a_locked_scheme_still_says_the_check_beside_its_state(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """算過、鎖住的方案：頂條紫色狀態旁照樣寫「檢查通過」；改了一格出問題照樣變紅字（算的鈕這時本來就停用，要先另存）。"""
    room = _run_control(result.scheme.model_copy(update={"scheme_id": "locked-room"}))
    _saved(tmp_path, room)
    _bundle(tmp_path, room, "2" * 32)
    with _serve(tmp_path) as base, _open(browser, f"{base}/?scheme=locked-room", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true' && openedId === 'locked-room'")
        expect(page.locator("#save-state")).to_have_text("算過了，內容鎖住留著比較")
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        _commit(page, "#speaker-left-x", "20")
        expect(page.locator("#check-state")).to_have_text(re.compile(r"^檢查沒過：平面圖 \d+ 條問題$"))
        expect(page.locator("#calculate")).to_be_disabled()
        _quiet_but_refused(watched)


def test_the_empty_screen_shows_no_check_sentence_and_no_message_line(tmp_path: Path, browser: Browser) -> None:
    """一份方案都沒有：空畫面，頂條那一句與訊息列都不出現（沒有方案可檢查）。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputStart === 'empty'")
        expect(page.locator("#empty-start")).to_be_visible()
        expect(page.locator("#check-state")).to_be_hidden()
        expect(page.locator("#messages")).to_be_hidden()
        expect(page.locator("#check")).to_have_count(0)
        _assert_quiet(watched)


def test_the_whole_page_with_the_check_in_the_top_bar_fits_in_two_screens(tmp_path: Path, browser: Browser) -> None:
    """整頁高度（量法同 5c）：1440×900，開一份切過區的方案、前牆選中、摺疊都收著；檢查那一段拿掉、搬上頂條之後不超過 2 屏。"""
    from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME, write_zone_naming_library
    write_zone_naming_library(tmp_path)
    (tmp_path / "schemes").mkdir(exist_ok=True)
    (tmp_path / "schemes" / "zone-naming-sample.json").write_text(json.dumps(ZONE_NAMING_SCHEME), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        page.goto(base, wait_until="networkidle")
        page.wait_for_function("document.body.dataset.inputReady === 'true' && openedId === 'zone-naming-sample' && !inputBusy")
        page.wait_for_load_state("networkidle")
        expect(page.locator("#elevation-wall-yL")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#check-state")).not_to_have_text("檢查中…")
        assert page.evaluate("[...document.querySelectorAll('details')].filter(node => node.open).length") == 0
        height = cast(int, page.evaluate("document.documentElement.scrollHeight"))
        assert height <= 2 * 900, f"整頁 {height} 像素，{height / 900:.2f} 屏"
        _assert_quiet(watched)
