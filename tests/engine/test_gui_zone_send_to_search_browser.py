"""真瀏覽器考「這間房貼什麼」那一頁的「拿去搜尋」從按下到離開這一頁（#1188 第 5、6 條）：存好之後、跳頁之前與跳頁途中
這一頁都還鎖著（改的東西不會沒存就被帶去搜尋頁）；搜尋頁一直沒打開不會鎖死；被還在路上的改動擋下的那一句，
那一批落地後還在。方案與材料庫都是考卷自己編的；只攔網路、伺服器是考卷自己起的。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from tests.engine._input_page_steps import scripted_open_scheme
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _write
from tests.engine.test_gui_zone_overview_browser import SAMPLE
from tests.engine.test_gui_zone_overview_faults_browser import _Held, _hold, _modes, _released, _wait_held
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _SELECTED, SEARCH, _apply, _click_zone, _open_overview, _prepare, _settled)


def _hold_gets(page: Page, pattern: str) -> list[Route]:
    """把符合的 GET 扣住不放（別的方法照放）；扣住的由考卷自己放行。"""
    held: list[Route] = []
    page.route(pattern, lambda route: held.append(route) if route.request.method == "GET" else route.continue_())
    return held


def _wait_routes(page: Page, held: list[Route], count: int) -> None:
    for _ in range(500):
        if len(held) >= count:
            return
        page.wait_for_timeout(20)
    raise AssertionError(f"等不到第 {count} 個被扣住的請求")


def _press_search_with_one_more_zone_ready(page: Page, sent: list[str]) -> tuple[dict[str, float], dict[str, float]]:
    """一區已經由程式最佳化；另一區切換做法後又選回原樣。回圖上另一區與套用鈕的位置（等一下拿真滑鼠去點）。"""
    _open_overview(page)
    _settled(page)
    _click_zone(page, "yL/z3")
    page.locator("#zone-panel-mode-search").check()
    _apply(page)
    _click_zone(page, "y0/auto2")
    page.locator("#zone-panel-mode-search").check()
    page.locator("#zone-panel-mode-keep_original").check()
    expect(page.locator("#zone-page-search")).to_have_text("拿去搜尋")
    expect(page.locator("#zone-panel-apply")).to_be_enabled()
    other = page.locator("#zone-map [data-wall='xL'][data-zone='auto1'] .zv-fill").bounding_box()
    button = page.locator("#zone-panel-apply").bounding_box()
    assert other is not None and button is not None
    page.on("request", lambda request: sent.append(request.url) if request.url.endswith("/api/zone-material/batch") else None)
    return cast(dict[str, float], other), cast(dict[str, float], button)


def _mouse_cannot_change_anything(page: Page, boxes: tuple[dict[str, float], dict[str, float]], sent: list[str]) -> None:
    expect(page.locator("#zone-panel-apply")).to_be_visible()
    for box in boxes:
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.wait_for_timeout(200)
    assert page.evaluate("document.getElementById('zone-page').inert") is True
    assert page.evaluate(_SELECTED) == ["y0/auto2"] and sent == []
    assert set(_modes(page, ["yL/z3", "y0/auto2"]).items()) == {("yL/z3", "search"), ("y0/auto2", "沒列")}


def test_nothing_can_be_changed_between_the_save_of_send_to_search_and_the_jump(tmp_path: Path, browser: Browser) -> None:
    """#1188 第 5 條：「拿去搜尋」存好之後還要重讀清單才跳頁。那一段（這裡把結果清單扣住，撐開成必中）這一頁還鎖著：
    真滑鼠點別區、按套用都沒作用、沒有送出換材料——不然那一批改動沒存就被帶去搜尋頁，搜尋頁讀到的比畫面上少。
    放行後照常跳到搜尋頁，磁碟上是按下那一刻那一份。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        sent: list[str] = []
        boxes = _press_search_with_one_more_zone_ready(page, sent)
        results = _hold_gets(page, "**/api/results")
        with page.expect_response(lambda r: r.url.endswith(f"/api/designs/{SAMPLE}") and r.request.method == "PUT") as design:
            page.locator("#zone-page-search").click()
        assert design.value.ok
        _wait_routes(page, results, 1)
        _mouse_cannot_change_anything(page, boxes, sent)
        results[0].continue_()
        page.wait_for_url(f"**/gradient?scheme={SAMPLE}")
        assert sent == [] and watched.page_errors == []
    stored = json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text(encoding="utf-8"))["zones"]
    assert stored == {"yL": {"z3": SEARCH}}


def test_nothing_can_be_changed_while_the_search_page_is_on_its_way(tmp_path: Path, browser: Browser) -> None:
    """#1188 第 5 條的尾巴：跳頁送出去了、搜尋頁還沒回來（這裡把那一頁扣住）——舊的這一頁還在畫面上，照樣鎖著，
    真滑鼠按套用沒有送出換材料。放行後到搜尋頁，磁碟上是按下那一刻那一份。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        sent: list[str] = []
        boxes = _press_search_with_one_more_zone_ready(page, sent)
        leaving = _hold_gets(page, "**/gradient?scheme=*")
        page.locator("#zone-page-search").click()
        _wait_routes(page, leaving, 1)
        # 跳頁扣在半路時瀏覽器不讓考卷讀頁面裡的值（會一直等到換頁）：只用真滑鼠按套用，從網路這一側看有沒有送出。
        page.mouse.click(boxes[1]["x"] + boxes[1]["width"] / 2, boxes[1]["y"] + boxes[1]["height"] / 2)
        page.wait_for_timeout(300)
        assert sent == []
        leaving[0].continue_()
        page.wait_for_url(f"**/gradient?scheme={SAMPLE}")
        assert sent == [] and watched.page_errors == []
    stored = json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text(encoding="utf-8"))["zones"]
    assert stored == {"yL": {"z3": SEARCH}}


def test_a_search_page_that_never_opens_gives_this_page_back_and_says_so(tmp_path: Path, browser: Browser) -> None:
    """#1188 第 5 條的鎖留到離開這一頁為止：搜尋頁要是一直沒打開（這裡讓那個請求回「沒有內容」，瀏覽器就留在原頁），
    不能永遠鎖死——等一陣子（這裡把頁面的時鐘往前撥，不真的等）就放開、照講存好了但沒打開，再按一次照常過去。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        answers = [204]
        page.route("**/gradient?scheme=*", lambda route: route.fulfill(status=answers[0], content_type="text/plain", body="gradient"))
        page.clock.install()
        with page.expect_response("**/gradient?scheme=*"):
            page.locator("#zone-page-search").click()
        page.wait_for_timeout(200)
        assert page.evaluate("[document.getElementById('zone-page').inert, inputBusy]") == [True, True]
        page.clock.fast_forward(60_000)
        _released(page)
        expect(page.locator("#zone-page-note")).to_have_text(
            "方案與每一區的做法都存好了，但等了一陣子搜尋頁還沒打開：再按一次「拿去搜尋」")
        expect(page.locator("#zone-page-note")).to_have_class("notice")
        assert page.url == f"{base}/"
        answers[0] = 200
        page.locator("#zone-page-search").click()
        page.wait_for_url(f"**/gradient?scheme={SAMPLE}")
        assert watched.page_errors == []
    stored = json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text(encoding="utf-8"))["zones"]
    assert stored == {"yL": {"z3": SEARCH}}


def test_the_words_of_a_blocked_send_to_search_are_still_there_after_what_blocked_it_lands(tmp_path: Path, browser: Browser) -> None:
    """#1188 第 6 條：「拿去搜尋」被還在路上的一批改動擋下（見上面那一題）。那一批落地、平面圖檢查通過之後，被擋的那一句還在
    這一頁上——那一次什麼都沒存、沒跳頁，不能在零點幾秒內變回一片乾淨，看起來像按了沒反應。
    之後再套用一批（新的動作）那一句才拿掉；再按一次「拿去搜尋」照常存、照常跳。"""
    _prepare(tmp_path)
    blocked = "上一個動作還沒做完，這一次沒有儲存；等它做完再按一次"
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        saved: list[str] = []
        page.on("request", lambda request: saved.append(request.url) if request.method == "PUT" else None)
        limits = _hold(page, "**/api/zone-limits")
        _click_zone(page, "y0/auto2")
        page.evaluate("void refreshZoneLimits()")
        _wait_held(page, limits, 1)
        page.locator("#zone-page-search").click()
        page.locator("#zone-panel-mode-search").check()
        batch = _hold(page, "**/api/zone-material/batch")
        plan = _hold(page, "**/api/plan")
        page.locator("#zone-panel-apply").click()
        _wait_held(page, batch, 1)
        limits.release()
        said = page.locator("#zone-page-said")
        expect(said).to_have_text(blocked)
        # 那一批落地（做法寫好了）、它的平面圖檢查還扣著：那一句在。
        batch.release()
        _wait_held(page, plan, 1)
        expect(page.locator("#zone-page-note")).to_have_text("1 區改成「由程式最佳化」")
        expect(said).to_have_text(blocked)
        # 平面圖檢查通過（輸入頁訊息列換成成功的一句）：那一句還在。
        with page.expect_response("**/api/plan") as checked:
            plan.release()
        assert checked.value.ok
        _settled(page)
        expect(page.locator("#messages")).to_have_class("ok")
        expect(said).to_be_visible()
        expect(said).to_have_text(blocked)
        assert saved == [] and "gradient" not in page.url
        assert _modes(page, ["yL/z3", "y0/auto2"]) == {"yL/z3": "search", "y0/auto2": "search"}
        # 新的一個動作：那一句拿掉，不永遠黏著。
        _click_zone(page, "xL/auto1")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        expect(said).to_have_count(0)
        page.route("**/gradient?scheme=*", lambda route: route.fulfill(status=200, content_type="text/plain", body="gradient"))
        page.locator("#zone-page-search").click()
        page.wait_for_url(f"**/gradient?scheme={SAMPLE}")
        assert watched.page_errors == []
    stored = json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text(encoding="utf-8"))["zones"]
    assert stored == {"yL": {"z3": SEARCH}, "y0": {"auto2": SEARCH}, "xL": {"auto1": SEARCH}}


BLOCKED = "上一個動作還沒做完，這一次沒有儲存；等它做完再按一次"
OTHER = "send-other"


def _blocked_with_the_batch_still_out(page: Page) -> tuple[_Held, _Held]:
    """走到「拿去搜尋被擋、那一句寫上去了、擋住它的那一批還扣著（整頁鎖著）」。回那一批與它的平面圖檢查兩個扣住器。"""
    _open_overview(page)
    _settled(page)
    _click_zone(page, "yL/z3")
    page.locator("#zone-panel-mode-search").check()
    _apply(page)
    limits = _hold(page, "**/api/zone-limits")
    _click_zone(page, "y0/auto2")
    page.evaluate("void refreshZoneLimits()")
    _wait_held(page, limits, 1)
    page.locator("#zone-page-search").click()
    page.locator("#zone-panel-mode-search").check()
    batch = _hold(page, "**/api/zone-material/batch")
    plan = _hold(page, "**/api/plan")
    page.locator("#zone-panel-apply").click()
    _wait_held(page, batch, 1)
    limits.release()
    expect(page.locator("#zone-page-said")).to_have_text(BLOCKED)
    return batch, plan


def _land(page: Page, batch: _Held, plan: _Held) -> None:
    """那一批落地、平面圖檢查通過、畫面放開。"""
    batch.release()
    _wait_held(page, plan, 1)
    with page.expect_response("**/api/plan") as checked:
        plan.release()
    assert checked.value.ok
    _settled(page)
    expect(page.locator("#messages")).to_have_class("ok")


def test_pressing_again_while_the_page_is_still_locked_does_not_drop_the_blocked_words(tmp_path: Path, browser: Browser) -> None:
    """#1188 第 6 條（審查重現）：被擋那一句剛寫上去、那一批還在路上（整頁鎖著），使用者再按一次「拿去搜尋」——按不到，
    點擊落在頁面底層。那一下不算「按到東西」：那一批落地後那一句還在，沒存、沒跳頁。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        saved: list[str] = []
        page.on("request", lambda request: saved.append(request.url) if request.method == "PUT" else None)
        batch, plan = _blocked_with_the_batch_still_out(page)
        assert page.evaluate("[document.getElementById('zone-page').inert, inputBusy]") == [True, True]
        box = page.locator("#zone-page-search").bounding_box()
        assert box is not None
        page.evaluate("""() => { window.trustedClicks = 0;
            document.addEventListener("click", event => { if (event.isTrusted) window.trustedClicks += 1; }, true); }""")
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_function("window.trustedClicks === 1")
        _land(page, batch, plan)
        expect(page.locator("#zone-page-said")).to_have_text(BLOCKED)
        assert saved == [] and "gradient" not in page.url and watched.page_errors == []


def test_the_blocked_words_go_away_at_the_next_real_press_and_not_at_a_click_the_page_sends_itself(
        tmp_path: Path, browser: Browser) -> None:
    """#1188 第 6 條：那一批落地後那一句留著。頁面程式自己發的點擊不是使用者按的，那一句照留；
    使用者真的按一下別的東西（這裡點圖上一區）才拿掉——不永遠黏著，也不必等下一批改動。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        batch, plan = _blocked_with_the_batch_still_out(page)
        _land(page, batch, plan)
        said = page.locator("#zone-page-said")
        expect(said).to_have_text(BLOCKED)
        page.evaluate("""() => { window.sentClicks = 0;
            document.addEventListener("click", () => { window.sentClicks += 1; }, true);
            document.getElementById("zone-page").click(); }""")
        assert page.evaluate("window.sentClicks") == 1
        expect(said).to_have_text(BLOCKED)
        _click_zone(page, "xL/auto1")
        expect(said).to_have_count(0)
        assert page.evaluate(_SELECTED) == ["xL/auto1"] and watched.page_errors == []


def test_the_blocked_words_do_not_follow_into_another_scheme(tmp_path: Path, browser: Browser) -> None:
    """#1188 第 6 條：那一句講的是原本那一份。頁面程式自己發「打開」換成另一份（使用者沒有按到東西，那一句還留著）：
    換過去之後這一頁不准還寫著上一份的「這一次沒有儲存」。"""
    _prepare(tmp_path)
    _write(tmp_path, ZONE_NAMING_SCHEME | {"scheme_id": OTHER})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        batch, plan = _blocked_with_the_batch_still_out(page)
        _land(page, batch, plan)
        said = page.locator("#zone-page-said")
        expect(said).to_have_text(BLOCKED)
        scripted_open_scheme(page, OTHER)
        page.wait_for_function("name => scheme.scheme_id === name && !inputBusy", arg=OTHER)
        _settled(page)
        expect(page.locator("#zone-map")).to_be_visible()
        expect(said).to_have_count(0)
        assert watched.page_errors == []


def test_a_scheme_swapped_in_by_the_page_itself_before_the_jump_is_not_what_the_search_page_opens(
        tmp_path: Path, browser: Browser) -> None:
    """存檔完成、還在等結果清單時，由頁面載入另一份方案。原方案應只存下已套用的區，並留下取消跳轉的原因。"""
    _prepare(tmp_path)
    _write(tmp_path, ZONE_NAMING_SCHEME | {"scheme_id": OTHER})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        sent: list[str] = []
        _press_search_with_one_more_zone_ready(page, sent)
        results = _hold_gets(page, "**/api/results")
        jumps: list[str] = []
        page.on("request", lambda request: jumps.append(request.url) if "/gradient" in request.url else None)
        page.route("**/gradient?scheme=*", lambda route: route.fulfill(status=200, content_type="text/plain", body="gradient"))
        with page.expect_response(lambda r: r.url.endswith(f"/api/designs/{SAMPLE}") and r.request.method == "PUT") as design:
            page.locator("#zone-page-search").click()
        assert design.value.ok
        _wait_routes(page, results, 1)
        assert page.evaluate("[document.getElementById('zone-page').inert, inputBusy]") == [True, True]
        assert sent == []
        assert page.evaluate("name => openScheme(name, true)", OTHER)
        page.wait_for_function("name => scheme.scheme_id === name", arg=OTHER)
        results[0].continue_()
        expect(page.locator("#zone-page-note")).to_have_text(f"存檔途中換了方案：「{SAMPLE}」照樣存好，這次不跳搜尋頁")
        page.wait_for_load_state("networkidle")
        assert jumps == [] and page.url == f"{base}/"
        assert page.evaluate("scheme.scheme_id") == OTHER and watched.page_errors == []
    stored = json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text(encoding="utf-8"))["zones"]
    assert stored == {"yL": {"z3": SEARCH}}
    assert not (tmp_path / "designs" / f"{OTHER}.json").exists()


def test_refused_apply_keeps_draft_and_never_sends_search(tmp_path: Path, browser: Browser) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _click_zone(page, "yL/z1")
        page.locator("#zone-panel-mode-search").check()
        sent = []
        page.on("request", lambda r: sent.append(r.url) if r.method == "PUT" or "/gradient?" in r.url else None)
        page.route("**/api/zone-material/batch", lambda route: route.fulfill(status=400, json={"error": "自編材料遭拒"}))
        page.get_by_role("button", name="套用並拿去搜尋").click()
        expect(page.locator("#zone-page-note")).to_contain_text("自編材料遭拒")
        expect(page.locator("#zone-panel")).to_have_attribute("data-zones", "yL/z1")
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        page.wait_for_load_state("networkidle")
        assert sent == [] and page.url == base + "/"


@pytest.mark.parametrize("endpoint", ["plan", "zone-limits"])
def test_combined_search_holds_selection_until_check_finishes(tmp_path: Path, browser: Browser, endpoint: str) -> None:
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "yL/z1")
        page.locator("#zone-panel-mode-search").check()
        other = page.locator("#zone-map [data-wall=yL][data-zone=z2] .zv-fill").bounding_box()
        assert other is not None
        held = _hold(page, f"**/api/{endpoint}")
        saved = []
        page.on("request", lambda r: saved.append(r.url) if r.method == "PUT" else None)
        page.get_by_role("button", name="套用並拿去搜尋").click()
        _wait_held(page, held, 1)
        page.mouse.click(other["x"] + other["width"] / 2, other["y"] + other["height"] / 2)
        expect(page.locator("#zone-panel")).to_have_count(0)
        assert page.locator("#zone-map").evaluate("map => Boolean(map.closest('[inert]'))")
        assert saved == []
        held.release()
        page.wait_for_url(f"**/gradient?scheme={SAMPLE}")
        assert len(saved) == 2 and watched.page_errors == []
    assert json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text())["zones"] == {"yL": {"z1": SEARCH}}
