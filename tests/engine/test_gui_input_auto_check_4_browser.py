"""#1265 單元 6 第三趟：複查兩錯與兩題只考現況的補題。

頁底兩顆用的判決要是按下那一刻的表單問過的：改了牆面表單、沒離開那一格就按「交給程式最佳化 ▸」，延後的那一次自動檢查
當場做掉、等它之後的限制判決，超過就不存、不跳；「照現在牆上貼的算一次」照它自己的規則（限制不擋）。按的是哪一顆在按下那一刻
就記住：等判決的時候使用者把焦點移到別格，判決回來不捲、不搶焦點。另兩題只考現況：自己有交接的格子（換種類、套預設）各剛好問
一次、切進實測草稿不問；平面圖檢查舊的那一次在新的那一次等分區縮放時回來，不准把舊結果畫上去。方案、數字都是考卷自己編的。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Route, expect

from tests.engine._surface_zone_cases import NEW
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_auto_check_2_browser import _bottom, _sent
from tests.engine.test_gui_input_auto_check_3_browser import (
    SETTLED, _ceiling_form, _ceiling_room, _quiet_but_refused_or_previewed, _requests)
from tests.engine.test_gui_input_auto_check_browser import ROOM, _commit, _hold, _let_go, _opened, _quiet_but_refused, _room, _seen

OVER = "檢查沒過：1 條超過限制"


def _saved_thickness(tmp_path: Path) -> float:
    """磁碟上那一份方案天花那層棉的厚度。"""
    document = json.loads((tmp_path / "schemes" / f"{ROOM}.json").read_text(encoding="utf-8"))
    return cast(float, document["scene"][NEW]["ceiling"][0]["material"]["layers"][0]["thickness_m"])


def test_search_pressed_right_after_a_wall_form_edit_past_the_limit_uses_the_verdict_on_that_edit(tmp_path: Path,
                                                                                                 browser: Browser) -> None:
    """複查發現 1：牆面表單把天花那層棉改厚到超過上限、沒離開那一格就直接按「交給程式最佳化 ▸」：不存、不跳，頂條紅字。"""
    _ceiling_room(tmp_path, 0.12)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        _ceiling_form(page)
        sent, left = _sent(page), []

        def leave(route: Route) -> None:
            left.append(route.request.url)
            route.fulfill(status=200, body="ok")

        page.route("**/gradient**", leave)
        field = page.locator("#layer-ceiling-0-thickness_m")
        field.focus()
        field.fill("0.27")
        page.locator("#zone-to-search").click()
        page.wait_for_timeout(1500)
        assert sent == [] and left == [], (sent, left)
        page.wait_for_function(SETTLED)
        expect(page.locator("#check-state")).to_have_text(OVER)
        assert _saved_thickness(tmp_path) == 0.12
        _quiet_but_refused(watched)


def test_calculate_pressed_right_after_a_wall_form_edit_past_the_limit_still_calculates(tmp_path: Path,
                                                                                       browser: Browser) -> None:
    """複查發現 1 的另一顆：同樣改超過、沒離開就按「照現在牆上貼的算一次」：限制不擋算，照樣存、照樣開算；頂條照講超過。"""
    _ceiling_room(tmp_path, 0.12)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        _ceiling_form(page)
        # 開算請求送到就算數（不真的算）：考卷自己回 409，頁面照講。
        page.route("**/api/runs", lambda route: route.fulfill(status=409, content_type="application/json",
                   body=json.dumps({"error": "考卷不開算"})) if route.request.method == "POST" else route.continue_())
        sent = _sent(page)
        field = page.locator("#layer-ceiling-0-thickness_m")
        field.focus()
        field.fill("0.27")
        page.locator("#calculate").click()
        expect(page.locator("#messages")).to_have_text("考卷不開算")
        assert [line.split(" ")[0] for line in sent] == ["PUT", "POST"], sent
        assert _saved_thickness(tmp_path) == 0.27
        page.wait_for_function(SETTLED)
        expect(page.locator("#check-state")).to_have_text(OVER)
        assert watched.page_errors == []
        assert all(item == "409 /api/runs" for item in watched.failed_responses), watched.failed_responses


def test_search_waiting_for_limits_leaves_alone_a_user_who_moved_to_another_box(tmp_path: Path, browser: Browser) -> None:
    """複查發現 2：限制判決還在路上時按「交給程式最佳化 ▸」，按下之後使用者點進房間高那一格、捲回頁頂：判決回來是超過，
    不存、不跳，但不捲、不搶焦點（按的是哪一顆在按下那一刻就記住）。"""
    _ceiling_room(tmp_path, 0.12)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _opened(page)
        _ceiling_form(page)
        limits = _hold(page, "**/api/zone-limits")
        _commit(page, "#layer-ceiling-0-thickness_m", "0.27")
        _seen(page, limits, "限制判決")
        _bottom(page)
        sent = _sent(page)
        page.locator("#zone-to-search").click()
        page.wait_for_timeout(100)
        page.locator("#room-Lz").focus()
        page.evaluate("window.scrollTo(0, 0)")
        _let_go(page, "**/api/zone-limits", limits)
        page.wait_for_function(SETTLED)
        page.wait_for_timeout(300)
        expect(page.locator("#check-state")).to_have_text(OVER)
        assert page.evaluate("document.activeElement?.id") == "room-Lz"
        assert page.evaluate("Math.round(window.scrollY)") == 0
        assert sent == []
        _quiet_but_refused(watched)


def test_controls_with_their_own_handoff_ask_once_and_a_measured_draft_asks_none(tmp_path: Path, browser: Browser) -> None:
    """只考現況（複查不擋合併第 1 條）：換種類、套預設那兩格自己有交接，各剛好一次平面圖檢查（不准再多一次）；
    換成實測阻抗只開草稿、還沒改到方案，零次。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        page.locator("#elevation-wall-x0").click()
        page.locator("#edit-wall-original").click()
        plans = _requests(page, "/api/plan")
        page.locator("#material-kind-x0").select_option("assembly")
        page.wait_for_function(SETTLED)
        page.wait_for_load_state("networkidle")
        assert plans == ["POST"], plans
        presets = cast(list[str], page.evaluate("() => inputDefaults.construction_presets.map(p => p.id)"))
        plans.clear()
        page.locator("#preset-x0").select_option(presets[-1])
        page.wait_for_function(SETTLED)
        page.wait_for_load_state("networkidle")
        assert plans == ["POST"], plans
        plans.clear()
        page.locator("#material-kind-x0").select_option("measured_impedance")
        expect(page.locator("#measured-draft-x0")).to_be_visible()
        page.wait_for_load_state("networkidle")
        assert plans == [], plans
        _quiet_but_refused_or_previewed(watched)


def test_an_old_plan_answer_landing_while_the_newer_check_waits_for_a_resize_is_not_drawn(tmp_path: Path,
                                                                                        browser: Browser) -> None:
    """只考現況（複查不擋合併第 2 條）：切過區的房間先把左喇叭搬到房外（那一次平面圖檢查扣在路上），再把房間加長到裝得下
    （分區縮放扣在路上，新的那一次在等它）。舊的那一次先回來：已經不是最新，訊息列不准寫它的「座標超出房間」、
    頂條不准換成它的結果；縮放回來、新的那一次問完，頂條是「檢查通過」。"""
    _room(tmp_path, deep_ceiling=True)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        plans: list[Route] = _hold(page, "**/api/plan")
        _commit(page, "#speaker-left-x", "7.2")
        _seen(page, plans, "搬喇叭那一次平面圖檢查")
        resizes = _hold(page, "**/api/zone-resize")
        # 打字那一下就問一次預覽與牆面倍數（分區還是舊尺寸，伺服器回不收的原因、寫進訊息列）：等那一句寫進訊息列才記，
        # 負載重時它晚到不算舊圖（只等回應到還不夠，頁面程式寫字在回應之後）。
        _commit(page, "#room-Lx", "7.5")
        page.wait_for_function("document.getElementById('messages').textContent.includes('外框必須正好蓋滿整面')")
        page.wait_for_load_state("networkidle")
        _seen(page, resizes, "加長那一次分區縮放")
        said = page.locator("#messages").inner_text()
        with page.expect_response("**/api/plan"):
            plans[0].continue_()
        page.wait_for_timeout(300)
        assert "座標超出房間" not in page.locator("#messages").inner_text(), page.locator("#messages").inner_text()
        assert page.locator("#messages").inner_text() == said
        expect(page.locator("#check-state")).to_have_text("檢查中…")
        page.unroute("**/api/plan")
        _let_go(page, "**/api/zone-resize", resizes)
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        _quiet_but_refused_or_previewed(watched)
