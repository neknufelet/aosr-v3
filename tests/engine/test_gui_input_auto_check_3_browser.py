"""#1265 單元 6 第二趟：審查五錯與三條保險行。

改牆面表單（厚度、流阻）也重問限制，改超過、改回來頂條都跟著換；切過區的房間連改兩格尺寸，頂條不停在「檢查中…」；
牆面表單裡會改到方案的格子（下拉、文字格、實測表增刪列）改定就檢查，還沒套用的實測草稿不發；頁底兩顆碰上平面圖檢查
連不上也停在問題上；停在問題上之前重看一次（使用者已經改好、點進別格就不搶焦點）。另三題：頂條寫「限制沒檢查成」、
個別那一顆字不在時焦點退回整句、按的那一顆自己問過平面圖，放開後不再問。方案、設計設定、數字都是考卷自己編的。"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, expect

from tests.engine._surface_zone_cases import NEW, room109_document
from tests.engine.test_gui_browser import Watched, _open, _serve, browser as browser
from tests.engine.test_gui_input_auto_check_2_browser import FOCUSED, _bottom, _search_room, _sent
from tests.engine.test_gui_input_auto_check_browser import (
    ROOM, _commit, _hold, _let_go, _opened, _quiet_but_refused, _room, _seen)
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_zone_elevation_browser import _write

SETTLED = "!inputBusy && document.getElementById('check-state').dataset.kind !== 'pending'"


def _requests(page: Page, path: str) -> list[str]:
    """記下之後每一個打到 ``path`` 的請求。"""
    seen: list[str] = []
    page.on("request", lambda request: seen.append(request.method) if request.url.endswith(path) else None)
    return seen


def _quiet_but_cut_off(watched: Watched) -> None:
    """頁面沒丟錯、伺服器沒回錯；主控台只有考卷自己斷線的那幾條（net::ERR_FAILED）。"""
    assert watched.page_errors == []
    assert watched.failed_responses == []
    assert all("ERR_FAILED" in text for text in watched.console_errors), watched.console_errors


def _quiet_but_refused_or_previewed(watched: Watched) -> None:
    """頁面沒丟錯；伺服器回的錯只有照規則拒收的兩種：平面圖檢查 422、表單預覽 400（房間尺寸剛改、分區還沒縮放，或某一格
    收不下），主控台每一條對得上一個。"""
    assert watched.page_errors == []
    allowed = {"422 /api/plan", "400 /api/input-preview"}
    assert all(item in allowed for item in watched.failed_responses), watched.failed_responses
    assert len(watched.console_errors) == len(watched.failed_responses), (watched.console_errors, watched.failed_responses)
    assert all(re.search(r"status of (422|400)", text) for text in watched.console_errors), watched.console_errors


def _ceiling_room(tmp_path: Path, thickness: float) -> None:
    """天花整面貼一層多孔棉（貼在原本那面固定阻抗上）、指定貼一種；房間預設總厚度上限 0.2 公尺；前牆整面交給程式最佳化。"""
    document = room109_document(ROOM)
    zones = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])
    zones["ceiling"] = [{"area": "whole_wall", "material": {"kind": "mounted_on_wall", "layers": [
        {"kind": "porous", "thickness_m": thickness, "flow_resistivity_pa_s_m2": 12500.0, "model": "komatsu",
         "material_class": "fibrous"}], "wall": {"kind": "constant_impedance", "impedance_pa_s_m": 7000.0}}}]
    _write(tmp_path, document)
    (tmp_path / "designs").mkdir(exist_ok=True)
    (tmp_path / "designs" / f"{ROOM}.json").write_text(json.dumps({
        "schema_version": "aosr.zone_design.v1", "scheme_id": ROOM, "left_right_symmetry": "off", "room_max_depth_m": 0.2,
        "zones": {"ceiling": {"whole_wall": {"mode": "fixed"}},
                  "x0": {"whole_wall": {"mode": "search", "allowed": ["none", "glass_wool_48k"]}}}}), encoding="utf-8")


def _ceiling_form(page: Page) -> None:
    page.locator("#elevation-wall-ceiling").click()
    page.locator("#edit-wall-original").click()


def test_a_wall_form_edit_past_the_depth_limit_turns_the_top_bar_red_and_holds_back_search(tmp_path: Path,
                                                                                         browser: Browser) -> None:
    """發現 1（改超過）：牆面表單把天花那層棉改厚到超過上限，離開那一格就重問限制：頂條紅字「1 條超過限制」、搜尋鈕停用；
    改流阻也重問限制。"""
    _ceiling_room(tmp_path, 0.12)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        _ceiling_form(page)
        asked = _requests(page, "/api/zone-limits")
        _commit(page, "#layer-ceiling-0-thickness_m", "0.26")
        page.wait_for_function(SETTLED)
        expect(page.locator("#check-state")).to_have_text("檢查沒過：1 條超過限制")
        expect(page.locator("#zone-to-search")).to_be_disabled()
        assert asked, "改厚度沒有重問限制"
        asked.clear()
        _commit(page, "#layer-ceiling-0-flow_resistivity_pa_s_m2", "15500")
        page.wait_for_function(SETTLED)
        assert asked, "改流阻沒有重問限制"
        expect(page.locator("#check-state")).to_have_text("檢查沒過：1 條超過限制")
        _quiet_but_refused(watched)


def test_a_wall_form_edit_back_under_the_depth_limit_clears_the_red(tmp_path: Path, browser: Browser) -> None:
    """發現 1（改回來）：開頁時天花那層棉超過上限；在牆面表單改薄到上限內，頂條回「檢查通過」、搜尋鈕按得下。"""
    _ceiling_room(tmp_path, 0.26)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        expect(page.locator("#check-state")).to_have_text("檢查沒過：1 條超過限制")
        expect(page.locator("#zone-to-search")).to_be_disabled()
        _ceiling_form(page)
        _commit(page, "#layer-ceiling-0-thickness_m", "0.12")
        page.wait_for_function(SETTLED)
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        expect(page.locator("#zone-to-search")).to_be_enabled()
        _quiet_but_refused(watched)


def test_two_size_edits_whose_resizes_finish_out_of_order_never_leave_the_top_bar_checking(tmp_path: Path,
                                                                                         browser: Browser) -> None:
    """發現 2：切過區的房間先改長、再改寬，兩次分區縮放倒著回來（後改的先回）：所有請求都結束之後，頂條是最新那一次的結果，
    不停在「檢查中…」；先改的那一次醒來時已經不是最新，不送平面圖檢查。"""
    _room(tmp_path, deep_ceiling=True)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        resizes, plans = _hold(page, "**/api/zone-resize"), _hold(page, "**/api/plan")
        _commit(page, "#room-Lx", "6.6")
        _seen(page, resizes, "第一次分區縮放")
        _commit(page, "#room-Ly", "4.55")
        for _ in range(250):
            if len(resizes) >= 2:
                break
            page.wait_for_timeout(20)
        assert len(resizes) >= 2, "第二次分區縮放沒有出去"
        resizes[1].continue_()
        _seen(page, plans, "後改那一次的平面圖檢查")
        sent_before = len(plans)
        resizes[0].continue_()
        page.unroute("**/api/zone-resize")
        _release(resizes[2:])
        page.wait_for_timeout(400)
        assert len(plans) == sent_before, "先改的那一次醒來時已經過時，還是送了平面圖檢查"
        _let_go(page, "**/api/plan", plans)
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        expect(page.locator("#check-state")).to_have_attribute("data-kind", "ok")
        assert page.evaluate("planVerdict.state") == "ok"
        _quiet_but_refused_or_previewed(watched)


def _perforated(page: Page) -> None:
    """前牆換成材料構造、套一個有穿孔板的預設（孔排法那一格是一般下拉、沒有自己的交接）。"""
    page.locator("#elevation-wall-x0").click()
    page.locator("#edit-wall-original").click()
    page.locator("#material-kind-x0").select_option("assembly")
    page.wait_for_function(SETTLED)
    presets = cast(list[dict[str, object]], page.evaluate(
        "() => inputDefaults.construction_presets.map(p => ({id: p.id, kinds: p.material.layers.map(l => l.kind)}))"))
    page.locator("#preset-x0").select_option(next(
        str(p["id"]) for p in presets if any(kind in ("perforated", "micro_perforated") for kind in cast(list[str], p["kinds"]))))
    page.wait_for_function(SETTLED)


def test_wall_form_dropdowns_text_boxes_and_measured_rows_check_but_an_unapplied_draft_does_not(tmp_path: Path,
                                                                                              browser: Browser) -> None:
    """發現 3：牆面表單裡改到方案的都自己檢查一次——孔排法下拉、實測表出處文字格、「增加一列」「刪除這列」；
    還沒套用的實測草稿（底構造數字格）改了不發。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        _perforated(page)
        plans, check = _requests(page, "/api/plan"), page.locator("#check-state")
        hole = page.locator("#wall-editor-x0 select[data-layer-field='hole_pattern']")
        restore = hole.input_value()
        hole.select_option("")
        page.wait_for_function(SETTLED)
        assert plans == ["POST"], plans
        expect(check).to_have_text("檢查沒過：平面圖 1 條問題")
        hole.select_option(restore)
        page.wait_for_function(SETTLED)
        expect(check).to_have_text("檢查通過")
        page.locator("#material-kind-x0").select_option("measured_impedance")
        page.locator("#measured-paste-x0").fill("63 1200 0\n125 1200 0\n250 1200 0")
        page.locator("#measured-source-x0").fill("考卷自己編的阻抗管")
        plans.clear()
        draft = page.locator("#measured-draft-x0 input[type=number]").first
        draft.fill("0.07")
        draft.blur()
        page.wait_for_timeout(300)
        assert plans == [], "還沒套用的草稿改了也發平面圖檢查"
        page.locator("#measured-apply-x0").click()
        page.wait_for_function("Boolean(document.getElementById('measured-add-x0')) && " + SETTLED)
        expect(check).to_have_text("檢查通過")
        steps: list[tuple[Callable[[], None], str]] = [
            (partial(_commit, page, "#measured-source-x0", ""), "檢查沒過：平面圖 1 條問題"),
            (partial(_commit, page, "#measured-source-x0", "考卷自己編的阻抗管"), "檢查通過"),
            (page.locator("#measured-add-x0").click, "檢查沒過：平面圖 1 條問題"),
            (page.locator("#measured-delete-x0-3").click, "檢查通過")]
        for press, words in steps:
            plans.clear()
            press()
            page.wait_for_function(SETTLED)
            assert plans == ["POST"], plans
            expect(check).to_have_text(words)
        _quiet_but_refused_or_previewed(watched)


def test_both_bottom_buttons_stop_at_the_plan_when_its_check_cannot_reach_the_server(tmp_path: Path,
                                                                                   browser: Browser) -> None:
    """發現 4：按頁底兩顆時平面圖檢查連不上（請求斷線）：一樣算平面圖沒過——不存、不開算、不跳，捲回第 1 段平面圖、
    焦點在頂條「平面圖沒檢查成」那一顆。"""
    _search_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        # 視窗 800 高（不是 900）：#1265 說明字瘦身後整頁變矮，900 高捲到底時平面圖還露出約 25 像素，「先捲離平面圖」做不到；
        # 800 高捲到底時平面圖在視窗上緣外約 75 像素。斷言不動。
        page.set_viewport_size({"width": 1440, "height": 800})
        _opened(page)
        sent = _sent(page)
        page.route("**/api/plan", lambda route: route.abort())
        for button in ("#calculate", "#zone-to-search"):
            _bottom(page)
            expect(page.locator("#plan-xy")).not_to_be_in_viewport()
            page.locator(button).click()
            page.wait_for_function(FOCUSED, arg="check-plan")
            expect(page.locator("#check-state")).to_have_text("檢查沒過：平面圖沒檢查成")
            expect(page.locator("#plan-xy")).to_be_in_viewport()
            page.wait_for_function("!inputBusy")
        assert sent == []
        _quiet_but_cut_off(watched)


def test_a_stop_at_the_problem_that_comes_back_late_leaves_alone_a_user_who_already_fixed_it(tmp_path: Path,
                                                                                            browser: Browser) -> None:
    """發現 5：按「算」被平面圖擋下，停在問題上之前要等限制判決；等的時候使用者已經改好、點進別格：判決回來不捲、不搶焦點。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _opened(page)
        _commit(page, "#speaker-left-x", "20")
        expect(page.locator("#check-state")).to_have_attribute("data-kind", "problem")
        _bottom(page)
        limits = _hold(page, "**/api/zone-limits")
        page.locator("#calculate").click()
        _seen(page, limits, "限制判決")
        page.wait_for_function("!inputBusy")
        _commit(page, "#speaker-left-x", "1.2")
        page.locator("#room-Lz").focus()
        # 焦點留在房間高那一格、畫面捲到頁底（使用者接著往下看）：判決回來不准捲回第 1 段。
        _bottom(page)
        expect(page.locator("#plan-xy")).not_to_be_in_viewport()
        _let_go(page, "**/api/zone-limits", limits)
        page.wait_for_function(SETTLED)
        page.wait_for_timeout(300)
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        assert page.evaluate("document.activeElement?.id") == "room-Lz"
        expect(page.locator("#plan-xy")).not_to_be_in_viewport()
        _quiet_but_refused(watched)


def test_a_limits_check_that_fails_says_so_in_the_top_bar(tmp_path: Path, browser: Browser) -> None:
    """保險行 1：限制判決沒檢查成（伺服器回 500）：頂條紅字「限制沒檢查成」，不是「檢查通過」。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        page.route("**/api/zone-limits", lambda route: route.fulfill(status=500, content_type="application/json",
                                                                       body='{"error": "考卷讓限制判決壞掉"}'))
        _commit(page, "#room-Lz", "3.05")
        expect(page.locator("#check-state")).to_have_text("檢查沒過：限制沒檢查成")
        expect(page.locator("#check-state")).to_have_attribute("data-kind", "problem")
        expect(page.locator("#check-limits")).to_be_visible()
        assert watched.page_errors == []
        assert watched.failed_responses and all(item == "500 /api/zone-limits" for item in watched.failed_responses)


def test_the_focus_falls_back_to_the_whole_sentence_when_its_button_is_not_there(tmp_path: Path, browser: Browser) -> None:
    """保險行 4：按「算」被平面圖擋下、等限制判決的時候又有一次平面圖檢查在路上（頂條「檢查中…」，沒有那一顆字）：
    停在問題上時焦點放在頂條整句上。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _opened(page)
        _commit(page, "#speaker-left-x", "20")
        expect(page.locator("#check-state")).to_have_attribute("data-kind", "problem")
        _bottom(page)
        limits = _hold(page, "**/api/zone-limits")
        page.locator("#calculate").click()
        _seen(page, limits, "限制判決")
        page.wait_for_function("!inputBusy")
        plans = _hold(page, "**/api/plan")
        # 不動焦點、直接讓房間高那一格發改定（別處來的一次平面圖檢查），按下的那一顆還是焦點所在。
        page.evaluate("document.getElementById('room-Lz').dispatchEvent(new Event('change', {bubbles: true}))")
        _seen(page, plans, "又一次平面圖檢查")
        expect(page.locator("#check-state")).to_have_text("檢查中…")
        _let_go(page, "**/api/zone-limits", limits)
        page.wait_for_function(FOCUSED, arg="check-state")
        _let_go(page, "**/api/plan", plans)
        expect(page.locator("#check-state")).to_have_attribute("data-kind", "problem")
        _quiet_but_refused(watched)


@pytest.mark.parametrize("cut", [False, True], ids=["plain-room", "cut-room-resizing"])
def test_a_button_that_checks_the_plan_itself_is_not_followed_by_a_second_check_after_release(tmp_path: Path,
                                                                                            browser: Browser, cut: bool) -> None:
    """保險行 5（含設計洞 3）：改了房間長、沒離開那一格就按「儲存」：按下那一刻那一格失焦、自動檢查延到放開；儲存自己問過
    平面圖，放開後不再多問一次。切過區的房間（分區縮放還在路上，儲存先等它）也一樣。"""
    _room(tmp_path, deep_ceiling=cut)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        plans = _requests(page, "/api/plan")
        page.locator("#room-Lx").fill("6.6")
        page.locator("#save").click()
        page.wait_for_function(SETTLED)
        page.wait_for_load_state("networkidle")
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        assert plans == ["POST"], plans
        _quiet_but_refused_or_previewed(watched)
