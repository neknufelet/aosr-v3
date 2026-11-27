"""#1265 單元 3 之二第二趟追加（第一趟另一家審查找到的四個真錯）：切牆、清掉、復原、改做法之後，畫面先重畫、限制判決後到。
這一段空檔裡不准拿上一版判決去畫（配對、沒給上限、逐組上限），也不准因為對不上就丟錯中斷整個動作；
清掉之後等圖面檢查的空檔把房間高度清空，檢查回來不准把清掉鈕塞回去。

方案資料都是考卷自己編的：109 號房（_surface_zone_cases）改成自編尺寸與切法（同 test_gui_input_wall_list_browser 的 _pair_room）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Error, Page, Route, expect

from aosr.reporting.scheme import Scheme
from tests.engine._zone_material_steps import pick_fixed_row
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._construction_cases import construction
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine._zone_panel_steps import choose_fixed, open_alone
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_input_wall_list_browser import WALL, WOOL, _box, _pair_room
from tests.engine.test_gui_zone_choices_browser import _sample_library
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _drag, _open_scheme, _write
from tests.engine.test_gui_zone_limits_browser import _deep_user_library
from tests.engine.test_gui_zone_modes_browser import _write_room109_design
from tests.engine.test_gui_zone_panel_show_all_browser import CEILING
from tests.engine._input_page_steps import limits_all_clear, recheck

Doc = dict[str, object]


def _settled(page: Page) -> None:
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def _verdict_shown(page: Page) -> None:
    page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")


def _in_flight_words(page: Page, wall: str) -> list[str]:
    """判決還在路上時右欄這一面每一區那一句：不准寫上一版判決的上限、配對（第 1、3 條同一個訊號），只寫還在核對。"""
    rows = page.locator(f"#elevation-list-zones .zone-item[data-zone^='{wall}/'] .what")
    expect(rows.first).to_be_visible()
    texts = rows.all_inner_texts()
    for text in texts:
        assert "沒給上限" not in text and "一對" not in text, text
    return texts


def _ran_to_the_end(page: Page) -> None:
    """整個動作做完了：圖面檢查回來寫了它自己那一句（綠字），不是中途丟錯被訊息列接住的那一句（橘字）。"""
    _settled(page)
    _verdict_shown(page)
    expect(page.locator("#messages")).to_have_class("ok")
    expect(page.locator("#messages")).not_to_contain_text("Cannot read")


def _y0_cut_in_two(tmp_path: Path) -> None:
    """_pair_room 那一間，左牆 y0 先切成兩塊（z1、auto1），兩塊都由程式最佳化：跟右牆整面配不成一對（沒配成）。
    清掉左牆之後左牆回到整面一區（做法沿用由程式最佳化），判決回來就跟右牆整面配成一對。"""
    _pair_room(tmp_path)
    path = tmp_path / "schemes" / "room109.json"
    document = cast(Doc, json.loads(path.read_text(encoding="utf-8")))
    zones = cast(dict[str, list[Doc]], cast(Doc, document["scene"])["surface_zones_by_wall"])
    zones["y0"] = [_box("z1", (0.0, 3.0), (0.0, 3.0)), _box("auto1", (3.0, 6.0), (0.0, 3.0))]
    path.write_text(Scheme.model_validate(document).model_dump_json(), encoding="utf-8")
    design_path = tmp_path / "designs" / "room109.json"
    design = cast(Doc, json.loads(design_path.read_text(encoding="utf-8")))
    search = {"mode": "search", "allowed": WOOL}
    cast(dict[str, Doc], design["zones"])["y0"] = {"z1": search, "auto1": search}
    design_path.write_text(json.dumps(design), encoding="utf-8")


def test_drawing_a_box_on_a_paired_wall_runs_to_the_end(tmp_path: Path, browser: Browser) -> None:
    """第 1 條之一（畫框）：左右兩面整面都由程式最佳化、配成一對；在左牆畫一個框。切完重畫的時候判決還是上一版（配的是左牆整面），
    不准拿它對新切法而丟錯：整個動作做完（存檔前的改動標記、圖面檢查、限制重判都跑到）。"""
    _pair_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _verdict_shown(page)
        page.locator("#elevation-wall-y0").click()
        expect(page.locator("#wall-zone-what-y0-whole_wall")).to_contain_text("和右牆整面一對")
        before = cast(int, page.evaluate("zoneLimitsGeneration"))
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        _drag(page, 6.0, 3.0, (0.5, 0.5), (1.5, 1.5))
        _settled(page)
        assert held, "限制檢查應該還在路上"
        words = _in_flight_words(page, "y0")
        assert len(words) > 1 and all("總厚度上限還在核對" in text for text in words), words
        _release(held)
        page.unroute("**/api/zone-limits")
        _ran_to_the_end(page)
        assert cast(int, page.evaluate("zoneLimitsGeneration")) > before
        assert len(cast(list[object], page.evaluate("scheme.scene.surface_zones_by_wall.y0"))) > 1
        expect(page.locator("#elevation-undo-clear")).to_have_count(0)
        expect(page.locator("#elevation-clear")).to_be_enabled()
        _assert_quiet(watched)


def test_clearing_a_paired_wall_runs_to_the_end_and_offers_undo(tmp_path: Path, browser: Browser) -> None:
    """第 1 條之二（清掉）：前牆左右兩條 auto1、auto2 由程式最佳化、配成一對；清掉前牆的切法。重畫時判決還是上一版（配的是
    已經不在的那兩條），不准丟錯：整個動作做完，復原鈕出現。"""
    _pair_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _verdict_shown(page)
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#wall-zone-what-x0-auto1")).to_contain_text("一對")
        _cut_by(page, "#elevation-clear")
        _ran_to_the_end(page)
        expect(page.locator("#elevation-undo-clear")).to_be_visible()
        expect(page.locator("#elevation-cleared")).to_contain_text("已清掉前牆 9 區")
        _assert_quiet(watched)


def test_undoing_a_clear_on_a_paired_wall_runs_to_the_end(tmp_path: Path, browser: Browser) -> None:
    """第 1 條之三（復原）：左牆兩塊清掉回到整面、判決回來跟右牆整面配成一對；按復原。重畫時判決還是上一版（配的是左牆整面），
    不准丟錯：左牆回到兩塊、整個動作做完。"""
    _y0_cut_in_two(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _verdict_shown(page)
        page.locator("#elevation-wall-y0").click()
        _cut_by(page, "#elevation-clear")
        _settled(page)
        _verdict_shown(page)
        expect(page.locator("#wall-zone-what-y0-whole_wall")).to_contain_text("和右牆整面一對")
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        page.locator("#elevation-undo-clear").click()
        _settled(page)
        assert held, "限制檢查應該還在路上"
        words = _in_flight_words(page, "y0")
        assert len(words) == 2 and all("總厚度上限還在核對" in text for text in words), words
        _release(held)
        page.unroute("**/api/zone-limits")
        _ran_to_the_end(page)
        expect(page.locator("#elevation-note")).to_contain_text("復原了：左牆回到清掉前的 2 區")
        assert [zone["id"] for zone in cast(list[Doc], page.evaluate("scheme.scene.surface_zones_by_wall.y0"))] == ["z1", "auto1"]
        _assert_quiet(watched)


def test_blanking_the_height_while_a_clear_is_being_checked_never_puts_the_clear_button_back(tmp_path: Path, browser: Browser) -> None:
    """第 2 條：清掉前牆之後圖面檢查攔著不放（頁面已解鎖），這時把房間高度清空：立面圖畫不出來，清掉／復原那一格空著；
    檢查回來不准把「清掉這一面的切法」塞回去（同 test_gui_input_clear_undo_browser 那一題清空高度的不變量）。"""
    _write(tmp_path, room109_document())
    held_plan: list[Route] = []
    held_limits: list[Route] = []
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, ROOM_109["Ly"], ROOM_109["Lz"], (0.3, 0.3), (1.3, 1.3))
        _settled(page)
        page.route("**/api/plan", lambda route: held_plan.append(route))
        page.route("**/api/zone-limits", lambda route: held_limits.append(route))
        _cut_by(page, "#elevation-clear")
        # 攔截的回呼是瀏覽器那邊排隊送回來的：機器忙時按完馬上看會是空的，給它跑到（5b 第三趟整套 -n 8 紅過一次）。
        for _ in range(200):
            if held_plan:
                break
            page.wait_for_timeout(25)
        assert held_plan, "圖面檢查應該還在路上"
        # 審查設計洞 1：切完那一刻就畫「已清掉…｜復原」，不等圖面檢查回來（檢查丟錯也看得到復原）。
        expect(page.locator("#elevation-undo-clear")).to_be_visible()
        page.locator("#room-Lz").click()
        page.keyboard.press("Control+A")
        page.keyboard.press("Backspace")
        expect(page.locator("#elevation-size")).to_have_text("先把房間的長、寬、高都填好正數，才畫得出這一面。")
        page.evaluate("() => { window.planBefore = shownPlan; }")
        _release(held_plan)
        page.unroute("**/api/plan")
        # 檢查回來那一刻（shownPlan 換成這一次的）之後同一串就輪到清掉鈕那一段重畫；換了就表示那一段已經跑完。
        page.wait_for_function("shownPlan !== window.planBefore")
        expect(page.locator("#elevation-clear-slot button")).to_have_count(0)
        expect(page.locator("#elevation-size")).to_have_text("先把房間的長、寬、高都填好正數，才畫得出這一面。")
        _release(held_limits)
        page.unroute("**/api/zone-limits")
        _settled(page)
        expect(page.locator("#elevation-clear-slot button")).to_have_count(0)
        _assert_quiet(watched)


def _hold_limits_and_apply(page: Page, wall: str, key: str, mode: str, material: str | None = None) -> list[Route]:
    """攔住限制檢查，在「這間房貼什麼」小框把那一區改成 mode（指定貼的話選 material）、套用、回輸入頁；判決還在路上就停。"""
    held: list[Route] = []
    page.route("**/api/zone-limits", lambda route: held.append(route))
    open_alone(page, wall, key)
    page.locator(f"#zone-panel-mode-{mode}").check()
    if material:
        pick_fixed_row(page, material)
    page.locator("#zone-panel-apply").click()
    expect(page.locator("#zone-panel")).to_have_count(0)
    page.wait_for_function("""() => { const zones = document.getElementById('zone-page');
        return !zones.inert && !inputBusy && !(Number(zones.dataset.changing) > 0) && zoneLimitsReceived !== zoneLimitsGeneration; }""")
    page.locator("#zone-page-close").click()
    assert held, "限制檢查應該還在路上"
    assert page.evaluate("zoneLimitsReceived !== zoneLimitsGeneration")
    return held


def test_keep_to_search_says_the_cap_is_being_checked_not_that_there_is_none(tmp_path: Path, browser: Browser) -> None:
    """第 3 條：前牆整面原本「不貼，保留原樣」，房間預設上限 15 公分；在小框改成由程式最佳化。判決回來之前右欄那一句不准說
    「沒給上限」（上一版判決對的是保留原樣，沒有上限可講），要說還在核對；判決回來才寫上限。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": 0.15,
                                     "zones": {"x0": {"whole_wall": {"mode": "keep_original"}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _verdict_shown(page)
        held = _hold_limits_and_apply(page, "x0", "whole_wall", "search")
        what = page.locator("#wall-zone-what-x0-whole_wall")
        expect(what).to_contain_text("總厚度上限還在核對")
        expect(what).not_to_contain_text("沒給上限")
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(what).to_contain_text("總厚度最多 15 公分（房間預設）")
        _assert_quiet(watched)


def test_search_to_fixed_drops_the_group_caps_before_the_verdict_comes_back(tmp_path: Path, browser: Browser) -> None:
    """第 4 條：前牆整面由程式最佳化、清單裡有一筆自己限制比房間預設深的（判決寫逐組上限）；在小框改成指定貼一種。
    判決回來之前右欄那一區底下不准還掛著上一版的逐組上限（逐組上限只給由程式最佳化的區）。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": 0.06,
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k", "none", "deep_rock_wool"]}}}})
    (tmp_path / "materials.json").write_text(_deep_user_library(), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _verdict_shown(page)
        expect(page.locator("#wall-zone-depths-x0-whole_wall")).to_have_count(1)
        held = _hold_limits_and_apply(page, "x0", "whole_wall", "fixed", "glass_wool_48k_50mm")
        expect(page.locator("#wall-zone-x0-whole_wall .tag")).to_have_text("指定貼一種")
        expect(page.locator("#wall-zone-depths-x0-whole_wall")).to_have_count(0)
        expect(page.locator("[id^=depth-target-x0-whole_wall-], #depth-cap-x0-whole_wall")).to_have_count(0)
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(page.locator("#wall-zone-depths-x0-whole_wall")).to_have_count(0)
        _assert_quiet(watched)



def test_a_new_deepest_on_a_search_zone_says_checking_until_the_verdict_comes_back(tmp_path: Path, browser: Browser) -> None:
    """第 3 條同一個訊號（右欄那一區的上限 capWords）：前牆整面由程式最佳化、照房間預設最多 15 公分；在小框把「最深」改成 10。
    判決回來之前右欄不准還寫上一版的「最多 15 公分」，寫還在核對；回來才寫 10 公分（這一塊）。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": 0.15,
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": WOOL}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _verdict_shown(page)
        what = page.locator("#wall-zone-what-x0-whole_wall")
        expect(what).to_contain_text("總厚度最多 15 公分（房間預設）")
        depths = page.locator("#wall-zone-depths-x0-whole_wall")
        expect(depths).to_have_count(1)  # 上一版判決寫了逐組上限（審查建議這一題順手釘住逐組上限那一道）
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        open_alone(page, "x0", "whole_wall")
        page.locator("#zone-panel-depth").fill("10")
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        page.wait_for_function("""() => { const zones = document.getElementById('zone-page');
            return !zones.inert && !inputBusy && !(Number(zones.dataset.changing) > 0) && zoneLimitsReceived !== zoneLimitsGeneration; }""")
        page.locator("#zone-page-close").click()
        assert held, "限制檢查應該還在路上"
        expect(what).to_contain_text("總厚度上限還在核對")
        expect(what).not_to_contain_text("15 公分")
        expect(depths).to_have_count(0)
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(what).to_contain_text("總厚度最多 10 公分（這一塊）")
        _assert_quiet(watched)


def test_a_new_deepest_on_a_search_zone_drops_the_group_caps_until_the_verdict_comes_back(tmp_path: Path, browser: Browser) -> None:
    """逐組上限（depthTargetLines）判決新鮮那一道（#1265 單元 3 之二收尾）：前牆整面由程式最佳化、照房間預設 6 公分，
    清單裡有一筆自己限制比房間預設深的（判決寫逐組上限）；在小框把「最深」改成 5，做法照樣是由程式最佳化。判決回來之前
    右欄那一區底下不准還掛著上一版（6 公分）的逐組上限；回來才寫 5 公分（這一塊）的那一份。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "room_max_depth_m": 0.06,
                                     "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k", "none", "deep_rock_wool"]}}}})
    (tmp_path / "materials.json").write_text(_deep_user_library(), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _verdict_shown(page)
        cap = page.locator("#depth-cap-x0-whole_wall")
        expect(cap).to_have_text("個案上限 0.06 公尺（房間預設）；每一組跟自己的上限取較嚴：")
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        open_alone(page, "x0", "whole_wall")
        page.locator("#zone-panel-depth").fill("5")
        page.locator("#zone-panel-apply").click()
        expect(page.locator("#zone-panel")).to_have_count(0)
        page.wait_for_function("""() => { const zones = document.getElementById('zone-page');
            return !zones.inert && !inputBusy && !(Number(zones.dataset.changing) > 0) && zoneLimitsReceived !== zoneLimitsGeneration; }""")
        page.locator("#zone-page-close").click()
        assert held, "限制檢查應該還在路上"
        expect(page.locator("#wall-zone-x0-whole_wall .tag")).to_have_text("由程式最佳化")
        expect(page.locator("#wall-zone-depths-x0-whole_wall")).to_have_count(0)
        expect(page.locator("[id^=depth-target-x0-whole_wall-], #depth-cap-x0-whole_wall")).to_have_count(0)
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(page.locator("#wall-zone-depths-x0-whole_wall")).to_have_count(1)
        expect(cap).to_have_text("個案上限 0.05 公尺（這一塊）；每一組跟自己的上限取較嚴：")
        _assert_quiet(watched)


def test_a_construction_put_back_by_a_new_method_is_not_named_until_the_verdict_comes_back(tmp_path: Path, browser: Browser) -> None:
    """裸構造身分（constructionAt 讀判決的 picked_constructions）同一個訊號（#1265 單元 3 之二收尾追加第 3 條）：天花整面指定貼
    一筆自編構造（伺服器認得是材料庫那一筆），再在小框改回由程式最佳化（材料放回原本的天花）。判決回來之前右欄那一句不准還說
    現在存的是那一筆自編構造（上一版判決對的是貼著它的時候）。"""
    built = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == CEILING)
    document = room109_document()
    cast(dict[str, list[Doc]], cast(Doc, document["scene"])["surface_zones_by_wall"])["ceiling"][0]["material"] = built
    _write(tmp_path, document)
    original = json.loads(json.dumps(document)) | {"scheme_id": "orig109"}
    (tmp_path / "schemes" / "orig109.json").write_text(json.dumps(original), encoding="utf-8")
    (tmp_path / "designs").mkdir(exist_ok=True)
    (tmp_path / "designs" / "room109.json").write_text(json.dumps({
        "schema_version": "aosr.zone_design.v1", "scheme_id": "room109", "original_scheme_id": "orig109",
        "left_right_symmetry": "off", "zones": {}}), encoding="utf-8")
    _sample_library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        assert page.request.put(f"{base}/api/materials/table_sample", data=construction("rigid", id="table_sample")).ok
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-ceiling").click()
        _verdict_shown(page)
        choose_fixed(page, "ceiling", "whole_wall", "table_sample")
        _verdict_shown(page)
        what = page.locator("#wall-zone-what-ceiling-whole_wall")
        held = _hold_limits_and_apply(page, "ceiling", "whole_wall", "search")
        expect(what).to_contain_text("總厚度上限還在核對")
        expect(what).not_to_contain_text("天花自編樣品")
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(what).not_to_contain_text("天花自編樣品")
        _assert_quiet(watched)


def test_turning_symmetry_on_does_not_blame_the_seat_before_the_verdict_comes_back(tmp_path: Path, browser: Browser) -> None:
    """左右對稱從關切到開（setSymmetry，#1265 單元 3 之二收尾追加第 4 條）：上一版判決是關著時問的（沒有鏡子）。新判決回來
    之前，選取框底下不准拿上一版說「請先檢查方案，讓主位對齊兩支喇叭的中點」（這一間是對稱的），立面圖上方那一句照講還在確認。"""
    _write_room109_design(tmp_path, {"left_right_symmetry": "off", "zones": {}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _verdict_shown(page)
        _drag(page, 4.0, 2.8, (0.3, 0.3), (1.3, 1.3))
        _ran_to_the_end(page)
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        page.locator("#elevation-symmetry-on").click()
        expect(page.locator("#elevation-axis-words")).to_have_text("正在確認鏡子在哪…")
        status = page.locator(".elevation-pair-status")
        expect(status).to_have_text("")
        assert held, "限制檢查應該還在路上"
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(status).not_to_contain_text("請先檢查方案")
        _assert_quiet(watched)


def test_a_room_resize_does_not_reuse_the_verdict_while_the_new_one_is_on_its_way(tmp_path: Path, browser: Browser) -> None:
    """改房間尺寸（applyRoomResize，#1265 單元 3 之二收尾追加第 4 條順手查的那一條路）：分區照新尺寸調好就重畫，限制判決之後
    才重問。新判決回來之前右欄不准拿上一版（舊尺寸）寫「一對」、上限，寫還在核對；回來才照新的寫。"""
    _pair_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _verdict_shown(page)
        page.locator("#elevation-wall-y0").click()
        what = page.locator("#wall-zone-what-y0-whole_wall")
        expect(what).to_contain_text("和右牆整面一對")
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        page.locator("#room-Lx").fill("6.2")
        page.locator("#room-Lx").press("Tab")
        page.wait_for_function("!document.getElementById('elevation-list').inert && !window.pendingRoomResize()")
        page.wait_for_function("zoneLimitsReceived !== zoneLimitsGeneration")
        assert held, "限制檢查應該還在路上"
        words = _in_flight_words(page, "y0")
        assert all("總厚度上限還在核對" in text for text in words), words
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(what).to_contain_text("和右牆整面一對")
        _assert_quiet(watched)


# 第 2 段右欄每畫一次配對就記一筆（那時開著哪一份、配到幾區）：換方案那一下是同一個工作裡畫完的，等畫面看不到，只能這樣記。
_RECORD_PAIR_MARKS = """() => { window.pairMarksSeen = []; const original = window.zonePairMarks;
    window.zonePairMarks = () => { const marks = original(); window.pairMarksSeen.push({id: scheme.scheme_id, marks: marks.size}); return marks; }; }"""


def test_opening_another_scheme_never_pairs_its_zones_with_the_last_schemes_verdict(tmp_path: Path, browser: Browser) -> None:
    """換方案（changeInputScheme，#1265 單元 3 之二收尾追加設計洞）：新方案的格子先畫、做法與判決後換，中間那一次重畫不准拿上一份
    方案的判決配新方案的區。上一份左右牆整面配成一對（只點名整面，新方案也有那兩區）；新方案關著左右對稱，一對都不該有。"""
    _pair_room(tmp_path)
    path = tmp_path / "schemes" / "room109.json"
    document = cast(Doc, json.loads(path.read_text(encoding="utf-8")))
    cast(dict[str, list[Doc]], cast(Doc, document["scene"])["surface_zones_by_wall"])["x0"] = [{"area": "whole_wall", "material": WALL}]
    path.write_text(Scheme.model_validate(document).model_dump_json(), encoding="utf-8")
    design_path = tmp_path / "designs" / "room109.json"
    design = cast(Doc, json.loads(design_path.read_text(encoding="utf-8")))
    cast(dict[str, object], design["zones"]).pop("x0", None)
    design_path.write_text(json.dumps(design), encoding="utf-8")
    plain = room109_document() | {"scheme_id": "plain109"}
    (tmp_path / "schemes" / "plain109.json").write_text(Scheme.model_validate(plain).model_dump_json(), encoding="utf-8")
    (tmp_path / "designs" / "plain109.json").write_text(json.dumps({
        "schema_version": "aosr.zone_design.v1", "scheme_id": "plain109", "left_right_symmetry": "off", "zones": {}}), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _verdict_shown(page)
        page.locator("#elevation-wall-y0").click()
        expect(page.locator("#wall-zone-what-y0-whole_wall")).to_contain_text("和右牆整面一對")
        page.evaluate(_RECORD_PAIR_MARKS)
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        _open_scheme(page, "plain109", checks_held=True)
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg="plain109")
        _settled(page)
        seen = cast(list[dict[str, object]], page.evaluate("window.pairMarksSeen"))
        assert any(entry["id"] == "plain109" for entry in seen), seen
        assert [entry for entry in seen if entry["id"] == "plain109" and entry["marks"]] == [], seen
        # 打開時第 2 段換到那一份的前牆（#1265 單元 5c）：點回左牆再看右欄那一區。
        page.locator("#elevation-wall-y0").click()
        expect(page.locator("#wall-zone-what-y0-whole_wall")).not_to_contain_text("一對")
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(page.locator("#wall-zone-what-y0-whole_wall")).not_to_contain_text("一對")
        _assert_quiet(watched)


def test_the_mirror_sentence_says_not_confirmed_when_the_check_fails_not_still_checking(tmp_path: Path, browser: Browser) -> None:
    """立面圖上方鏡子那一句（inputAxisWords，#1265 單元 3 之二收尾追加設計洞）：判決在路上寫「正在確認鏡子在哪…」；
    限制檢查失敗（連不上）就照實講沒確認成，不准一直寫正在確認。"""
    _write_room109_design(tmp_path, {"zones": {}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _verdict_shown(page)
        words = page.locator("#elevation-axis-words")
        expect(words).to_contain_text("鏡子：喇叭中線")
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        page.locator("#elevation-symmetry-off").click()
        page.locator("#elevation-symmetry-on").click()
        expect(words).to_have_text("正在確認鏡子在哪…")
        assert held, "限制檢查應該還在路上"
        for route in held:  # 先斷再 unroute（unroute 會把還扣著的放掉）；關掉那一下問的那一次已經被蓋掉、頁面自己放棄了，不用再斷
            try:
                route.abort()
            except Error as error:
                assert "already handled" in str(error), error
        page.unroute("**/api/zone-limits")
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimits.status === 'error'")
        expect(words).not_to_contain_text("正在確認")
        expect(words).to_contain_text("鏡子在哪沒確認成")
        # 失敗之後再問一次（關掉再打開左右對稱）：新的判決在路上時照樣寫「正在確認」，不拿上一次失敗那份講（收尾植錯：
        # 那一句只看「有判決、不是 checked」會活下來，這一段釘住 zoneLimitsFresh 那一道）。
        held.clear()
        page.route("**/api/zone-limits", lambda route: held.append(route))
        page.locator("#elevation-symmetry-off").click()
        page.locator("#elevation-symmetry-on").click()
        expect(words).to_have_text("正在確認鏡子在哪…")
        assert held, "限制檢查應該還在路上"
        _release(held)
        page.unroute("**/api/zone-limits")
        _verdict_shown(page)
        expect(words).to_contain_text("鏡子：喇叭中線")
        assert watched.page_errors == []


def test_a_running_scheme_whose_progress_cannot_be_read_still_gets_its_limits_checked(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二收尾第二趟 B3：打開一份正在計算的方案、接回計算進度失敗（伺服器回錯）。打開時已經把上一份的判決作廢，
    限制檢查照樣要發，整頁不准卡在「還沒檢查限制／正在確認鏡子在哪…／總厚度上限還在核對」；進度讀不到的原因照樣寫在訊息列。"""
    _write_room109_design(tmp_path, {"zones": {"x0": {"whole_wall": {"mode": "search", "allowed": WOOL}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        _settled(page)
        page.evaluate("activeRuns = [{run_id: 'fake-run-1', scheme_id: 'room109', status: 'running', elapsed_s: 3, display_text: '計算中'}]")
        page.route("**/api/runs/fake-run-1", lambda route: route.fulfill(status=500, content_type="application/json",
                                                                          body=json.dumps({"error": "考卷編的：讀不到進度"})))
        asked: list[str] = []
        page.on("request", lambda request: asked.append(request.url) if request.url.endswith("/api/zone-limits") else None)
        page.locator("#scheme-menu-button").click()
        page.locator('#home-schemes [data-scheme="room109"] .home-open').click()
        page.wait_for_function("document.getElementById('save-id').value === 'room109'")
        _settled(page)
        # 等新的判決：開頁那一次的判決世代號也相等，只等 _verdict_shown 會在進度請求還在路上時就斷言（5b 第二趟審查 6B）。
        page.wait_for_function("zoneLimitsFresh()")
        _verdict_shown(page)
        assert asked, "打開之後應該問過限制"
        expect(page.locator("#messages")).to_contain_text("考卷編的：讀不到進度")
        page.locator("#elevation-wall-x0").click()
        limits_all_clear(page)
        expect(page.locator("#elevation-axis-words")).to_contain_text("鏡子：喇叭中線")
        expect(page.locator("#wall-zone-what-x0-whole_wall")).not_to_contain_text("總厚度上限還在核對")
        expect(page.locator("#messages")).to_contain_text("考卷編的：讀不到進度")


def test_the_mirror_sentence_says_checking_again_while_a_retry_after_a_failure_is_on_its_way(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二收尾第二趟 B4 之一：限制檢查失敗（鏡子那一句寫沒確認成）之後重問一次（#1265 單元 6 起沒有「檢查」鈕，改定一格自動重問）；重試還在路上時那一句要回到
    「正在確認鏡子在哪…」，不准停在上一次的「沒確認成」。回來之後照新的判決寫。"""
    _write_room109_design(tmp_path, {"zones": {}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _verdict_shown(page)
        words = page.locator("#elevation-axis-words")
        page.route("**/api/zone-limits", lambda route: route.abort())
        recheck(page)
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimits.status === 'error'")
        expect(words).to_contain_text("鏡子在哪沒確認成")
        page.unroute("**/api/zone-limits")
        held: list[Route] = []
        page.route("**/api/zone-limits", lambda route: held.append(route))
        recheck(page)
        page.wait_for_function("zoneLimitsReceived !== zoneLimitsGeneration")
        expect(words).to_have_text("正在確認鏡子在哪…")
        assert held, "重試應該還在路上"
        _release(held)
        page.unroute("**/api/zone-limits")
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimits.status === 'checked'")
        expect(words).to_contain_text("鏡子：喇叭中線")
        assert watched.page_errors == []


def test_the_mirror_sentence_asks_for_the_room_size_when_the_height_is_blanked_not_still_checking(tmp_path: Path, browser: Browser) -> None:
    """#1265 單元 3 之二收尾第二趟 B4 之二：判決新鮮、鏡子在哪已經知道，把房間高度格清空（只打字、還沒離開，預覽也還沒回來）：
    這時沒有發限制請求，那一句不准寫「正在確認鏡子在哪…」，要講房間尺寸沒填好。填回去就回到鏡子那一句。"""
    _write_room109_design(tmp_path, {"zones": {}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _verdict_shown(page)
        words = page.locator("#elevation-axis-words")
        expect(words).to_contain_text("鏡子：喇叭中線")
        held: list[Route] = []
        page.route("**/api/input-preview", lambda route: held.append(route))
        asked: list[str] = []
        page.on("request", lambda request: asked.append(request.url) if request.url.endswith("/api/zone-limits") else None)
        height = page.locator("#room-Lz")
        typed = height.input_value()
        height.fill("")
        expect(words).to_have_text("先把房間的長、寬、高都填好正數，才算得出鏡子在哪")
        assert asked == [] and page.evaluate("zoneLimitsFresh()") is True
        height.fill(typed)
        expect(words).to_contain_text("鏡子：喇叭中線")
        _release(held)
        page.unroute("**/api/input-preview")
        assert watched.page_errors == []


def _with_an_extra_message(route: Route) -> None:
    """切牆的回覆照伺服器的原樣，只多一句 message：清掉（沒有鏡像那一份）走到組那一句時會丟錯——拿來撐開「分區已經換了、
    中途丟錯」那一下（#1265 單元 3 之二收尾第二趟 D1；伺服器現在不會這樣回，考的是改寫分區之前就先標判決在路上）。"""
    response = route.fetch()
    body = cast(Doc, response.json())
    body["message"] = "考卷編的：伺服器多寫了一句"
    route.fulfill(response=response, json=body)


def test_a_cut_that_throws_after_rewriting_the_zones_leaves_no_stale_pairing(tmp_path: Path, browser: Browser) -> None:
    """D1：前牆左右兩條 auto1、auto2 配成一對；清掉前牆，回覆讓 applyCut 在改寫分區之後丟錯。那時判決要已經標成在路上：
    再點一次前牆重畫，不准拿上一版的配對對新切法而丟錯；改定一格重問之後照常。"""
    _pair_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        _verdict_shown(page)
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#wall-zone-what-x0-auto1")).to_contain_text("一對")
        page.route("**/api/zone-cut", _with_an_extra_message)
        page.locator("#elevation-clear").click()
        _settled(page)
        assert cast(list[Doc], page.evaluate("structuredClone(scheme.scene.surface_zones_by_wall.x0)"))[0]["area"] == "whole_wall"
        assert page.evaluate("zoneLimitsFresh()") is False
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#wall-zone-what-x0-whole_wall")).to_be_visible()
        assert watched.page_errors == []
        page.unroute("**/api/zone-cut")
        recheck(page)
        _verdict_shown(page)
        expect(page.locator("#elevation-list-zones .zone-item")).to_have_count(1)
        assert watched.page_errors == []


_PLAN_FAILS = {
    # 圖面檢查自己丟錯（回的不是 JSON）：那一句是 reply 寫的「伺服器的回覆讀不懂…」。
    "plan-throws": ({"status": 502, "content_type": "text/plain", "body": "bad gateway"}, "伺服器的回覆讀不懂（狀態碼 502）"),
    # 圖面檢查回 400（refreshPlan 回 false，自己已經寫了訊息列）。
    "plan-400": ({"status": 400, "content_type": "application/json", "body": json.dumps({"error": "考卷編的：圖面檢查有問題"})},
                 "考卷編的：圖面檢查有問題"),
}


@pytest.mark.parametrize("plan_fails", sorted(_PLAN_FAILS))
def test_a_running_scheme_whose_progress_and_plan_both_fail_keeps_both_reasons(tmp_path: Path, browser: Browser,
                                                                                plan_fails: str) -> None:
    """#1265 單元 3 之二收尾第三趟 E2：打開正在計算的方案，接回計算進度失敗、圖面檢查也失敗。訊息列兩邊的原因都要看得到：
    圖面檢查丟錯不准吞掉進度那一句，圖面檢查寫好的那一句也不准被進度那一句蓋掉。限制檢查照樣要問（第二趟 B3）。"""
    fulfill, plan_words = _PLAN_FAILS[plan_fails]
    _write_room109_design(tmp_path, {"zones": {"x0": {"whole_wall": {"mode": "search", "allowed": WOOL}}}})
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        _settled(page)
        page.evaluate("activeRuns = [{run_id: 'fake-run-1', scheme_id: 'room109', status: 'running', elapsed_s: 3, display_text: '計算中'}]")
        page.route("**/api/runs/fake-run-1", lambda route: route.fulfill(status=500, content_type="application/json",
                                                                          body=json.dumps({"error": "考卷編的：讀不到進度"})))
        page.route("**/api/plan", lambda route: route.fulfill(**fulfill))
        asked: list[str] = []
        page.on("request", lambda request: asked.append(request.url) if request.url.endswith("/api/zone-limits") else None)
        page.locator("#scheme-menu-button").click()
        page.locator('#home-schemes [data-scheme="room109"] .home-open').click()
        page.wait_for_function("document.getElementById('save-id').value === 'room109'")
        _settled(page)
        # 等新的判決：開頁那一次的判決世代號也相等，只等 _verdict_shown 會在進度請求還在路上時就斷言（5b 第二趟審查 6B）。
        page.wait_for_function("zoneLimitsFresh()")
        _verdict_shown(page)
        assert asked, "打開之後應該問過限制"
        messages = page.locator("#messages")
        expect(messages).to_contain_text("考卷編的：讀不到進度")
        expect(messages).to_contain_text(plan_words)
        expect(messages).to_have_class("notice")


def test_a_zones_link_to_a_running_scheme_whose_progress_cannot_be_read_still_opens_the_zone_page(tmp_path: Path,
                                                                                                  browser: Browser) -> None:
    """#1265 單元 3 之二收尾第三趟 E2（審查設計洞 2）：用 ?zones= 連結打開正在計算的方案，只有接回進度失敗。方案、做法、
    限制檢查都好了，照樣進「這間房貼什麼」；進度讀不到的原因照樣寫在訊息列（回輸入頁看得到）。
    開頁時先接回的是另一筆（清單第一筆），打開這一份時才接它自己那一筆、那一次失敗。"""
    _write_room109_design(tmp_path, {"zones": {"x0": {"whole_wall": {"mode": "search", "allowed": WOOL}}}})
    running = [{"run_id": "fake-other", "scheme_id": "other", "status": "running", "elapsed_s": 3, "display_text": "計算中"},
               {"run_id": "fake-run-1", "scheme_id": "room109", "status": "running", "elapsed_s": 3, "display_text": "計算中"}]
    with _serve(tmp_path) as base:
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        try:
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/api/runs", lambda route: route.fulfill(status=200, content_type="application/json",
                                                                  body=json.dumps({"running": running}))
                       if route.request.method == "GET" else route.continue_())
            page.route("**/api/runs/fake-*", lambda route: route.fulfill(status=500, content_type="application/json",
                                                                         body=json.dumps({"error": "考卷編的：讀不到進度"})))
            page.goto(f"{base}/?zones=room109")
            page.wait_for_function("document.getElementById('save-id').value === 'room109'")
            _settled(page)
            _verdict_shown(page)
            expect(page.locator("#zone-page")).to_be_visible()
            expect(page.locator("#zone-page h1")).to_have_text("這間房貼什麼")
            page.locator("#zone-page-close").click()
            expect(page.locator("#messages")).to_contain_text("考卷編的：讀不到進度")
            assert errors == []
        finally:
            page.close()
