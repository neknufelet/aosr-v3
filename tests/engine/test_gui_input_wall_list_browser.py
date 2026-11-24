"""第 2 段右欄清單「前牆這 N 區」（#1265 單元 3 審查真錯 5、設計洞 2、3）：

- 每一區那一句講「跟誰一對」用「這間房貼什麼」那一頁同一份配對（伺服器的 pairs），兩頁說法一致；
  限制檢查還沒回來或沒做成時不寫「沒給上限」。
- 「從 N 種裡挑」跟那一頁同一支算（「鋼板直接貼在棉上」那一類程式不會挑，不算進 N）。
- 改房間尺寸還在路上：清掉／復原那一格、右欄清單跟立面圖一起鎖住。
- 牆名換名擱著（游標在牆面表單裡）時點別一面的牆名鈕：換名落在對的那一面、打的字不丟。

方案資料都是考卷自己編的：109 號房的長寬高（_surface_zone_cases）改成自編尺寸與切法、對稱房（_zone_symmetry_room）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from tests.engine._input_page_steps import open_fact_editor, reveal
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import ROOM_109, room109_document
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser
from tests.engine.test_gui_wall_words import _arrive, _late_rename
from tests.engine._zone_symmetry_room import NAME as REFERENCE_ROOM
from tests.engine.test_gui_vpr_reference_browser import LISTED, _prepare
from tests.engine.test_gui_zone_elevation_browser import _drag, _open_scheme, _write

Doc = dict[str, object]
WOOL = ["glass_wool_48k", "none"]
WALL = {"kind": "constant_impedance", "impedance_pa_s_m": 40000.0}


def _settled(page: Page) -> None:
    page.wait_for_function("!document.querySelector('main').inert && !inputBusy")


def _box(key: str, u: tuple[float, float], v: tuple[float, float]) -> Doc:
    return {"area": "rectangle", "id": key, "u_m": list(u), "v_m": list(v), "material": WALL}


def _pair_room(tmp_path: Path) -> None:
    """6 × 4 × 3 公尺、兩支喇叭左右對稱（軸在正中）；前牆兩個畫的框 z1、z2（左右對稱）加七塊自動補的，
    z1 由程式最佳化、z2 保留原樣（幾何對稱但做法不同，沒配成一對）；左右兩條 auto1、auto2 都由程式最佳化（配成一對）；
    左右牆整面都由程式最佳化（配成一對）。"""
    document = room109_document()
    scene = cast(Doc, document["scene"])
    scene["room_m"] = {"Lx": 6.0, "Ly": 4.0, "Lz": 3.0}
    document["speakers"] = {"left": {"x": 1.0, "y": 1.0, "z": 1.0}, "right": {"x": 1.0, "y": 3.0, "z": 1.0}}
    receivers = cast(Doc, document["receiver_set"])
    points = cast(list[Doc], receivers["points"])[:2]
    points[0]["position_m"], points[1]["position_m"] = [3.5, 2.0, 1.0], [3.5, 2.0, 1.2]
    receivers["points"] = points
    zones: dict[str, list[Doc]] = {wall: [{"area": "whole_wall", "material": WALL}] for wall in ("floor", "ceiling", "xL", "y0", "yL")}
    zones["x0"] = [_box("z1", (0.5, 1.5), (0.5, 2.5)), _box("z2", (2.5, 3.5), (0.5, 2.5)), _box("auto1", (0.0, 0.5), (0.0, 3.0)),
                   _box("auto2", (3.5, 4.0), (0.0, 3.0)), _box("auto3", (1.5, 2.5), (0.0, 3.0)), _box("auto4", (0.5, 1.5), (0.0, 0.5)),
                   _box("auto5", (0.5, 1.5), (2.5, 3.0)), _box("auto6", (2.5, 3.5), (0.0, 0.5)), _box("auto7", (2.5, 3.5), (2.5, 3.0))]
    scene["surface_zones_by_wall"] = zones
    scene.pop("impedance_pa_s_per_m_by_wall", None)
    _write(tmp_path, document)
    search = {"mode": "search", "allowed": WOOL}
    (tmp_path / "designs").mkdir(exist_ok=True)
    (tmp_path / "designs" / "room109.json").write_text(json.dumps({
        "schema_version": "aosr.zone_design.v1", "scheme_id": "room109", "room_max_depth_m": 0.15,
        "zones": {"x0": {"z1": search, "z2": {"mode": "keep_original"}, "auto1": search, "auto2": search},
                  "y0": {"whole_wall": search}, "yL": {"whole_wall": search}}}), encoding="utf-8")


def _list_sentences(page: Page, wall: str) -> dict[str, str]:
    """右欄清單每一區那一句（區代號 → 字）。"""
    return cast(dict[str, str], page.evaluate("""wall => Object.fromEntries([...document.querySelectorAll('#elevation-list-zones .zone-item')]
        .map(row => [row.dataset.zone.split('/')[1], document.getElementById(`wall-zone-what-${row.dataset.zone.replace('/', '-')}`).firstChild.textContent])
        .filter(([key]) => key !== undefined))""", wall))


def test_the_list_names_the_same_pairs_as_the_zone_page(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 5 之一：右欄每一區講「和○牆○一對」「沒配成一對」照「這間房貼什麼」那一頁圖上掛的同一份；
    幾何對稱但做法不同的 z1 不說成一對，自動補的 auto1 與左右整面配成的照寫。"""
    _pair_room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        said: dict[str, dict[str, str]] = {}
        for wall in ("x0", "y0"):
            page.locator(f"#elevation-wall-{wall}").click()
            expect(page.locator(f"#elevation-wall-{wall}")).to_have_attribute("aria-pressed", "true")
            said[wall] = _list_sentences(page, wall)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        checked = 0
        for wall, sentences in said.items():
            for key, sentence in sentences.items():
                spot = page.locator(f"#zone-map [data-wall='{wall}'][data-zone='{key}']")
                partner, missed = spot.get_attribute("data-pair"), spot.get_attribute("data-missed")
                if partner:
                    other_wall, other_key = partner.split("/")
                    words = cast(str, page.evaluate("([w, k]) => k === 'whole_wall' ? `和${wallNames[w]}整面一對` : `和${wallNames[w]}${zoneLabels(w).get(k).numberText}一對`",
                                                    [other_wall, other_key]))
                    assert words in sentence, (wall, key, sentence)
                    checked += 1
                elif missed:
                    assert "沒配成一對" in sentence and "左右對稱" not in sentence, (wall, key, sentence)
                    checked += 1
                else:
                    assert "一對" not in sentence and "左右對稱" not in sentence, (wall, key, sentence)
        assert said["x0"]["z1"] and page.locator("#zone-map [data-wall='x0'][data-zone='z1']").get_attribute("data-missed") == "true"
        assert page.locator("#zone-map [data-wall='x0'][data-zone='auto1']").get_attribute("data-pair") == "x0/auto2"
        assert checked >= 4
        _assert_quiet(watched)


def test_the_list_does_not_say_no_cap_while_the_limits_are_unknown(tmp_path: Path, browser: Browser) -> None:
    """審查真錯 5 之二：限制檢查還在路上（攔著）時右欄寫「總厚度上限還在核對」、沒做成（伺服器回 500）時寫
    「總厚度上限沒核對成（限制檢查沒做成）」，都不寫「總厚度沒給上限」（左邊那一行明明寫最多 15 公分）；回來了照寫上限。"""
    _pair_room(tmp_path)
    held: list[Route] = []
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.route("**/api/zone-limits", lambda route: held.append(route))
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#elevation-depth")).to_have_text("貼的東西總厚度最多 15 公分（照房間預設）")
        what = page.locator("#wall-zone-what-x0-z1")
        expect(what).to_contain_text("總厚度上限還在核對")
        expect(what).not_to_contain_text("沒給上限")
        assert held
        held.pop().fulfill(status=500, content_type="application/json", body=json.dumps({"error": "考卷故意讓它壞"}))
        expect(what).to_contain_text("總厚度上限沒核對成（限制檢查沒做成）")
        expect(what).not_to_contain_text("沒給上限")
        page.unroute("**/api/zone-limits")
        _release(held)
        page.evaluate("refreshZoneLimits()")
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration && zoneLimits?.status === 'checked'")
        expect(what).to_contain_text("總厚度最多 15 公分（房間預設）")
        assert watched.page_errors == []  # 故意回 500，主控台會記一筆載入失敗，這一題只看頁面錯誤


def test_the_list_counts_choices_the_same_way_as_the_zone_page(tmp_path: Path, browser: Browser) -> None:
    """設計洞 2：清單裡有「鋼板直接貼在棉上」那一筆（程式不會挑）：右欄「從 N 種裡挑」的 N 與那一頁小框
    「程式從下面勾的 N 種裡選」同一個數。"""
    _prepare(tmp_path, {"x0/a": LISTED}, left_right_symmetry="off")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, REFERENCE_ROOM)
        page.locator("#elevation-wall-x0").click()
        listed = page.locator("#wall-zone-what-x0-a").inner_text()
        page.locator("#zone-goto-x0-a").click()
        expect(page.locator("#zone-panel")).to_have_attribute("data-zones", "x0/a")
        panel = page.locator("#zone-panel-search-from").inner_text()
        count = panel.removeprefix("程式從下面勾的 ").split(" 種")[0]
        assert count.isdigit() and int(count) < len(LISTED), panel
        assert f"從 {count} 種裡挑" in listed, (listed, panel)
        _assert_quiet(watched)


def test_the_clear_slot_and_the_list_are_locked_while_the_room_is_resizing(tmp_path: Path, browser: Browser) -> None:
    """設計洞 3：改房間尺寸、分區跟著調的請求還在路上（攔著）：清掉／復原那一格與右欄清單跟立面圖一起鎖住
    （同時點「清掉這一面的切法」「改這一區 ▸」會跟調尺寸同時改同一份分區）；回來就解鎖。"""
    _write(tmp_path, room109_document())
    held: list[Route] = []
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _open_scheme(page, "room109")
        page.locator("#elevation-wall-x0").click()
        _drag(page, ROOM_109["Ly"], ROOM_109["Lz"], (0.3, 0.3), (1.3, 1.3))
        _settled(page)
        failed: list[str] = []
        page.on("response", lambda response: failed.append(f"{response.status} {response.url}") if response.status >= 400 else None)
        page.route("**/api/zone-resize", lambda route: held.append(route))
        page.locator("#room-Lz").fill("3.4")
        page.locator("#room-Lz").press("Tab")
        locked = "id => Boolean(document.getElementById(id).closest('[inert]'))"
        page.wait_for_function(f"({locked})('elevation')")
        assert held
        assert page.evaluate(locked, "elevation-clear-slot") and page.evaluate(locked, "elevation-list")
        page.unroute("**/api/zone-resize")
        _release(held)
        page.wait_for_function(f"!({locked})('elevation')")
        assert not page.evaluate(locked, "elevation-clear-slot") and not page.evaluate(locked, "elevation-list")
        # 改房間高度那一下，預覽（/api/input-preview）帶新高度、舊的分區送出去，伺服器拒收 400「外框必須正好蓋滿整面」：
        # 不扣調尺寸那一支也一樣（實測過，見單元 3 之二回報第 0 節），是改之前就有的，回報第 8 節另提；
        # 這一題只放過這一條，其餘請求不准失敗、頁面不准出錯。
        assert all(item.startswith("400 ") and item.endswith("/api/input-preview") for item in failed), failed
        assert watched.page_errors == []


def test_a_wall_tab_click_while_a_rename_waits_lands_on_that_wall_and_keeps_the_typing(tmp_path: Path, browser: Browser) -> None:
    """審查考卷的洞：伺服器回的牆名晚到、游標還在地板那一格（換名擱著），人打了 2000 再點前牆那一面的牆名鈕：
    點到的就是那一面（選中、編輯處換成那一面），牆名換成新的那一份，地板那一格打的 2000 照留。"""
    with _serve(tmp_path) as base, _open(browser, f"{base}/") as watched:
        page = watched.page
        expect(page.locator("#wall-editor-x0")).to_contain_text("前牆材料")
        reveal(page, "#wall-floor")
        held = _late_rename(page)
        floor = page.locator("#wall-floor")
        floor.focus()
        floor.select_text()
        _arrive(page, held)
        page.keyboard.press("Delete")
        page.keyboard.insert_text("2000")
        assert floor.input_value() == "2000"
        page.locator("#elevation-wall-x0").click()
        expect(page.locator("#elevation-wall-x0")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#elevation-wall-floor")).to_have_attribute("aria-pressed", "false")
        expect(page.locator("#elevation-wall-x0")).to_contain_text("x 起點牆")
        expect(page.locator("#wall-editor-x0")).to_contain_text("x 起點牆材料")
        assert floor.input_value() == "2000"
        assert all("422" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []
