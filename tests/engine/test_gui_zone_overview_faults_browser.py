"""真瀏覽器考「這間房貼什麼」那一頁（#1176 第 C 支）出狀況的時候：伺服器回得慢、回一半斷掉、拒收，畫面該停在哪裡、
該講什麼；以及平常走不太到的那幾條路（開機範本沒切過區、牆上有還沒套用的實測草稿、選的跟原牆一樣、清單裡有這一面貼不了的、
重切之後選的區不在了、換了一份方案）。獨立審查第一輪（11-16）點出來的每一條各有一題。

資料全是合成的：區名考卷那間假房間（``_zone_naming_room``）配合成材料庫（``_sample_material_library``），同
``test_gui_zone_overview_tasks_browser``；開機範本用伺服器自己的那一份。"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from tests.engine._input_page_steps import open_save_as, open_scheme, reveal
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._sample_material_library import SAMPLE_USER_ENTRIES
from tests.engine._zone_panel_steps import choose_fixed, expect_mode
from tests.engine._zone_naming_room import ZONE_NAMING_LIBRARY, ZONE_NAMING_SCHEME
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _open_scheme, _write
from tests.engine.test_gui_zone_overview_browser import SAMPLE, WOOLS
from tests.engine.test_gui_zone_overview_tasks_browser import (
    _ALLOWED, _GROUPS, _SELECTED, SEARCH, VPR, _apply, _click_zone, _library, _open_overview, _prepare, _save, _settled)

OTHER = "zone-naming-other"
_MODES = """() => Object.fromEntries(Object.entries(zoneDesign?.zones ?? {}).flatMap(([wall, zones]) => Object.entries(zones)
    .map(([key, setting]) => [`${wall}/${key}`, setting.mode])))"""
# 規則是少數那一組時，由程式最佳化的那 8 區（見 _minority_rule）。
_SEARCHING = [f"xL/auto{n}" for n in range(1, 7)] + ["y0/auto1", "y0/auto2"]


class _Held:
    """把符合的請求扣住不放；``release`` 把扣住的全部放行，之後的也直接放行。"""

    def __init__(self, page: Page, pattern: str) -> None:
        self.routes: list[Route] = []
        self._open = False
        page.route(pattern, self._handle)

    def _handle(self, route: Route) -> None:
        if self._open:
            route.continue_()
        else:
            self.routes.append(route)

    def release(self) -> None:
        self._open = True
        for route in self.routes:
            route.continue_()


def _hold(page: Page, pattern: str) -> _Held:
    return _Held(page, pattern)


def _wait_held(page: Page, held: _Held, count: int) -> None:
    for _ in range(500):
        if len(held.routes) >= count:
            return
        page.wait_for_timeout(20)
    raise AssertionError(f"等不到第 {count} 個被扣住的請求")


def _released(page: Page) -> None:
    """一批改動送完、畫面放開（平面圖檢查可能還沒回來）。"""
    page.wait_for_function("!document.getElementById('zone-page').inert && !inputBusy")


def _modes(page: Page, zones: list[str]) -> dict[str, str]:
    modes = cast(dict[str, str], page.evaluate(_MODES))
    return {zone: modes.get(zone, "沒列") for zone in zones}


def _minority_rule(page: Page) -> None:
    """規則是「不貼，保留原樣」但照著它的只剩 4 區，由程式最佳化的另外指定有 8 區（比規則那一組多）。"""
    _click_zone(page, "yL/auto1")
    page.locator("#zone-panel-material").select_option(VPR)
    _apply(page)
    for zones in (["xL/auto1", "xL/auto2", "xL/auto3"], ["xL/auto4", "xL/auto5", "xL/auto6"], ["y0/auto1", "y0/auto2"]):
        for zone in zones:
            _click_zone(page, zone)
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
    expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "keep_original")
    assert set(_modes(page, _SEARCHING).values()) == {"search"}


def test_what_is_picked_while_the_plan_check_is_out_stays_when_it_comes_back(tmp_path: Path, browser: Browser) -> None:
    """套用之後平面圖檢查（/api/plan）還沒回來，畫面已經放開：使用者點了下一區、點了一個選項。檢查回來時，
    剛點的區與選項都還在（晚到的那一步不准把它清掉）。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        held = _hold(page, "**/api/plan")
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-keep_original").check()
        page.locator("#zone-panel-apply").click()
        _wait_held(page, held, 1)
        _released(page)
        # 套用那一步已經做完：面板收起來了（不必等平面圖檢查）。
        expect(page.locator("#zone-panel")).to_have_count(0)
        _click_zone(page, "y0/auto2")
        page.locator("#zone-panel-mode-search").check()
        held.release()
        _settled(page)
        assert page.evaluate(_SELECTED) == ["y0/auto2"]
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        assert watched.page_errors == []


def test_changing_the_rule_twice_while_the_plan_check_is_slow_never_writes_into_zones_set_apart(tmp_path: Path, browser: Browser) -> None:
    """規則是少數那一組時改規則，平面圖檢查慢：等待中「其他牆面」就是剛設的那一種（不准先翻成別組）；這時再改一次規則，
    寫到的是跟著規則的那幾區，原本另外指定的 8 區一區都不准動。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _minority_rule(page)
        held = _hold(page, "**/api/plan")
        page.locator("#zone-rule-mode").select_option("fixed")
        page.locator("#zone-rule-material").select_option(VPR)
        _wait_held(page, held, 1)
        _released(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "fixed")
        expect(page.locator("#zone-rule-material")).to_have_value(VPR)
        expect(page.locator("#zone-special")).to_have_attribute("data-count", "12")
        page.locator("#zone-rule-mode").select_option("keep_original")
        _wait_held(page, held, 2)
        _released(page)
        assert set(_modes(page, _SEARCHING).values()) == {"search"}
        held.release()
        _settled(page)
        assert set(_modes(page, _SEARCHING).values()) == {"search"}
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "keep_original")
        expect(page.locator("#zone-special")).to_have_attribute("data-count", "12")
        assert watched.page_errors == []


def test_a_plan_check_that_fails_once_leaves_the_rule_as_set_and_says_so_on_the_page(tmp_path: Path, browser: Browser) -> None:
    """改規則後平面圖檢查斷一次（連不上）：「其他牆面」停在使用者剛設的那一種，新頁上看得到一句白話講連不上
    （訊息列在藏起來的輸入頁裡，只寫在那裡等於沒講）。下一次改動成功，那一句就拿掉。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _minority_rule(page)
        aborted: list[str] = []

        def abort_once(route: Route) -> None:
            if aborted:
                route.continue_()
            else:
                aborted.append(route.request.url)
                route.abort()

        page.route("**/api/plan", abort_once)
        page.locator("#zone-rule-mode").select_option("fixed")
        page.locator("#zone-rule-material").select_option(VPR)
        _settled(page)
        assert len(aborted) == 1
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "fixed")
        expect(page.locator("#zone-rule-material")).to_have_value(VPR)
        expect(page.locator("#zone-special")).to_have_attribute("data-count", "12")
        expect(page.locator("#zone-page-said")).to_be_visible()
        expect(page.locator("#zone-page-said")).to_have_text("連不上本機的網頁伺服器，請確認它還開著")
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        expect(page.locator("#zone-page-said")).to_have_count(0)
        # 輸入頁自己講過的警示不帶進這一頁：回輸入頁、訊息列寫了一句警示，再打開這一頁，沒有那一句。
        page.locator("#zone-page-close").click()
        page.evaluate("say('考卷寫在輸入頁訊息列的一句警示', 'notice')")
        expect(page.locator("#messages")).to_have_text("考卷寫在輸入頁訊息列的一句警示")
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(page.locator("#zone-page-said")).to_have_count(0)
        assert watched.page_errors == []


def test_a_refused_save_from_send_to_search_is_explained_on_the_page(tmp_path: Path, browser: Browser) -> None:
    """開機載入的範本，資料夾裡已經有同名的方案：新頁把牆面由程式最佳化、一面指定貼一種，按「拿去搜尋」，存檔照規矩不蓋掉同名的那一份。
    被擋的原因要寫在新頁上（不跳頁、不是只寫在藏起來的輸入頁）。"""
    with _serve(tmp_path) as base:
        example = json.loads(urllib.request.urlopen(f"{base}/api/example").read())["scheme"]
    _write(tmp_path, example)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        # 再指定貼一區：方案跟資料夾裡那一份真的不一樣了（一樣的話存檔算沒變動、不會被擋）。
        _click_zone(page, "x0/whole_wall")
        page.locator("#zone-panel-material").select_option("glass_wool_48k_50mm")
        _apply(page)
        expect(page.locator("#zone-page-search")).to_be_enabled()
        page.locator("#zone-page-search").click()
        said = page.locator("#zone-page-said")
        expect(said).to_be_visible()
        expect(said).to_contain_text("已經有叫")
        assert "gradient" not in page.url and page.locator("main").is_hidden()
        assert watched.page_errors == []


def test_the_startup_template_without_zones_takes_a_rule_change(tmp_path: Path, browser: Browser) -> None:
    """開機載入的範本沒切過區（方案裡還是每面牆一個阻抗、沒有分區那一格）：在新頁改規則成由程式最佳化，照樣寫得進去——
    方案長出分區那一格、四面牆都由程式最佳化，天花地板不動。"""
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        assert page.evaluate("scheme.scene.surface_zones_by_wall === undefined || scheme.scene.surface_zones_by_wall === null")
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "keep_original")
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "search")
        expect(page.locator("#zone-page-note")).to_have_text("4 區改成「由程式最佳化」")
        assert page.evaluate(_MODES) == {f"{wall}/whole_wall": "search" for wall in ("x0", "xL", "y0", "yL")}
        assert page.evaluate("Object.keys(scheme.scene.surface_zones_by_wall).sort()") == ["ceiling", "floor", "x0", "xL", "y0", "yL"]
        assert page.evaluate("'impedance_pa_s_per_m_by_wall' in scheme.scene") is False
        assert watched.page_errors == [] and watched.console_errors == []


def test_walls_made_only_of_doors_and_windows_still_let_the_ceiling_be_picked(tmp_path: Path, browser: Browser) -> None:
    """四面牆每一區都是門（沒有算進「其他牆面」的區，那條規則是空的）：點天花一區，面板照樣出來，整頁不准空白。"""
    scheme = json.loads(json.dumps(ZONE_NAMING_SCHEME))
    door = next(entry["material"] for entry in ZONE_NAMING_LIBRARY if entry["id"] == "sample_door_solid_26kg")
    for wall in ("x0", "xL", "y0", "yL"):
        for zone in scheme["scene"]["surface_zones_by_wall"][wall]:
            zone["material"] = door
    _write(tmp_path, scheme)
    _library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        expect(page.locator("#zone-rule-mode")).to_have_count(0)
        _click_zone(page, "ceiling/auto1")
        expect(page.locator("#zone-panel h2")).to_have_text("已選 1 區")
        expect(page.locator("#zone-panel-same")).to_have_count(0)
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        assert page.evaluate(_MODES) == {"ceiling/auto1": "search"}
        expect(page.locator("#zone-map")).to_be_visible()
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_drawing_failure_keeps_the_last_screen_and_the_way_back(tmp_path: Path, browser: Browser) -> None:
    """這一頁組到一半丟了例外（考卷把畫喇叭記號那一步換成會丟的）：
    ① 第一次打開就畫不出來（沒有上一次的樣子可以留）：只有標題、一句講畫不出來、「回房間方案輸入」，按了回得去；
    ② 畫好之後才壞：畫面停在上一次的樣子（圖、規則都在）、最上面加那一句，「回房間方案輸入」照樣按得到；
    修好之後再開，那一句就不在了。例外照樣進瀏覽器的錯誤紀錄，不吞掉。"""
    _prepare(tmp_path)
    breaks = """() => { window.realMarkerItems = window.elevationGeometry.markerItems;
        window.elevationGeometry.markerItems = () => { throw new Error('考卷故意丟的'); }; }"""
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        # 打開方案時送出的限制檢查要先回來、先畫好：晚到的那一次若落在弄壞之後、關新頁之前，會在新頁上多畫（多丟）一次，
        # 錯誤就不是三筆（雲端 81182ee9 第二跑紅過：四筆）。
        page.wait_for_function("() => zoneLimitsGeneration > 0 && zoneLimitsShown === zoneLimitsGeneration")
        page.evaluate(breaks)
        page.locator("#zone-page-open").click()
        broken = page.locator("#zone-page-broken")
        expect(broken).to_be_visible()
        expect(broken).to_contain_text("這一頁畫不出來")
        expect(page.locator("#zone-page h1")).to_have_text("這間房貼什麼")
        expect(page.locator("#zone-map")).to_have_count(0)
        page.locator("#zone-page-close").click()
        expect(page.locator("main")).to_be_visible()
        expect(page.locator("#zone-page")).to_be_hidden()
        page.evaluate("() => { window.elevationGeometry.markerItems = window.realMarkerItems; }")
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(broken).to_have_count(0)
        _settled(page)
        page.evaluate(breaks)
        _click_zone(page, "yL/z3")
        expect(broken).to_be_visible()
        expect(page.locator("#zone-map")).to_have_count(1)
        expect(page.locator("#zone-map [data-zone]")).to_have_count(29)
        expect(page.locator("#zone-rule")).to_be_visible()
        # 壞著的時候再點一次：那一句還是只有一句。
        _click_zone(page, "y0/auto2")
        expect(broken).to_have_count(1)
        page.locator("#zone-page-close").click()
        expect(page.locator("main")).to_be_visible()
        # 三次畫不出來的例外都進了瀏覽器的錯誤紀錄（丟出來那一步排在後面，等到齊）。
        for _ in range(250):
            if len(watched.page_errors) >= 3:
                break
            page.wait_for_timeout(20)
        assert len(watched.page_errors) == 3 and all("考卷故意丟的" in text for text in watched.page_errors), watched.page_errors


def test_opening_another_scheme_forgets_the_rule_and_the_selection_of_the_last_one(tmp_path: Path, browser: Browser) -> None:
    """兩份方案的區代號一模一樣。第一份把「其他牆面」設成由程式最佳化、還選著一區；回輸入頁打開第二份（它只有一區由程式最佳化、
    清單跟第一份的規則同一份，其餘都沒貼）：第二份的規則照它自己的多數推（不貼，保留原樣），上一份選的區也不算數。"""
    _prepare(tmp_path)
    _write(tmp_path, {**ZONE_NAMING_SCHEME, "scheme_id": OTHER})
    (tmp_path / "designs").mkdir(exist_ok=True)
    (tmp_path / "designs" / f"{OTHER}.json").write_text(json.dumps(
        {"schema_version": "aosr.zone_design.v1", "scheme_id": OTHER, "zones": {"xL": {"auto1": SEARCH}}}), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "search")
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED))["xL/auto1"] == WOOLS
        _click_zone(page, "xL/auto2")
        expect(page.locator("#zone-panel")).to_have_count(1)
        expect(page.locator("#zone-page-unsaved")).to_have_count(1)
        page.locator("#zone-page-close").click()
        _open_scheme(page, OTHER)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(page.locator("#zone-panel")).to_have_count(0)
        assert page.evaluate(_SELECTED) == []
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "keep_original")
        # 上一份改了沒存的那一句也不帶過來（這一份沒改過）。
        expect(page.locator("#zone-page-unsaved")).to_have_count(0)
        assert [group["zones"] for group in cast(list[dict[str, object]], page.evaluate(_GROUPS))][-1] == ["xL/auto1"]
        assert watched.page_errors == [] and watched.console_errors == []


def test_nothing_on_the_page_reacts_while_a_batch_is_on_its_way(tmp_path: Path, browser: Browser) -> None:
    """套用送出去、伺服器還沒回（一批換材料扣在路上）：這時在圖上點別區、再按一次套用都沒作用——選的還是那一區、
    只送出過一批。回來之後照常。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-keep_original").check()
        other = page.locator("#zone-map [data-wall='y0'][data-zone='auto2'] .zv-fill").bounding_box()
        button = page.locator("#zone-panel-apply").bounding_box()
        assert other is not None and button is not None
        held = _hold(page, "**/api/zone-material/batch")
        page.locator("#zone-panel-apply").click()
        _wait_held(page, held, 1)
        for box in (other, button):
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(200)
        assert page.evaluate(_SELECTED) == ["yL/z3"] and len(held.routes) == 1
        held.release()
        expect(page.locator("#zone-panel")).to_have_count(0)
        _settled(page)
        assert _modes(page, ["yL/z3", "y0/auto2"]) == {"yL/z3": "keep_original", "y0/auto2": "沒列"}
        # 放開之後點得到。
        _click_zone(page, "y0/auto2")
        assert page.evaluate(_SELECTED) == ["y0/auto2"]
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_page_is_locked_while_send_to_search_is_saving_and_what_is_stored_is_what_was_there_at_the_press(tmp_path: Path, browser: Browser) -> None:
    """按了「拿去搜尋」、存方案那一步還沒回來（#1187 的存檔鎖）：這一頁跟輸入頁一起鎖住——圖上點別區、按套用都沒作用、
    沒有送出換材料。放行之後存下去的是按下那一刻那一份（面板上點了、還沒套用的那一區沒被寫進去），照常跳到搜尋頁。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        # 面板上再點好一區與做法、先不套用：等一下存檔途中去按它的套用。
        _click_zone(page, "y0/auto2")
        page.locator("#zone-panel-mode-search").check()
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        other = page.locator("#zone-map [data-wall='xL'][data-zone='auto1'] .zv-fill").bounding_box()
        button = page.locator("#zone-panel-apply").bounding_box()
        assert other is not None and button is not None
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url) if request.url.endswith("/api/zone-material/batch") else None)
        held = _hold(page, f"**/api/schemes/{SAMPLE}")
        page.locator("#zone-page-search").click()
        _wait_held(page, held, 1)
        assert held.routes[0].request.method == "PUT"
        for box in (other, button):
            page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(200)
        assert page.evaluate(_SELECTED) == ["y0/auto2"] and sent == []
        assert page.evaluate("document.getElementById('zone-page').inert") is True
        held.release()
        page.wait_for_url(f"**/gradient?scheme={SAMPLE}")
        assert sent == [] and watched.page_errors == []
    stored = json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text(encoding="utf-8"))["zones"]
    assert stored == {"yL": {"z3": SEARCH}}


def test_a_send_to_search_blocked_by_a_change_still_on_its_way_says_so_on_the_page(tmp_path: Path, browser: Browser) -> None:
    """按了「拿去搜尋」、它還在等限制檢查回來（這時還沒鎖頁），使用者又套用了一批、那一批還在路上：限制檢查回來後存檔被擋
    （#1187：上一個動作還沒做完）。那一句要寫在這一頁上（輸入頁的訊息列與按鈕旁那一行這時都是藏起來的），不存、不跳頁。"""
    _prepare(tmp_path)
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
        page.locator("#zone-panel-mode-search").check()
        page.evaluate("void refreshZoneLimits()")
        _wait_held(page, limits, 1)
        page.locator("#zone-page-search").click()
        batch = _hold(page, "**/api/zone-material/batch")
        page.locator("#zone-panel-apply").click()
        _wait_held(page, batch, 1)
        limits.release()
        said = page.locator("#zone-page-said")
        expect(said).to_be_visible()
        expect(said).to_have_text("上一個動作還沒做完，這一次沒有儲存；等它做完再按一次")
        assert saved == [] and "gradient" not in page.url
        batch.release()
        _settled(page)
        assert saved == [] and "gradient" not in page.url and watched.page_errors == []


def test_a_refused_batch_changes_nothing_and_the_page_names_the_zone_and_the_reason(tmp_path: Path, browser: Browser) -> None:
    """一批換材料被伺服器拒收：一區都不改、選的區與面板照留，新頁那一行寫是哪一區、為什麼（伺服器指名了區就用牆名與號碼叫它；
    沒指名就照它給的那一句）。"""
    _prepare(tmp_path)
    replies = [{"error": "考卷故意拒收的一批", "wall": "yL", "zone": "z3", "reason": "考卷說這一區不能換"}, {"error": "考卷故意拒收的一批"}]
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        before = page.evaluate("JSON.stringify([zoneDesign, scheme])")
        page.route("**/api/zone-material/batch", lambda route: route.fulfill(
            status=400, content_type="application/json", body=json.dumps(replies.pop(0), ensure_ascii=False)))
        _click_zone(page, "yL/z3")
        _click_zone(page, "y0/auto2")
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-apply").click()
        note = page.locator("#zone-page-note")
        expect(note).to_have_text("沒有改：前牆 ③ 主位正前方那塊：考卷說這一區不能換；一區都沒改")
        expect(note).to_have_class("notice")
        _released(page)
        page.locator("#zone-panel-apply").click()
        expect(note).to_have_text("沒有改：考卷故意拒收的一批")
        _released(page)
        assert page.evaluate(_SELECTED) == ["yL/z3", "y0/auto2"]
        expect(page.locator("#zone-panel-mode-search")).to_be_checked()
        assert page.evaluate("JSON.stringify([zoneDesign, scheme])") == before
        assert watched.page_errors == []


def test_a_refused_rule_change_puts_other_walls_back_as_it_was_and_says_why(tmp_path: Path, browser: Browser) -> None:
    """把「其他牆面」選成由程式最佳化，伺服器拒收那一批：一區都沒改，「其他牆面」那一條與它的下拉都回到原本的那一種
    （下拉不准停在剛選、但沒寫成的那一項），結果列上面那一行講為什麼。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        before = page.evaluate("JSON.stringify([zoneDesign, scheme])")
        page.route("**/api/zone-material/batch", lambda route: route.fulfill(
            status=400, content_type="application/json", body=json.dumps({"error": "考卷故意拒收的一批"}, ensure_ascii=False)))
        page.locator("#zone-rule-mode").select_option("search")
        note = page.locator("#zone-page-note")
        expect(note).to_have_text("沒有改：考卷故意拒收的一批")
        expect(note).to_have_class("notice")
        _settled(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "keep_original")
        expect(page.locator("#zone-rule-mode")).to_have_value("keep_original")
        expect(page.locator("#zone-rule")).to_have_class("zv-rule still")
        assert page.evaluate("JSON.stringify([zoneDesign, scheme])") == before
        assert watched.page_errors == []


def test_a_wall_with_an_unapplied_measured_draft_is_not_changed_and_the_page_says_why(tmp_path: Path, browser: Browser) -> None:
    """左牆（整面一區）在輸入頁開了實測阻抗的草稿、還沒套用：新頁上改左牆不送出，那一行寫先套用或取消草稿；
    別面牆照改。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        # #1265 單元 3 起牆面表單只露選中那一面、收在第 2 段「換原本的牆」裡：先選那一面、打開。
        reveal(page, "#material-kind-x0").select_option("measured_impedance")
        expect(page.locator("#measured-draft-x0")).to_be_visible()
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url) if request.url.endswith("/api/zone-material/batch") else None)
        _click_zone(page, "x0/whole_wall")
        page.locator("#zone-panel-mode-search").check()
        page.locator("#zone-panel-apply").click()
        note = page.locator("#zone-page-note")
        expect(note).to_have_text("左牆有還沒套用的實測阻抗草稿：先套用或取消")
        expect(note).to_have_class("notice")
        assert sent == [] and page.evaluate(_SELECTED) == ["x0/whole_wall"]
        page.locator("#zone-panel-cancel").click()
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        assert len(sent) == 1 and _modes(page, ["yL/z3", "x0/whole_wall"]) == {"yL/z3": "search", "x0/whole_wall": "沒列"}
        assert watched.page_errors == []


def _bare_vpr_scheme() -> dict[str, object]:
    """假房間改一份：右牆第一塊空牆本身就是合成庫那一筆 VPR 的構造（不是貼上去的，原牆就是它）。"""
    scheme = json.loads(json.dumps(ZONE_NAMING_SCHEME))
    built = next(entry["material"] for entry in SAMPLE_USER_ENTRIES if entry["id"] == VPR)
    next(zone for zone in scheme["scene"]["surface_zones_by_wall"]["xL"] if zone.get("id") == "auto1")["material"] = built
    return cast(dict[str, object], scheme)


def test_a_pick_that_is_the_wall_itself_writes_no_depth(tmp_path: Path, browser: Browser) -> None:
    """#1258 2b 審查：小框指定貼一種、選的跟原牆一模一樣，做法變成「不貼，保留原樣」；「最深」那一格打的值不寫進去
    （不貼不加層：舊表那一格寫「不加層」、小框的不貼也不畫最深，寫進去就看不到也清不掉）。"""
    _write(tmp_path, _bare_vpr_scheme())
    _library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        _click_zone(page, "xL/auto1")
        page.locator("#zone-panel-material").select_option(VPR)
        page.locator("#zone-panel-depth").fill("12")
        _apply(page)
        assert page.evaluate("structuredClone(zoneDesign.zones.xL.auto1)") == {"mode": "keep_original"}
        assert watched.page_errors == [] and watched.console_errors == []


def test_picking_what_the_wall_already_is_becomes_keep_original_and_says_so_on_the_list(tmp_path: Path, browser: Browser) -> None:
    """選的那一筆跟這一區原牆一模一樣（伺服器不疊一層、放回原樣）：新頁把做法記成「不貼，保留原樣」，不是指定貼。
    #1265 單元 3 之二第二趟起舊做法表拆了（這一題原名 …_and_drops_the_old_table_remark）：「選的跟原本一樣，當成不貼」那一句的
    新家是第 2 段右欄那一區（id 照舊），由小框套用時記（從輸入頁「改這一區 ▸」開的跟從新頁點區開的是同一個小框）。
    舊題最後考「新頁改過之後舊表那一句拿掉」：舊表那一句沒了（丙）；新頁再選同一筆之後那一句照樣在（沒被新頁那一下抹掉）。
    「改成別的做法就拿掉」由 test_gui_zone_choices_browser.py 的 test_picking_what_the_ceiling_already_is… 末段考。"""
    scheme = _bare_vpr_scheme()
    _write(tmp_path, scheme)
    _library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, SAMPLE)
        page.locator("#elevation-wall-xL").click()
        choose_fixed(page, "xL", "auto1", VPR)
        expect(page.locator("#zone-same-xL-auto1")).to_contain_text("跟原本一樣，當成不貼")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        _settled(page)
        # 新頁：另一塊空牆旁邊那塊（auto1）再選同一筆 → 照樣當成不貼。
        _click_zone(page, "xL/auto1")
        page.locator("#zone-panel-material").select_option(VPR)
        _apply(page)
        assert _modes(page, ["xL/auto1"]) == {"xL/auto1": "keep_original"}
        expect(page.locator("#zone-map [data-wall='xL'][data-zone='auto1']")).to_have_attribute("data-shown", "still")
        page.locator("#zone-page-close").click()
        expect_mode(page, "xL", "auto1", "keep_original")
        expect(page.locator("#zone-same-xL-auto1")).to_contain_text("跟原本一樣，當成不貼")
        assert watched.page_errors == [] and watched.console_errors == []


def test_the_list_written_to_a_face_drops_what_cannot_go_there_and_keeps_ids_the_library_lacks(tmp_path: Path, browser: Browser) -> None:
    """由程式最佳化時寫進每一區的清單照畫面上那一份，但這一面貼不了的拿掉：
    ① 清單裡只有天花才用的一筆：寫到牆區時一種都不剩，就只放「不貼」（不准寫一份空清單）；
    ② 清單裡有材料庫找不到的代號（別台機器存的）：原樣留著，不因為這一頁改了別區就被丟掉。"""
    only_ceiling = "ceiling_gypsum9mm_gap300"
    for name, listed, expected in (("ceiling", [only_ceiling], ["none"]),
                                   ("unknown", ["glass_wool_48k", "sample_missing_from_library"], ["glass_wool_48k", "sample_missing_from_library"])):
        folder = tmp_path / name
        folder.mkdir()
        zone = {"ceiling": {"z1": {"mode": "search", "allowed": listed}}} if name == "ceiling" else {"y0": {"auto1": {"mode": "search", "allowed": listed}}}
        _prepare(folder, {"schema_version": "aosr.zone_design.v1", "scheme_id": SAMPLE, "zones": zone})
        with _serve(folder) as base, _open(browser, base) as watched:
            page = watched.page
            _open_overview(page)
            _settled(page)
            assert list(cast(dict[str, list[str]], page.evaluate(_ALLOWED)).values()) == [listed]
            _click_zone(page, "yL/auto1")
            page.locator("#zone-panel-mode-search").check()
            expect(page.locator("#zone-panel-search-from")).to_contain_text(f"程式從下面勾的 {len(expected)} 種裡選")
            assert page.locator("#zone-panel-list [data-state=all]").evaluate_all("rows => rows.map(row => row.dataset.choice)") == expected
            _apply(page)
            assert cast(dict[str, list[str]], page.evaluate(_ALLOWED))["yL/auto1"] == expected
            assert watched.page_errors == []


def _outside_library_scheme() -> dict[str, object]:
    """假房間改一份：右牆六塊空牆都貼上前牆兩塊板那一種（合成的、不在材料庫裡），它就成了牆面裡最多區的做法。"""
    scheme = json.loads(json.dumps(ZONE_NAMING_SCHEME))
    walls = scheme["scene"]["surface_zones_by_wall"]
    pasted = next(zone["material"] for zone in walls["yL"] if zone["id"] == "z1")
    for zone in walls["xL"]:
        if zone["id"].startswith("auto"):
            zone["material"] = pasted
    return cast(dict[str, object], scheme)


def test_reset_is_refused_with_a_reason_when_the_rule_is_pasted_with_something_outside_the_library(tmp_path: Path, browser: Browser) -> None:
    """規則貼的不是材料庫裡的一筆（沒有代號可以照貼）：按「改回跟其他牆面一樣」不送出，那一行寫為什麼、要先做什麼。"""
    _write(tmp_path, _outside_library_scheme())
    _library(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "fixed")
        count = page.locator("#zone-special").get_attribute("data-count")
        before = page.evaluate("JSON.stringify([zoneDesign, scheme])")
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url) if request.url.endswith("/api/zone-material/batch") else None)
        page.locator("#zone-special-reset").click()
        expect(page.locator("#zone-page-note")).to_have_text("其他牆面現在貼的不是材料庫裡的一筆，沒辦法照著改回去：先在「其他牆面」選一種")
        expect(page.locator("#zone-page-note")).to_have_class("notice")
        expect(page.locator("#zone-special")).to_have_attribute("data-count", cast(str, count))
        assert sent == [] and page.evaluate("JSON.stringify([zoneDesign, scheme])") == before
        assert watched.page_errors == [] and watched.console_errors == []


def test_keeping_what_is_pasted_now_applies_without_asking_the_server_to_change_anything(tmp_path: Path, browser: Browser) -> None:
    """選到的都貼著東西、下拉停在「照現在貼的，不換」：套用按得下去，只把做法記成指定貼一種，貼著的東西一個字不動、
    不送換材料的請求（一塊貼著材料庫認不得的、一塊原本設成由程式最佳化但方案裡還貼著的，一起選）。"""
    scheme = _outside_library_scheme()
    _write(tmp_path, scheme)
    _library(tmp_path)
    (tmp_path / "designs").mkdir(exist_ok=True)
    (tmp_path / "designs" / f"{SAMPLE}.json").write_text(json.dumps(
        {"schema_version": "aosr.zone_design.v1", "scheme_id": SAMPLE, "zones": {"xL": {"auto2": {"mode": "keep_original"}}}}), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        materials = page.evaluate("JSON.stringify(scheme.scene.surface_zones_by_wall)")
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url) if request.url.endswith("/api/zone-material/batch") else None)
        _click_zone(page, "xL/auto1")
        _click_zone(page, "xL/auto2")
        page.locator("#zone-panel-mode-fixed").check()
        expect(page.locator("#zone-panel-material")).to_have_value("")
        expect(page.locator("#zone-panel-material option").first).to_have_text("照現在貼的，不換")
        expect(page.locator("#zone-panel-apply")).to_be_enabled()
        _apply(page)
        expect(page.locator("#zone-page-note")).to_have_text("2 區改成「指定貼一種」")
        assert _modes(page, ["xL/auto1", "xL/auto2"]) == {"xL/auto1": "fixed", "xL/auto2": "fixed"}
        assert sent == [] and page.evaluate("JSON.stringify(scheme.scene.surface_zones_by_wall)") == materials
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_zone_that_is_gone_after_a_recut_drops_out_of_the_selection_with_what_was_picked_for_it(tmp_path: Path, browser: Browser) -> None:
    """新頁選著前牆③、面板上剛點了「不貼」，回輸入頁把那一塊從立面圖刪掉（重切）：再開新頁，選取裡沒有它、面板收著；
    這時點別區，面板照那一區現在的做法起（上一區剛點的選項不准帶過來）。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-keep_original").check()
        page.locator("#zone-page-close").click()
        page.locator("#elevation-wall-yL").click()
        page.locator("#elevation [data-zone-id=z3] rect").click()
        _cut_by(page, "#elevation-delete")
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(page.locator("#zone-map [data-wall='yL'][data-zone='z3']")).to_have_count(0)
        expect(page.locator("#zone-panel")).to_have_count(0)
        expect(page.locator("#zone-slot")).to_be_visible()
        assert page.evaluate(_SELECTED) == []
        _click_zone(page, "yL/z1")
        expect(page.locator("#zone-panel h2")).to_have_text("已選 1 區")
        expect(page.locator("#zone-panel-mode-fixed")).to_be_checked()
        expect(page.locator("#zone-panel-mode-keep_original")).not_to_be_checked()
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_red_limit_on_this_wall_shows_in_the_right_list_without_unfolding_anything(tmp_path: Path, browser: Browser) -> None:
    """這一面有超過深度上限的區（紅字）：紅字攤在第 2 段右欄清單上方，不藏在摺疊裡（#1265 單元 3 之二第二趟起舊做法表拆了，
    這一題原名 test_the_old_table_unfolds_by_itself_when_a_limit_is_red：舊表收著、有紅字自己打開；「自己打開」那一件丙）。"""
    _prepare(tmp_path, {"schema_version": "aosr.zone_design.v1", "scheme_id": SAMPLE, "zones": {}, "wall_max_depth_m": {"yL": 0.01}})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        open_scheme(page, SAMPLE)
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg=SAMPLE)
        page.locator("#elevation-wall-yL").click()
        page.wait_for_function("zoneLimitsShown === zoneLimitsGeneration")
        expect(page.locator("#elevation-list-notes .limit-red").first).to_be_visible()
        expect(page.locator("#elevation-list-notes .notice")).to_have_count(0)
        assert watched.page_errors == []


def test_editing_the_list_keeps_other_walls_handed_to_the_computer(tmp_path: Path, browser: Browser) -> None:
    """「其他牆面」是由程式最佳化、但照著它的區比「不貼」的少：這時改清單（拿掉一種），規則還是由程式最佳化、還是那幾區
    （清單換了不等於規則換了）。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        for zone in [f"xL/auto{n}" for n in range(1, 7)] + [f"y0/auto{n}" for n in range(1, 5)]:
            _click_zone(page, zone)
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "search")
        followers = [zone for zone, mode in cast(dict[str, str], page.evaluate(_MODES)).items() if mode == "search"]
        assert len(followers) == 3
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-remove-glass_wool_96k").click()
        _apply(page)
        expect(page.locator("#zone-rule")).to_have_attribute("data-mode", "search")
        expect(page.locator("#zone-rule-mode")).to_have_value("search")
        expect(page.locator("#zone-list-who")).to_have_text("現在管 3 區：沒另外指定的牆區")
        assert cast(dict[str, list[str]], page.evaluate(_ALLOWED)) == {zone: [item for item in WOOLS if item != "glass_wool_96k"] for zone in followers}
        assert watched.page_errors == [] and watched.console_errors == []


_PLAN_LABELS = """() => JSON.stringify(['plan-xy', 'plan-xz', 'zoom-xy', 'zoom-xz'].map(id =>
    [...document.getElementById(id).querySelectorAll('text')].map(text => `${text.textContent}@${text.getAttribute('x')},${text.getAttribute('y')}`)))"""


def test_plan_labels_are_drawn_again_on_the_way_back_after_a_change_made_on_the_page(tmp_path: Path, browser: Browser) -> None:
    """這一頁開著時輸入頁是藏起來的，平面圖上的字量不到寬度、避不開彼此。在這一頁改一區（會重畫平面圖）再回輸入頁：
    一回去就照同一份再畫一次（圖上的字是新畫的那一批），字的位置跟改之前一樣（改材料不影響平面圖上的字），
    不必再問伺服器、訊息列也不被蓋掉。假房間的字本來就不擠，所以這一題考的是「回去時真的重畫了」，不是擠不擠。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        asked: list[str] = []
        page.on("request", lambda request: asked.append(request.url) if request.url.endswith("/api/plan") else None)
        page.locator("#zone-page-close").click()
        before = cast(str, page.evaluate(_PLAN_LABELS))
        assert "@" in before
        page.locator("#zone-page-open").click()
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        assert len(asked) == 1
        page.evaluate("say('考卷寫在訊息列的一句', 'ok')")
        page.evaluate("for (const text of document.querySelectorAll('#plan-xy text, #zoom-xy text')) text.dataset.drawnHidden = 'yes'")
        assert page.locator("[data-drawn-hidden]").count() > 0
        # 包一層畫平面圖的那一支：記下每次被叫的時候輸入頁是不是藏著的。
        page.evaluate("""() => { window.planDrawnWhileHidden = []; const draw = window.drawPlan;
            window.drawPlan = (...given) => { window.planDrawnWhileHidden.push(document.querySelector('main').hidden); return draw(...given); }; }""")
        page.locator("#zone-page-close").click()
        # 回輸入頁那一次是在輸入頁已經看得到之後才畫的（先畫再顯示等於白畫：藏著的時候字量不到寬度）。
        assert page.evaluate("window.planDrawnWhileHidden") == [False]
        assert page.locator("[data-drawn-hidden]").count() == 0
        assert page.evaluate(_PLAN_LABELS) == before
        assert len(asked) == 1
        expect(page.locator("#messages")).to_have_text("考卷寫在訊息列的一句")
        assert watched.page_errors == [] and watched.console_errors == []


def test_a_line_says_changes_are_not_in_the_file_yet_until_they_are_saved(tmp_path: Path, browser: Browser) -> None:
    """這一頁沒有存檔鍵，改動當場套到畫面、但要存檔才進檔案。1400×1000 下：剛打開、還沒改過時沒有那一句；改規則之後
    第一屏看得到一句白話講還沒存、兩條存的路（回輸入頁按儲存、或按拿去搜尋）；標題列與底列都還是一行。
    回輸入頁再進來（還沒存）照樣在；在輸入頁按「儲存」成功之後再進來就沒有了。之後改清單、套用到選中的區又出現。"""
    _prepare(tmp_path)
    context = browser.new_context(viewport={"width": 1400, "height": 1000})
    with _serve(tmp_path) as base:
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base)
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        _open_overview(page)
        _settled(page)
        line = page.locator("#zone-page-unsaved")
        expect(line).to_have_count(0)
        head = page.locator("#zone-page .zv-head").bounding_box()
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        expect(line).to_be_visible()
        text = cast(str, line.text_content())
        assert text == "改的還沒存進檔案：回「房間方案輸入」按「儲存」，或按底下的「拿去搜尋」（會先存再跳到搜尋頁，不會自己開始算；到那一頁按「開始搜尋」才會算）"
        assert not any("a" <= char.lower() <= "z" for char in text)
        box, back, now = line.bounding_box(), page.locator("#zone-page-close").bounding_box(), page.locator("#zone-page .zv-head").bounding_box()
        assert box is not None and back is not None and head is not None and now is not None
        # 第一屏看得到、整句在視窗裡、一行；標題列沒被撐高；「回房間方案輸入」還在它右邊、沒被擠出去。
        assert box["y"] >= 0 and box["y"] + box["height"] <= 1000 and box["x"] >= 0 and box["height"] < 40
        assert now["height"] == head["height"] and back["x"] >= box["x"] + box["width"] and back["x"] + back["width"] <= 1400
        for width in (1280, 1400, 1440):
            page.set_viewport_size({"width": width, "height": 1000})
            assert page.locator("#zone-page .zv-head").evaluate("""head => {
                const bounds = head.getBoundingClientRect();
                const text = document.createRange();
                text.selectNodeContents(head.querySelector(".zv-unsaved"));
                return [...head.children].every(child => {
                    const r = child.getBoundingClientRect();
                    return r.top >= bounds.top && r.bottom <= bounds.bottom && r.right <= bounds.right;
                }) && text.getClientRects().length === 1;
            }"""), width
        tops = {round(cast(dict[str, float], page.locator(f"#zone-result-{key}").bounding_box())["y"]) for key in ("search", "fixed", "still")}
        assert max(tops) - min(tops) < 8
        # 回輸入頁再進來，還沒存：照樣在。
        page.locator("#zone-page-close").click()
        page.locator("#zone-page-open").click()
        expect(line).to_be_visible()
        # 輸入頁按儲存（方案與每一區的做法都存成）：再進來就沒有了。
        page.locator("#zone-page-close").click()
        _save(page, design=True)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(line).to_have_count(0)
        # 改清單也算改動；存過之後再改，那一句再出現。
        page.locator("#zone-list-select").click()
        page.locator("#zone-panel-remove-glass_wool_96k").click()
        _apply(page)
        expect(line).to_be_visible()
        page.locator("#zone-page-close").click()
        _save(page, design=True)
        page.locator("#zone-page-open").click()
        expect(line).to_have_count(0)
        _click_zone(page, "yL/z1")
        page.locator("#zone-panel-mode-keep_original").check()
        _apply(page)
        expect(line).to_be_visible()
        assert errors == []
    context.close()


def test_the_not_saved_line_stays_when_a_save_did_not_really_store_everything(tmp_path: Path, browser: Browser) -> None:
    """「改的還沒存進檔案」那一句只在真的都存成之後才拿掉：
    ① 輸入頁按「儲存」，方案存了、每一區的做法被伺服器拒收（500）：訊息列講做法沒存成，再進新頁那一句還在；
    ② 「另存新名字」撞到已經有的名字（409）、什麼都沒存：那一句還在；
    ③ 放行之後正常存一次：那一句不見（前面不是因為它永遠不拿掉）。"""
    _prepare(tmp_path)
    _write(tmp_path, {**ZONE_NAMING_SCHEME, "scheme_id": OTHER})
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_overview(page)
        _settled(page)
        line = page.locator("#zone-page-unsaved")
        _click_zone(page, "yL/z3")
        page.locator("#zone-panel-mode-search").check()
        _apply(page)
        expect(line).to_be_visible()
        refused: list[str] = []

        def refuse_design(route: Route) -> None:
            if route.request.method == "PUT":
                refused.append(route.request.url)
                route.fulfill(status=500, content_type="application/json", body=json.dumps({"error": "考卷故意不收的做法"}, ensure_ascii=False))
            else:
                route.continue_()

        page.route("**/api/designs/*", refuse_design)
        page.locator("#zone-page-close").click()
        with page.expect_response(lambda r: r.url.endswith(f"/api/schemes/{SAMPLE}") and r.request.method == "PUT") as scheme:
            page.locator("#save").click()
        assert scheme.value.ok
        expect(page.locator("#messages")).to_contain_text("每一區的做法沒存成：考卷故意不收的做法")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
        assert len(refused) == 1
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(line).to_be_visible()
        # 另存撞名：方案那一步就被擋（409），做法也沒送。
        page.locator("#zone-page-close").click()
        open_save_as(page)
        page.locator("#save-as-id").fill(OTHER)
        with page.expect_response(lambda r: r.url.endswith(f"/api/schemes/{OTHER}") and r.request.method == "PUT") as taken:
            page.locator("#save-as").click()
        assert taken.value.status == 409
        expect(page.locator("#save-as-note")).to_contain_text("已經有叫")
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
        assert len(refused) == 1
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(line).to_be_visible()
        # 放行，正常存一次：那一句不見。
        page.unroute("**/api/designs/*")
        page.locator("#zone-page-close").click()
        _save(page, design=True)
        page.locator("#zone-page-open").click()
        expect(page.locator("#zone-map")).to_be_visible()
        expect(line).to_have_count(0)
        assert watched.page_errors == []
    assert json.loads((tmp_path / "designs" / f"{SAMPLE}.json").read_text(encoding="utf-8"))["zones"] == {"yL": {"z3": SEARCH}}


# 捲到最底之後：底列的上緣，與清單卡、左右兩欄各自最後一個東西的下緣（px，照視窗量）。
_BOTTOMS = """() => { scrollTo(0, document.documentElement.scrollHeight);
    const bottom = node => node.getBoundingClientRect().bottom;
    return {bar: document.getElementById('zone-result').getBoundingClientRect().top, list: bottom(document.getElementById('zone-list')),
            side: bottom(document.querySelector('#zone-page .zv-side').lastElementChild),
            map: bottom(document.querySelector('#zone-page .zv-mapwrap').lastElementChild), view: innerHeight,
            end: Math.abs(scrollY + innerHeight - document.documentElement.scrollHeight) < 1}; }"""
# 叫瀏覽器把清單卡捲進畫面、靠視窗下緣（跳到某一格、按 Tab 走到某一格時瀏覽器做的就是這件事）：卡的下緣與底列的上緣。
_BROUGHT_INTO_VIEW = """() => { const card = document.getElementById('zone-list');
    scrollTo(0, 0); card.scrollIntoView({block: 'end'});
    return {list: card.getBoundingClientRect().bottom, bar: document.getElementById('zone-result').getBoundingClientRect().top}; }"""


def test_scrolled_to_the_end_nothing_is_left_under_the_bottom_bar(tmp_path: Path, browser: Browser) -> None:
    """底列黏在視窗下緣。視窗開矮（1400×700）讓左右兩欄都比視窗長：
    ① 瀏覽器把清單卡捲進畫面、靠視窗下緣時（跳到某一格、Tab 走到某一格），卡要停在底列上面，不准有一截被底列壓住；
    ② 捲到最底：清單卡、右欄最後一個東西、左邊那張圖都整個在底列上面。
    小框打開、清單卡收起來時兩樣都再考一次。底列仍是一行。"""
    _prepare(tmp_path)
    context = browser.new_context(viewport={"width": 1400, "height": 700})
    with _serve(tmp_path) as base:
        page = context.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(base)
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        _open_overview(page)
        _settled(page)
        page.locator("#zone-rule-mode").select_option("search")
        _settled(page)
        for opened in (False, True):
            if opened:
                page.locator("#zone-list-select").click()
                expect(page.locator("#zone-panel")).to_be_visible()
            brought = cast(dict[str, float], page.evaluate(_BROUGHT_INTO_VIEW))
            assert brought["list"] <= brought["bar"], (opened, brought)
            found = cast(dict[str, float], page.evaluate(_BOTTOMS))
            assert found["end"] and found["bar"] < found["view"], found
            assert found["list"] <= found["bar"] and found["side"] <= found["bar"] and found["map"] <= found["bar"], (opened, found)
            tops = {round(cast(dict[str, float], page.locator(f"#zone-result-{key}").bounding_box())["y"]) for key in ("search", "fixed", "still")}
            assert max(tops) - min(tops) < 8
        assert errors == []
    context.close()
