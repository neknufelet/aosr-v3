"""#1265 單元 6：改了就自動檢查，檢查結果搬到頂條（#1265 示意稿第 36 列）。

拿掉「檢查」那一顆與那一段；訊息列（``#messages``）搬到頂條「打開」那一句底下，跟頂條其他字一起藏在開頁那道門後面。
頂條狀態旁一句（``#check-state``）合成平面圖檢查與限制檢查：還在路上寫「檢查中…」、都沒問題寫「檢查通過」、有問題逐種寫
一顆可按的字（按了捲到問題那裡）。第 2 段右欄不再寫「限制都沒超過」。方案、設計設定、數字都是考卷自己編的。"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

from playwright.sync_api import Browser, Page, Route, expect

from tests.engine._input_page_steps import reveal
from tests.engine._surface_zone_cases import NEW, ROOM_109, room109_document, room109_quarters
from tests.engine._zone_panel_steps import choose_mode
from tests.engine.test_gui_browser import Watched, _assert_quiet, _assert_quiet_but_plan_refused, _open, _serve, browser as browser
from tests.engine.test_gui_input_clear_undo_browser import _release
from tests.engine.test_gui_zone_elevation_browser import _cut_by, _drag, _write

ROOM = "check-room"
LY, LZ = ROOM_109["Ly"], ROOM_109["Lz"]
READY = "document.body.dataset.inputReady === 'true'"
UNLOCKED = "!document.querySelector('main').inert && !inputBusy"
PROBLEM = re.compile(r"^檢查沒過：")
# 天花右上那塊貼的：多孔層加空腔共 0.3 公尺，貼在原本那面固定阻抗的天花上。
DEEP: dict[str, object] = {"kind": "mounted_on_wall", "layers": [
    {"kind": "porous", "thickness_m": 0.12, "flow_resistivity_pa_s_m2": 10500.0, "model": "komatsu", "material_class": "fibrous"},
    {"kind": "air", "thickness_m": 0.18}], "wall": {"kind": "constant_impedance", "impedance_pa_s_m": 8000.0}}


def _room(tmp_path: Path, design: dict[str, object] | None = None, *, deep_ceiling: bool = False) -> None:
    """109 號房的長寬高、六面整面一區，名字換成考卷自己的；``deep_ceiling`` 時天花切成四塊、右上那塊貼 ``DEEP``（0.3 公尺），
    每一塊都指定貼一種。左右對稱關著（這幾題不考鏡子）。"""
    document = room109_document(ROOM)
    zones: dict[str, object] = {}
    if deep_ceiling:
        room109_quarters(document, "ceiling", (8000.0, 8000.0, 8000.0, 8000.0))
        ceiling = cast(dict[str, list[dict[str, object]]], cast(dict[str, object], document["scene"])[NEW])["ceiling"]
        ceiling[3]["material"] = DEEP
        zones["ceiling"] = {str(zone["id"]): {"mode": "fixed"} for zone in ceiling}
    _write(tmp_path, document)
    folder = tmp_path / "designs"
    folder.mkdir(exist_ok=True)
    (folder / f"{ROOM}.json").write_text(json.dumps({"schema_version": "aosr.zone_design.v1", "scheme_id": ROOM,
                                                     "left_right_symmetry": "off", **({"zones": zones} if zones else {}),
                                                     **(design or {})}), encoding="utf-8")


def _opened(page: Page) -> None:
    page.wait_for_function(READY)
    page.wait_for_function(f"openedId === '{ROOM}'")
    page.wait_for_function(UNLOCKED)


def _quiet_but_refused(watched: Watched) -> None:
    """頁面沒丟錯；主控台只有伺服器照規則拒收（平面圖檢查沒過回 422）那幾行，每一行對得上一次（共用的那一支）。"""
    _assert_quiet_but_plan_refused(watched)


def _hold(page: Page, pattern: str) -> list[Route]:
    """攔住之後每一個符合的請求（由考卷放行）。"""
    held: list[Route] = []
    page.route(pattern, lambda route: held.append(route))
    return held


def _seen(page: Page, held: list[Route], what: str) -> None:
    """等攔下的請求出現（最多約 10 秒）。"""
    for _ in range(500):
        if held:
            return
        page.wait_for_timeout(20)
    raise AssertionError(f"{what}沒有出去")


def _let_go(page: Page, pattern: str, held: list[Route]) -> None:
    page.unroute(pattern)
    _release(held)


def _commit(page: Page, selector: str, value: str) -> None:
    """改一格、離開那一格（瀏覽器發 change，算改定）；收起來的先打開。"""
    field = reveal(page, selector)
    field.fill(value)
    field.blur()


def _checks_by_itself(page: Page, edit: Callable[[], None], after: str | re.Pattern[str] = "檢查通過") -> None:
    """改一下、不按任何鈕：平面圖檢查自己出去（攔著時頂條寫「檢查中…」，不留上一句），放行後頂條換成結果。"""
    check = page.locator("#check-state")
    held = _hold(page, "**/api/plan")
    edit()
    _seen(page, held, "平面圖檢查")
    expect(check).to_have_text("檢查中…")
    expect(check).to_have_attribute("data-kind", "pending")
    _let_go(page, "**/api/plan", held)
    expect(check).to_have_text(after)


def test_the_check_button_is_gone_and_the_message_line_sits_in_the_top_bar_behind_the_door(tmp_path: Path, browser: Browser) -> None:
    """頁面上沒有「檢查」鈕；訊息列與頂條那一句住在頂條，開頁那道門關著（讀檔還在路上）時跟頂條其他字一起藏著。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        held = _hold(page, f"**/api/schemes/{ROOM}")
        page.goto(base)
        _seen(page, held, "讀方案檔")
        assert page.evaluate("document.body.dataset.inputStart") == "loading"
        expect(page.locator("#messages")).to_be_hidden()
        expect(page.locator("#check-state")).to_be_hidden()
        _let_go(page, f"**/api/schemes/{ROOM}", held)
        _opened(page)
        expect(page.locator("#check")).to_have_count(0)
        expect(page.locator("main section h2", has_text=re.compile("^檢查$"))).to_have_count(0)
        expect(page.locator("#input-bars #messages")).to_have_count(1)
        # 頂條那一句緊接在存檔狀態膠囊後面（同一排）。
        assert page.evaluate("document.getElementById('save-state').nextElementSibling.id") == "check-state"
        check = page.locator("#input-bars .bar-row #check-state")
        expect(check).to_be_visible()
        expect(check).to_have_text("檢查通過")
        expect(check).to_have_attribute("data-kind", "ok")
        # 訊息列照寫伺服器那一句（存檔、另存照抄它），但「檢查通過」頂條已經寫了，不顯示第二次。
        expect(page.locator("#messages")).to_have_text("檢查通過")
        expect(page.locator("#messages")).to_be_hidden()
        # 第 2 段右欄不寫「限制都沒超過」；限制那一半也真的判過、沒有超過。
        expect(page.locator("#zone-limits-summary")).to_have_count(0)
        assert "限制都沒超過" not in page.locator("body").inner_text()
        assert page.evaluate("zoneLimitsFresh() && zoneLimits.status === 'checked' && zoneLimits.red === 0")
        _assert_quiet(watched)


def test_every_edit_checks_by_itself_without_pressing_anything(tmp_path: Path, browser: Browser) -> None:
    """長寬高、喇叭座標、切一區、清掉一面、撤銷：每一種改動都不用按鈕，頂條先「檢查中…」再「檢查通過」。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        page.locator("#elevation-wall-x0").click()
        _checks_by_itself(page, lambda: _commit(page, "#room-Lz", "3.05"))
        _checks_by_itself(page, lambda: _commit(page, "#speaker-left-x", "1.2"))
        _checks_by_itself(page, lambda: _commit(page, "#receiver-main-y", "2.2"))
        _checks_by_itself(page, lambda: _drag(page, LY, LZ, (0.3, 0.3), (1.3, 1.3)))
        _checks_by_itself(page, lambda: _cut_by(page, "#elevation-clear"))
        _checks_by_itself(page, lambda: page.locator("#elevation-undo-clear").click())
        # 「這間房貼什麼」那一頁改一區、套用、回輸入頁：一樣自己檢查（套用要等頁面安定，攔不住，改數出去了幾次）。
        zone = cast(str, page.locator("#elevation-svg [data-zone-id]").first.get_attribute("data-zone-id"))
        asked: list[str] = []
        page.on("request", lambda request: asked.append(request.method) if request.url.endswith("/api/plan") else None)
        choose_mode(page, "x0", zone, "keep_original")
        assert asked == ["POST"]
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        assert cast(float, page.evaluate("scheme.scene.room_m.Lz")) == 3.05
        _assert_quiet(watched)


# 改定就自動檢查的每一區（頁面程式 AUTO_CHECK_AREAS）：從那一區裡第一個輸入格（沒有就那一區自己）發一次會冒泡的 change。
AREAS = ("room-fields", "speakers", "receivers", "scattering", "use-scattering", "source-model", "furniture-sofa-fields",
         "furniture-table-fields", "furniture-cloud-fields", "cabinet-fields")
CHANGE_IN = """([area, pick]) => {
  const box = document.getElementById(area), target = box.matches(pick) ? box : box.querySelector(pick) ?? box;
  target.dispatchEvent(new Event("change", {bubbles: true}));
}"""


def test_every_field_area_checks_by_itself_and_the_wall_form_only_for_number_boxes(tmp_path: Path, browser: Browser) -> None:
    """每一區改定都自己檢查（含收起來、還沒打開的家具格）；牆面表單只有數字格走這條路（下拉、勾選各自交接完才問，不多問一次）。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        for area in AREAS:
            _checks_by_itself(page, partial(page.evaluate, CHANGE_IN, [area, "input, select"]))
        _checks_by_itself(page, lambda: page.evaluate(CHANGE_IN, ["walls", "input[type=number]"]))
        asked: list[str] = []
        page.on("request", lambda request: asked.append(request.url) if request.url.endswith("/api/plan") else None)
        # 牆面表單裡不是數字格的東西發 change（下拉、勾選各自有自己的交接，這條路不再多問一次）。
        page.evaluate("""() => document.getElementById("walls").dispatchEvent(new Event("change"))""")
        page.wait_for_timeout(300)
        assert asked == []
        expect(page.locator("#check-state")).to_have_text("檢查通過")
        _assert_quiet(watched)


def test_an_automatic_check_that_cannot_reach_the_server_says_so_without_wiping_the_words_beside_the_buttons(
        tmp_path: Path, browser: Browser) -> None:
    """自動那一次連不上伺服器（平面圖檢查的請求斷線）：不往外丟錯、不清掉按鈕旁邊那幾句（存檔、打開、開新的、另存那幾行），
    頂條那一句不停在「檢查中…」，寫「平面圖沒檢查成」（念得出原因）；訊息列寫連不上。接回來再改一格就回「檢查通過」。"""
    _room(tmp_path)
    beside = ("save-feedback", "open-note", "new-scheme-note", "save-as-note")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        page.evaluate("ids => ids.forEach(id => { document.getElementById(id).textContent = '旁邊那一句'; })", list(beside))
        page.route("**/api/plan", lambda route: route.abort())
        _commit(page, "#room-Lz", "3.05")
        check = page.locator("#check-state")
        expect(check).to_have_text("檢查沒過：平面圖沒檢查成")
        expect(check).to_have_attribute("data-kind", "problem")
        expect(page.locator("#check-plan")).to_have_attribute("aria-label", re.compile("連不上"))
        expect(page.locator("#messages")).to_contain_text("連不上")
        assert page.evaluate("ids => ids.map(id => document.getElementById(id).textContent)", list(beside)) == ["旁邊那一句"] * 4
        page.unroute("**/api/plan")
        _commit(page, "#room-Lz", "3.1")
        expect(check).to_have_text("檢查通過")
        assert watched.page_errors == []
        # 斷線那一次瀏覽器主控台照例記一行「載入失敗」（net::ERR_FAILED），伺服器沒有回過任何錯。
        assert watched.failed_responses == []
        assert len(watched.console_errors) == 1 and "ERR_FAILED" in watched.console_errors[0], watched.console_errors


def test_a_click_right_after_typing_lands_and_the_check_goes_after_the_press(tmp_path: Path, browser: Browser) -> None:
    """打完一格直接點牆名鈕（那一格在按下那一刻失焦、發 change）：自動檢查等放開、那一下點完才出去，
    回來清掉平面圖、版面跳了也不會讓那一下點空——點到的就是那一面。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        page.locator("#elevation-wall-x0").click()
        page.evaluate("""() => {
          window.order = [];
          for (const kind of ["pointerdown", "pointerup"]) document.addEventListener(kind, () => order.push(kind), true);
          const fetched = window.fetch;
          window.fetch = (url, ...rest) => { if (String(url).includes("/api/plan")) order.push("plan"); return fetched(url, ...rest); };
        }""")
        page.locator("#room-Ly").fill("")
        page.locator("#elevation-wall-floor").click()
        expect(page.locator("#elevation-wall-floor")).to_have_attribute("aria-pressed", "true")
        expect(page.locator("#check-state")).to_have_text(PROBLEM)
        assert page.evaluate("order") == ["pointerdown", "pointerup", "plan"]
        _quiet_but_refused(watched)


def test_a_plan_problem_turns_the_top_bar_red_and_jumps_to_the_plan(tmp_path: Path, browser: Browser) -> None:
    """喇叭座標超出房間：頂條紅字寫平面圖幾條問題、念得出原因；捲到頁底再點它，第 1 段平面圖回到眼前；改回來就回「檢查通過」。"""
    _room(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _opened(page)
        check = page.locator("#check-state")
        _commit(page, "#speaker-left-x", "20")
        expect(check).to_have_text(re.compile(r"^檢查沒過：平面圖 \d+ 條問題$"))
        expect(check).to_have_attribute("data-kind", "problem")
        go = page.locator("#check-state #check-plan")
        expect(go).to_have_attribute("aria-label", re.compile("座標超出房間"))
        expect(page.locator("#messages")).to_contain_text("座標超出房間")
        expect(page.locator("#messages")).to_be_visible()
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        expect(page.locator(".room-plan")).not_to_be_in_viewport()
        go.click()
        expect(page.locator("#plan-xy")).to_be_in_viewport()
        _commit(page, "#speaker-left-x", "1.2")
        expect(check).to_have_text("檢查通過")
        expect(page.locator("#check-plan")).to_have_count(0)
        _quiet_but_refused(watched)


def _red_ceiling(page: Page) -> None:
    """房間預設總厚度上限填 20 公分：天花右上那塊（0.3 公尺）就超過。"""
    _commit(page, "#depth-room", "20")
    page.wait_for_function("zoneLimitsFresh() && zoneLimits.red > 0")


def test_limits_over_turn_the_top_bar_red_select_that_wall_and_jump_to_the_red_zone(tmp_path: Path, browser: Browser) -> None:
    """超過限制：頂條紅字寫幾條超過限制；點了選天花那一面、捲到標紅的那一塊；右欄只留紅的明細，沒有「限制都沒超過」。"""
    _room(tmp_path, deep_ceiling=True)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        _opened(page)
        check = page.locator("#check-state")
        expect(check).to_have_text("檢查通過")
        _red_ceiling(page)
        red = cast(int, page.evaluate("zoneLimits.red"))
        expect(check).to_have_text(f"檢查沒過：{red} 條超過限制")
        expect(page.locator("#check-limits")).to_have_attribute("aria-label", re.compile("超過"))
        assert page.evaluate("document.getElementById('elevation-wall-ceiling').getAttribute('aria-pressed')") != "true"
        page.evaluate("window.scrollTo(0, 0)")
        page.locator("#check-limits").click()
        expect(page.locator("#elevation-wall-ceiling")).to_have_attribute("aria-pressed", "true")
        zone = page.locator("#elevation [data-zone-id=u1v1]")
        expect(zone).to_have_class(re.compile(r"\bzone-red\b"))
        expect(zone).to_be_in_viewport()
        expect(page.locator("#zone-limits-summary")).to_contain_text(f"有 {red} 條超過限制")
        # 拿掉上限：限制那一半回到沒超過，頂條「檢查通過」，右欄那一句不在。
        _commit(page, "#depth-room", "")
        expect(check).to_have_text("檢查通過")
        expect(page.locator("#zone-limits-summary")).to_have_count(0)
        assert "限制都沒超過" not in page.locator("body").inner_text()
        _assert_quiet(watched)


def test_both_kinds_of_problem_at_once_are_both_named(tmp_path: Path, browser: Browser) -> None:
    """平面圖有問題、又超過限制：頂條兩種都講，各一顆可按的字。"""
    _room(tmp_path, deep_ceiling=True)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        _red_ceiling(page)
        _commit(page, "#speaker-left-x", "20")
        check = page.locator("#check-state")
        expect(check).to_have_text(re.compile(r"^檢查沒過：平面圖 \d+ 條問題；\d+ 條超過限制$"))
        expect(page.locator("#check-state button")).to_have_count(2)
        _quiet_but_refused(watched)


def test_the_top_bar_does_not_keep_the_old_sentence_while_a_check_is_on_its_way(tmp_path: Path, browser: Browser) -> None:
    """紅字之後改好、檢查還在路上（攔住平面圖或限制請求）：頂條寫「檢查中…」，不留舊的紅字；回來才換「檢查通過」。"""
    _room(tmp_path, deep_ceiling=True)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _opened(page)
        check = page.locator("#check-state")
        _commit(page, "#speaker-left-x", "20")
        expect(check).to_have_text(PROBLEM)
        _checks_by_itself(page, lambda: _commit(page, "#speaker-left-x", "1.2"))
        _red_ceiling(page)
        expect(check).to_have_text(PROBLEM)
        held = _hold(page, "**/api/zone-limits")
        _commit(page, "#depth-room", "")
        _seen(page, held, "限制檢查")
        expect(check).to_have_text("檢查中…")
        _let_go(page, "**/api/zone-limits", held)
        expect(check).to_have_text("檢查通過")
        _quiet_but_refused(watched)
