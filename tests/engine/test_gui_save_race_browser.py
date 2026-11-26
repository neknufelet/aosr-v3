"""真瀏覽器考首頁存檔途中換方案（#1187）：按「儲存」「另存新名字」「計算這個方案」到伺服器回覆之前，
「打開」「開一份新的」（從範本開一份新的）換不了表單；存進去的方案與每一區做法是按下那一刻那一份；另存成功後首頁「已存的方案」馬上有新名字。
兩份方案與做法都是考卷自己編的（房間尺寸、深度上限、清單都不是任何真資料）；只攔網路、伺服器是考卷自己起的。"""
import json
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Browser, Locator, Page, Request, Route, expect

from tests.engine._input_page_steps import archive_then_recreate, open_example_scheme, open_save_as, open_scheme, scheme_open_button
from tests.engine._gui_cache import gui_startup_identity_memo as gui_startup_identity_memo
from tests.engine._surface_zone_cases import zone_document
from tests.engine.test_gui_browser import _open, _serve, browser as browser
from tests.engine.test_gui_input_open_last_browser import LATEST as OL_LATEST, MIDDLE as OL_MIDDLE, STAMPS as OL_STAMPS
from tests.engine.test_gui_input_open_last_browser import _opened as _ol_opened, _template as _ol_template, _write as _ol_write
from tests.engine.test_gui_input_setup_browser import _filled
from tests.engine.test_gui_zone_elevation_browser import _open_scheme, _write

A, B, COPY = "race-a", "race-b", "race-copy"
ROOM_A = {"Lx": 5.2, "Ly": 3.0, "Lz": 2.5}
ROOM_B = {"Lx": 5.6, "Ly": 3.2, "Lz": 2.6}
# 兩份都是六面整面一區（切法一樣，伺服器收得下把 B 的做法寫進 A）；做法內容不一樣，看得出是誰的。
DESIGN_A = {"schema_version": "aosr.zone_design.v1", "scheme_id": A, "room_max_depth_m": 0.11,
            "zones": {"x0": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_48k", "none"]}}}}
DESIGN_B = {"schema_version": "aosr.zone_design.v1", "scheme_id": B, "room_max_depth_m": 0.23,
            "zones": {"y0": {"whole_wall": {"mode": "keep_original"}},
                      "xL": {"whole_wall": {"mode": "search", "allowed": ["glass_wool_24k", "glass_wool_96k", "none"]}}}}
UNLOCKED = "!document.querySelector('main').inert"
# 鎖頁期間由頁面程式按「開一份新的」（#1265 單元 5b，取代「載入範本」）被擋下時，那一塊裡寫的那一句。
REFUSED_NEW = "上一個動作還沒做完，沒有開新的；等它做完再按一次「開一份新的」"


def _prepare(tmp_path: Path) -> None:
    for name, room, design in ((A, ROOM_A, DESIGN_A), (B, ROOM_B, DESIGN_B)):
        document = zone_document() | {"scheme_id": name}
        cast(dict[str, object], document["scene"])["room_m"] = dict(room)
        _write(tmp_path, document)
        (tmp_path / "designs").mkdir(exist_ok=True)
        (tmp_path / "designs" / f"{name}.json").write_text(json.dumps(design), encoding="utf-8")


def _files(tmp_path: Path, name: str) -> tuple[bytes, bytes]:
    return ((tmp_path / "schemes" / f"{name}.json").read_bytes(), (tmp_path / "designs" / f"{name}.json").read_bytes())


def _is_put(request: Request, path: str) -> bool:
    return request.method == "PUT" and urlsplit(request.url).path == path


def _settle(page: Page) -> None:
    page.wait_for_function(UNLOCKED)
    page.wait_for_load_state("networkidle")


def _control_save(page: Page, tmp_path: Path) -> tuple[bytes, bytes]:
    """打開 A、沒人打擾地存一次：磁碟上那兩個檔就是「按下儲存那一刻 A 的方案與做法」的逐位答案。"""
    _open_scheme(page, A)
    page.wait_for_load_state("networkidle")
    with page.expect_response(lambda r: _is_put(r.request, f"/api/designs/{A}")) as design:
        page.locator("#save").click()
    assert design.value.ok, design.value.text()
    _settle(page)
    return _files(tmp_path, A)


def _hold_first(page: Page, pattern: str, method: str) -> list[Route]:
    """攔住第一個符合的請求不放（之後同一路徑的照放），回傳的清單裡那一個由考卷放行。"""
    held: list[Route] = []

    def handler(route: Route) -> None:
        if route.request.method == method and not held:
            held.append(route)
        else:
            route.continue_()

    page.route(pattern, handler)
    return held


def _requests(page: Page) -> list[str]:
    seen: list[str] = []
    page.on("request", lambda request: seen.append(f"{request.method} {urlsplit(request.url).path}"))
    return seen


def _press(page: Page, button: str) -> None:
    """不用滑鼠、直接對按鈕發一次點擊：鎖頁（inert）擋得住滑鼠與鍵盤，擋不住頁面程式自己發的點擊
    （首頁第一屏的「打開來改」就是這樣借「打開」），所以程式裡那道鎖也要擋得住。"""
    page.locator(button).evaluate("node => node.click()")


def _try_open_b(page: Page, button: Locator) -> None:
    """存檔卡在半路時去按「打開」B；等到有結果才回：被擋下（旁邊寫了原因）或真的換成 B 了。"""
    button.evaluate("node => node.click()")
    page.wait_for_function(
        "name => document.getElementById('open-note').textContent !== '' ||"
        " (document.getElementById('save-id').value === name && !document.querySelector('main').inert)", arg=B)


def _assert_open_was_refused(page: Page, seen: list[str]) -> None:
    assert page.evaluate("document.querySelector('main').inert") is True
    assert page.locator("#open-note").inner_text() == f"上一個動作還沒做完，沒有打開「{B}」；等它做完再按一次「打開」"
    assert page.locator("#open-note").get_attribute("class") == "notice"
    assert page.evaluate("[scheme.scheme_id, zoneDesign.scheme_id, document.getElementById('save-id').value]") == [A, A, A]
    assert f"GET /api/schemes/{B}" not in seen and f"GET /api/designs/{B}" not in seen


def test_opening_another_scheme_while_saving_is_refused_and_the_saved_design_is_the_pressed_one(
        tmp_path: Path, browser: Browser) -> None:
    """存 A 的方案那個請求卡住時按「打開」B：打開被擋下、旁邊照講；放行後 A 的方案檔與做法檔逐位等於
    沒人打擾時存的那一份，B 的兩個檔沒被動，畫面停在 A、訊息列是存好了，被擋的那一句留在「打開」旁邊。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        control = _control_save(page, tmp_path)
        b_before = _files(tmp_path, B)
        open_b = scheme_open_button(page, B)
        held = _hold_first(page, f"**/api/schemes/{A}", "PUT")
        seen = _requests(page)
        with page.expect_request(lambda r: _is_put(r, f"/api/schemes/{A}")):
            page.locator("#save").click()
        _try_open_b(page, open_b)
        _assert_open_was_refused(page, seen)
        with page.expect_response(lambda r: _is_put(r.request, f"/api/designs/{A}")) as design:
            held[0].continue_()
        assert design.value.ok, design.value.text()
        _settle(page)
        assert _files(tmp_path, A) == control
        assert _files(tmp_path, B) == b_before
        assert page.evaluate("[scheme.scheme_id, zoneDesign.scheme_id, zoneDesign.room_max_depth_m]") == [A, A, 0.11]
        assert page.locator("#save-id").input_value() == A
        expect(page.locator("#messages")).to_have_class("ok")
        expect(page.locator("#open-note")).to_contain_text(f"沒有打開「{B}」")
        assert f"GET /api/schemes/{B}" not in seen
        assert watched.page_errors == [] and watched.console_errors == []


def test_loading_a_template_while_saving_as_is_refused_and_the_copy_is_the_pressed_scheme(
        tmp_path: Path, browser: Browser) -> None:
    """另存卡在第一步（圖面檢查）時按「開一份新的」（從範本開一份新的）：被擋下、旁邊照講；放行後新名字那份方案與做法就是按下那一刻的 A。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        scheme_a, design_a = (json.loads(content) for content in _control_save(page, tmp_path))
        open_save_as(page)
        page.locator("#save-as-id").fill(COPY)
        held = _hold_first(page, "**/api/plan", "POST")
        seen = _requests(page)
        with page.expect_request("**/api/plan"):
            page.locator("#save-as").click()
        _press(page, "#new-scheme-go")
        page.wait_for_function("document.getElementById('new-scheme-note').textContent !== ''")
        assert page.evaluate("document.querySelector('main').inert") is True
        assert page.locator("#new-scheme-note").inner_text() == REFUSED_NEW
        assert page.evaluate("[scheme.scheme_id, zoneDesign.scheme_id]") == [A, A]
        assert not any(line.startswith("GET /api/example") for line in seen)
        # 鎖頁擋不到的路在半路改了一格（同最後一題）：另存也是先抄、後等，新名字那份不帶這一格。
        with page.expect_request("**/api/validate"):
            page.locator("#room-Ly").evaluate(
                "node => { node.value = '3.3'; node.dispatchEvent(new Event('input', {bubbles: true})); }")
        assert page.evaluate("scheme.scene.room_m.Ly") == 3.3
        held[0].continue_()
        page.wait_for_function("name => document.getElementById('save-as-note').textContent.includes(name)", arg=COPY)
        _settle(page)
        stored_scheme, stored_design = (json.loads(content) for content in _files(tmp_path, COPY))
        assert stored_scheme == scheme_a | {"scheme_id": COPY}
        assert stored_design == design_a | {"scheme_id": COPY}
        assert page.evaluate("[scheme.scheme_id, zoneDesign.scheme_id, document.getElementById('save-id').value]") == [COPY, COPY, COPY]
        expect(page.locator("#save-as-note")).to_have_class("ok")
        expect(page.locator("#new-scheme-note")).to_contain_text("沒有開新的")
        assert watched.page_errors == [] and watched.console_errors == []


def test_calculate_starts_the_scheme_that_was_on_the_form_when_pressed(tmp_path: Path, browser: Browser) -> None:
    """按「計算這個方案」、存方案那個請求卡住時按「打開」B：被擋下；放行後開算的那一筆是 A，A 的兩個檔也是 A 的。
    開算的請求由考卷接住回一句錯（不真的算）；只考送出去的是哪一份。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        control = _control_save(page, tmp_path)
        open_b = scheme_open_button(page, B)
        started = _catch_start(page)
        held = _hold_first(page, f"**/api/schemes/{A}", "PUT")
        seen = _requests(page)
        with page.expect_request(lambda r: _is_put(r, f"/api/schemes/{A}")):
            page.locator("#calculate").click()
        _try_open_b(page, open_b)
        _assert_open_was_refused(page, seen)
        with page.expect_request(lambda r: r.method == "POST" and urlsplit(r.url).path == "/api/runs"):
            held[0].continue_()
        expect(page.locator("#messages")).to_have_text("考卷接住了這一次開算")
        _settle(page)
        assert started == [{"scheme_id": A}]
        assert _files(tmp_path, A) == control
        assert page.locator("#save-id").input_value() == A
        assert watched.page_errors == []


def test_save_as_shows_the_new_name_on_the_home_screen_without_a_reload(tmp_path: Path, browser: Browser) -> None:
    """另存成功後，首頁第一屏「已存的方案」那一塊不用重新整理就有新名字（資料早就存好了，只是畫面沒重讀）。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, A)
        page.wait_for_load_state("networkidle")
        expect(page.locator(f"#home-schemes [data-scheme='{A}']")).to_have_count(1)
        expect(page.locator(f"#home-schemes [data-scheme='{COPY}']")).to_have_count(0)
        open_save_as(page)
        page.locator("#save-as-id").fill(COPY)
        with page.expect_response(lambda r: _is_put(r.request, f"/api/schemes/{COPY}")) as saved:
            page.locator("#save-as").click()
        assert saved.value.ok, saved.value.text()
        expect(page.locator(f"#home-schemes [data-scheme='{COPY}']")).to_have_count(1)
        expect(page.locator(f'[data-scheme="{COPY}"] .home-open')).to_have_attribute("aria-current", "true")
        assert watched.page_errors == [] and watched.console_errors == []


def test_fields_changed_behind_the_lock_do_not_leak_into_the_save(tmp_path: Path, browser: Browser) -> None:
    """鎖頁擋不到的路（頁面程式自己改格子、自己改做法）在存檔半路改了東西：存進去的還是按下那一刻那一份。
    卡在第一步（圖面檢查）時把房間長度那一格改掉並發 input 事件、把做法的房間深度上限改掉；
    先確認兩樣真的改進頁面上那一份了（不然這一題是空的），放行後磁碟上兩個檔逐位等於沒人打擾時存的。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        control = _control_save(page, tmp_path)
        held = _hold_first(page, "**/api/plan", "POST")
        with page.expect_request("**/api/plan"):
            page.locator("#save").click()
        with page.expect_request("**/api/validate"):
            page.evaluate("""() => {
                const field = document.getElementById('room-Lx');
                field.value = '5.5';
                field.dispatchEvent(new Event('input', {bubbles: true}));
                zoneDesign.room_max_depth_m = 0.37;
                zoneDesign.zones.x0.whole_wall.allowed.push('glass_wool_96k');
            }""")
        assert page.evaluate("[scheme.scene.room_m.Lx, zoneDesign.room_max_depth_m]") == [5.5, 0.37]
        with page.expect_response(lambda r: _is_put(r.request, f"/api/designs/{A}")) as design:
            held[0].continue_()
        assert design.value.ok, design.value.text()
        _settle(page)
        assert _files(tmp_path, A) == control
        assert json.loads(control[0])["scene"]["room_m"]["Lx"] == ROOM_A["Lx"]
        assert watched.page_errors == []


def test_saving_while_another_scheme_is_being_opened_is_refused(tmp_path: Path, browser: Browser) -> None:
    """鎖的另一半：「打開」B 卡在半路（表單正在換）時按「另存新名字」「儲存」「計算這個方案」，三樣都被擋下照講、
    一個存檔或開算的請求都沒送；放行後照樣換成 B（別人拿著的鎖沒被存檔那幾支放掉），磁碟上四個檔沒被動。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _open_scheme(page, A)
        page.wait_for_load_state("networkidle")
        before = (_files(tmp_path, A), _files(tmp_path, B))
        open_save_as(page)
        page.locator("#save-as-id").fill(COPY)
        open_b = scheme_open_button(page, B)
        held = _hold_first(page, f"**/api/schemes/{B}", "GET")
        seen = _requests(page)
        with page.expect_request(f"**/api/schemes/{B}"):
            open_b.click()
        _press(page, "#save-as")
        expect(page.locator("#save-as-note")).to_have_text("沒有另存：上一個動作還沒做完，等它做完再按一次「另存新名字」")
        _press(page, "#save")
        expect(page.locator("#messages")).to_have_text("上一個動作還沒做完，這一次沒有儲存；等它做完再按一次")
        _press(page, "#calculate")
        expect(page.locator("#messages")).to_have_text("上一個動作還沒做完，這一次沒有開算；等它做完再按一次")
        assert page.evaluate("[document.querySelector('main').inert, inputBusy, scheme.scheme_id]") == [True, True, A]
        held[0].continue_()
        page.wait_for_function("name => document.getElementById('save-id').value === name", arg=B)
        _settle(page)
        # 被擋下的三次一個請求都沒送；圖面檢查只有打開 B 成功後自己問的那一次。
        assert not [line for line in seen if line.startswith(("PUT ", "POST /api/runs"))]
        assert seen.count("POST /api/plan") == 1
        assert (_files(tmp_path, A), _files(tmp_path, B)) == before
        assert not (tmp_path / "schemes" / f"{COPY}.json").exists()
        assert page.evaluate("[scheme.scheme_id, zoneDesign.scheme_id, zoneDesign.room_max_depth_m]") == [B, B, 0.23]
        assert watched.page_errors == [] and watched.console_errors == []


def _catch_start(page: Page) -> list[object]:
    """開算的請求由考卷接住回一句錯（不真的算），回傳的清單記每一次送出去的內容。"""
    started: list[object] = []

    def catch_start(route: Route) -> None:
        if route.request.method != "POST":
            route.continue_()
            return
        started.append(route.request.post_data_json)
        route.fulfill(status=409, content_type="application/json", body=json.dumps({"error": "考卷接住了這一次開算"}))

    page.route("**/api/runs", catch_start)
    return started


def test_calculate_starts_the_pressed_name_when_the_name_changes_behind_the_lock(tmp_path: Path, browser: Browser) -> None:
    """按「計算這個方案」、存方案那個請求卡住時，鎖頁擋不到的路把表單上的方案代號改掉：開算送出的仍是按下那一刻那一份的名字
    （上面那題靠鎖擋下「打開」；這一題鎖還在、名字被頁面程式自己改掉，考的是開算用存下去那份的名字、不讀畫面）。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        control = _control_save(page, tmp_path)
        started = _catch_start(page)
        held = _hold_first(page, f"**/api/schemes/{A}", "PUT")
        with page.expect_request(lambda r: _is_put(r, f"/api/schemes/{A}")):
            page.locator("#calculate").click()
        page.evaluate("name => { document.getElementById('save-id').value = name; scheme.scheme_id = name; }", B)
        assert page.evaluate("scheme.scheme_id") == B
        with page.expect_request(lambda r: r.method == "POST" and urlsplit(r.url).path == "/api/runs"):
            held[0].continue_()
        expect(page.locator("#messages")).to_have_text("考卷接住了這一次開算")
        _settle(page)
        assert started == [{"scheme_id": A}]
        assert _files(tmp_path, A) == control
        assert watched.page_errors == []


# 模擬鎖漏掉（例如不在 <main> 裡、鎖頁蓋不到的頁面）：存檔半路由頁面程式把表單換成 B，像「打開」成功那樣整組換。
_SWAP_TO_B = """([name, design]) => {
    const next = structuredClone(scheme);
    next.scheme_id = name;
    scheme = next; inputPreviewScheme = next; formScheme = next;
    document.getElementById('save-id').value = name;
    openedId = name;
    applyZoneDesign({design, stored: true, broken: false, note: ''});
}"""
_PAGE_NAMES = ("[scheme.scheme_id, document.getElementById('save-id').value, openedId,"
               " zoneDesign.scheme_id, zoneDesign.room_max_depth_m]")


def test_a_save_as_reply_does_not_rename_a_page_that_was_swapped_behind_the_lock(tmp_path: Path, browser: Browser) -> None:
    """第二道守衛之一（app.js::pressedCopy 的 samePage）：另存卡在存方案那一步時表單被換成 B（鎖漏掉），
    回覆回來不准把畫面上的 B 改名成新名字、也不准把「現在開著哪一份」改掉；新名字那份照樣是按下那一刻的 A。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        scheme_a, design_a = (json.loads(content) for content in _control_save(page, tmp_path))
        open_save_as(page)
        page.locator("#save-as-id").fill(COPY)
        held = _hold_first(page, f"**/api/schemes/{COPY}", "PUT")
        with page.expect_request(lambda r: _is_put(r, f"/api/schemes/{COPY}")):
            page.locator("#save-as").click()
        page.evaluate(_SWAP_TO_B, [B, DESIGN_B])
        held[0].continue_()
        page.wait_for_function("name => document.getElementById('save-as-note').textContent.includes(name)", arg=COPY)
        _settle(page)
        assert page.evaluate(_PAGE_NAMES) == [B, B, B, B, 0.23]
        stored_scheme, stored_design = (json.loads(content) for content in _files(tmp_path, COPY))
        assert stored_scheme == scheme_a | {"scheme_id": COPY}
        assert stored_design == design_a | {"scheme_id": COPY}
        assert watched.page_errors == []


def test_a_save_reply_does_not_put_the_old_design_back_on_a_page_swapped_behind_the_lock(
        tmp_path: Path, browser: Browser) -> None:
    """第二道守衛之二（zone_modes.js::prepareZoneDesignSave 回來那一行的比對）：儲存卡在存做法那一步時表單被換成 B，
    晚到的回覆不准把 A 的做法寫回頁面；A 的兩個檔照樣逐位等於按下那一刻的。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        control = _control_save(page, tmp_path)
        held = _hold_first(page, f"**/api/designs/{A}", "PUT")
        with page.expect_request(lambda r: _is_put(r, f"/api/designs/{A}")):
            page.locator("#save").click()
        page.evaluate(_SWAP_TO_B, [B, DESIGN_B])
        with page.expect_response(lambda r: _is_put(r.request, f"/api/designs/{A}")) as design:
            held[0].continue_()
        assert design.value.ok, design.value.text()
        _settle(page)
        assert page.evaluate(_PAGE_NAMES) == [B, B, B, B, 0.23]
        assert page.evaluate("zoneDesign.zones") == DESIGN_B["zones"]
        assert _files(tmp_path, A) == control
        assert watched.page_errors == []


def test_a_save_reply_does_not_change_which_scheme_is_open_on_a_page_swapped_behind_the_lock(
        tmp_path: Path, browser: Browser) -> None:
    """第二道守衛在「儲存」那一處（app.js::storeScheme 的 samePage）：儲存卡在存方案那一步時表單被換成 B，
    回覆回來不准把「現在開著哪一份」改回 A（改回去的話，下一次按儲存會把 B 當成新檔、或拿 A 的名字去比）；
    上一題攔的是後面存做法那一步，這一處在那之前就跑完了，所以另開一題攔存方案。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        control = _control_save(page, tmp_path)
        held = _hold_first(page, f"**/api/schemes/{A}", "PUT")
        with page.expect_request(lambda r: _is_put(r, f"/api/schemes/{A}")):
            page.locator("#save").click()
        page.evaluate(_SWAP_TO_B, [B, DESIGN_B])
        with page.expect_response(lambda r: _is_put(r.request, f"/api/designs/{A}")) as design:
            held[0].continue_()
        assert design.value.ok, design.value.text()
        _settle(page)
        assert page.evaluate(_PAGE_NAMES) == [B, B, B, B, 0.23]
        assert _files(tmp_path, A) == control
        assert watched.page_errors == []


def test_the_cursor_goes_back_to_where_it_was_after_a_save(tmp_path: Path, browser: Browser) -> None:
    """鎖頁會讓游標離開原本那一格：存完放回去。游標在房間長度那一格時存檔（頁面程式發的點擊不搬游標），
    卡住時游標不在任何一格，放行後回到那一格、接著打的字進得去；用滑鼠按「儲存」則回到「儲存」那一顆。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _control_save(page, tmp_path)
        assert page.evaluate("document.activeElement.id") == "save"
        held = _hold_first(page, f"**/api/schemes/{A}", "PUT")
        page.locator("#room-Lx").click()
        assert page.evaluate("document.activeElement.id") == "room-Lx"
        with page.expect_request(lambda r: _is_put(r, f"/api/schemes/{A}")):
            _press(page, "#save")
        assert page.evaluate("document.activeElement.tagName") == "BODY"
        with page.expect_response(lambda r: _is_put(r.request, f"/api/designs/{A}")):
            held[0].continue_()
        _settle(page)
        assert page.evaluate("document.activeElement.id") == "room-Lx"
        page.keyboard.press("ControlOrMeta+A")
        page.keyboard.insert_text("5.3")
        assert page.locator("#room-Lx").input_value() == "5.3"
        assert watched.page_errors == []


def test_a_field_changed_during_the_save_gets_its_preview_once_the_page_unlocks(tmp_path: Path, browser: Browser) -> None:
    """鎖著的時候預覽不問（晚到的也作廢）：存檔半路改了一格，解鎖後要替目前這份補問一次，
    不然伺服器換算的那一份（預覽）停在改之前的值。"""
    _prepare(tmp_path)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        _control_save(page, tmp_path)
        assert page.evaluate("inputPreview.field_values['room-Lx'].value") == ROOM_A["Lx"]
        held = _hold_first(page, f"**/api/schemes/{A}", "PUT")
        seen = _requests(page)
        with page.expect_request(lambda r: _is_put(r, f"/api/schemes/{A}")):
            page.locator("#save").click()
        with page.expect_request("**/api/validate"):
            page.locator("#room-Lx").evaluate(
                "node => { node.value = '5.5'; node.dispatchEvent(new Event('input', {bubbles: true})); }")
        assert "POST /api/input-preview" not in seen
        with page.expect_response(lambda r: _is_put(r.request, f"/api/designs/{A}")):
            held[0].continue_()
        page.wait_for_function("inputPreview.field_values['room-Lx'].value === 5.5")
        _settle(page)
        assert seen.count("POST /api/input-preview") == 1
        assert watched.page_errors == []


def test_the_refused_words_beside_the_button_outlive_the_preview_asked_after_unlock(tmp_path: Path, browser: Browser) -> None:
    """#1193：表單有一格收不下（預覽回 400）時按「另存新名字」，卡在圖面檢查那一步去按「開一份新的」：被擋下、旁邊照講。
    放行後另存被拒、頁面解鎖、替目前這份補問一次預覽（也是 400，訊息列再換一次）：等那一次回覆處理完，
    「開一份新的」那一塊裡被擋的那一句與「另存新名字」旁邊沒有另存的原因都還在，訊息列講的是表單的問題。"""
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        # 開機那一份範例就夠：地板換成材料構造、第一層厚度清空，預覽就回 400（房間尺寸亂填預覽不拒）。
        _filled(page, "#material-kind-floor", "assembly")
        with page.expect_response("**/api/input-preview") as preview:
            page.locator("#layer-floor-0-thickness_m").fill("")
        assert preview.value.status == 400
        expect(page.locator("#messages")).to_have_class("notice")
        open_save_as(page)
        page.locator("#save-as-id").fill(COPY)
        held = _hold_first(page, "**/api/plan", "POST")
        asked_again: list[Route] = []
        page.route("**/api/input-preview", lambda route: asked_again.append(route))
        with page.expect_request("**/api/plan"):
            page.locator("#save-as").click()
        _press(page, "#new-scheme-go")
        page.wait_for_function("document.getElementById('new-scheme-note').textContent !== ''")
        refused = REFUSED_NEW
        assert page.locator("#new-scheme-note").inner_text() == refused
        # 補問那個請求攔在半路：這時另存自己的話都寫完了，才開始盯訊息列，放行後它一被改寫就是補問的回覆處理完了。
        with page.expect_request("**/api/input-preview"):
            held[0].continue_()
        page.wait_for_function(UNLOCKED)
        not_saved = page.locator("#save-as-note").inner_text()
        assert not_saved.startswith("沒有另存：") and page.locator("#new-scheme-note").inner_text() == refused
        page.evaluate("""() => {
            window.saidAfterUnlock = false;
            new MutationObserver(() => { window.saidAfterUnlock = true; }).observe(
                document.getElementById("messages"), {childList: true, characterData: true, subtree: true, attributes: true});
        }""")
        assert len(asked_again) == 1
        with page.expect_response("**/api/input-preview") as reply:
            asked_again[0].continue_()
        assert reply.value.status == 400
        page.wait_for_function("window.saidAfterUnlock")
        assert page.locator("#new-scheme-note").inner_text() == refused
        assert page.locator("#new-scheme-note").get_attribute("class") == "notice"
        assert page.locator("#save-as-note").inner_text() == not_saved
        assert page.locator("#save-as-note").get_attribute("class") == "notice"
        assert page.locator("#messages").get_attribute("class") == "notice"
        assert "地板材料構造 › 第 1 層" in page.locator("#messages").inner_text()
        assert not (tmp_path / "schemes" / f"{COPY}.json").exists()
        assert watched.page_errors == []


def _watch_messages(page: Page) -> None:
    """從這一刻起盯訊息列：它一被改寫（字或 class），window.saidSince 就是 true。"""
    page.evaluate("""() => {
        window.saidSince = false;
        new MutationObserver(() => { window.saidSince = true; }).observe(
            document.getElementById("messages"), {childList: true, characterData: true, subtree: true, attributes: true});
    }""")


@pytest.mark.parametrize("button", ["#save", "#calculate"])
def test_the_refused_words_beside_the_button_outlive_a_save_the_server_turns_down(
        tmp_path: Path, browser: Browser, button: str) -> None:
    """#1188 第 7 條：「儲存」（「計算」先存的那一步同一條路）卡在圖面檢查時按「開一份新的」：被擋下、那一塊裡照講。
    放行後伺服器不收這一次存檔（開著的那一份從選單封存了、表單留著，資料夾裡又有了同名的方案，存檔當成新建，照規矩不蓋掉）：
    訊息列寫伺服器的原因，「開一份新的」那一塊裡被擋的那一句還在——不能因為存檔自己也失敗，那一下就看起來像沒按到。"""
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        example = open_example_scheme(page, base, tmp_path)
        name = cast(str, example["scheme_id"])
        before = archive_then_recreate(page, tmp_path, name)
        with page.expect_response("**/api/input-preview") as preview:
            page.locator("#room-Lx").fill(str(cast(dict[str, dict[str, float]], example["scene"])["room_m"]["Lx"] + 0.5))
        assert preview.value.ok
        page.wait_for_load_state("networkidle")
        held = _hold_first(page, "**/api/plan", "POST")
        with page.expect_request("**/api/plan"):
            page.locator(button).click()
        _press(page, "#new-scheme-go")
        page.wait_for_function("document.getElementById('new-scheme-note').textContent !== ''")
        refused = REFUSED_NEW
        assert page.locator("#new-scheme-note").inner_text() == refused
        with page.expect_response(lambda r: _is_put(r.request, f"/api/schemes/{name}")) as put:
            held[0].continue_()
        assert not put.value.ok
        turned_down = cast(str, put.value.json()["error"])
        expect(page.locator("#messages")).to_have_text(turned_down)
        _settle(page)
        assert page.locator("#messages").get_attribute("class") == "notice"
        assert page.locator("#messages").inner_text() == turned_down
        assert page.locator("#new-scheme-note").inner_text() == refused
        assert page.locator("#new-scheme-note").get_attribute("class") == "notice"
        assert (tmp_path / "schemes" / f"{name}.json").read_bytes() == before
        assert watched.page_errors == []


def test_why_it_was_not_opened_stays_beside_open_when_the_preview_asked_afterwards_fails_too(
        tmp_path: Path, browser: Browser) -> None:
    """#1188 第 8 條：表單有一格收不下（預覽回 400）時去「打開」一份現在檢查不過的方案：沒打開、原因寫在「打開」旁邊。
    表單換不過去之後頁面替原本那一份補問一次預覽（也是 400，訊息列再換一次）：等那一次回覆處理完，
    「沒有打開…」那一句還在。"""
    _prepare(tmp_path)
    outside = json.loads((tmp_path / "schemes" / f"{A}.json").read_text(encoding="utf-8")) | {"scheme_id": "race-outside"}
    speaker = next(iter(outside["speakers"]))
    outside["speakers"][speaker]["x"] = ROOM_A["Lx"] + 3
    (tmp_path / "schemes" / "race-outside.json").write_text(json.dumps(outside), encoding="utf-8")
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        # 開機那一份範例：地板換成材料構造、第一層厚度清空，預覽就回 400（同 #1193 那一題）。
        _filled(page, "#material-kind-floor", "assembly")
        with page.expect_response("**/api/input-preview") as preview:
            page.locator("#layer-floor-0-thickness_m").fill("")
        assert preview.value.status == 400
        shown = page.locator("#save-id").input_value()
        # 存著的那一份讀出來就檢查不過，「打開」自己不問預覽：之後唯一的一次預覽就是補問的那一次，攔在半路。
        asked: list[Route] = []
        page.route("**/api/input-preview", lambda route: asked.append(route))
        open_scheme(page, "race-outside")
        page.wait_for_function("document.getElementById('open-note').textContent !== ''")
        not_opened = page.locator("#open-note").inner_text()
        assert not_opened.startswith("沒有打開「race-outside」：") and page.locator("#save-id").input_value() == shown
        page.wait_for_function(UNLOCKED)
        for _ in range(500):
            if asked:
                break
            page.wait_for_timeout(20)
        assert len(asked) == 1, "打開失敗後沒有替原本那一份補問預覽"
        _watch_messages(page)
        with page.expect_response("**/api/input-preview") as reply:
            asked[0].continue_()
        assert reply.value.status == 400
        page.wait_for_function("window.saidSince")
        assert page.locator("#open-note").inner_text() == not_opened
        assert page.locator("#open-note").get_attribute("class") == "notice"
        assert page.locator("#messages").get_attribute("class") == "notice"
        assert "地板材料構造 › 第 1 層" in page.locator("#messages").inner_text()
        assert page.locator("#save-id").input_value() == shown
        assert watched.page_errors == []


_RUN_ID = "a" * 32


def _held(page: Page, held: list[Route], what: str) -> None:
    """等攔下的請求出現（最多約 10 秒）。"""
    for _ in range(500):
        if held:
            return
        page.wait_for_timeout(20)
    raise AssertionError(f"{what}沒有出去")


def _hold_follow_and_save_plan(page: Page, mode: str, attack: dict[str, bool]) -> tuple[list[Route], list[Route]]:
    """接進度那一步（儲存：/api/runs；另存：/api/runs/<代號>）與 ``attack`` 打開後的第一次圖面檢查都攔下來。"""
    held_follow: list[Route] = []
    held_save_plan: list[Route] = []

    def on_runs(route: Route) -> None:
        if mode == "save" and attack["on"]:
            held_follow.append(route)
        else:
            route.continue_()

    def on_run_item(route: Route) -> None:
        if mode == "save":
            route.fulfill(status=200, content_type="application/json", body=json.dumps(
                {"run_id": _RUN_ID, "scheme_id": "s-other", "status": "stopped", "display_text": "已停止", "stderr_tail": []}))
        else:
            held_follow.append(route)

    def on_plan(route: Route) -> None:
        if attack["on"] and not held_save_plan:
            held_save_plan.append(route)
        else:
            route.continue_()

    page.route("**/api/runs", on_runs)
    page.route(f"**/api/runs/{_RUN_ID}", on_run_item)
    page.route("**/api/plan", on_plan)
    return held_follow, held_save_plan


def _turned_down_while_another_scheme_finishes_opening(tmp_path: Path, browser: Browser, mode: str) -> dict[str, str]:
    """#1265 單元 5c 第三趟（複查第 1 條）：開一份有計算在跑的方案、接進度那一步還沒回來時改出一格檢查不過的（房長 -1），
    按「儲存」或「另存新名字」；存檔那一次圖面檢查攔在半路，等「打開」尾端自己的那一次圖面檢查也送出去了才放行存檔那一次。
    存檔那一次回來時已經不是最新的一次——帶著送出去那一份的檢查照樣要算數。回訊息列、頂條存檔那一句、另存旁那一句。"""
    with _serve(tmp_path) as base:
        doc = _ol_template(base)
        _ol_write(tmp_path, "s-main", doc, OL_STAMPS[OL_LATEST])
        _ol_write(tmp_path, "s-other", doc, OL_STAMPS[OL_MIDDLE])
        with _open(browser, "about:blank", viewport_width=1440) as watched:
            page = watched.page
            attack = {"on": False}
            held_follow, held_save_plan = _hold_follow_and_save_plan(page, mode, attack)
            page.goto(base)
            _ol_opened(page, "s-main")
            page.wait_for_load_state("networkidle")
            attack["on"] = True
            page.evaluate(f'() => {{ activeRuns = [{{run_id: "{_RUN_ID}", scheme_id: "s-other", status: "running", display_text: "計算中"}}]; }}')
            page.locator("#scheme-menu-button").click()
            page.locator('#home-schemes [data-scheme="s-other"] button.home-open').click()
            _held(page, held_follow, "接進度那一步的請求")
            page.locator("#room-Lx").fill("-1")
            if mode == "save":
                page.locator("#save").click()
            else:
                page.locator("#save-as-toggle").click()
                page.locator("#save-as-id").fill("copy-one")
                page.locator("#save-as").click()
            _held(page, held_save_plan, "存檔那一次圖面檢查")
            held_open_plan: list[Route] = []
            page.unroute("**/api/plan")
            page.route("**/api/plan", lambda route: held_open_plan.append(route))
            followed = {"running": []} if mode == "save" else {
                "run_id": _RUN_ID, "scheme_id": "s-other", "status": "running", "display_text": "計算中", "stderr_tail": []}
            held_follow[0].fulfill(status=200, content_type="application/json", body=json.dumps(followed))
            _held(page, held_open_plan, "「打開」尾端那一次圖面檢查")
            with page.expect_response("**/api/plan") as checked:
                held_save_plan[0].continue_()
            assert checked.value.status == 422, checked.value.status
            # 存檔那一支做完（解鎖、頂條或另存旁邊寫好）才讀；「打開」那一次還攔著，蓋不到這幾處。
            done = "#save-feedback" if mode == "save" else "#save-as-note"
            page.wait_for_function("id => !inputBusy && document.querySelector(id).textContent !== ''", arg=done)
            said = cast(dict[str, str], page.evaluate("""() => Object.fromEntries(
                ['save-feedback', 'save-as-note', 'messages'].flatMap(id => [[id, document.getElementById(id).textContent],
                    [id + '-class', document.getElementById(id).className]]))"""))
            held_open_plan[0].continue_()
            page.wait_for_load_state("networkidle")
            assert not (tmp_path / "schemes" / "copy-one.json").exists()
            assert watched.page_errors == []
            return said


def test_a_save_turned_down_by_the_plan_check_says_why_even_when_a_newer_check_is_on_its_way(
        tmp_path: Path, browser: Browser) -> None:
    """複查第 1 條（儲存）：頂條存檔那一句是紅字、寫圖面檢查的原因，不是上一次留下的綠字「檢查通過」。"""
    said = _turned_down_while_another_scheme_finishes_opening(tmp_path, browser, "save")
    assert said["save-feedback-class"] == "notice", said
    assert "座標超出房間" in said["save-feedback"] and "檢查通過" not in said["save-feedback"], said
    assert said["messages"] == said["save-feedback"], said


def test_a_save_as_turned_down_by_the_plan_check_says_why_even_when_a_newer_check_is_on_its_way(
        tmp_path: Path, browser: Browser) -> None:
    """複查第 1 條（另存新名字）：另存旁邊寫「沒有另存：」加上圖面檢查的原因，不是斷尾；訊息列也是那個原因，不是清空。"""
    said = _turned_down_while_another_scheme_finishes_opening(tmp_path, browser, "saveAs")
    assert said["save-as-note-class"] == "notice", said
    assert said["save-as-note"].startswith("沒有另存：") and "座標超出房間" in said["save-as-note"], said
    assert "座標超出房間" in said["messages"], said
