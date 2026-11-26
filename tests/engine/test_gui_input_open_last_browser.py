"""#1265 單元 5c：開頁直接開上一份、拿掉 #home 兩張大卡、第 2 段打開方案時先看前牆、空畫面「看算過的結果」。

有存過的方案時，網址沒帶 ``?scheme=``／``?zones=`` 就開最近存的那一份（跟頂條選單「最近的方案」第一列同一個排序）；
帶的那一份在就開它，不在就寫那一句、照樣開最近那一份；最近那一份打不開就講為什麼、開下一份；全都打不開進空畫面講清楚。
開頁那道門（body 的 data-input-start）等開好那一份才開，``inputReady``（頁面載完的旗子）同一刻設，開機範本一刻都不露出來。
方案的名字、存的時間、喇叭擺哪都是考卷自己編的；只攔網路，伺服器是考卷自己起的。"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from aosr.reporting.result import SchemeResult
from tests.engine._input_page_steps import open_results, open_scheme
from tests.engine.test_gui_archive import _bundle
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser, result as result

READY = "document.body.dataset.inputReady === 'true'"
UNLOCKED = "!document.querySelector('main').inert && !inputBusy"
# 三份存的時間不同：最近的那一份名字排在字母中間，排序寫錯（照名字、舊到新）不會碰巧過。
OLDEST, LATEST, MIDDLE = "zulu-room", "mid-room", "alpha-room"
STAMPS = {OLDEST: 1_000_000_000, MIDDLE: 1_000_000_600, LATEST: 1_000_001_200}


def _template(base: str, template: str = "reference") -> dict[str, object]:
    reply = json.loads(urllib.request.urlopen(f"{base}/api/example?template={template}").read())
    return cast(dict[str, object], reply["scheme"])


def _write(data_dir: Path, name: str, document: dict[str, object] | str, stamp: int | None = None) -> Path:
    path = data_dir / "schemes" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document if isinstance(document, str) else json.dumps(document | {"scheme_id": name}), encoding="utf-8")
    if stamp is not None:
        os.utime(path, (stamp, stamp))
    return path


def _three(data_dir: Path, base: str, broken: tuple[str, ...] = ()) -> None:
    """三份參考房，房間長度各不同（打開的是哪一份看得出來）；``broken`` 那幾份存成伺服器讀不懂的檔。"""
    for index, name in enumerate((OLDEST, MIDDLE, LATEST)):
        scheme = _template(base)
        cast(dict[str, dict[str, float]], scheme["scene"])["room_m"]["Lx"] = 5.2 + index / 5
        _write(data_dir, name, '{"scheme_id": "壞的"' if name in broken else scheme, STAMPS[name])


def _opened(page: Page, name: str) -> None:
    page.wait_for_function(READY)
    page.wait_for_function("name => openedId === name && document.getElementById('save-id').value === name", arg=name)
    page.wait_for_function(UNLOCKED)


def _menu_order(page: Page) -> list[str]:
    page.locator("#scheme-menu-button").click()
    names = cast(list[str], page.locator("#home-schemes [data-scheme]").evaluate_all("rows => rows.map(row => row.dataset.scheme)"))
    page.keyboard.press("Escape")
    return names


def test_the_page_opens_the_latest_saved_scheme_the_same_one_the_menu_lists_first(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base:
        _three(tmp_path, base)
        with _open(browser, base, viewport_width=1440) as watched:
            page = watched.page
            _opened(page, LATEST)
            assert page.locator("#room-Lx").input_value() == "5.6"
            assert _menu_order(page) == [LATEST, MIDDLE, OLDEST]
            expect(page.locator("#open-note")).to_have_text("")
            expect(page.locator("#scheme-section")).to_be_visible()
            # 沒存的開機範本不在表單上：狀態列不是「範本還沒存」。
            expect(page.locator("#save-state")).to_have_text("都存好了（還沒算過）")
            _assert_quiet(watched)


@pytest.mark.parametrize("query", ["scheme", "zones"])
def test_a_link_opens_the_scheme_it_names_when_it_is_saved(tmp_path: Path, browser: Browser, query: str) -> None:
    with _serve(tmp_path) as base:
        _three(tmp_path, base)
        with _open(browser, f"{base}/?{query}={OLDEST}", viewport_width=1440) as watched:
            page = watched.page
            _opened(page, OLDEST)
            assert page.locator("#room-Lx").input_value() == "5.2"
            expect(page.locator("#open-note")).to_have_text("")
            if query == "zones":
                expect(page.locator("#zone-page")).to_be_visible()
            else:
                expect(page.locator("#zone-page")).to_be_hidden()
            _assert_quiet(watched)


@pytest.mark.parametrize("query", ["scheme", "zones"])
def test_a_link_to_a_scheme_that_is_not_saved_says_so_and_opens_the_latest(tmp_path: Path, browser: Browser, query: str) -> None:
    with _serve(tmp_path) as base:
        _three(tmp_path, base)
        with _open(browser, f"{base}/?{query}=ghost-room", viewport_width=1440) as watched:
            page = watched.page
            _opened(page, LATEST)
            page.wait_for_load_state("networkidle")
            note = page.locator("#open-note")
            expect(note).to_be_visible()
            expect(note).to_have_text("沒有打開「ghost-room」：已存的方案裡沒有這一份（可能封存了或改了名字）。"
                                      f"先開了「{LATEST}」；要別份從「換一個方案」另選")
            expect(page.locator("#zone-page")).to_be_hidden()
            expect(page.locator("#scheme-section")).to_be_visible()
            _assert_quiet(watched)


def test_an_unreadable_latest_scheme_is_said_and_the_next_one_opens(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base:
        _three(tmp_path, base, broken=(LATEST,))
        with _open(browser, base, viewport_width=1440) as watched:
            page = watched.page
            _opened(page, MIDDLE)
            page.wait_for_load_state("networkidle")
            note = page.locator("#open-note")
            expect(note).to_be_visible()
            expect(note).to_contain_text(f"沒有打開「{LATEST}」：")
            expect(note).to_contain_text(f"先開了「{MIDDLE}」")
            text = note.inner_text()
            assert "保留原方案" not in text
            # 冒號後面有原因（伺服器讀不懂那一份的那一句），不是空的。
            reason = text.split(f"沒有打開「{LATEST}」：", 1)[1].split("。先開了", 1)[0]
            assert reason.strip(), text
            shots = os.environ.get("AOSR_SHOTS")
            if shots:
                (Path(shots) / "unreadable-latest-note.txt").write_text(text, encoding="utf-8")
            assert page.locator("#room-Lx").input_value() == "5.4"
            assert watched.page_errors == []


@pytest.mark.parametrize("link", ["", "?scheme=ghost-room"])
def test_when_no_saved_scheme_opens_the_empty_screen_says_why_for_each(tmp_path: Path, browser: Browser, link: str) -> None:
    """三份都打不開：空畫面卡片寫每一份的原因，份數只算存著的那 3 份（網址帶的不在的那一份另寫一句、不算進去）；
    「打開」旁邊不留打開途中寫的那一句（那一句帶著開機範本的字眼）。"""
    with _serve(tmp_path) as base:
        _three(tmp_path, base, broken=(OLDEST, MIDDLE, LATEST))
        with _open(browser, f"{base}/{link}", viewport_width=1440) as watched:
            page = watched.page
            page.wait_for_function("document.body.dataset.inputStart === 'empty'")
            page.wait_for_load_state("networkidle")
            note = page.locator("#empty-list-note")
            expect(note).to_be_visible()
            expect(note).to_contain_text("存著的 3 份方案都打不開")
            text = note.inner_text()
            for name in (OLDEST, MIDDLE, LATEST):
                expect(note).to_contain_text(f"沒有打開「{name}」：")
                reason = text.split(f"沒有打開「{name}」：", 1)[1].split("；", 1)[0]
                assert reason.strip() and "保留原方案" not in reason, text
            if link:
                assert text.startswith("沒有打開「ghost-room」：已存的方案裡沒有這一份"), text
            else:
                assert "ghost-room" not in text
            expect(page.locator("#open-note")).to_have_text("")
            shots = os.environ.get("AOSR_SHOTS")
            if shots:
                (Path(shots) / f"all-broken-note{'-link' if link else ''}.txt").write_text(text, encoding="utf-8")
            expect(page.locator("#scheme-section")).to_be_hidden()
            expect(page.locator("#empty-start")).to_be_visible()
            assert page.evaluate("document.body.dataset.inputReady") is None
            assert watched.page_errors == []


# inputReady 設好那一刻：開著哪一份、方案名字那格寫什麼、門開了沒（開頁前就裝上，連開頁那一次都記得到）。
_WATCH_READY = """document.addEventListener('input-ready', () => {
  window.readyAt = [openedId, document.getElementById('save-id').value, document.body.dataset.inputStart ?? null];
});"""


def test_the_gate_stays_shut_until_the_latest_scheme_is_on_the_form(tmp_path: Path, browser: Browser) -> None:
    """讀最近那一份還在路上：第 1 段藏著、inputReady 沒設、只看得到「讀取中…」；讀到了、表單是那一份，才開門、設旗子。"""
    with _serve(tmp_path) as base:
        _three(tmp_path, base)
        with _open(browser, "about:blank", viewport_width=1440) as watched:
            page = watched.page
            page.add_init_script(_WATCH_READY)
            held: list[Route] = []
            page.route(f"{base}/api/schemes/{LATEST}", lambda route: held.append(route) if route.request.method == "GET" else route.continue_())
            page.goto(base, wait_until="commit")
            page.wait_for_function(f"document.body && document.body.dataset.inputStart === 'loading'")
            deadline = time.monotonic() + 30
            while not held and time.monotonic() < deadline:
                page.wait_for_timeout(50)
            assert held, "讀最近那一份的請求沒有出去"
            assert page.evaluate("document.body.dataset.inputStart") == "loading"
            assert page.evaluate("document.body.dataset.inputReady") is None
            expect(page.locator("#scheme-section")).to_be_hidden()
            expect(page.locator("#boot-note")).to_be_visible()
            expect(page.locator("#boot-note")).to_have_text("讀取中…")
            assert page.evaluate("openedId") is None
            held[0].continue_()
            _opened(page, LATEST)
            assert page.evaluate("window.readyAt") == [LATEST, LATEST, None]
            _assert_quiet(watched)


def test_a_running_calculation_is_picked_up_while_the_latest_scheme_opens(tmp_path: Path, browser: Browser) -> None:
    """開頁時有一份在算、但最近存的是另一份：開最近那一份，進度照接回、寫的是在算的那一份的名字，停止鈕在頁底那一行。"""
    script = tmp_path / "sleep-runner.py"
    script.write_text("import time\ntime.sleep(60)\n")
    with _serve(tmp_path, (sys.executable, str(script))) as base:
        _write(tmp_path, "busy-room", _template(base), STAMPS[OLDEST])
        reply = json.loads(urllib.request.urlopen(urllib.request.Request(
            f"{base}/api/runs", data=json.dumps({"scheme_id": "busy-room"}).encode(), method="POST",
            headers={"Content-Type": "application/json"})).read())
        run_id = reply["run_id"]
        try:
            _write(tmp_path, LATEST, _template(base), int(time.time()) + 60)
            with _open(browser, base, viewport_width=1440) as watched:
                page = watched.page
                _opened(page, LATEST)
                page.wait_for_function("() => !document.querySelector('#stop').disabled")
                expect(page.locator("#run-state")).to_be_visible()
                expect(page.locator("#run-state")).to_contain_text("busy-room")
                expect(page.locator("#followed-stop #stop")).to_be_visible()
                expect(page.locator("#save-state")).to_have_text("都存好了（還沒算過）")
                shots = os.environ.get("AOSR_SHOTS")
                if shots:
                    page.screenshot(path=str(Path(shots) / "running-other.png"), full_page=True)
                _assert_quiet(watched)
        finally:
            urllib.request.urlopen(urllib.request.Request(f"{base}/api/runs/{run_id}/stop", data=b"{}", method="POST",
                                                          headers={"Content-Type": "application/json"}))


def test_the_two_big_home_cards_are_gone_and_the_ways_on_are_still_there(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base:
        _three(tmp_path, base)
        with _open(browser, base, viewport_width=1440) as watched:
            page = watched.page
            _opened(page, LATEST)
            for gone in ("#home", "#home-title", "#home-facts", "#home-picture", "#home-room", "#home-result",
                         "#home-materials", "#home-layout"):
                expect(page.locator(gone)).to_have_count(0)
            layout = page.locator('.input-nav a[href="/search-start"]')
            expect(layout).to_be_visible()
            expect(layout).to_have_text("找喇叭和座位的位置")
            expect(page.locator("#nav-zones")).to_be_visible()
            open_results(page)
            expect(page.locator("#results-title")).to_have_text("算過的結果與比較")
            page.locator("#results-close").click()
            layout.click()
            page.wait_for_url(f"{base}/search-start")
            _assert_quiet(watched)


def test_the_scheme_menu_still_says_what_each_scheme_has(tmp_path: Path, browser: Browser) -> None:
    """/api/home 只剩選單那一句白話在用（拿掉大卡後）：每一列底下照樣寫那一份貼了什麼。"""
    with _serve(tmp_path) as base:
        _three(tmp_path, base)
        with _open(browser, base, viewport_width=1440) as watched:
            page = watched.page
            _opened(page, LATEST)
            page.wait_for_load_state("networkidle")
            page.locator("#scheme-menu-button").click()
            expect(page.locator(f'#home-schemes [data-scheme="{LATEST}"]')).to_contain_text("沒貼吸音材料")
            _assert_quiet(watched)


def _facing_y0(scheme: dict[str, object]) -> dict[str, object]:
    """同一間參考房，兩支喇叭改靠 y 起點牆、主位面向它（面向 −y 時左手邊是 +x，左喇叭放 x 大的那邊）。"""
    speakers = cast(dict[str, dict[str, float]], scheme["speakers"])
    speakers["left"] |= {"x": 3.6, "y": 0.8}
    speakers["right"] |= {"x": 2.4, "y": 0.8}
    spots = {"main": (3.0, 2.6), "front": (3.0, 2.5), "back": (3.0, 2.7), "left": (3.1, 2.6), "right": (2.9, 2.6),
             "up": (3.0, 2.6), "down": (3.0, 2.6)}
    for point in cast(dict[str, list[dict[str, object]]], scheme["receiver_set"])["points"]:
        position = cast(list[float], point["position_m"])
        point["position_m"] = [*spots[cast(str, point["receiver_id"])], position[2]]
    return scheme


def _pressed(page: Page) -> list[str]:
    return cast(list[str], page.locator("#elevation-walls button[aria-pressed='true']").evaluate_all(
        "buttons => buttons.map(button => button.id.replace('elevation-wall-', ''))"))


@pytest.mark.parametrize("front", ["x0", "y0"])
def test_opening_a_scheme_shows_the_front_wall_first(tmp_path: Path, browser: Browser, front: str) -> None:
    """喇叭靠 x 起點牆、靠 y 起點牆各一次：打開時第 2 段選的是前牆那一面。"""
    shape = front
    with _serve(tmp_path) as base:
        scheme = _template(base)
        _write(tmp_path, LATEST, scheme if shape == "x0" else _facing_y0(scheme), STAMPS[LATEST])
        other = _template(base)
        _write(tmp_path, OLDEST, other if shape == "y0" else _facing_y0(other), STAMPS[OLDEST])
        with _open(browser, base, viewport_width=1440) as watched:
            page = watched.page
            _opened(page, LATEST)
            assert _pressed(page) == [front]
            expect(page.locator(f"#elevation-wall-{front} .wall-name")).to_have_text("前牆")
            expect(page.locator("#elevation-svg")).to_have_attribute("data-wall", front)
            expect(page.locator("#elevation-size b")).to_have_text("前牆")
            # 換一面看，再打開另一份（前牆在另一面）：又是那一份的前牆。
            page.locator("#elevation-wall-ceiling").click()
            open_scheme(page, OLDEST)
            _opened(page, OLDEST)
            turned = "y0" if front == "x0" else "x0"
            assert _pressed(page) == [turned]
            expect(page.locator(f"#elevation-wall-{turned} .wall-name")).to_have_text("前牆")
            _assert_quiet(watched)


def _no_front(scheme: dict[str, object]) -> dict[str, object]:
    """兩支喇叭的中點離 x 起點牆、y 起點牆、y 終點牆一樣遠（都 2 公尺）：最近的牆不只一面，分不出前後，牆名照座標名。"""
    speakers = cast(dict[str, dict[str, float]], scheme["speakers"])
    speakers["left"] |= {"x": 2.0, "y": 1.4}
    speakers["right"] |= {"x": 2.0, "y": 2.6}
    return scheme


def test_without_a_front_wall_nothing_is_picked_and_a_picked_wall_stays(tmp_path: Path, browser: Browser) -> None:
    """分不出前後（牆名是座標名）：開頁那一份不選任何一面（照舊寫「選一面牆打開立面圖。」）；從一份有前牆的換過來，
    留著原本選的那一面（打開時不動它）。"""
    with _serve(tmp_path) as base:
        _write(tmp_path, LATEST, _no_front(_template(base)), STAMPS[LATEST])
        _write(tmp_path, OLDEST, _template(base), STAMPS[OLDEST])
        with _open(browser, base, viewport_width=1440) as watched:
            page = watched.page
            _opened(page, LATEST)
            assert _pressed(page) == []
            expect(page.locator("#elevation-size")).to_have_text("選一面牆打開立面圖。")
            expect(page.locator("#elevation-wall-x0 .wall-name")).to_have_text("x 起點牆")
            open_scheme(page, OLDEST)
            _opened(page, OLDEST)
            assert _pressed(page) == ["x0"]
            page.locator("#elevation-wall-yL").click()
            open_scheme(page, LATEST)
            _opened(page, LATEST)
            assert _pressed(page) == ["yL"]
            _assert_quiet(watched)


def test_an_empty_screen_with_results_opens_the_drawer_from_the_card(tmp_path: Path, browser: Browser,
                                                                    result: SchemeResult) -> None:
    for run_id, name in (("4" * 32, "gone-one"), ("5" * 32, "gone-two")):
        _bundle(tmp_path, result.model_copy(update={"scheme": result.scheme.model_copy(update={"scheme_id": name})}), run_id)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        page.wait_for_function("document.body.dataset.inputStart === 'empty'")
        page.wait_for_load_state("networkidle")
        link = page.locator("#empty-results")
        expect(link).to_be_visible()
        expect(link).to_have_text("看算過的結果（2 筆）")
        shots = os.environ.get("AOSR_SHOTS")
        if shots:
            page.screenshot(path=str(Path(shots) / "empty-with-results.png"))
        link.click()
        expect(page.locator("#results-panel")).to_be_visible()
        rows = page.locator("#results-list tr")
        expect(rows).to_have_count(2)
        expect(page.locator("#results-list")).to_contain_text("gone-one")
        expect(page.locator("#results-list")).to_contain_text("gone-two")
        page.locator("#results-close").click()
        expect(page.locator("#results-panel")).to_be_hidden()
        expect(link).to_be_focused()
        # 結果都封存掉：卡片那一句收起來，抽屜裡「還沒有算過的結果」那句不指到看不見的頁底按鈕。
        link.click()
        for run_id in ("4" * 32, "5" * 32):
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator(f'#results-list tr[data-run-id="{run_id}"]').get_by_role("button", name="封存").click()
            expect(page.locator(f'#results-list tr[data-run-id="{run_id}"]')).to_have_count(0)
        empty = page.locator("#results-empty")
        expect(empty).to_be_visible()
        expect(empty).to_have_text("還沒有算過的結果；先從範本開一份方案，算好就列在這裡。")
        page.locator("#results-close").click()
        expect(link).to_be_hidden()
        assert page.evaluate("document.body.dataset.inputReady") is None
        assert watched.page_errors == []


def test_the_whole_page_with_a_zoned_scheme_fits_in_two_screens(tmp_path: Path, browser: Browser) -> None:
    """整頁高度（老闆 11-24 定的驗收：整頁不超過 2 屏）：1440×900，開一份切過區的方案、前牆選中、摺疊都收著。"""
    from tests.engine._zone_naming_room import ZONE_NAMING_SCHEME, write_zone_naming_library
    write_zone_naming_library(tmp_path)
    _write(tmp_path, "zone-naming-sample", ZONE_NAMING_SCHEME)
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.set_viewport_size({"width": 1440, "height": 900})
        page.goto(base, wait_until="networkidle")
        _opened(page, "zone-naming-sample")
        page.wait_for_load_state("networkidle")
        expect(page.locator("#elevation-wall-yL")).to_have_attribute("aria-pressed", "true")
        assert page.evaluate("[...document.querySelectorAll('details')].filter(node => node.open).length") == 0
        height = page.evaluate("document.documentElement.scrollHeight")
        assert height <= 2 * 900, f"整頁 {height} 像素，{height / 900:.2f} 屏"
        _assert_quiet(watched)
