"""#1265 單元 5c 第二趟：開頁那一串的四個錯與空畫面封存入口（審查第 1～4 條、設計洞 3～5）。

- 門還關著、正在試下一份時，「打開」旁邊不寫字、也不出現「保留原方案」（開頁時沒有原方案）；門的樣式連那一行一起藏。
- 開頁途中晚回來的平面圖（開機範本那一份）整筆丟掉：不畫、不改訊息、不清「打開」旁邊那一句。
- ``inputReady``（頁面載完的旗子）設下那一刻，「打開」旁邊那一句已經寫好、該開的「這間房貼什麼」已經開了。
- 空畫面搬回只有結果、沒有方案檔的包：不去打開那份方案，講人話、不露伺服器的檔案路徑，「看算過的結果」跟著更新。
- 空畫面只有舊的單筆封存結果時，卡片上照樣有一句打得開抽屜。
- 第三趟：存檔的圖面檢查不被序號丟掉（在存檔考卷檔）；空畫面搬回的回覆缺欄照樣打開；搬回後清單重讀失敗講得出來。
方案的名字、喇叭擺哪都是考卷自己編的；只攔網路，伺服器是考卷自己起的。"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Page, Route, expect

from aosr.reporting.result import SchemeResult
from tests.engine.test_gui_archive import _bundle
from tests.engine.test_gui_browser import _open, _serve, browser as browser, result as result
from tests.engine.test_gui_input_open_last_browser import LATEST, MIDDLE, OLDEST, STAMPS, _opened, _template, _three, _write

# 開頁時「打開」旁邊、頂條存檔那一句、訊息列、抽屜上方那一句：開頁那一串寫字只會寫到這幾處。
_NOTES = ("open-note", "save-feedback", "messages", "results-note", "new-scheme-note", "empty-list-note")
_GATE = """() => {
  const shown = id => getComputedStyle(document.getElementById(id)).display !== 'none';
  return {inputStart: document.body.dataset.inputStart ?? null, inputReady: document.body.dataset.inputReady ?? null,
          openNote: document.getElementById('open-note').textContent,
          openNoteShown: shown('open-note'), saveFeedbackShown: shown('save-feedback'),
          kept: ['open-note', 'save-feedback', 'messages'].some(id => document.getElementById(id).textContent.includes('保留原方案'))};
}"""


def test_trying_the_next_scheme_behind_the_gate_writes_nothing_beside_open(tmp_path: Path, browser: Browser) -> None:
    """審查第 1 條：最近那一份打不開、正在讀下一份（門還是 loading）：「打開」旁邊是空的、門的樣式把它跟頂條存檔那一句藏著，
    訊息列也不寫「保留原方案」；開好了才寫一句「沒有打開…；先開了…」。"""
    with _serve(tmp_path) as base:
        _three(tmp_path, base, broken=(LATEST,))
        with _open(browser, "about:blank", viewport_width=1440) as watched:
            page = watched.page
            held: list[Route] = []
            page.route(f"{base}/api/schemes/{MIDDLE}",
                       lambda route: held.append(route) if route.request.method == "GET" else route.continue_())
            page.goto(base, wait_until="commit")
            deadline = time.monotonic() + 30
            while not held and time.monotonic() < deadline:
                page.wait_for_timeout(50)
            assert held, "讀第二份的請求沒有出去"
            state = page.evaluate(_GATE)
            held[0].continue_()
            assert state["inputStart"] == "loading" and state["inputReady"] is None, state
            assert state["openNote"] == "", state
            assert not state["kept"], state
            # 兩層分開考：程式不寫（上面），門的樣式也藏（這裡；空的那一格高度本來就是 0，量高度考不出來）。
            assert not state["openNoteShown"] and not state["saveFeedbackShown"], state
            _opened(page, MIDDLE)
            page.wait_for_load_state("networkidle")
            note = page.locator("#open-note")
            expect(note).to_be_visible()
            text = note.inner_text()
            assert text.startswith(f"沒有打開「{LATEST}」：") and f"。先開了「{MIDDLE}」" in text, text
            assert "保留原方案" not in text and "方案與預覽換不過去" not in text, text
            assert watched.page_errors == []


def test_an_empty_screen_link_to_a_missing_scheme_is_said_on_the_card(tmp_path: Path, browser: Browser) -> None:
    """設計洞 5：一份方案都沒有、網址帶了不在的 ?scheme=：那一句寫在空畫面卡片上；頂條「打開」旁邊不寫、也藏著
    （「換一個方案」那顆在空畫面看不到，不准指到它）。"""
    with _serve(tmp_path) as base, _open(browser, f"{base}/?scheme=ghost-room", viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputStart === 'empty'")
        page.wait_for_load_state("networkidle")
        card = page.locator("#empty-list-note")
        expect(card).to_be_visible()
        expect(card).to_have_text("沒有打開「ghost-room」：已存的方案裡沒有這一份（可能封存了或改了名字）。")
        state = page.evaluate(_GATE)
        assert state["openNote"] == "" and not state["openNoteShown"], state
        assert state["inputReady"] is None, state
        assert watched.page_errors == []


def test_a_late_plan_for_the_startup_template_is_dropped(tmp_path: Path, browser: Browser) -> None:
    """審查第 2 條：開頁途中頁面程式改了開機範本（家具），那一次的平面圖晚到、已經開好要開的那一份：整筆丟掉——
    畫的照樣是開好的那一份（長 7.2 公尺、沒有家具），「打開」旁邊那一句（網址那一份不在）照留。"""
    with _serve(tmp_path) as base:
        doc = _template(base)
        cast(dict[str, dict[str, float]], doc["scene"])["room_m"]["Lx"] = 7.2
        _write(tmp_path, "target-room", doc, STAMPS[LATEST])
        with _open(browser, "about:blank", viewport_width=1440) as watched:
            page = watched.page
            held_results: list[Route] = []
            held_edit: list[Route] = []
            held_template_plan: list[Route] = []
            plan_calls: list[float] = []

            def on_plan(route: Route) -> None:
                sent = cast(dict[str, dict[str, dict[str, float]]], route.request.post_data_json)
                lx = sent["scene"]["room_m"]["Lx"]
                plan_calls.append(lx)
                if lx != 7.2 and len(plan_calls) >= 2:
                    held_template_plan.append(route)
                else:
                    route.continue_()

            page.route("**/api/results", lambda route: held_results.append(route))
            page.route("**/api/input-edit", lambda route: held_edit.append(route))
            page.route("**/api/plan", on_plan)
            page.goto(f"{base}/?scheme=ghost-room", wait_until="commit")
            page.wait_for_function("document.querySelectorAll('#home-schemes .home-open').length > 0")
            page.locator("#furniture-sofa").evaluate("n => { n.checked = true; n.dispatchEvent(new Event('change')); }")
            page.wait_for_function("inputBusy")
            deadline = time.monotonic() + 15
            while (not held_results or not held_edit) and time.monotonic() < deadline:
                page.wait_for_timeout(20)
            assert held_results and held_edit, (held_results, held_edit)
            held_results[0].continue_()
            page.wait_for_timeout(150)
            held_edit[0].continue_()
            _opened(page, "target-room")
            deadline = time.monotonic() + 15
            while not held_template_plan and time.monotonic() < deadline:
                page.wait_for_timeout(20)
            assert held_template_plan, f"沒攔到開機範本那一次的平面圖：{plan_calls}"
            before = page.evaluate("[document.getElementById('messages').textContent, document.getElementById('plan-xy').innerHTML]")
            held_template_plan[0].continue_()
            page.wait_for_load_state("networkidle")
            page.wait_for_timeout(400)
            after = page.evaluate("""() => ({
                openedId, roomLx: document.getElementById('room-Lx').value,
                shownPlanLx: shownPlan?.room?.Lx ?? null, shownPlanFurniture: (shownPlan?.furniture ?? []).length,
                openNote: document.getElementById('open-note').textContent,
                messages: document.getElementById('messages').textContent,
                planXY: document.getElementById('plan-xy').innerHTML})""")
            assert after["openedId"] == "target-room" and after["roomLx"] == "7.2", after
            assert after["shownPlanLx"] == 7.2 and after["shownPlanFurniture"] == 0, after
            assert after["openNote"].startswith("沒有打開「ghost-room」："), after
            # 不畫、不改訊息列：丟掉之前與之後一字不差。
            assert [after["messages"], after["planXY"]] == before
            assert watched.page_errors == []


# 第一次 input-ready 那一刻：「打開」旁邊寫了什麼、「這間房貼什麼」那一頁開了沒、門開了沒（開頁前就裝上）；之後再發只數次數。
_WATCH_READY = """document.addEventListener('input-ready', () => {
  window.readyCount = (window.readyCount ?? 0) + 1;
  if (window.readyAt !== undefined) return;
  const zones = document.getElementById('zone-page');
  window.readyAt = {openNote: document.getElementById('open-note').textContent,
                    zonePage: !zones.hidden && getComputedStyle(zones).display !== 'none',
                    inputStart: document.body.dataset.inputStart ?? null, openedId};
});"""


@pytest.mark.parametrize("link", ["scheme=ghost-room", f"zones={OLDEST}"])
def test_input_ready_is_set_after_the_note_and_the_zone_page(tmp_path: Path, browser: Browser, link: str) -> None:
    """設計洞 4：inputReady 是「全部就緒」，不是「門開了」——設下那一刻「打開」旁邊那一句已經寫好、?zones= 要開的那一頁已經開了。"""
    with _serve(tmp_path) as base:
        _three(tmp_path, base)
        with _open(browser, "about:blank", viewport_width=1440) as watched:
            page = watched.page
            page.add_init_script(_WATCH_READY)
            page.goto(f"{base}/?{link}")
            page.wait_for_function("window.readyAt !== undefined")
            page.wait_for_load_state("networkidle")
            ready = page.evaluate("window.readyAt")
            assert page.evaluate("window.readyCount") == 1
            assert ready["inputStart"] is None, ready
            if link.startswith("scheme="):
                assert ready["openedId"] == LATEST, ready
                assert ready["openNote"].startswith("沒有打開「ghost-room」：已存的方案裡沒有這一份"), ready
                assert f"先開了「{LATEST}」" in ready["openNote"], ready
                assert not ready["zonePage"], ready
            else:
                assert ready["openedId"] == OLDEST and ready["openNote"] == "", ready
                assert ready["zonePage"], ready
            assert watched.page_errors == []


def _all_notes(page_text: dict[str, str]) -> str:
    return "\n".join(page_text.values())


def test_moving_back_a_package_with_only_results_on_the_empty_screen(tmp_path: Path, browser: Browser,
                                                                     result: SchemeResult) -> None:
    """審查第 4 條：空畫面只有結果 → 從「看算過的結果」封存 → 從「看已封存的方案」搬回：那一包沒有方案檔，不去打開它；
    卡片上講人話（結果搬回了、在哪裡看），「看算過的結果（N 筆）」跟著更新，哪一句都不露伺服器上的檔案路徑。"""
    run_id = "4" * 32
    _bundle(tmp_path, result.model_copy(update={"scheme": result.scheme.model_copy(update={"scheme_id": "gone-one"})}), run_id)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        asked: list[str] = []
        page.on("request", lambda request: asked.append(request.url) if "/api/schemes/gone-one" in request.url else None)
        page.wait_for_function("document.body.dataset.inputStart === 'empty'")
        page.wait_for_load_state("networkidle")
        page.locator("#empty-results").click()
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator(f'#results-list tr[data-run-id="{run_id}"]').get_by_role("button", name="封存").click()
        page.wait_for_function("document.querySelectorAll('#archived-packages-list tr').length === 1")
        expect(page.locator("#empty-results")).to_be_hidden()
        asked.clear()
        page.locator("#archived summary").click()
        page.locator("#archived-packages-list tr button", has_text="搬回").first.click()
        page.wait_for_function("document.getElementById('results-drawer').hidden")
        page.wait_for_load_state("networkidle")
        notes = cast(dict[str, str], page.evaluate(
            "ids => Object.fromEntries(ids.map(id => [id, document.getElementById(id).textContent]))", list(_NOTES)))
        assert page.evaluate("document.body.dataset.inputStart") == "empty"
        assert page.evaluate("document.body.dataset.inputReady") is None
        assert asked == [], asked
        assert notes["new-scheme-note"] == "「gone-one」的 1 筆結果已經搬回；這一包裡沒有方案檔，沒有打開方案。結果在下面「看算過的結果」裡", notes
        expect(page.locator("#new-scheme-note")).to_be_visible()
        link = page.locator("#empty-results")
        expect(link).to_be_visible()
        expect(link).to_have_text("看算過的結果（1 筆）")
        everything = _all_notes(notes)
        assert "保留原方案" not in everything, notes
        for path in {str(tmp_path), str(tmp_path.resolve())}:
            assert path not in everything, notes
        assert "No such file" not in everything and "Errno" not in everything, notes
        assert not page.evaluate(_GATE)["openNoteShown"]
        assert watched.page_errors == []


def test_an_empty_screen_with_only_single_archived_results_opens_the_drawer(tmp_path: Path, browser: Browser,
                                                                            result: SchemeResult) -> None:
    """設計洞 3：一份方案、一包封存、一筆沒封存的結果都沒有，只有舊的單筆封存結果（archive/results/）：卡片上照樣有一句打開抽屜，
    抽屜裡看得到那一筆、搬得回來；搬回之後那一句換成「看算過的結果（1 筆）」。"""
    run_id = "6" * 32
    _bundle(tmp_path / "archive", result.model_copy(update={"scheme": result.scheme.model_copy(update={"scheme_id": "old-one"})}), run_id)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.wait_for_function("document.body.dataset.inputStart === 'empty'")
        page.wait_for_load_state("networkidle")
        expect(page.locator("#empty-results")).to_be_hidden()
        link = page.locator("#empty-archived")
        expect(link).to_be_visible()
        expect(link).to_have_text("看已封存的結果（1 筆）")
        link.click()
        expect(page.locator("#results-panel")).to_be_visible()
        expect(page.locator("#archived")).to_have_attribute("open", "")
        row = page.locator("#archived-list tr")
        expect(row).to_have_count(1)
        expect(row).to_contain_text("old-one")
        row.get_by_role("button", name="搬回").click()
        expect(page.locator("#archived-list tr")).to_have_count(0)
        page.locator("#results-close").click()
        expect(link).to_be_hidden()
        expect(page.locator("#empty-results")).to_have_text("看算過的結果（1 筆）")
        expect(page.locator("#empty-results")).to_be_focused()
        assert page.evaluate("document.body.dataset.inputReady") is None
        assert watched.page_errors == []


# ---- #1265 單元 5c 第三趟：空畫面搬回的兩個出口（複查第 2、3 條） ----

def _archive_by_api(base: str, name: str) -> None:
    request = urllib.request.Request(f"{base}/api/schemes/{name}/archive", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
    urllib.request.urlopen(request).read()


def _move_back_on_the_empty_screen(page: Page, fail: dict[str, bool]) -> dict[str, str]:
    """空畫面 → 「看已封存的方案」→ 搬回第一包；``fail["on"]`` 在按「搬回」前一刻打開（之後的清單請求回 500）。
    等抽屜關上、網路靜下來才讀卡片、訊息列與門。"""
    page.wait_for_function("document.body.dataset.inputStart === 'empty'")
    page.wait_for_load_state("networkidle")
    page.locator("#empty-archived").click()
    fail["on"] = True
    page.locator("#archived-packages-list tr button", has_text="搬回").first.click()
    page.wait_for_function("document.getElementById('results-drawer').hidden && !inputBusy")
    page.wait_for_load_state("networkidle")
    return cast(dict[str, str], page.evaluate("""() => ({
        inputStart: document.body.dataset.inputStart ?? null, openedId: openedId ?? null,
        note: document.getElementById('new-scheme-note').textContent,
        noteClass: document.getElementById('new-scheme-note').className,
        messages: document.getElementById('messages').textContent,
        messagesClass: document.getElementById('messages').className,
        emptyResultsHidden: document.getElementById('empty-results').hidden})"""))


def _fail_with(page: Page, pattern: str) -> dict[str, bool]:
    fail = {"on": False}
    page.route(pattern, lambda route: route.fulfill(status=500, content_type="application/json",
               body=json.dumps({"error": "清單壞了"})) if fail["on"] else route.continue_())
    return fail


def test_moving_back_on_the_empty_screen_opens_the_scheme_when_the_reply_has_no_paths(tmp_path: Path, browser: Browser) -> None:
    """複查第 2 條：搬回的回覆少了 ``paths``（成員路徑）與 ``result_ids``（結果代號）兩欄：當成包裡有方案檔，照樣打開它、離開空畫面，
    卡片上不留「沒有打開：」斷尾，頁面程式不丟錯。"""
    with _serve(tmp_path) as base:
        _write(tmp_path, "has-scheme", _template(base), STAMPS[LATEST])
        _archive_by_api(base, "has-scheme")
        with _open(browser, "about:blank", viewport_width=1440) as watched:
            page = watched.page

            def on_restore(route: Route) -> None:
                reply = route.fetch()
                body = json.loads(reply.text())
                body.pop("paths", None)
                body.pop("result_ids", None)
                route.fulfill(status=reply.status, content_type="application/json", body=json.dumps(body))

            page.route("**/api/archive/packages/*/restore", on_restore)
            page.goto(base)
            state = _move_back_on_the_empty_screen(page, {"on": False})
            assert state["openedId"] == "has-scheme" and state["inputStart"] is None, state
            assert "沒有打開" not in state["note"], state
            assert watched.page_errors == []


def test_moving_back_only_results_on_the_empty_screen_says_when_the_lists_cannot_be_reread(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """複查第 3 條：空畫面搬回只有結果的包、之後重讀結果清單回 500：卡片上用紅字講結果搬回了、清單更新不了、請重新整理，
    帶上真正的原因；不再用綠字指去看不到的「看算過的結果」。"""
    _bundle(tmp_path, result.model_copy(update={"scheme": result.scheme.model_copy(update={"scheme_id": "res-only"})}), "2" * 32)
    with _serve(tmp_path) as base:
        _archive_by_api(base, "res-only")
        with _open(browser, "about:blank", viewport_width=1440) as watched:
            page = watched.page
            fail = _fail_with(page, "**/api/results")
            page.goto(base)
            state = _move_back_on_the_empty_screen(page, fail)
            assert state["inputStart"] == "empty", state
            assert state["noteClass"] == "notice", state
            assert "結果已經搬回，但清單更新不了，請重新整理這一頁：" in state["note"] and "清單壞了" in state["note"], state
            assert "看算過的結果" not in state["note"], state
            assert watched.page_errors == []


def test_moving_back_a_scheme_on_the_empty_screen_says_when_the_lists_cannot_be_reread(tmp_path: Path, browser: Browser) -> None:
    """複查第 3 條（有方案檔那一條出口）：空畫面搬回有方案檔的包、重讀結果清單回 500：方案照樣打開（伺服器上那一份是好的），
    開好之後訊息列用紅字講清單更新不了、請重新整理、真正的原因——不被打開時那一次圖面檢查的「檢查通過」蓋掉。"""
    with _serve(tmp_path) as base:
        _write(tmp_path, "has-scheme", _template(base), STAMPS[LATEST])
        _archive_by_api(base, "has-scheme")
        with _open(browser, "about:blank", viewport_width=1440) as watched:
            page = watched.page
            fail = _fail_with(page, "**/api/results")
            page.goto(base)
            state = _move_back_on_the_empty_screen(page, fail)
            assert state["openedId"] == "has-scheme" and state["inputStart"] is None, state
            assert state["messagesClass"] == "notice", state
            assert "已經搬回，但清單更新不了，請重新整理這一頁：" in state["messages"] and "清單壞了" in state["messages"], state
            assert "沒有打開" not in state["note"], state
            assert watched.page_errors == []
