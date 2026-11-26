"""#1265 單元 5b 第二趟：「從範本開一份新的」與空畫面，審查（第二趟派工單附的那一份）五錯各一題。

1. 有沒存的改動時開新的：「離開前詢問」（#leave-edits）的點擊、鎖頁中的 Esc 都不准把那一塊關掉，失敗的原因看得到。
2. 從空畫面開新的、從封存區搬回、開頁判定完：第 1 段剛看得到時平面圖照同一份重畫，字不疊在收音點上。
3. 空畫面刪光封存再關抽屜：游標落在卡片裡選中的範本，不掉到頁面底層（body）。
4. 存檔、重讀清單、打開一路鎖著：第二下「開一份新的」與存、另存、計算、打開都照鎖頁那樣擋；沒打開時寫真正的原因。
5. 開頁判定完成之前（還不知道有沒有存過的方案）：第 1、2 段、頁底、頂條按鈕與「這間房貼什麼」都不出現，只有「讀取中…」。
方案名字與內容都是考卷自己編的或範本本身；只攔網路，伺服器是考卷自己起的。"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Error, Page, Route, expect

from tests.engine._input_page_steps import fill_new_scheme, new_from_template, open_example_scheme
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser as browser

UNLOCKED = "!document.querySelector('main').inert && !inputBusy"
BUSY = "上一個動作還沒做完"
# 「聆聽區側面放大」（#zoom-xz）上每一個字框壓到哪幾個不是它自己的圓點（審查第 2 條的量法）。
OVERLAPS = """() => {
  const svg = document.getElementById('zoom-xz');
  const circles = [...svg.querySelectorAll('circle:not(.changed-ring)')];
  const hits = [];
  for (const g of svg.querySelectorAll('g[data-keys]')) {
    const text = g.querySelector('text'), own = g.querySelector('circle:not(.changed-ring)');
    if (!text) continue;
    const box = text.getBBox();
    for (const c of circles) {
      if (c === own) continue;
      const x = Number(c.getAttribute('cx')), y = Number(c.getAttribute('cy')), r = Number(c.getAttribute('r'));
      if (box.x < x + r && x - r < box.x + box.width && box.y < y + r && y - r < box.y + box.height)
        hits.push([text.textContent, c.parentElement.dataset.keys]);
    }
  }
  return hits;
}"""


def _folder(data_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted((data_dir / "schemes").glob("*.json"))}


def _empty_start(page: Page) -> None:
    page.wait_for_function("document.body.dataset.inputStart === 'empty'")


def _zoom_overlaps(page: Page) -> list[list[str]]:
    page.locator("#view-zoom-xz").click()
    return cast(list[list[str]], page.evaluate(OVERLAPS))


def _store(base: str, data_dir: Path, name: str) -> None:
    scheme = json.loads(urllib.request.urlopen(f"{base}/api/example").read())["scheme"] | {"scheme_id": name}
    (data_dir / "schemes").mkdir(exist_ok=True)
    (data_dir / "schemes" / f"{name}.json").write_text(json.dumps(scheme), encoding="utf-8")


def _archive(base: str, name: str) -> None:
    request = urllib.request.Request(f"{base}/api/schemes/{name}/archive", data=b"{}", method="POST",
                                     headers={"Content-Type": "application/json"})
    assert json.loads(urllib.request.urlopen(request).read())["message"]


def test_the_box_stays_open_through_the_leave_question_and_says_why_nothing_opened(
        tmp_path: Path, browser: Browser) -> None:
    """審查第 1 條：開著那一份有沒存的改動，按「開一份新的」→ #leave-edits 按「不存走」→ 存檔在路上時按 Esc：
    那一塊一直開著；伺服器拒收（422）之後原因寫在那一塊、看得到，游標在名字格，表單與資料夾都沒變。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        folder = _folder(tmp_path)
        page.locator("#room-Lx").fill("5.5")
        held: list[Route] = []
        page.route("**/api/schemes/fresh", lambda route: held.append(route) if route.request.method == "PUT" else route.continue_())
        fill_new_scheme(page, "fresh", "stage_one")
        page.locator("#new-scheme-go").click()
        expect(page.locator("#leave-edits")).to_be_visible()
        with page.expect_request(lambda request: request.method == "PUT"):
            page.locator("#leave-edits").get_by_role("button", name="不存走", exact=True).click()
        expect(page.locator("#template-box")).to_be_visible()
        assert page.evaluate("document.querySelector('main').inert") is True
        page.keyboard.press("Escape")
        expect(page.locator("#template-box")).to_be_visible()
        held[0].fulfill(status=422, json={"error": "伺服器不收這一份"})
        page.wait_for_function(UNLOCKED)
        expect(page.locator("#template-box")).to_be_visible()
        expect(page.locator("#new-scheme-note")).to_be_visible()
        expect(page.locator("#new-scheme-note")).to_have_text("沒有開新的：伺服器不收這一份")
        expect(page.locator("#new-scheme-id")).to_be_focused()
        assert page.locator("#room-Lx").input_value() == "5.5" and page.evaluate("openedId") == "example"
        assert _folder(tmp_path) == folder
        statuses = [text.split("status of ")[1][:3] for text in watched.console_errors]
        assert statuses == ["422"], watched.console_errors
        assert watched.page_errors == []


def test_the_plan_is_redrawn_once_section_one_shows_so_no_label_sits_on_a_receiver(
        tmp_path: Path, browser: Browser) -> None:
    """審查第 2 條：空畫面開新的、空畫面從封存區搬回、有存過的方案時開頁，三種都是第 1 段藏著時先畫了圖；
    第 1 段看得到之後重畫，「聆聽區側面放大」上沒有字壓在收音點上（跟用 ?scheme= 正常打開同一份一樣）。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        new_from_template(page, "from-empty", "reference")
        assert _zoom_overlaps(page) == []
        page.goto(f"{base}/?scheme=from-empty", wait_until="networkidle")
        page.wait_for_function(f"openedId === 'from-empty' && {UNLOCKED}")
        assert _zoom_overlaps(page) == []
        # 有存過的方案、網址沒帶名字：開頁判定完才把第 1 段攤出來。
        page.goto(base, wait_until="networkidle")
        page.wait_for_function(f"document.body.dataset.inputReady === 'true' && {UNLOCKED}")
        assert _zoom_overlaps(page) == []
        _assert_quiet(watched)
    with _serve(tmp_path) as base:
        _archive(base, "from-empty")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        page.locator("#empty-archived").click()
        page.locator('#archived-packages-list tr[data-scheme-id="from-empty"]').get_by_role("button", name="搬回").click()
        page.wait_for_function(f"openedId === 'from-empty' && document.body.dataset.inputReady === 'true' && {UNLOCKED}")
        assert _zoom_overlaps(page) == []
        _assert_quiet(watched)


def test_closing_the_drawer_after_the_last_archive_is_gone_puts_the_cursor_on_the_chosen_template(
        tmp_path: Path, browser: Browser) -> None:
    """審查第 3 條：空畫面打開封存抽屜、永久刪掉最後一包、關抽屜：「看已封存的方案」那一句已經藏起來，游標落在
    卡片裡選中的範本單選；再按 Tab 走到卡片下一格（新方案的名字），不掉到頁面底層。"""
    with _serve(tmp_path) as base:
        _store(base, tmp_path, "kept")
        _archive(base, "kept")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        page.locator("#empty-archived").click()
        page.once("dialog", lambda dialog: dialog.accept())
        page.locator('#archived-packages-list tr[data-scheme-id="kept"]').get_by_role("button", name="永久刪除").click()
        expect(page.locator("#empty-archived")).to_be_hidden()
        page.locator("#results-close").click()
        expect(page.locator("#results-drawer")).to_be_hidden()
        expect(page.locator("#new-template-reference")).to_be_focused()
        page.keyboard.press("Tab")
        expect(page.locator("#new-scheme-id")).to_be_focused()
        _assert_quiet(watched)


def test_saving_reloading_the_list_and_opening_stay_locked_so_a_second_press_is_refused(
        tmp_path: Path, browser: Browser) -> None:
    """審查第 4 條：空畫面開新的「first-a」，存好之後重讀清單卡住：整頁還鎖著；這時頁面程式按「開一份新的」（改名
    first-b）、「儲存」、「另存新名字」、「計算」、「打開」，各自照鎖頁的說法講、什麼都不送；放行之後只有 first-a 一份檔、
    開的是它。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        held: list[Route] = []
        page.route(f"{base}/api/schemes", lambda route: held.append(route) if route.request.method == "GET" and not held
                   else route.continue_())
        fill_new_scheme(page, "first-a", "reference")
        with page.expect_request(f"{base}/api/schemes"):
            page.locator("#new-scheme-go").click()
        assert page.evaluate("[inputBusy, document.querySelector('main').inert]") == [True, True]
        puts: list[str] = []
        page.on("request", lambda request: puts.append(request.url) if request.method in ("PUT", "POST") and "/api/schemes" in request.url
                or "/api/runs" in request.url else None)
        page.evaluate("document.getElementById('new-scheme-id').value = 'first-b'")
        page.locator("#new-scheme-go").evaluate("node => node.click()")
        expect(page.locator("#new-scheme-note")).to_have_text(f"{BUSY}，沒有開新的；等它做完再按一次「開一份新的」")
        page.locator("#save").evaluate("node => node.click()")
        expect(page.locator("#messages")).to_have_text(f"{BUSY}，這一次沒有儲存；等它做完再按一次")
        page.evaluate("document.getElementById('save-as-id').value = 'first-c'")
        page.locator("#save-as").evaluate("node => node.click()")
        expect(page.locator("#save-as-note")).to_have_text(f"沒有另存：{BUSY}，等它做完再按一次「另存新名字」")
        page.locator("#calculate").evaluate("node => node.click()")
        expect(page.locator("#messages")).to_have_text(f"{BUSY}，這一次沒有開算；等它做完再按一次")
        assert page.evaluate("openScheme('first-a')") is False
        expect(page.locator("#open-note")).to_have_text(f"{BUSY}，沒有打開「first-a」；等它做完再按一次「打開」")
        assert page.evaluate("[inputBusy, document.querySelector('main').inert]") == [True, True]
        held[0].continue_()
        page.wait_for_function(f"openedId === 'first-a' && document.body.dataset.inputReady === 'true' && {UNLOCKED}")
        page.wait_for_load_state("networkidle")
        assert set(_folder(tmp_path)) == {"first-a.json"} and puts == [], puts
        expect(page.locator("#save-id")).to_have_value("first-a")
        expect(page.locator("#open-note")).to_have_text("已從範本「參考房」開一份新的「first-a」，存好了")
        _assert_quiet(watched)


def test_a_saved_scheme_that_will_not_open_says_the_real_reason(tmp_path: Path, browser: Browser) -> None:
    """審查第 4 條另一半：存好了、打開時讀那一份回 500：那一塊寫「已經存好，但沒有打開」配伺服器那一句，
    不抄訊息列上別人留的舊句（例如「檢查通過」）；那一塊與空畫面照留，游標在名字格。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        page.evaluate("document.getElementById('messages').textContent = '檢查通過'")
        page.route(f"{base}/api/schemes/second", lambda route: route.fulfill(status=500, json={"error": "讀不出這一份"})
                   if route.request.method == "GET" else route.continue_())
        fill_new_scheme(page, "second", "reference")
        page.locator("#new-scheme-go").click()
        note = page.locator("#new-scheme-note")
        expect(note).to_contain_text("「second」已經存好，但沒有打開：")
        page.wait_for_function(UNLOCKED)
        expect(note).to_contain_text("讀不出這一份")
        expect(note).not_to_contain_text("檢查通過")
        expect(note).to_be_visible()
        expect(page.locator("#new-scheme-id")).to_be_focused()
        assert page.evaluate("document.body.dataset.inputStart") == "empty"
        assert set(_folder(tmp_path)) == {"second.json"}
        assert watched.page_errors == []


def test_before_the_page_knows_whether_schemes_exist_only_a_loading_line_shows(
        tmp_path: Path, browser: Browser) -> None:
    """審查第 5 條：開頁時結果清單卡住（還沒判定有沒有存過的方案）：沒存的開機範本整頁不出現——第 1、2 段、頁底、
    「這間房貼什麼」、儲存／另存／計算都看不到、點不到，只有一句「讀取中…」；判定是空資料夾後進空畫面，「這間房貼什麼」
    那一頁沒被打開、main 沒被藏。開頁讀不到範例時「讀取中…」換成讀不到的原因，不一直轉。"""
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        held: list[Route] = []
        page.route(f"{base}/api/results", lambda route: held.append(route))
        page.goto(base, wait_until="domcontentloaded")
        page.wait_for_function("() => document.getElementById('room-Lx') !== null && typeof scheme === 'object'"
                               " && !document.querySelector('main').inert && document.getElementById('plan-xy').childElementCount > 0")
        for gone in ("#scheme-section", "#wall-section", "#input-footer", "#nav-zones", "#save", "#save-as-toggle",
                     "#calculate", "#show-results", "#scheme-menu-button"):
            expect(page.locator(gone)).to_be_hidden()
        expect(page.locator("#boot-note")).to_be_visible()
        expect(page.locator("#boot-note")).to_have_text("讀取中…")
        try:
            page.locator("#nav-zones").click(timeout=500)
            clicked = True
        except Error:
            clicked = False
        assert clicked is False
        held[0].continue_()
        _empty_start(page)
        assert page.evaluate("[document.getElementById('zone-page').hidden, document.querySelector('main').hidden]") == [True, False]
        expect(page.locator("#boot-note")).to_be_hidden()
        expect(page.locator("#empty-start")).to_be_visible()
        _assert_quiet(watched)
    with _serve(tmp_path) as base, _open(browser, "about:blank", viewport_width=1440) as watched:
        page = watched.page
        page.route(f"{base}/api/example", lambda route: route.fulfill(status=500, json={"error": "範例壞了"}))
        page.goto(base, wait_until="networkidle")
        expect(page.locator("#boot-note")).to_contain_text("範例壞了")
        expect(page.locator("#scheme-section")).to_be_hidden()
        assert watched.page_errors == []


def test_restoring_on_the_empty_screen_stays_locked_until_it_opens_and_says_the_real_reason(
        tmp_path: Path, browser: Browser) -> None:
    """第二趟追加（回報第 6 點第 2 條，跟第 4 條同一套）：空畫面從封存區搬回，搬回、重讀清單、打開一路鎖著。
    搬回的那一份讀檔被攔成 500：卡片那一塊寫「已經搬回，但沒有打開」配伺服器那一句，不抄訊息列的舊句「檢查通過」；
    另一包搬回、重讀清單卡住時，頁面程式按「開一份新的」照鎖頁的說法擋、一個請求都不送，放行後開的是搬回的那一份。"""
    with _serve(tmp_path) as base:
        for name in ("broken", "kept"):
            _store(base, tmp_path, name)
            _archive(base, name)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        page.route(f"{base}/api/schemes/broken", lambda route: route.fulfill(status=500, json={"error": "讀不出這一份"})
                   if route.request.method == "GET" else route.continue_())
        page.evaluate("document.getElementById('messages').textContent = '檢查通過'")
        page.locator("#empty-archived").click()
        page.locator('#archived-packages-list tr[data-scheme-id="broken"]').get_by_role("button", name="搬回").click()
        note = page.locator("#new-scheme-note")
        expect(note).to_contain_text("「broken」已經搬回，但沒有打開：")
        page.wait_for_function(UNLOCKED)
        expect(note).to_contain_text("讀不出這一份")
        expect(note).not_to_contain_text("檢查通過")
        expect(note).to_be_visible()
        assert page.evaluate("document.body.dataset.inputStart") == "empty"
        held: list[Route] = []
        page.route(f"{base}/api/schemes", lambda route: held.append(route) if route.request.method == "GET" and not held
                   else route.continue_())
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url) if request.method == "PUT" or "template=" in request.url else None)
        page.locator("#empty-archived").click()
        with page.expect_request(f"{base}/api/schemes"):
            page.locator('#archived-packages-list tr[data-scheme-id="kept"]').get_by_role("button", name="搬回").click()
        assert page.evaluate("[inputBusy, document.querySelector('main').inert]") == [True, True]
        page.evaluate("document.getElementById('new-scheme-id').value = 'second-try'")
        page.locator("#new-scheme-go").evaluate("node => node.click()")
        expect(note).to_have_text(f"{BUSY}，沒有開新的；等它做完再按一次「開一份新的」")
        held[0].continue_()
        page.wait_for_function(f"openedId === 'kept' && document.body.dataset.inputReady === 'true' && {UNLOCKED}")
        page.wait_for_load_state("networkidle")
        assert sent == [], sent
        assert set(_folder(tmp_path)) == {"broken.json", "kept.json"}
        expect(page.locator("#save-id")).to_have_value("kept")
        # 不變式四（第三趟）：搬回成功之後，鎖著時被擋下的那一句不留在那一塊。
        expect(note).to_have_text("")
        assert watched.page_errors == []


# 第三趟（審查 u5b2 三錯同一個缺口）：「開一份新的」從按下到結束一次只准一個，「存了再走」等清單的那段沒鎖也一樣。
REFUSED_NEW = f"{BUSY}，沒有開新的；等它做完再按一次「開一份新的」"


def _save_then_leave_with_the_list_held(page: Page, base: str, tmp_path: Path, name: str) -> list[Route]:
    """開著的那一份改了房間長度，從選單開新的「name」，詢問裡按「存了再走」：舊方案存好、鎖已解（save 故意的），
    重讀結果清單那一趟被攔住，回傳攔住的那一趟。"""
    open_example_scheme(page, base, tmp_path)
    page.locator("#room-Lx").fill("5.5")
    held: list[Route] = []
    page.route(f"{base}/api/results", lambda route: held.append(route) if not held else route.continue_())
    fill_new_scheme(page, name, "stage_one")
    page.locator("#new-scheme-go").click()
    expect(page.locator("#leave-edits")).to_be_visible()
    with page.expect_request(f"{base}/api/results"):
        page.locator("#leave-edits").get_by_role("button", name="存了再走", exact=True).click()
    page.wait_for_function("!inputBusy")
    return held


def _clickable(page: Page, selector: str) -> bool:
    try:
        page.locator(selector).click(timeout=500)
        return True
    except Error:
        return False


def test_a_second_press_while_the_old_scheme_is_being_saved_is_refused_and_sends_nothing(
        tmp_path: Path, browser: Browser) -> None:
    """不變式一（審查 u5b2 第 1 條甲）：「存了再走」存好舊方案、還在重讀清單（這時整頁沒鎖）：那一塊 inert，真按點不到；
    頁面程式改名 fresh-2 再按「開一份新的」照鎖頁那句擋下、不送請求；程式按「取消」、按 Esc 都不關那一塊。
    放行後只存出 fresh-1、開的是它，那一塊的提示欄是空的。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url) if request.method == "PUT" and "fresh" in request.url else None)
        held = _save_then_leave_with_the_list_held(page, base, tmp_path, "fresh-1")
        assert page.evaluate("[inputBusy, document.getElementById('new-scheme').inert, newSchemePending]") == [False, True, True]
        assert _clickable(page, "#new-scheme-go") is False
        assert _clickable(page, "#new-template-reference") is False
        note = page.locator("#new-scheme-note")
        # 頁面程式寫進名字格、發 input：名字格的即時提示不准在這段換掉提示欄（真打字 inert 擋著）。
        page.evaluate("(() => { const field = document.getElementById('new-scheme-id');"
                      " field.value = '中文'; field.dispatchEvent(new Event('input')); })()")
        expect(note).to_have_text("")
        page.evaluate("document.getElementById('new-scheme-id').value = 'fresh-2'")
        page.locator("#new-scheme-go").evaluate("node => node.click()")
        expect(note).to_have_text(REFUSED_NEW)
        page.locator("#new-scheme-cancel").evaluate("node => node.click()")
        page.keyboard.press("Escape")
        expect(page.locator("#template-box")).to_be_visible()
        held[0].continue_()
        page.wait_for_function(f"openedId === 'fresh-1' && {UNLOCKED}")
        page.wait_for_load_state("networkidle")
        assert sent == [f"{base}/api/schemes/fresh-1"], sent
        assert set(_folder(tmp_path)) == {"example.json", "fresh-1.json"}
        expect(note).to_have_text("")
        expect(page.locator("#template-box")).to_be_hidden()
        _assert_quiet(watched)


def test_cancel_while_the_old_scheme_is_being_saved_does_not_hide_the_reason(tmp_path: Path, browser: Browser) -> None:
    """不變式一、三（審查 u5b2 第 1 條乙）：同一段空窗頁面程式按「取消」不起作用；接著真按「換一個方案」把那一塊收起來；
    放行後新的那一份伺服器不收（422）：那一塊打開回來、原因寫在那一塊裡、看得到，游標回名字格。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        page.route(f"{base}/api/schemes/fresh-c", lambda route: route.fulfill(status=422, json={"error": "伺服器不收這一份"})
                   if route.request.method == "PUT" else route.continue_())
        held = _save_then_leave_with_the_list_held(page, base, tmp_path, "fresh-c")
        page.locator("#new-scheme-cancel").evaluate("node => node.click()")
        expect(page.locator("#template-box")).to_be_visible()
        # 這段沒鎖：真按「換一個方案」把那一塊收起來、打開選單；沒開成的時候那一塊要打開回來、選單收掉。
        page.locator("#scheme-menu-button").click()
        expect(page.locator("#template-box")).to_be_hidden()
        held[0].continue_()
        note = page.locator("#new-scheme-note")
        expect(note).to_have_text("沒有開新的：伺服器不收這一份")
        expect(page.locator("#scheme-menu")).to_be_hidden()
        page.wait_for_function(UNLOCKED)
        expect(page.locator("#template-box")).to_be_visible()
        expect(note).to_be_visible()
        expect(page.locator("#new-scheme-id")).to_be_focused()
        assert page.evaluate("openedId") == "example"
        assert watched.page_errors == []


def test_a_failed_save_in_the_leave_question_says_so_in_the_box_and_cancel_says_nothing(
        tmp_path: Path, browser: Browser) -> None:
    """不變式三（審查 u5b2 第 2 條）：房間長度改成 -1，「存了再走」存不成：那一塊開著、提示欄寫「沒有開新的」配舊方案
    沒存成的原因（跟存檔回饋列同一句），看得到；再按一次、詢問裡按「取消」：提示欄是空的——兩條出口分得出來；
    取消之後游標回「開一份新的」，不掉到頁面底層。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        folder = _folder(tmp_path)
        page.locator("#room-Lx").fill("-1")
        fill_new_scheme(page, "fresh-x", "stage_one")
        page.locator("#new-scheme-go").click()
        expect(page.locator("#leave-edits")).to_be_visible()
        page.locator("#leave-edits").get_by_role("button", name="存了再走", exact=True).click()
        note = page.locator("#new-scheme-note")
        expect(note).to_contain_text("沒有開新的：開著的這一份沒有存成")
        page.wait_for_function(UNLOCKED)
        page.wait_for_load_state("networkidle")
        reason = page.locator("#save-feedback").text_content() or ""
        assert reason != ""
        expect(note).to_contain_text(reason)
        expect(page.locator("#template-box")).to_be_visible()
        expect(note).to_be_visible()
        page.locator("#new-scheme-go").click()
        expect(page.locator("#leave-edits")).to_be_visible()
        page.locator("#leave-edits").get_by_role("button", name="取消", exact=True).click()
        expect(page.locator("#leave-edits")).to_be_hidden()
        page.wait_for_function("!newSchemePending")
        expect(note).to_have_text("")
        expect(page.locator("#template-box")).to_be_visible()
        expect(page.locator("#new-scheme-go")).to_be_focused()
        assert _folder(tmp_path) == folder
        assert watched.page_errors == []


@pytest.mark.parametrize("case", ["design", "edited"])
def test_saved_but_still_unsaved_edits_say_the_real_reason(tmp_path: Path, browser: Browser, case: str) -> None:
    """不變式三（「存了再走」存好方案、表單卻還有沒存的）：design＝每一區的做法沒存成（頁面程式把做法標成讀不到），
    那一塊寫「沒有存成」配存檔回饋列那一句（做法沒存成的原因），不寫「又改了別的地方」；edited＝重讀清單那段沒鎖時
    真的又改了房間長度，那一塊寫「又改了別的地方」。兩種都沒存出新的那一份。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        if case == "design":
            open_example_scheme(page, base, tmp_path)
            page.locator("#room-Lx").fill("5.5")
            page.evaluate("zoneDesignBroken = true")
            fill_new_scheme(page, "fresh-d", "stage_one")
            page.locator("#new-scheme-go").click()
            page.locator("#leave-edits").get_by_role("button", name="存了再走", exact=True).click()
        else:
            held = _save_then_leave_with_the_list_held(page, base, tmp_path, "fresh-d")
            page.locator("#room-Lx").fill("5.7")
            held[0].continue_()
        note = page.locator("#new-scheme-note")
        expect(note).to_contain_text("沒有開新的：開著的這一份沒有存成")
        page.wait_for_function(f"!newSchemePending && {UNLOCKED}")
        if case == "design":
            expect(note).to_contain_text("每一區的做法沒存成")
            expect(note).not_to_contain_text("又改了別的地方")
        else:
            expect(note).to_contain_text("存的時候又改了別的地方")
        expect(page.locator("#template-box")).to_be_visible()
        assert "fresh-d.json" not in _folder(tmp_path)
        assert watched.page_errors == []


@pytest.mark.parametrize("taker", ["save", "save-as"])
def test_after_the_leave_question_a_lock_taken_meanwhile_refuses_the_new_scheme(
        tmp_path: Path, browser: Browser, taker: str) -> None:
    """不變式二：「存了再走」存好、重讀清單那段空窗裡頁面程式按了「儲存」或「另存新名字」（copy-b；它的存檔被攔住、鎖著）：
    詢問回來時看到 inputBusy，照鎖頁那句擋下、不送新方案的請求，那一塊開著寫那一句；放行之後資料夾沒有新的那一份。
    「儲存」做完那一塊照開著、那一句照留；「另存新名字」存好會自己收掉頂條的彈出塊（storeSchemeAs，原本就這樣），不在這裡守。"""
    stored = "example" if taker == "save" else "copy-b"
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        sent: list[str] = []
        page.on("request", lambda request: sent.append(request.url) if "fresh-b" in request.url or "template=" in request.url else None)
        held = _save_then_leave_with_the_list_held(page, base, tmp_path, "fresh-b")
        puts: list[Route] = []
        page.route(f"{base}/api/schemes/{stored}", lambda route: puts.append(route) if route.request.method == "PUT"
                   else route.continue_())
        page.evaluate("document.getElementById('save-as-id').value = 'copy-b'")
        with page.expect_request(lambda request: request.method == "PUT" and request.url.endswith(f"/{stored}")):
            page.locator(f"#{taker}").evaluate("node => node.click()")
        assert page.evaluate("inputBusy") is True
        held[0].continue_()
        note = page.locator("#new-scheme-note")
        expect(note).to_have_text(REFUSED_NEW)
        page.wait_for_function("!newSchemePending")
        expect(page.locator("#template-box")).to_be_visible()
        puts[0].continue_()
        page.wait_for_function(UNLOCKED)
        page.wait_for_load_state("networkidle")
        assert sent == [], sent
        assert set(_folder(tmp_path)) == {"example.json", f"{stored}.json"}
        expect(note).to_have_text(REFUSED_NEW)
        if taker == "save":
            expect(page.locator("#template-box")).to_be_visible()
        assert page.evaluate("openedId") == stored
        assert watched.page_errors == []


def test_opening_the_box_again_starts_with_an_empty_note(tmp_path: Path, browser: Browser) -> None:
    """不變式四：名字格空著按「開一份新的」（寫了沒開成的原因）→「取消」關掉 → 從選單再打開那一塊：提示欄是空的。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_example_scheme(page, base, tmp_path)
        fill_new_scheme(page, "", "reference")
        page.locator("#new-scheme-go").click()
        note = page.locator("#new-scheme-note")
        expect(note).to_contain_text("沒有開新的：")
        page.locator("#new-scheme-cancel").click()
        expect(page.locator("#template-box")).to_be_hidden()
        page.locator("#scheme-menu-button").click()
        page.locator("#choose-template").click()
        expect(page.locator("#template-box")).to_be_visible()
        expect(note).to_have_text("")
        _assert_quiet(watched)


def test_a_refusal_during_a_successful_new_scheme_does_not_come_back_when_the_box_reopens(
        tmp_path: Path, browser: Browser) -> None:
    """不變式四（審查 u5b2 第 3 條）：空畫面開新的「first-a」、重讀清單卡住時頁面程式再按一次（被擋、寫了鎖頁那句）；
    first-a 打開之後提示欄是空的，從選單再打開那一塊也是空的。空畫面搬回那一條見上面搬回那一題最後兩行。"""
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _empty_start(page)
        held: list[Route] = []
        page.route(f"{base}/api/schemes", lambda route: held.append(route) if route.request.method == "GET" and not held
                   else route.continue_())
        fill_new_scheme(page, "first-a", "reference")
        with page.expect_request(f"{base}/api/schemes"):
            page.locator("#new-scheme-go").click()
        page.evaluate("document.getElementById('new-scheme-id').value = 'first-b'")
        page.locator("#new-scheme-go").evaluate("node => node.click()")
        note = page.locator("#new-scheme-note")
        expect(note).to_have_text(REFUSED_NEW)
        held[0].continue_()
        page.wait_for_function(f"openedId === 'first-a' && {UNLOCKED}")
        expect(note).to_have_text("")
        page.locator("#scheme-menu-button").click()
        page.locator("#choose-template").click()
        expect(page.locator("#template-box")).to_be_visible()
        expect(note).to_have_text("")
        assert set(_folder(tmp_path)) == {"first-a.json"}
        _assert_quiet(watched)
