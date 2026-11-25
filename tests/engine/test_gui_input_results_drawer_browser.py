"""#1265 單元 4：輸入頁的結果清單搬進右側「算過的結果與比較」抽屜。資料都是考卷自己編的（控制組結果改名複製）。

一列＝一筆結果（一次計算）；勾兩筆比較（先勾的是 A，第三筆擋下）；評分設定不同的兩筆標紅；封存、已封存、
永久刪除的確認框字不變；存檔鎖住期間抽屜裡的按鈕按不了；頁面中間不再有第二份結果清單。"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, Dialog, Locator, Page, Route, expect

from aosr.gui.labels import RESULT_RUN_LABELS
from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine._input_page_steps import check_result, open_archived, open_results, open_scheme, result_row
from tests.engine.test_gui_archive import _bundle, _saved
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser, result
from tests.engine.test_gui_result_standing import changed_snapshot

FIRST, SECOND, THIRD = "1" * 32, "2" * 32, "3" * 32


def _renamed(result: SchemeResult, name: str) -> SchemeResult:
    return result.model_copy(update={"scheme": result.scheme.model_copy(update={"scheme_id": name})})


def _three(root: Path, result: SchemeResult) -> None:
    """三筆正常算完、評分設定都跟現在相同的結果，三個不同名字。"""
    for run_id, name in ((FIRST, "room-one"), (SECOND, "room-two"), (THIRD, "room-three")):
        _bundle(root, _renamed(result, name), run_id)


def _active_inside_panel(page: Page) -> bool:
    return bool(page.evaluate("() => document.getElementById('results-panel').contains(document.activeElement)"))


def _dialogs(page: Page) -> list[str]:
    messages: list[str] = []

    def accept(dialog: Dialog) -> None:
        messages.append(dialog.message)
        dialog.accept()

    page.on("dialog", accept)
    return messages


def test_drawer_opens_closes_three_ways_and_traps_tab(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _three(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        # 頁面中間不再有第二份結果清單：舊的段落、A／B 兩格、舊比較連結、「選為 A」都不在了。
        for gone in ("#result-section", "#compare-a", "#compare-b", "#compare-link"):
            expect(page.locator(gone)).to_have_count(0)
        expect(page.get_by_role("button", name="選為 A")).to_have_count(0)
        expect(page.get_by_role("button", name="選為 B")).to_have_count(0)
        expect(page.locator("#results-panel")).to_be_hidden()
        open_results(page)
        panel = page.locator("#results-panel")
        expect(panel).to_have_attribute("role", "dialog")
        expect(panel).to_have_attribute("aria-modal", "true")
        expect(page.locator("#results-title")).to_have_text("算過的結果與比較")
        assert "一列就是一份方案" not in panel.inner_text()
        assert _active_inside_panel(page)
        # Tab、Shift+Tab 各繞兩圈以上都不跑出抽屜。
        for key in ["Tab"] * 14 + ["Shift+Tab"] * 14:
            page.keyboard.press(key)
            assert _active_inside_panel(page), key
        # Esc 關，游標回頂條那顆鈕。
        page.keyboard.press("Escape")
        expect(panel).to_be_hidden()
        expect(page.locator("#show-results")).to_be_focused()
        # 關閉鈕關。
        open_results(page)
        page.locator("#results-close").click()
        expect(panel).to_be_hidden()
        expect(page.locator("#show-results")).to_be_focused()
        # 點遮罩（抽屜左邊那一片）關。
        open_results(page)
        page.mouse.click(200, 450)
        expect(panel).to_be_hidden()
        expect(page.locator("#show-results")).to_be_focused()
        _assert_quiet(watched)


def test_empty_drawer_says_so_without_compare_row(tmp_path: Path, browser: Browser) -> None:
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        open_results(page)
        expect(page.locator("#results-empty")).to_have_text(
            "還沒有算過的結果；在頁底按「照現在牆上貼的算一次」，算好就列在這裡。")
        expect(page.locator("#results-list tr")).to_have_count(0)
        expect(page.locator("#compare-row")).to_be_hidden()
        expect(page.locator("#compare-go")).to_be_hidden()
        _assert_quiet(watched)


def test_running_and_failed_rows_say_their_state(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _bundle(tmp_path, _renamed(result, "room-failed"), FIRST, "failed")
    _bundle(tmp_path, _renamed(result, "room-running"), SECOND)
    _bundle(tmp_path, _renamed(result, "room-done"), THIRD)
    running_text = RESULT_RUN_LABELS["running"][0]

    def mark_running(route: Route) -> None:
        # 計算中的那一筆：伺服器的計算行程是考卷造不出來的，清單那一格照伺服器的字改。
        body = route.fetch().json()
        for row in body["results"]:
            if row["run_id"] == SECOND:
                row.update(run_status="running", status_text=running_text)
        route.fulfill(json=body)

    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        page.route(f"{base}/api/results", mark_running)
        page.reload(wait_until="networkidle")
        expect(result_row(page, FIRST).locator(".run-status")).to_have_text(RESULT_RUN_LABELS["failed"][0])
        expect(result_row(page, SECOND).locator(".run-status")).to_have_text(running_text)
        expect(result_row(page, THIRD).locator(".run-status")).to_have_count(0)
        for run_id in (FIRST, SECOND):
            assert "not-finished" in (result_row(page, run_id).get_attribute("class") or "").split()
        assert "not-finished" not in (result_row(page, THIRD).get_attribute("class") or "").split()
        _assert_quiet(watched)


def test_same_scheme_twice_is_two_rows_and_open_one_is_marked(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    name = result.scheme.scheme_id
    _saved(tmp_path, result)
    _bundle(tmp_path, result, FIRST, "failed")
    _bundle(tmp_path, result, SECOND)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        for run_id in (FIRST, SECOND):
            expect(result_row(page, run_id)).to_contain_text(name)
            expect(result_row(page, run_id).locator(".opened-mark")).to_have_count(0)
        open_scheme(page, name)
        expect(page.locator("#save-id")).to_have_value(name)
        for run_id in (FIRST, SECOND):
            expect(result_row(page, run_id).locator(".opened-mark")).to_have_text("（現在開著）")
            expect(result_row(page, run_id).get_by_role("link", name="看", exact=True)).to_have_attribute(
                "href", f"/results/{run_id}")
        _assert_quiet(watched)


def test_check_two_to_compare_third_is_blocked(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _three(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        open_results(page)
        go, hint = page.locator("#compare-go"), page.locator("#compare-hint")
        expect(hint).to_have_text("勾兩筆就能比較")
        expect(go).to_have_attribute("aria-disabled", "true")
        check_result(page, SECOND)
        expect(hint).to_contain_text("再勾一筆就能比較")
        expect(go).to_have_attribute("aria-disabled", "true")
        assert go.get_attribute("href") is None
        expect(result_row(page, SECOND).locator(".compare-mark")).to_have_text("A")
        check_result(page, FIRST)
        expect(go).to_have_attribute("aria-disabled", "false")
        expect(go).to_have_attribute("href", f"/compare/{SECOND}/{FIRST}")
        expect(result_row(page, FIRST).locator(".compare-mark")).to_have_text("B")
        # 第三筆擋下：勾選格停用、照講怎麼換；網址不變。
        third = result_row(page, THIRD).get_by_role("checkbox")
        expect(third).to_be_disabled()
        expect(hint).to_contain_text("先取消勾選其中一筆")
        expect(go).to_have_attribute("href", f"/compare/{SECOND}/{FIRST}")
        # 取消 A：A 那一格空出來、B 不動；再勾第三筆就補進 A。
        result_row(page, SECOND).get_by_role("checkbox").uncheck()
        expect(go).to_have_attribute("aria-disabled", "true")
        expect(result_row(page, FIRST).locator(".compare-mark")).to_have_text("B")
        expect(third).to_be_enabled()
        check_result(page, THIRD)
        expect(go).to_have_attribute("href", f"/compare/{THIRD}/{FIRST}")
        # 重讀清單（存檔、封存、算完都會）之後勾選照留。
        page.evaluate("() => loadResultList()")
        expect(go).to_have_attribute("href", f"/compare/{THIRD}/{FIRST}")
        expect(result_row(page, THIRD).get_by_role("checkbox")).to_be_checked()
        _assert_quiet(watched)


def test_different_scoring_settings_turn_both_rows_red(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _bundle(tmp_path, _renamed(result, "room-one"), FIRST)
    _bundle(tmp_path, _renamed(result, "room-two"), SECOND)
    other = _renamed(result, "room-old-scoring")
    _bundle(tmp_path, other.model_copy(update={"purpose_settings": changed_snapshot(other)}), THIRD)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        warning = page.locator("#compare-warning")
        check_result(page, FIRST)
        check_result(page, SECOND)
        for run_id in (FIRST, SECOND, THIRD):
            expect(result_row(page, run_id)).not_to_have_class("registry-differs")
        expect(warning).to_be_hidden()
        plain = result_row(page, THIRD).evaluate("node => getComputedStyle(node).backgroundColor")
        result_row(page, SECOND).get_by_role("checkbox").uncheck()
        check_result(page, THIRD)
        for run_id in (FIRST, THIRD):
            expect(result_row(page, run_id)).to_have_class("registry-differs")
            assert result_row(page, run_id).evaluate("node => getComputedStyle(node).backgroundColor") != plain
        expect(result_row(page, SECOND)).not_to_have_class("registry-differs")
        expect(warning).to_be_visible()
        expect(warning).to_have_text("評分設定不同，放在一起比會不準")
        # 不擋比較。
        expect(page.locator("#compare-go")).to_have_attribute("href", f"/compare/{FIRST}/{THIRD}")
        _assert_quiet(watched)


def test_archive_restore_and_delete_from_drawer_keep_confirm_words(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    name = "room-archive"
    item = _renamed(result, name)
    _saved(tmp_path, item)
    _bundle(tmp_path, item, FIRST)
    _bundle(tmp_path, _renamed(result, "room-stays"), SECOND)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        messages = _dialogs(page)
        result_row(page, FIRST).get_by_role("button", name="封存").click()
        expect(result_row(page, FIRST)).to_have_count(0)
        assert messages[0] == (f"會把「{name}」的設定檔和 1 筆結果一起收起來，下拉和清單都看不到，名字會空出來，"
                               "之後可以從封存區整包搬回。確定封存嗎？")
        # 訊息列被抽屜蓋著：回話照抄在抽屜上方。
        expect(page.locator("#results-note")).to_have_text(page.locator("#messages").inner_text())
        expect(page.locator("#archived summary")).to_have_text("看已封存的結果（1 包方案、0 筆舊的單筆封存）")
        open_archived(page)
        package = page.locator(f'#archived-packages-list tr[data-scheme-id="{name}"]')
        package.get_by_role("button", name="搬回", exact=True).click()
        expect(result_row(page, FIRST)).to_be_visible()
        expect(page.locator("#archived")).to_be_hidden()
        # 再封存一次、永久刪除：確認框的字跟搬家前一字不差（數字從伺服器的封存清單來）。
        result_row(page, FIRST).get_by_role("button", name="封存").click()
        expect(result_row(page, FIRST)).to_have_count(0)
        listed = page.request.get(f"{base}/api/archive").json()["packages"][0]
        open_archived(page)
        package.get_by_role("button", name="永久刪除", exact=True).click()
        expect(package).not_to_be_attached()
        assert messages[-1] == (f"會永久刪除封存的「{name}」（{listed['archived_text']} 封存、{listed['result_count']} 筆結果、"
                                f"{listed['size_text']}）。這個資料夾沒有備份，刪了救不回來。確定刪除嗎？")
        expect(result_row(page, SECOND)).to_be_visible()
        _assert_quiet(watched)


def test_menu_show_archived_opens_drawer_on_archive(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    name = "room-archive"
    _saved(tmp_path, _renamed(result, name))
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        assert page.request.post(f"{base}/api/schemes/{name}/archive", data={}).ok
        page.reload(wait_until="networkidle")
        page.wait_for_function("document.body.dataset.inputReady === 'true'")
        page.locator("#scheme-menu-button").click()
        page.locator("#show-archived").click()
        expect(page.locator("#results-panel")).to_be_visible()
        expect(page.locator("#archived")).to_have_attribute("open", "")
        expect(page.locator("#archived summary")).to_be_focused()
        expect(page.locator("#archived-packages-list")).to_contain_text(name)
        page.keyboard.press("Escape")
        expect(page.locator("#results-panel")).to_be_hidden()
        expect(page.locator("#show-results")).to_be_focused()
        _assert_quiet(watched)


def _click_center(page: Page, target: Locator) -> None:
    box = target.bounding_box()
    assert box is not None
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def test_drawer_buttons_do_nothing_while_save_holds_the_page(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _bundle(tmp_path, _renamed(result, "room-one"), FIRST)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        held: list[Route] = []
        page.route("**/api/schemes/*", lambda route: held.append(route) if route.request.method == "PUT"
                   else route.continue_())
        asked: list[str] = []
        page.on("request", lambda request: asked.append(request.url) if "archive" in request.url else None)
        dialogs: list[str] = []

        def dismiss(dialog: Dialog) -> None:
            dialogs.append(dialog.message)
            dialog.dismiss()

        page.on("dialog", dismiss)
        archive = result_row(page, FIRST).get_by_role("button", name="封存")
        # 存檔途中（伺服器還沒回）：整頁鎖住，抽屜裡的封存按了沒有反應——不問確認、不送封存。
        page.evaluate("() => { action(save); }")
        page.wait_for_function("document.querySelector('main').inert && inputBusy")
        for _ in range(100):  # 存檔先跑圖面檢查，再送方案：等送方案那一趟被攔住
            if held:
                break
            page.wait_for_timeout(100)
        assert len(held) == 1
        assert page.evaluate("document.querySelector('main').inert && inputBusy")
        _click_center(page, archive)
        page.wait_for_timeout(500)
        assert dialogs == [] and asked == []
        held[0].continue_()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
        expect(page.locator("#messages")).to_have_text("方案已儲存")
        # 解鎖之後同一顆按得到（這次按取消，不封存）。
        with page.expect_event("dialog"):
            _click_center(page, archive)
        assert len(dialogs) == 1 and "確定封存嗎" in dialogs[0]
        expect(result_row(page, FIRST)).to_be_visible()
        _assert_quiet(watched)
