"""#1265 單元 4：輸入頁的結果清單搬進右側「算過的結果與比較」抽屜。資料都是考卷自己編的（控制組結果改名複製）。

一列＝一筆結果（一次計算）；勾兩筆比較（先勾的是 A，第三筆擋下）；評分設定不同的兩筆標紅；封存、已封存、
永久刪除的確認框字不變；存檔鎖住期間抽屜裡的按鈕按不了；頁面中間不再有第二份結果清單。"""
from __future__ import annotations

import os
import re
from pathlib import Path

from playwright.sync_api import Browser, Dialog, Locator, Page, Route, expect

from aosr.gui.labels import RESULT_RUN_LABELS
from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine._input_page_steps import check_result, open_archived, open_example_scheme, open_results, open_scheme, result_row
from tests.engine.test_gui_archive import _bundle, _saved
from tests.engine.test_gui_browser import _assert_quiet, _open, _serve, browser, result
from tests.engine.test_gui_result_standing import changed_snapshot

FIRST, SECOND, THIRD = "1" * 32, "2" * 32, "3" * 32
# 評分不同的紅底：比單一類別名（同一列另有 not-finished 等類別時也比得準，第三趟審查建議 2）。
REGISTRY_DIFFERS = re.compile(r"(^|\s)registry-differs(\s|$)")


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
        open_example_scheme(page, base, tmp_path)
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
        # 取消 A：B 遞補成 A（「先勾的是 A」照字面成立，第二趟 F4）；再勾第三筆是 B。網址、標記、提示句三處一致。
        result_row(page, SECOND).get_by_role("checkbox").uncheck()
        expect(go).to_have_attribute("aria-disabled", "true")
        expect(result_row(page, FIRST).locator(".compare-mark")).to_have_text("A")
        expect(hint).to_contain_text("已勾 A：room-one")
        expect(third).to_be_enabled()
        check_result(page, THIRD)
        expect(result_row(page, FIRST).locator(".compare-mark")).to_have_text("A")
        expect(result_row(page, THIRD).locator(".compare-mark")).to_have_text("B")
        expect(go).to_have_attribute("href", f"/compare/{FIRST}/{THIRD}")
        expect(hint).to_have_text("已勾兩筆（A 是先勾的）；要換一筆，先取消勾選其中一筆")
        # 重讀清單（存檔、封存、算完都會）之後勾選照留。
        page.evaluate("() => loadResultList()")
        expect(go).to_have_attribute("href", f"/compare/{FIRST}/{THIRD}")
        expect(result_row(page, THIRD).get_by_role("checkbox")).to_be_checked()
        _assert_quiet(watched)


def test_archive_two_picks_of_same_scheme_clears_both_even_if_reload_fails(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """第三趟 G1：勾的 A、B 是同一份方案的兩筆，封存那一份時兩格一起清空；封存之後重讀清單失敗（不會再整張重畫）
    也一樣。再勾別筆它是 A，網址不指到已封存的那一筆。"""
    first, second, other = FIRST, SECOND, THIRD
    same = _renamed(result, "same-scheme")
    _saved(tmp_path, same)
    _bundle(tmp_path, same, first)
    _bundle(tmp_path, same, second)
    _bundle(tmp_path, _renamed(result, "other-scheme"), other)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _dialogs(page)
        check_result(page, first)
        check_result(page, second)
        page.route("**/api/results", lambda route: route.fulfill(
            status=500, content_type="application/json", body='{"error":"清單暫時讀不到"}'))
        result_row(page, first).get_by_role("button", name="封存").click()
        expect(page.locator("#results-note")).to_contain_text("清單更新不了")
        assert page.evaluate("() => [compareChoice.a?.run_id ?? null, compareChoice.b?.run_id ?? null]") == [None, None]
        for run_id in (first, second):
            expect(result_row(page, run_id).locator(".compare-mark")).to_have_text("")
            expect(result_row(page, run_id).get_by_role("checkbox")).not_to_be_checked()
        expect(page.locator("#compare-hint")).to_have_text("勾兩筆就能比較")
        check_result(page, other)
        expect(result_row(page, other).locator(".compare-mark")).to_have_text("A")
        expect(page.locator("#compare-hint")).to_contain_text("已勾 A：other-scheme")
        go = page.locator("#compare-go")
        expect(go).to_have_attribute("aria-disabled", "true")
        assert go.get_attribute("href") is None
        # 主控台只准有攔下的那幾次 500（瀏覽器自己印的「讀取資源失敗」），頁面本身不准拋錯。
        assert all("500" in text for text in watched.console_errors), watched.console_errors
        assert watched.page_errors == []


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
            expect(result_row(page, run_id)).not_to_have_class(REGISTRY_DIFFERS)
        expect(warning).to_be_hidden()
        plain = result_row(page, THIRD).evaluate("node => getComputedStyle(node).backgroundColor")
        result_row(page, SECOND).get_by_role("checkbox").uncheck()
        check_result(page, THIRD)
        for run_id in (FIRST, THIRD):
            expect(result_row(page, run_id)).to_have_class("registry-differs")
            assert result_row(page, run_id).evaluate("node => getComputedStyle(node).backgroundColor") != plain
        expect(result_row(page, SECOND)).not_to_have_class(REGISTRY_DIFFERS)
        expect(warning).to_be_visible()
        expect(warning).to_have_text("評分設定不同，放在一起比會不準")
        # 不擋比較。
        expect(page.locator("#compare-go")).to_have_attribute("href", f"/compare/{FIRST}/{THIRD}")
        # 取消其中一筆：紅底與那一句一起收起（審查建議 3）。
        result_row(page, THIRD).get_by_role("checkbox").uncheck()
        for run_id in (FIRST, THIRD):
            expect(result_row(page, run_id)).not_to_have_class(REGISTRY_DIFFERS)
        expect(warning).to_be_hidden()
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


def _order(page: Page, body: str, key: str) -> list[str]:
    """抽屜裡那張表由上到下每一列的代號（結果列 data-run-id、整包 data-package-id）。"""
    return list(page.locator(f"{body} tr").evaluate_all(f"rows => rows.map(row => row.dataset.{key})"))


def test_redraw_keeps_focus_on_same_button_of_same_row(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """第二趟 F1：抽屜開著時重讀清單（存檔解鎖、算完、封存之後都會），游標留在同一筆的同一顆，不准跳去「看」。"""
    _three(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        row = result_row(page, SECOND)
        for target in (row.get_by_role("button", name="封存"), row.get_by_role("link", name="看", exact=True),
                       row.get_by_role("checkbox")):
            target.focus()
            page.evaluate("() => loadResultList().then(() => null)")
            expect(target).to_be_focused()
        # 結果列重畫完、封存清單還在路上（伺服器慢）的那一段：游標已經放回去了，不等封存清單回來。
        held: list[Route] = []
        page.route("**/api/archive", lambda route: held.append(route))
        target = row.get_by_role("button", name="封存")
        target.focus()
        page.evaluate("() => { window.reloaded = loadResultList(); }")
        for _ in range(100):
            if held:
                break
            page.wait_for_timeout(50)
        assert len(held) == 1
        expect(target).to_be_focused()
        held[0].continue_()
        page.unroute("**/api/archive")
        page.evaluate("() => window.reloaded.then(() => null)")
        expect(target).to_be_focused()
        _assert_quiet(watched)


def test_row_gone_moves_focus_to_next_row_same_button_or_close(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """第二趟 F3：封存、搬回把游標所在那一列拿掉時，游標落到下一列的同一顆；沒有下一列就落到「關閉 ✕」，不准掉出抽屜。"""
    _three(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _dialogs(page)
        order = _order(page, "#results-list", "runId")
        result_row(page, order[1]).get_by_role("button", name="封存").focus()
        page.keyboard.press("Enter")
        expect(result_row(page, order[1])).to_have_count(0)
        expect(result_row(page, order[2]).get_by_role("button", name="封存")).to_be_focused()
        page.keyboard.press("Enter")
        expect(result_row(page, order[2])).to_have_count(0)
        expect(page.locator("#results-close")).to_be_focused()
        # 已封存那一塊：搬回第一包，游標落到下一包的「搬回」；搬回最後一包，落到關閉鈕。
        open_archived(page)
        packages = _order(page, "#archived-packages-list", "packageId")
        assert len(packages) == 2

        def restore(package: str) -> Locator:
            row = page.locator(f'#archived-packages-list tr[data-package-id="{package}"]')
            return row.get_by_role("button", name="搬回", exact=True)

        restore(packages[0]).focus()
        page.keyboard.press("Enter")
        expect(restore(packages[0])).to_have_count(0)
        expect(restore(packages[1])).to_be_focused()
        page.keyboard.press("Enter")
        expect(restore(packages[1])).to_have_count(0)
        expect(page.locator("#results-close")).to_be_focused()
        # 固定的那一顆被藏起來：游標在「全部清除封存」上按確認，清完那一顆與整塊都藏起來，游標落到關閉鈕。
        result_row(page, order[0]).get_by_role("button", name="封存").click()
        expect(result_row(page, order[0])).to_have_count(0)
        open_archived(page)
        page.wait_for_load_state("networkidle")
        # 藏起來的同一拍就要放好（不是等之後哪一次重讀清單才順手放回）：記下藏起來那一下的游標。
        page.evaluate("""() => new MutationObserver((changes, watch) => {
          window.whenHidden = document.activeElement.id || document.activeElement.tagName; watch.disconnect();
        }).observe(document.getElementById('clear-archive'), {attributes: true, attributeFilter: ['hidden']})""")
        page.locator("#clear-archive").focus()
        page.keyboard.press("Enter")
        expect(page.locator("#clear-archive")).to_be_hidden()
        assert page.evaluate("() => window.whenHidden") == "results-close"
        expect(page.locator("#results-close")).to_be_focused()
        _assert_quiet(watched)


def _stamp(root: Path, run_id: str, seconds: int) -> None:
    """結果清單照結果檔的修改時間新到舊排：把那一筆的時間釘住，抽屜的順序就固定。"""
    for path in (root / "results").glob(f"{run_id}*"):
        os.utime(path, (seconds, seconds))


def test_row_gone_lands_on_first_surviving_row_below(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """第三趟 G2：那一筆不在了，游標落到「原本在它下面、現在還在的第一筆」的同一顆，不是照數字位置：
    同一份方案好幾筆一起被封存時不跳過真正的下一筆；上面同時插進新的一筆，也不落到原本在它上面的那一筆。"""
    a1, a2, b1, c1, d1, new = "1" * 32, "2" * 32, "3" * 32, "4" * 32, "5" * 32, "6" * 32
    same = _renamed(result, "room-one")
    _saved(tmp_path, same)
    for index, (item, run_id) in enumerate(((_renamed(result, "room-four"), d1), (_renamed(result, "room-three"), c1),
                                            (_renamed(result, "room-two"), b1), (same, a2), (same, a1))):
        _bundle(tmp_path, item, run_id)
        _stamp(tmp_path, run_id, 1700000000 + index * 60)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _dialogs(page)
        open_results(page)
        assert _order(page, "#results-list", "runId") == [a1, a2, b1, c1, d1]
        # 游標在 a2（第 2 列）按封存：a1、a2 同一份方案一起收走，剩 [b1, c1, d1]，落到 b1，不是第 2 列的 c1。
        result_row(page, a2).get_by_role("button", name="封存").focus()
        page.keyboard.press("Enter")
        expect(result_row(page, a2)).to_have_count(0)
        assert _order(page, "#results-list", "runId") == [b1, c1, d1]
        expect(result_row(page, b1).get_by_role("button", name="封存")).to_be_focused()
        # 游標在 c1（第 2 列）的「看」：c1 不在了、最上面同時多了新的一筆，剩 [new, b1, d1]，落到 d1，不是第 2 列的 b1。
        result_row(page, c1).get_by_role("link", name="看", exact=True).focus()
        for path in (tmp_path / "results").glob(f"{c1}*"):
            path.unlink()
        _bundle(tmp_path, _renamed(result, "room-new"), new)
        _stamp(tmp_path, new, 1700001000)
        page.evaluate("() => loadResultList().then(() => null)")
        assert _order(page, "#results-list", "runId") == [new, b1, d1]
        expect(result_row(page, d1).get_by_role("link", name="看", exact=True)).to_be_focused()
        _assert_quiet(watched)


def test_archive_block_folds_after_clear_then_new_archive(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """第三趟順手（審查建議 1）：已封存那一塊清空時一併收合，之後再封存一筆，它出現時是收著的，沒按「看已封存的」不自己展開。"""
    _three(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        _dialogs(page)
        result_row(page, FIRST).get_by_role("button", name="封存").click()
        expect(result_row(page, FIRST)).to_have_count(0)
        open_archived(page)
        page.wait_for_load_state("networkidle")
        page.locator("#clear-archive").click()
        expect(page.locator("#archived")).to_be_hidden()
        result_row(page, SECOND).get_by_role("button", name="封存").click()
        expect(result_row(page, SECOND)).to_have_count(0)
        expect(page.locator("#archived")).to_be_visible()
        assert page.locator("#archived").evaluate("node => node.open") is False
        _assert_quiet(watched)


def test_locked_page_ignores_escape_and_unlock_keeps_focus(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """第二趟 F2＋F3：整頁鎖著（inert）時 Esc 不關抽屜；存檔、換方案解鎖後，游標還在抽屜裡同一筆的同一顆。"""
    _bundle(tmp_path, _renamed(result, "room-one"), FIRST)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        held: list[Route] = []
        page.route("**/api/schemes/*", lambda route: held.append(route) if route.request.method == "PUT"
                   else route.continue_())
        archive = result_row(page, FIRST).get_by_role("button", name="封存")
        archive.focus()
        page.evaluate("() => { action(save); }")
        page.wait_for_function("document.querySelector('main').inert && inputBusy")
        for _ in range(100):  # 存檔先跑圖面檢查，再送方案：等送方案那一趟被攔住
            if held:
                break
            page.wait_for_timeout(100)
        assert len(held) == 1
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        expect(page.locator("#results-panel")).to_be_visible()
        held[0].continue_()
        page.wait_for_function("!document.querySelector('main').inert && !inputBusy")
        expect(page.locator("#messages")).to_have_text("方案已儲存")
        expect(archive).to_be_focused()
        # 換方案（開檔、範本共用的交接）鎖頁再解開：鎖著那一段伺服器慢、游標已掉回 body；解鎖後放回同一顆。
        page.wait_for_load_state("networkidle")
        previews: list[Route] = []
        page.route("**/api/input-preview", lambda route: previews.append(route))
        archive.focus()
        page.evaluate("() => { window.changed = changeInputScheme(async () => structuredClone(scheme)); }")
        page.wait_for_function("document.querySelector('main').inert && document.activeElement === document.body")
        for _ in range(100):
            if previews:
                break
            page.wait_for_timeout(50)
        assert len(previews) == 1
        previews[0].continue_()
        page.unroute("**/api/input-preview")
        # 解鎖那一下就放回去（不是等之後哪一次重讀清單才順手放回）。
        assert page.evaluate("""async () => {
          const changed = await window.changed, active = document.activeElement;
          return [changed, active.tagName, active.textContent];
        }""") == [True, "BUTTON", "封存"]
        expect(archive).to_be_focused()
        _assert_quiet(watched)


def test_show_archived_with_nothing_archived_says_so(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """第二趟 F5：沒有封存時頂條「看已封存的」照樣打開抽屜、寫一句還沒有，不展開空表、不出滑動提示；
    之後從頂條結果鈕重開，那一塊照有沒有封存決定；真有表格時窄畫面才出滑動提示。"""
    _bundle(tmp_path, _renamed(result, "room-one"), FIRST)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=600) as watched:
        page = watched.page
        _dialogs(page)
        none, archived, say = page.locator("#archive-none"), page.locator("#archived"), page.locator("#archive-say")
        scroll_note = page.locator(".archive-scroll-note")

        def show_archived() -> None:
            page.wait_for_function("document.body.dataset.inputReady === 'true'")
            page.locator("#scheme-menu-button").click()
            page.locator("#show-archived").click()
            expect(page.locator("#results-panel")).to_be_visible()

        show_archived()
        expect(none).to_be_visible()
        expect(none).to_have_text("還沒有封存的結果")
        expect(none).to_be_focused()
        for gone in (archived, say):
            expect(gone).to_be_hidden()
        assert scroll_note.evaluate("node => getComputedStyle(node).display") == "none"
        page.keyboard.press("Escape")
        open_results(page)
        for gone in (none, archived, say):
            expect(gone).to_be_hidden()
        page.keyboard.press("Escape")
        show_archived()
        expect(none).to_be_visible()
        result_row(page, FIRST).get_by_role("button", name="封存").click()
        expect(result_row(page, FIRST)).to_have_count(0)
        expect(none).to_be_hidden()
        open_archived(page)
        expect(say).to_be_visible()
        expect(scroll_note).to_be_visible()
        _assert_quiet(watched)


def test_results_note_keeps_line_breaks_like_messages(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    """第二趟 F6：抽屜上方的回話跟訊息列一樣保留換行（全部清除封存有跳過的包時是兩行）。"""
    _bundle(tmp_path, _renamed(result, "room-one"), FIRST)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        open_results(page)
        page.evaluate("() => say('第一行\\n第二行', 'notice')")
        assert page.locator("#messages").inner_text() == "第一行\n第二行"
        assert page.locator("#results-note").inner_text() == "第一行\n第二行"
        _assert_quiet(watched)
