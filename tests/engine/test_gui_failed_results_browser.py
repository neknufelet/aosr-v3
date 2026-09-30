"""瀏覽器能看到失敗產物，但不把它當完成結果或比較勝負。"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser, expect

from aosr.reporting.result import SchemeResult
from tests.engine.test_gui_browser import _open, _serve, browser, result
from tests.engine.test_gui_compare_routes import pair
from tests.engine.test_gui_failed_results import _client, _path, _runner, _start, _wait_state


def _populate(tmp_path: Path, result: SchemeResult, other: SchemeResult) -> tuple[str, str]:
    runner = _runner(tmp_path, {result.scheme.scheme_id: (result, 3, False),
                                other.scheme.scheme_id: (other, 0, False)})
    with _client(tmp_path, runner) as client:
        failed, done = _start(client, result), _start(client, other)
        _wait_state(client, failed, "failed")
        _wait_state(client, done, "done")
    return failed, done


def test_home_marks_failed_row_and_compare_selection(
        tmp_path: Path, browser: Browser, result: SchemeResult,
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    failed, done = _populate(tmp_path, result, pair[1])
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        failed_row = page.locator("#results-list tr").filter(has=page.locator(f'a[href="/results/{failed}"]'))
        done_row = page.locator("#results-list tr").filter(has=page.locator(f'a[href="/results/{done}"]'))
        expect(failed_row).to_contain_text("失敗：內容可能不完整")
        assert "not-finished" in (failed_row.get_attribute("class") or "").split()
        assert "not-finished" not in (done_row.get_attribute("class") or "").split()
        failed_row.get_by_role("button", name="選為 A", exact=True).click()
        expect(page.locator("#compare-a")).to_contain_text("失敗：內容可能不完整")
        assert watched.page_errors == []


def test_result_notice_visible_only_for_failed(
        tmp_path: Path, browser: Browser, result: SchemeResult,
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    failed, done = _populate(tmp_path, result, pair[1])
    with _serve(tmp_path) as base:
        with _open(browser, f"{base}/results/{failed}") as watched:
            expect(watched.page.locator("#run-notice")).to_be_visible()
            expect(watched.page.locator("#run-notice")).to_contain_text("失敗（離開碼 3）")
            expect(watched.page.locator("#content")).to_be_visible()
            assert watched.page_errors == []
        with _open(browser, f"{base}/results/{done}") as watched:
            expect(watched.page.locator("#content")).to_be_visible()
            expect(watched.page.locator("#run-notice")).to_be_hidden()
            assert watched.page_errors == []


def test_failed_compare_has_notice_and_no_better_elements(
        tmp_path: Path, browser: Browser, result: SchemeResult,
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    failed, done = _populate(tmp_path, result, pair[1])
    with _serve(tmp_path) as base, _open(browser, f"{base}/compare/{failed}/{done}") as watched:
        page = watched.page
        expect(page.locator("#run-notices")).to_be_visible()
        expect(page.locator("#run-notices")).to_contain_text("A：這一筆計算回報失敗")
        expect(page.locator("#table-verdict")).to_contain_text("A：失敗")
        assert "比較好：" not in page.locator("#table-verdict").inner_text()
        # 頁面上找不到任何一格被標成比較好（問有沒有，不數個數）。
        assert page.evaluate("() => document.querySelector('.better') === null")
        expect(page.locator("#content")).to_be_visible()
        assert watched.page_errors == []


def test_failure_notices_visible_above_rejected_result_and_compare(
        tmp_path: Path, browser: Browser, result: SchemeResult,
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    failed, done = _populate(tmp_path, result, pair[1])
    _path(tmp_path, "results", failed).write_text("{}", encoding="utf-8")
    with _serve(tmp_path) as base:
        for route, notice in ((f"/results/{failed}", "#run-notice"),
                              (f"/compare/{failed}/{done}", "#run-notices")):
            with _open(browser, f"{base}{route}") as watched:
                page = watched.page
                expect(page.locator("#rejection")).to_be_visible()
                expect(page.locator(notice)).to_be_visible()
                expect(page.locator(notice)).to_contain_text("失敗")
                notice_box = page.locator(notice).bounding_box()
                rejection_box = page.locator("#rejection").bounding_box()
                assert notice_box is not None and rejection_box is not None
                assert notice_box["y"] < rejection_box["y"]
                assert watched.page_errors == []
