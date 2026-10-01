"""首頁能按封存、搬回，並保住比較選擇與卡片內的版面。"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Locator, Page, expect

from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine.test_gui_archive import _bundle
from tests.engine.test_gui_browser import BUTTON_LOOK_JS, _assert_quiet, _open, _serve, browser, result
from tests.engine.test_gui_failed_results import _path, _runner


def _row(page: Page, run_id: str, archived: bool = False) -> Locator:
    listing = "archived-list" if archived else "results-list"
    return page.locator(f'#{listing} tr[data-run-id="{run_id}"]')


def _assert_text_on_one_line(node: Locator) -> None:
    assert node.evaluate("""node => {
      const text = node.firstChild;
      if (!text || text.nodeType !== Node.TEXT_NODE) return false;
      const range = document.createRange(); range.selectNodeContents(text);
      const rects = [...range.getClientRects()];
      return rects.some(rect => rect.width > 0) && rects.every(rect => rect.top === rects[0].top);
    }""")


def _long_name_results(root: Path, result: SchemeResult) -> None:
    long = result.model_copy(update={"scheme": result.scheme.model_copy(
        update={"scheme_id": "reference-room-original2"})})
    # 30 字、沒有連字號：本機字型下 32 字在 1440 寬只剩約 1.6 像素餘裕，雲端字型稍寬就會紅；
    # 30 字留約 15 像素，照樣把表格擠到日期與時刻上下疊（日期小段可折的錯法才會現形）。
    unbroken = result.model_copy(update={"scheme": result.scheme.model_copy(
        update={"scheme_id": "classroom_reference_original_2"})})
    for run_id, item, status in (("a" * 32, long, "done"), ("b" * 32, result, "failed"),
                                 ("c" * 32, result, "none"), ("d" * 32, unbroken, "done")):
        _bundle(root, item, run_id, status)
        stamp = datetime(2026, 9, 30, 18, 47).timestamp()
        os.utime(_path(root, "results", run_id), (stamp, stamp))


def test_archive_restore_buttons_status_and_empty_section(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    run_id = "a" * 32
    _bundle(tmp_path, result, run_id, "failed")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        expect(page.locator("#archived")).to_be_hidden()
        row = _row(page, run_id)
        archive = row.get_by_role("button", name="封存", exact=True)
        expect(archive).to_be_visible()
        assert archive.evaluate(BUTTON_LOOK_JS) == row.get_by_role("button", name="選為 A").evaluate(BUTTON_LOOK_JS)
        archive.click()
        page.wait_for_function("id => document.querySelector(`#results-list tr[data-run-id='${id}']`) === null",
                               arg=run_id)
        expect(page.locator("#archived")).to_be_visible()
        expect(page.locator("#archived summary")).to_contain_text("1 筆")
        assert page.locator("#archived").evaluate("node => !node.open")
        page.locator("#archived summary").click()
        archived = _row(page, run_id, True)
        expect(archived).to_be_visible()
        expect(archived.locator(".run-status")).to_have_text("失敗：內容可能不完整")
        assert "not-finished" in (archived.get_attribute("class") or "").split()
        assert archived.evaluate("node => node.querySelector('a') === null")
        expect(page.locator("#messages")).to_contain_text("已封存")
        archived.get_by_role("button", name="搬回", exact=True).click()
        expect(_row(page, run_id)).to_be_visible()
        expect(page.locator("#archived")).to_be_hidden()
        expect(page.locator("#messages")).to_contain_text("已搬回")
        _assert_quiet(watched)


@pytest.mark.parametrize("side", ["A", "B"])
def test_archive_clears_selected_compare_side(
        tmp_path: Path, browser: Browser, result: SchemeResult, side: str) -> None:
    run_id, other_id = "a" * 32, "b" * 32
    _bundle(tmp_path, result, run_id)
    _bundle(tmp_path, result, other_id)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        other_side = "B" if side == "A" else "A"
        _row(page, run_id).get_by_role("button", name=f"選為 {side}", exact=True).click()
        _row(page, other_id).get_by_role("button", name=f"選為 {other_side}", exact=True).click()
        expect(page.locator("#compare-link")).to_be_visible()
        _row(page, run_id).get_by_role("button", name="封存", exact=True).click()
        expect(page.locator(f"#compare-{side.lower()}")).to_have_text(f"{side}：未選")
        expect(page.locator(f"#compare-{other_side.lower()}")).to_contain_text(result.scheme.scheme_id)
        expect(page.locator("#compare-link")).to_be_hidden()
        _assert_quiet(watched)


def test_archive_hides_latest_result_link(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    runner = _runner(tmp_path, {result.scheme.scheme_id: (result, 0, False)})
    with _serve(tmp_path, runner=runner) as base, _open(browser, base) as watched:
        page = watched.page
        name = result.scheme.scheme_id
        response = page.request.put(f"{base}/api/schemes/{name}", data=result.scheme.model_dump(mode="json"))
        assert response.status == 200
        page.reload(wait_until="networkidle")
        page.locator("#scheme-list").select_option(name)
        page.locator("#open-scheme").click()
        expect(page.locator("#save-id")).to_have_value(name)
        page.locator("#calculate").click()
        expect(page.locator("#result-link")).to_be_visible(timeout=20_000)
        page.locator("#speaker-left-x").fill("1.7")
        expect(page.locator("#result-stale")).to_be_visible()
        expect(page.locator("#result-stale")).to_have_text("設定已修改，這份結果是修改前算的")
        run_id = str(page.locator("#result-link").get_attribute("href")).rsplit("/", 1)[1]
        _row(page, run_id).get_by_role("button", name="封存", exact=True).click()
        expect(page.locator("#result-link")).to_be_hidden()
        expect(page.locator("#result-stale")).to_be_hidden()
        _assert_quiet(watched)


def test_long_scheme_table_fits_card_and_date_stays_on_one_line(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    _long_name_results(tmp_path, result)
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        for run_id, name in (("a" * 32, "reference-room-original2"),
                             ("d" * 32, "classroom_reference_original_2")):
            row = _row(page, run_id)
            expect(row).to_contain_text(name)
            _assert_text_on_one_line(row.locator("td").first)
            date = row.locator(".finished-date")
            expect(date).to_have_text("2026-09-30")
            _assert_text_on_one_line(date)
        assert page.locator("table:has(#results-list)").evaluate("""table => {
          const card = table.closest('section');
          return card.scrollWidth <= card.clientWidth;
        }""")
        # 1100 寬、兩個長名字都在時表格一定被擠到最窄（跟字型寬窄無關）：
        # 只驗名字與日期各自不斷行，不驗凸不凸出；驗完回 1440 寬。
        page.set_viewport_size({"width": 1100, "height": 900})
        for run_id in ("a" * 32, "d" * 32):
            squeezed = _row(page, run_id)
            _assert_text_on_one_line(squeezed.locator("td").first)
            _assert_text_on_one_line(squeezed.locator(".finished-date"))
        page.set_viewport_size({"width": 1440, "height": 900})
        _row(page, "d" * 32).get_by_role("button", name="封存", exact=True).click()
        expect(page.locator("#archived")).to_be_visible()
        # 沒有斷點的長名字撐寬第一欄時，帶連字號的名字永遠不會被擠；封存它之後再量一次，
        # 名字格若可折，reference-room-original2 這時就會在連字號斷成兩行。
        remaining = _row(page, "a" * 32)
        expect(remaining).to_contain_text("reference-room-original2")
        _assert_text_on_one_line(remaining.locator("td").first)
        page.locator("#archived summary").click()
        archived = _row(page, "d" * 32, archived=True)
        expect(archived).to_contain_text("classroom_reference_original_2")
        _assert_text_on_one_line(archived.locator("td").first)
        _assert_text_on_one_line(archived.locator(".finished-date"))
        assert page.locator("table:has(#archived-list)").evaluate("""table => {
          const card = table.closest('section');
          return card.scrollWidth <= card.clientWidth;
        }""")
        _assert_quiet(watched)


@pytest.mark.parametrize("action", ["archive", "restore"])
def test_archive_error_shows_server_message_without_changing_row(
        tmp_path: Path, browser: Browser, result: SchemeResult, action: str) -> None:
    run_id = "a" * 32
    _bundle(tmp_path, result, run_id)
    _bundle(tmp_path / "archive", result, run_id)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        if action == "restore":
            page.locator("#archived summary").click()
        expected = ("封存區已經有同代號，沒有搬動任何檔案" if action == "archive" else
                    "結果或計算資料夾已經有同代號，沒有搬動任何檔案")
        row = _row(page, run_id, archived=action == "restore")
        row.get_by_role("button", name="封存" if action == "archive" else "搬回", exact=True).click()
        expect(page.locator("#messages")).to_have_text(expected)
        expect(row).to_be_visible()
        assert watched.page_errors == []
        assert all("409" in message for message in watched.console_errors)
