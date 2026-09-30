"""首頁能按封存、搬回，並保住比較選擇與卡片內的版面。"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import cast

import pytest
from playwright.sync_api import Browser, Locator, Page, expect

from aosr.reporting.result import SchemeResult
from tests.engine.test_gui_archive import _bundle
from tests.engine.test_gui_browser import BUTTON_LOOK_JS, _assert_quiet, _open, _serve, browser, result
from tests.engine.test_gui_failed_results import _path, _runner


def _row(page: Page, run_id: str, archived: bool = False) -> Locator:
    listing = "archived-list" if archived else "results-list"
    return page.locator(f'#{listing} tr[data-run-id="{run_id}"]')


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
        run_id = str(page.locator("#result-link").get_attribute("href")).rsplit("/", 1)[1]
        _row(page, run_id).get_by_role("button", name="封存", exact=True).click()
        expect(page.locator("#result-link")).to_be_hidden()
        expect(page.locator("#result-stale")).to_be_hidden()
        _assert_quiet(watched)


def test_long_scheme_table_fits_card_and_date_stays_on_one_line(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    long = result.model_copy(update={"scheme": result.scheme.model_copy(
        update={"scheme_id": "reference-room-original2"})})
    for run_id, item, status in (("a" * 32, long, "done"), ("b" * 32, result, "failed"),
                                 ("c" * 32, result, "none")):
        _bundle(tmp_path, item, run_id, status)
        stamp = datetime(2026, 9, 30, 18, 47).timestamp()
        os.utime(_path(tmp_path, "results", run_id), (stamp, stamp))
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        page = watched.page
        row = _row(page, "a" * 32)
        expect(row).to_contain_text("reference-room-original2")
        geometry = cast(dict[str, float], page.locator("table:has(#results-list)").evaluate("""table => ({
          right: table.getBoundingClientRect().right,
          cardRight: table.closest('section').getBoundingClientRect().right,
          width: table.getBoundingClientRect().width,
          cardWidth: table.closest('section').getBoundingClientRect().width
        })"""))
        assert geometry["right"] <= geometry["cardRight"]
        assert geometry["width"] <= geometry["cardWidth"]
        date = row.locator("td").nth(1).locator(".finished-date")
        expect(date).to_have_text("2026-09-30")
        assert date.evaluate("""node => {
          const range = document.createRange(); range.selectNodeContents(node);
          const rects = [...range.getClientRects()];
          return rects.length > 0 && rects.every(rect => Math.abs(rect.top - rects[0].top) < 1);
        }""")
        _assert_quiet(watched)


def test_archive_error_shows_server_message_without_changing_row(
        tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    run_id = "a" * 32
    _bundle(tmp_path, result, run_id)
    _bundle(tmp_path / "archive", result, run_id)
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        expected = page.request.post(f"{base}/api/results/{run_id}/archive", data={}).json()["error"]
        _row(page, run_id).get_by_role("button", name="封存", exact=True).click()
        expect(page.locator("#messages")).to_have_text(expected)
        expect(_row(page, run_id)).to_be_visible()
        assert watched.page_errors == []
        assert all("409" in message for message in watched.console_errors)
