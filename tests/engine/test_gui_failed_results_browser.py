"""瀏覽器能看到失敗產物，但不把它當完成結果或比較勝負。"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from playwright.sync_api import Browser, Locator, Page, expect

from aosr.gui.labels import RESULT_RUN_LABELS, RUN_EXIT_TEXT
from aosr.reporting.result import SchemeResult, save_result
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine.test_gui_browser import _open, _serve, browser, result
from tests.engine.test_gui_compare_routes import pair
from tests.engine.test_gui_compare_browser import _has_lines
from tests.engine.test_gui_failed_results import _client, _path, _runner, _start, _wait_file, _wait_state


def _populate(tmp_path: Path, result: SchemeResult, other: SchemeResult) -> tuple[str, str]:
    runner = _runner(tmp_path, {result.scheme.scheme_id: (result, 3, False),
                                other.scheme.scheme_id: (other, 0, False)})
    with _client(tmp_path, runner) as client:
        failed, done = _start(client, result), _start(client, other)
        _wait_state(client, failed, "failed")
        _wait_state(client, done, "done")
    return failed, done


def _populate_four(tmp_path: Path, result: SchemeResult, other: SchemeResult
                   ) -> tuple[str, str, str, str]:
    failed, done = _populate(tmp_path, result, other)
    runner = _runner(tmp_path, {other.scheme.scheme_id: (other, 3, True)})
    with _client(tmp_path, runner) as client:
        stopped = _start(client, other)
        try:
            _wait_file(_path(tmp_path, "results", stopped))
            response = client.post(f"/api/runs/{stopped}/stop", json={})
            assert response.status_code == 200 and response.json()["status"] == "stopped"
        finally:
            client.post(f"/api/runs/{stopped}/stop", json={})
    untracked = "d" * 32
    save_result(result, _path(tmp_path, "results", untracked))
    return failed, stopped, done, untracked


def _result_row(page: Page, run_id: str) -> Locator:
    return page.locator("#results-list tr").filter(has=page.locator(f'a[href="/results/{run_id}"]'))


def test_home_marks_four_statuses_and_compare_selection(
        tmp_path: Path, browser: Browser, result: SchemeResult,
        pair: tuple[SchemeResult, SchemeResult]) -> None:
    failed, stopped, done, untracked = _populate_four(tmp_path, result, pair[1])
    with _serve(tmp_path) as base, _open(browser, base) as watched:
        page = watched.page
        failed_row, done_row = _result_row(page, failed), _result_row(page, done)
        for run_id, text in ((failed, "失敗：內容可能不完整"), (stopped, "已停止：內容可能不完整")):
            row = _result_row(page, run_id)
            expect(row.locator("td").first.locator(".run-status")).to_have_text(text)
            assert "not-finished" in (row.get_attribute("class") or "").split()
        for run_id in (done, untracked):
            row = _result_row(page, run_id)
            expect(row).to_be_visible()
            assert "not-finished" not in (row.get_attribute("class") or "").split()
            assert row.locator("td").first.evaluate("node => node.querySelector('.run-status') === null")
        assert failed_row.evaluate("node => getComputedStyle(node).backgroundColor") != (
            done_row.evaluate("node => getComputedStyle(node).backgroundColor"))
        _result_row(page, stopped).get_by_role("button", name="選為 A", exact=True).click()
        expect(page.locator("#compare-a")).to_contain_text("已停止：內容可能不完整")
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


def test_two_unfinished_compare_shows_both_notices(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult]) -> None:
    failed, stopped, _, _ = _populate_four(tmp_path, *pair)
    with _serve(tmp_path) as base, _open(browser, f"{base}/compare/{failed}/{stopped}") as watched:
        notices = watched.page.locator("#run-notices")
        expect(notices).to_contain_text("A：這一筆計算回報失敗（離開碼 3）")
        expect(notices).to_contain_text("B：這一筆計算被停止")
        expect(watched.page.locator("#table-verdict")).to_contain_text("兩份計算都沒有正常完成")
        assert watched.page_errors == []


# 頁面載入前包住畫字入口；只記下載用、沒有掛進頁面的畫布。
PNG_TEXT_RECORDER = """(() => {
  window.pngTexts = [];
  const original = CanvasRenderingContext2D.prototype.fillText;
  CanvasRenderingContext2D.prototype.fillText = function(text, x, y, ...rest) {
    if (!this.canvas.isConnected) window.pngTexts.push({text: String(text), x, y,
      width: this.measureText(String(text)).width, color: this.fillStyle,
      font: this.font, canvasWidth: this.canvas.width, canvasHeight: this.canvas.height});
    return original.call(this, text, x, y, ...rest);
  };
  const drawImage = CanvasRenderingContext2D.prototype.drawImage;
  CanvasRenderingContext2D.prototype.drawImage = function(image, x, y, ...rest) {
    if (!this.canvas.isConnected) window.pngChartTop = y;
    return drawImage.call(this, image, x, y, ...rest);
  };
})();"""


def _assert_png_texts(page: Page, expected: list[str]) -> None:
    texts = cast(list[dict[str, object]], page.evaluate("() => window.pngTexts"))
    joined = "".join(str(item["text"]) for item in texts)
    for notice in expected:
        assert notice in joined
    if expected:
        assert any(str(item["text"]).startswith("A：") and "失敗" in str(item["text"])
                   for item in texts)
        warning_color = page.locator("#run-notices .notice").first.evaluate(
            "node => getComputedStyle(node).color")
        color = page.evaluate("""color => {
          const ctx = document.createElement('canvas').getContext('2d');
          ctx.fillStyle = color; return ctx.fillStyle;
        }""", warning_color)
        warnings = [item for item in texts if item["color"] == color]
        assert "".join(str(item["text"]) for item in warnings) == "".join(expected)
        chart_top = page.evaluate("() => window.pngChartTop")
        assert all(float(cast(float, item["y"])) < chart_top for item in warnings)
        assert float(cast(float, warnings[0]["y"])) > float(cast(float, texts[0]["y"]))
    else:
        assert all("沒有正常完成" not in str(item["text"]) and "失敗" not in str(item["text"])
                   for item in texts)
    for item in texts:
        x, y, width = (float(cast(float, item[key])) for key in ("x", "y", "width"))
        assert 0 <= x and x + width <= float(cast(float, item["canvasWidth"]))
        font_size = float(str(item["font"]).split("px")[0])
        assert 0 < y and y + font_size * 0.25 <= float(cast(float, item["canvasHeight"]))


@pytest.mark.parametrize("width, scale", [(1440, 1), (390, 2)])
def test_png_download_keeps_all_failure_notices_and_finished_control(
        tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult],
        width: int, scale: int) -> None:
    failed, stopped, done, untracked = _populate_four(tmp_path, *pair)
    runner = _runner(tmp_path, {pair[0].scheme.scheme_id: (pair[0], 0, False)})
    with _client(tmp_path, runner) as client:
        finished_a = _start(client, pair[0])
        _wait_state(client, finished_a, "done")
    failed_notice = "A：" + RESULT_RUN_LABELS["failed"][2].format(exit_text=RUN_EXIT_TEXT.format(code=3))
    stopped_notice = "B：" + RESULT_RUN_LABELS["stopped"][2]
    with _serve(tmp_path) as base:
        page = browser.new_page(viewport={"width": width, "height": 900}, device_scale_factor=scale)
        try:
            page.add_init_script(PNG_TEXT_RECORDER)
            for left, right, expected in ((failed, done, [failed_notice]),
                                           (failed, stopped, [failed_notice, stopped_notice]),
                                           (untracked, done, []), (finished_a, done, [])):
                page.goto(f"{base}/compare/{left}/{right}", wait_until="networkidle")
                _has_lines(page)
                page.evaluate("() => { window.pngTexts = []; }")
                with page.expect_download() as event:
                    page.get_by_role("button", name="下載曲線圖片（PNG）").click()
                assert event.value.path().read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
                _assert_png_texts(page, expected)
        finally:
            page.close()


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
