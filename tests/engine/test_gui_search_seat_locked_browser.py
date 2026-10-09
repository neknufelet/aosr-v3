"""本次沙箱伺服器的座位鎖定搜尋頁；截圖檢查文字、推出標示與橫向溢出。"""
from pathlib import Path

from playwright.sync_api import Browser

from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._seat_locked_cases import SEAT_LINE, locked_store
from tests.engine._search_run_cases import FakeCompute, run
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser


def test_locked_seat_line_derived_label_and_screenshot(tmp_path: Path, browser: Browser) -> None:
    store, registry = locked_store(tmp_path)
    run(store, registry, FakeCompute(store))
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.locator("#stage").wait_for()
        assert SEAT_LINE in page.locator("#stage").inner_text()
        assert "聆聽距離（由座位推出）：" in page.locator("#search-best").inner_text()
        data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
        assert {block["key"] for block in data["blocks"]} == {
            "stage", "counts", "timings", "updated", "search-best", "refine-best", "crossover", "reasons", "modal", "identity"}
        assert page.locator("#stage").evaluate("e => e.scrollWidth <= e.clientWidth")
        assert page.locator("#search-best").evaluate("e => e.scrollWidth <= e.clientWidth")
        page.screenshot(path=str(tmp_path / "559-seat-locked-search.png"), full_page=True)
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
