"""搜尋頁「目前階段」那塊的新喇叭一行與座位鎖定那一行（#972 由 test_gui_search_seat_locked_browser 併入）；
真瀏覽器與本次伺服器截圖、不橫向溢出。"""
from pathlib import Path

from playwright.sync_api import Browser

from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._search_furniture_cases import FurnitureFlowCompute
from aosr.search.run import start_search
from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, run
from tests.engine._seat_locked_cases import SEAT_LINE, locked_store
from tests.engine._search_speaker_setup_cases import flow_project, store_for
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser


def test_search_speaker_line_in_stage_and_screenshot(tmp_path: Path, browser: Browser) -> None:
    store, registry = store_for(tmp_path, flow_project("desk"))
    start_search(store, compute=FurnitureFlowCompute(store), probe=lambda: store.identity,
                 registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    text = "喇叭：書架喇叭、放桌面（代表模型，非實際型號）；箱體 寬 0.21 × 深 0.28 × 高 0.355 m，聲學中心離箱底 0.205 m"
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.locator("#stage").wait_for()
        assert text in page.locator("#stage").inner_text()
        data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
        assert {block["key"] for block in data["blocks"]} == {
            "stage", "counts", "timings", "updated", "search-best", "refine-best", "crossover", "stability", "reasons", "modal", "identity"}
        assert page.locator("#stage").evaluate("e => e.scrollWidth <= e.clientWidth")
        page.screenshot(path=str(tmp_path / "559-step5-search-speaker.png"), full_page=True)
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


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
            "stage", "counts", "timings", "updated", "search-best", "refine-best", "crossover", "stability", "reasons", "modal", "identity"}
        assert page.locator("#stage").evaluate("e => e.scrollWidth <= e.clientWidth")
        assert page.locator("#search-best").evaluate("e => e.scrollWidth <= e.clientWidth")
        page.screenshot(path=str(tmp_path / "559-seat-locked-search.png"), full_page=True)
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
