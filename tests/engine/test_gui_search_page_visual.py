"""1440 寬進度頁：各卡、身分長碼、原因都在畫面內且不重疊。"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Browser

from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._search_run_cases import FakeCompute, make_store, run
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser


def test_progress_cards_fit_desktop_without_overlap(tmp_path: Path, browser: Browser) -> None:
    store, registry = make_store(tmp_path, budget=3)
    run(store, registry, FakeCompute(store))
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.locator("#identity").wait_for()
        boxes = page.locator("#search-content section").evaluate_all(
            "nodes => nodes.map(n => { const r=n.getBoundingClientRect(); return {id:n.id,x:r.x,y:r.y,w:r.width,h:r.height,scroll:n.scrollWidth,client:n.clientWidth}; })")
        assert boxes
        for item in boxes:
            assert item["x"] >= 0 and item["x"] + item["w"] <= 1440
            assert item["scroll"] <= item["client"]
            for other in boxes:
                if item["id"] != other["id"]:
                    assert item["x"] + item["w"] <= other["x"] or other["x"] + other["w"] <= item["x"] or \
                        item["y"] + item["h"] <= other["y"] or other["y"] + other["h"] <= item["y"]
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
