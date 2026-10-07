"""真瀏覽器核交接提醒、距離、輪詢、正常內容不標紅，並留本次暫存截圖。"""
from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.search import crossover_sensitivity as module
from aosr.search.crossover_record import INCOMPLETE, STALE, VERDICTS, summary_path, write_summary
from tests.engine._crossover_cases import evaluate, prepared
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser


def test_crossover_browser_verdicts_distances_and_screenshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            browser: Browser) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        block = page.locator("#crossover")
        block.wait_for()
        shown = block.inner_text()
        assert VERDICTS[summary.verdict] in shown
        assert "試算 9" in shown and "喇叭 left 相距" in shown and "主位相距" in shown
        assert not block.locator("p.notice").all()
        block.screenshot(path=str(tmp_path / "688-crossover-sensitive.png"))
        for verdict in ("stable", "unverified"):
            changed = summary.model_copy(update={"verdict": verdict})
            write_summary(store.path, changed)
            page.wait_for_function("text => document.getElementById('crossover').textContent.includes(text)", arg=VERDICTS[verdict], timeout=12000)
            assert not block.locator("p.notice").all()
        write_summary(store.path, summary.model_copy(update={"completed": False}))
        page.wait_for_function("text => document.getElementById('crossover').textContent.includes(text)", arg=INCOMPLETE, timeout=12000)
        assert not block.locator("p.notice").all()
        block.screenshot(path=str(tmp_path / "688-crossover-incomplete.png"))
        store.status_path.write_text(status.model_copy(update={"asked": status.asked + 1}).model_dump_json())
        page.wait_for_function("text => document.getElementById('crossover').textContent.includes(text)", arg=STALE, timeout=12000)
        assert not block.locator("p.notice").all()
        block.screenshot(path=str(tmp_path / "688-crossover-stale.png"))
        summary_path(store.path).write_text("{")
        page.wait_for_function("() => document.querySelector('#crossover p.notice') !== null", timeout=12000)
        assert "讀不到" in block.inner_text()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
