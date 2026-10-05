"""結果頁與比較頁真的畫出能力表的兩份原文清單。"""
from __future__ import annotations

from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.gui.capability_view import capability_lists
from aosr.reporting.result import SchemeResult, save_result
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> tuple[SchemeResult, SchemeResult]:
    return (shared_control_result(tmp_path_factory, worker_id, "wall-1"),
            shared_control_result(tmp_path_factory, worker_id, "wall-2"))


@pytest.mark.parametrize("kind", ["result", "compare"])
def test_capability_lists_visible_on_both_pages(tmp_path: Path, browser: Browser, pair: tuple[SchemeResult, SchemeResult], kind: str) -> None:
    a, b = "b" * 32, "c" * 32
    (tmp_path / "results").mkdir()
    for run_id, result in zip((a, b), pair, strict=True):
        save_result(result, tmp_path / "results" / f"{run_id}.json")
    route = f"/results/{a}" if kind == "result" else f"/compare/{a}/{b}"
    expected = capability_lists(load_capabilities(config_path("capabilities.toml")))
    with _serve(tmp_path) as base, _open(browser, base + route) as watched:
        page = watched.page
        page.locator("#not-modeled li").first.wait_for()
        for field, items in expected.items():
            actual = page.locator(f"#{field.replace('_', '-')} li").all_inner_texts()
            assert actual == list(dict.fromkeys(items)) if kind == "compare" else actual == list(items)
        assert "模型沒算到" in page.locator("body").inner_text()
        assert "要人工確認" in page.locator("body").inner_text()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
