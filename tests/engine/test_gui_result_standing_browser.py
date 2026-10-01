"""#577 真瀏覽器檢查四級提示、重算按鈕與首頁欄名。"""
from pathlib import Path

import pytest
from playwright.sync_api import Browser

from tests.engine._gui_cache import gui_startup_identity_memo
from aosr.reporting.result import SchemeResult, save_result
from tests.engine.test_gui_browser import RUN_ID, _assert_quiet, _open, _save, _serve, browser
from tests.engine.test_gui_result_standing import TEXT, variant_result
from tests.engine.test_scheme_repair import result


@pytest.mark.parametrize("standing", ["needs_physics", "remeasured"])
def test_browser_standing_and_rerun(tmp_path: Path, browser: Browser,
                                    result: SchemeResult, standing: str) -> None:
    _save(tmp_path, variant_result(result, standing))
    with _serve(tmp_path) as base, _open(browser, f"{base}/results/{RUN_ID}",
                                         viewport_width=1440) as watched:
        page = watched.page
        page.locator("#content").wait_for(state="visible")
        assert page.locator("#fingerprint-status").inner_text() == TEXT[standing]
        assert page.locator("#fingerprint-rerun").is_visible() == (standing == "needs_physics")
        _assert_quiet(watched)


def test_browser_home_physics_and_settings_columns(tmp_path: Path, browser: Browser,
                                                    result: SchemeResult) -> None:
    _save(tmp_path, result)
    for standing, digit in (("remeasured", "a"), ("needs_physics", "b")):
        saved = variant_result(result, standing)
        renamed = SchemeResult.model_validate_json(saved.model_dump_json().replace(
            result.scheme.scheme_id, f"{standing}-wall"))
        save_result(renamed, tmp_path / "results" / f"{digit * 32}{'.json'}")
    with _serve(tmp_path) as base, _open(browser, base, viewport_width=1440) as watched:
        watched.page.locator("#results-list tr").first.wait_for()
        section = watched.page.locator("#results-list").locator("xpath=ancestor::section")
        assert section.evaluate("element => element.scrollWidth <= element.clientWidth")
        headers = watched.page.locator("#results-list").locator("xpath=../thead//th").all_inner_texts()
        assert "物理" in headers and "評分設定" in headers
        assert "計算版本" not in headers and "品質登記簿" not in headers
        _assert_quiet(watched)
