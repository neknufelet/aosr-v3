"""#577 真瀏覽器檢查四級提示、重算按鈕與首頁欄名。"""
from pathlib import Path

from playwright.sync_api import Browser

from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._input_page_steps import open_results
from aosr.reporting.result import SchemeResult, save_result
from tests.engine.test_gui_browser import RUN_ID, _assert_quiet, _open, _save, _serve, browser
from tests.engine.test_gui_result_standing import TEXT, variant_result
from tests.engine.test_scheme_repair import result


def test_browser_standing_and_rerun(tmp_path: Path, browser: Browser, result: SchemeResult) -> None:
    # #972：原本另跑 remeasured 一格（純文字、沒有重算鈕）。那一格的字與「不給重算」由路由考卷
    # test_gui_result_standing::test_result_standing_text_and_rerun[remeasured] 守；網頁在指紋相同時不畫重算鈕由
    # test_gui_browser::test_result_page_shows_fingerprint_rerun_only_when_different 守。
    standing = "needs_physics"
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
        # #1265 單元 4：結果清單搬進結果抽屜，欄名變成每一列小字裡的「評分：」「物理：」兩段。
        open_results(watched.page)
        watched.page.locator("#results-list tr").first.wait_for()
        section = watched.page.locator("#results-list").locator("xpath=ancestor::section")
        assert section.evaluate("element => element.scrollWidth <= element.clientWidth")
        for line in watched.page.locator("#results-list .res-line").all_inner_texts():
            assert "物理：" in line and "評分：" in line
            assert "計算版本" not in line and "品質登記簿" not in line
        _assert_quiet(watched)
