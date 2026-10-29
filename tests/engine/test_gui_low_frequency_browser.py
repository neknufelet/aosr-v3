"""#558 第二段 B：真伺服器與瀏覽器展開低頻詳情；截圖寫進這題自己的暫存目錄，要親眼看就用 --basetemp 指一個資料夾。"""
from pathlib import Path

from playwright.sync_api import Browser

from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_load_result_memo, gui_startup_identity_memo
from tests.engine.test_gui_browser import RUN_ID, _assert_quiet, _open, _save, _serve, browser
from tests.engine.test_gui_compare_routes import _files
from tests.engine.test_gui_compare_view import _renamed
from tests.engine.test_low_frequency_response_display import assert_details, low_frequency
from tests.engine.test_low_frequency_response_reload import saved as saved

OTHER_ID = "b" * 32


def test_low_frequency_row_and_expanded_details(tmp_path: Path, browser: Browser, saved: SchemeResult) -> None:
    """結果頁與比較頁各開一次（#972 由兩格合成一題、共用一台伺服器）：低頻那一列寫「已算代價」，展開詳情逐項照評估。"""
    _save(tmp_path, saved)
    with _serve(tmp_path) as base:
        _files(tmp_path, _renamed(saved, "other"), OTHER_ID)
        for comparison in (False, True):
            path = f"/compare/{RUN_ID}/{OTHER_ID}" if comparison else f"/results/{RUN_ID}"
            with _open(browser, base + path, viewport_width=1440) as watched:
                page = watched.page
                page.locator("#content").wait_for(state="visible")
                row = page.locator("#categories tr").filter(has=page.get_by_text("低頻頻響", exact=True))
                assert "已算代價" in row.inner_text(), path
                details = row.locator("details")
                details.locator("summary").click()
                text = details.inner_text()
                assert_details(text, low_frequency(saved))
                assert "理想聲源" not in text
                shot = tmp_path / ("compare.png" if comparison else "result.png")
                page.locator("#categories").locator("..").screenshot(path=str(shot))
                assert shot.stat().st_size > 0
                _assert_quiet(watched)
