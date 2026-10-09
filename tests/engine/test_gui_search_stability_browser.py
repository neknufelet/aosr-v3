"""真無頭瀏覽器驗共用文字、輪詢、逐行標紅與搜尋頁截圖。"""
from pathlib import Path

import pytest
from playwright.sync_api import Browser

from aosr.search.outer_status import OuterStatus
from aosr.search.placement_stability_record import STALE, summary_path, write_summary
from aosr.search.report_stability import ABSENT, stability_report
from tests.engine._stability_attach_cases import ShiftCompute, attach
from tests.engine._stability_report_cases import report_case
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser


@pytest.mark.parametrize("scenario,width", [("done", 1440), ("running", 900), ("skipped", 1440)])
def test_stability_browser_states_and_screenshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                browser: Browser, scenario: str, width: int) -> None:
    store, registry, status, summary, compute = report_case(tmp_path, monkeypatch)
    if scenario == "running":
        write_summary(store.path, compute.snapshots[1])
    elif scenario == "skipped":
        status = status.model_copy(update={"outer": OuterStatus(conclusion="user_stopped")})
        store.status_path.write_text(status.model_dump_json())
        summary = attach(store, registry, status, ShiftCompute(store))
    expected = stability_report(store, status)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=width) as watched:
        page = watched.page
        block = page.locator("#stability")
        block.wait_for()
        assert block.locator("p").all_text_contents() == list(expected.lines)
        assert not block.locator("p.notice").all()
        if scenario == "done":
            assert summary.arithmetic is not None and summary.arithmetic.score_winner != summary.arithmetic.minimax_winner
            assert any(p.model_discontinuity for p in summary.points)
            assert "模型不連續" in block.inner_text()
        elif scenario == "running":
            running = compute.snapshots[1]
            assert f"已算 {running.computed_points}／共 {running.total_points} 點" in block.inner_text()
        else:
            assert summary.reason_text in block.inner_text()
        page.screenshot(path=str(tmp_path / f"699-stability-{scenario}.png"), full_page=True)
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_stability_browser_polling_and_warning_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, browser: Browser) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch, fail_after=1)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}") as watched:
        page = watched.page
        block = page.locator("#stability")
        block.wait_for()
        assert block.locator("p.notice").all_text_contents() == [line for line in stability_report(store, status).lines
                                                                 if "讀不到" in line or "失敗" in line]
        # 停止原文同樣由寫入端記錄，暫時與舊了不能被「中斷」字眼誤標紅。
        from aosr.search.placement_stability_attach import record_stability_error
        record_stability_error(store, status, KeyboardInterrupt())
        page.wait_for_function("() => document.querySelector('#stability').textContent.includes('狀態：停止')", timeout=12000)
        assert not block.locator("p.notice").all()
        store.status_path.write_text(status.model_copy(update={"outer": OuterStatus(conclusion="refine_budget")}).model_dump_json())
        page.wait_for_function("text => document.querySelector('#stability').textContent.includes(text)", arg=STALE, timeout=12000)
        assert not block.locator("p.notice").all()
        summary_path(store.path).unlink()
        page.wait_for_function("text => document.querySelector('#stability').textContent.includes(text)", arg=ABSENT, timeout=12000)
        assert not block.locator("p.notice").all()
        write_summary(store.path, summary)
        summary_path(store.path).write_text("{")
        page.wait_for_function("() => document.querySelector('#stability p.notice') !== null", timeout=12000)
        assert "讀不到" in block.inner_text()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
