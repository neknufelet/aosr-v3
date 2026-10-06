"""真伺服器與瀏覽器：搜尋各塊、缺檔、輪詢與過期資料標示。"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

from playwright.sync_api import Browser, Route

from aosr.search.run import RefineStatus, SearchStatus
from aosr.search.modal_record import ModalRole, ModalSummary, write_summary
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._search_run_cases import FakeCompute, make_store, run
from tests.engine.test_gui_browser import _assert_quiet, _assert_text_is_formatted, _open, _serve, browser


def test_progress_sections_and_polling_update(tmp_path: Path, browser: Browser) -> None:
    store, registry = make_store(tmp_path, budget=3)
    status = run(store, registry, FakeCompute(store))
    descriptor = store.open_folder_lock()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        running = status.model_copy(update={"state": "running", "message": "搜尋進行中"})
        store.status_path.write_text(running.model_dump_json())
        with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}") as watched:
            page = watched.page
            page.locator("#stage").wait_for()
            data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
            for block in data["blocks"]:
                shown = page.locator(f"#{block['key']}").inner_text()
                assert block["title"] in shown
                # 每塊的固定原文在頁上；最後更新的距今秒數會前進，另外核對。
                if block["key"] != "updated":
                    assert all(line in shown for line in block["lines"]), shown
            assert "搜尋：進行中" in page.locator("#stage").inner_text()
            store.status_path.write_text(status.model_dump_json())
            page.wait_for_function("() => document.getElementById('stage').textContent.includes('因預算停止')", timeout=12000)
            assert "本次預算內最佳" in page.locator("#search-best").inner_text()
            assert "距今" in page.locator("#updated").inner_text()
            _assert_text_is_formatted(page)
            _assert_quiet(watched)
    finally:
        os.close(descriptor)


def test_missing_or_broken_data_remains_honest(tmp_path: Path, browser: Browser) -> None:
    store, _ = make_store(tmp_path)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}") as watched:
        page = watched.page
        page.locator("#stage").wait_for()
        assert "讀不到：檔案不存在（status.json）" in page.locator("#stage").inner_text()
        assert "讀不到" in page.locator("#counts").inner_text()
        assert store.identity.physics_identity in page.locator("#identity").inner_text()
        store.status_path.write_text(SearchStatus(refine=RefineStatus(state="running")).model_dump_json())
        page.wait_for_function("() => document.getElementById('stage').textContent.includes('上次中斷了')", timeout=12000)
        assert "細算中斷" in page.locator("#stage").inner_text()
        store.status_path.write_text("{")
        page.wait_for_function("() => document.getElementById('stage').textContent.includes('JSON 資料損壞')", timeout=12000)
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_server_unreachable_keeps_old_data_with_timestamp(tmp_path: Path, browser: Browser) -> None:
    store, _ = make_store(tmp_path)
    store.status_path.write_text(SearchStatus(state="budget_exhausted").model_dump_json())
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}") as watched:
        page = watched.page
        page.locator("#stage").wait_for()
        previous = page.locator("#search-content").inner_text()
        def offline(route: Route) -> None:
            route.abort("failed")
        page.route("**/api/searches/*", offline)
        page.wait_for_function("() => document.getElementById('connection').textContent.includes('讀不到伺服器，上面是')", timeout=12000)
        assert page.locator("#search-content").inner_text() == previous
        assert "的資料" in page.locator("#connection").inner_text()
        _assert_text_is_formatted(page)
        assert watched.page_errors == []
        assert all("ERR_FAILED" in text for text in watched.console_errors)


def test_search_list_links_to_individual_progress(tmp_path: Path, browser: Browser) -> None:
    store, _ = make_store(tmp_path)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches") as watched:
        page = watched.page
        link = page.locator(f"#search-list a[href='/searches/{store.search_id}']")
        link.wait_for()
        link.click()
        page.locator("#stage").wait_for()
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_modal_attachment_polling_and_screenshot(tmp_path: Path, browser: Browser) -> None:
    store, registry = make_store(tmp_path, budget=3)
    run(store, registry, FakeCompute(store))
    summary = ModalSummary(cache_dir=str(tmp_path / "modal-cache"), roles=(
        ModalRole(role="baseline", state="running", reason_text="原方案診斷中"),
        ModalRole(role="search_best", trial_number=0),
        ModalRole(role="refine_best", trial_number=1, temporary=True)))
    write_summary(store.path, summary)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.locator("#modal").wait_for()
        assert "不計分、不改名次" in page.locator("#modal").inner_text()
        assert "上次沒做完（可能進行中或被中斷）" in page.locator("#modal").inner_text()
        finished = summary.model_copy(update={"completed": True, "roles": (
            ModalRole(role="baseline", state="diagnosed_not_scored"),
            ModalRole(role="search_best", trial_number=0, state="stopped", reason_text="已停止，搜尋結果保留"),
            ModalRole(role="refine_best", trial_number=1, state="failed", temporary=True, reason_text="求解器錯誤原文"))})
        write_summary(store.path, finished)
        page.wait_for_function("() => document.getElementById('modal').textContent.includes('已診斷不計分')", timeout=12000)
        shown = page.locator("#modal").inner_text()
        assert "原方案" in shown and "搜尋第一名" in shown and "細算第一名" in shown
        assert "已停止" in shown and "求解器錯誤原文" in shown
        modal_box, stage_box = page.locator("#modal").bounding_box(), page.locator("#stage").bounding_box()
        assert modal_box is not None and stage_box is not None
        assert modal_box["width"] == stage_box["width"]
        page.screenshot(path=str(tmp_path / "683-modal-progress.png"), full_page=True)
        page.locator("#modal").screenshot(path=str(tmp_path / "683-modal-block.png"))
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
