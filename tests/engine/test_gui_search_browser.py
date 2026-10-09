"""真伺服器與瀏覽器：搜尋各塊、缺檔、輪詢與過期資料標示。"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest
from playwright.sync_api import Browser, Route

from aosr.search.run import RefineStatus, SearchStatus
from aosr.search.modal_record import ModalSummary, role_inputs, write_summary
from aosr.search.outer_status import snapshot_of
from tests.engine._search_modal_cases import prepared
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._search_run_cases import FakeCompute, make_store, run
from tests.engine._search_blocked_cases import SavedFurnitureCompute, blocked_store
FURNITURE_MODEL_NOTE = "家具模型：近似"  # 決策紙第 13 條原文。
from aosr.reporting.scheme import Scheme
from aosr.reporting.result import save_result
from tests.engine.test_scheme_pipeline import _run_control
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


@pytest.mark.parametrize("blocked", [False, True])
def test_furniture_notes_stay_in_existing_search_blocks_and_screenshot(
    tmp_path: Path, browser: Browser, blocked: bool,
) -> None:
    store, registry = blocked_store(tmp_path, budget=3, blocked=blocked)
    status = run(store, registry, SavedFurnitureCompute(store) if blocked else FakeCompute(store))
    assert status.best_trial is not None
    from aosr.search import layout, ledger

    row = next(item for item in ledger.read_for(store).rows if item.trial_number == status.best_trial)
    params = layout.LayoutParams(*(row.params_m[key] for key in ("front_distance", "spacing", "listening_distance")))
    placement = layout.place(store.project, store.settings.layout, params)
    scheme = layout.to_scheme(store.project, placement, f"{store.search_id}-trial-{status.best_trial:06d}")
    # 完整管線與真正家具幾何，沿用既有 FEM 控制組替身；截圖不留半份結果的讀取錯。
    save_result(_run_control(Scheme.model_validate(scheme.model_dump())), store.candidate_path(status.best_trial))
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.locator("#search-best").wait_for()
        for key in ("search-best", "crossover", "stability"):
            assert FURNITURE_MODEL_NOTE in page.locator(f"#{key}").inner_text()
        assert FURNITURE_MODEL_NOTE not in page.locator("#refine-best").inner_text()
        page.wait_for_function("() => document.querySelector('#best-frequency canvas') !== null")
        assert FURNITURE_MODEL_NOTE in page.locator("#best-frequency").inner_text()
        assert set(page.locator("#live-best .chart-error").all_inner_texts()) == {""}
        assert ("原方案不符合擺位要求" in page.locator("#stage").inner_text()) == blocked
        if blocked:
            assert "原方案：跳過；原方案不符合擺位要求" in page.locator("#modal").inner_text()
            assert "原方案：未開始" not in page.locator("#modal").inner_text()
        data = page.request.get(f"{base}/api/searches/{store.search_id}").json()
        assert {block["key"] for block in data["blocks"]} == {
            "stage", "counts", "timings", "updated", "search-best", "refine-best",
            "crossover", "stability", "reasons", "modal", "identity",
        }
        page.screenshot(path=str(tmp_path / f"559-furniture-{'blocked' if blocked else 'clear'}.png"), full_page=True)
        _assert_text_is_formatted(page)
        _assert_quiet(watched)


def test_modal_attachment_polling_and_screenshot(tmp_path: Path, browser: Browser) -> None:
    store, _, status = prepared(tmp_path)
    roles = tuple(i.record for i in role_inputs(store, status))
    summary = ModalSummary(cache_dir=str(tmp_path / "modal-cache"), conclusion=status.outer.conclusion,
        snapshot=snapshot_of(status), roles=(roles[0].model_copy(update={"state": "running"}), *roles[1:]))
    write_summary(store.path, summary)
    with _serve(tmp_path) as base, _open(browser, f"{base}/searches/{store.search_id}", viewport_width=1440) as watched:
        page = watched.page
        page.locator("#modal").wait_for()
        assert "不計分、不改名次" in page.locator("#modal").inner_text()
        assert "上次沒做完（可能進行中或被中斷）" in page.locator("#modal").inner_text()
        assert "開始過、沒有收尾紀錄" in page.locator("#modal").inner_text()
        assert "這是上次收尾時的診斷，之後搜尋又動過" not in page.locator("#modal").inner_text()
        page.locator("#modal").screenshot(path=str(tmp_path / "683-modal-no-completion.png"))
        finished = summary.model_copy(update={"completed": True, "roles": (
            roles[0].model_copy(update={"state": "diagnosed_not_scored"}),
            roles[1].model_copy(update={"state": "stopped", "reason_text": "已停止，搜尋結果保留"}),
            roles[2].model_copy(update={"state": "failed", "reason_text": "求解器錯誤原文"}))})
        write_summary(store.path, finished)
        page.wait_for_function("() => document.getElementById('modal').textContent.includes('已診斷不計分')", timeout=12000)
        shown = page.locator("#modal").inner_text()
        assert "原方案" in shown and "搜尋第一名" in shown and "細算第一名" in shown
        assert "已停止" in shown and "求解器錯誤原文" in shown
        assert "這是上次收尾時的診斷，之後搜尋又動過" not in shown
        store.status_path.write_text(status.model_copy(update={"asked": status.asked + 1}).model_dump_json())
        page.wait_for_function("() => document.getElementById('modal').textContent.includes('這是上次收尾時的診斷，之後搜尋又動過')", timeout=12000)
        page.locator("#modal").screenshot(path=str(tmp_path / "683-modal-stale.png"))
        modal_box, stage_box = page.locator("#modal").bounding_box(), page.locator("#stage").bounding_box()
        assert modal_box is not None and stage_box is not None
        assert modal_box["width"] == stage_box["width"]
        page.screenshot(path=str(tmp_path / "683-modal-progress.png"), full_page=True)
        page.locator("#modal").screenshot(path=str(tmp_path / "683-modal-block.png"))
        _assert_text_is_formatted(page)
        _assert_quiet(watched)
