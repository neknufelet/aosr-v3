"""搜尋進度唯讀考卷：帳本、行程鎖與各塊讀取失敗分開。"""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import pytest

from aosr.gui.search_view import build_search_view, folder_process
from aosr.search import ledger
from aosr.search.outer_status import OuterStatus, snapshot_of
from aosr.search.run import RefineStatus, SearchStatus
from tests.engine._search_run_cases import FakeCompute, make_store, run


def _view(path: Path) -> dict[str, object]:
    return build_search_view(path, server_physics="現在物理", server_program="現在程式").model_dump()


def _text(view: dict[str, object]) -> str:
    return json.dumps(view, ensure_ascii=False)


def _snapshot(path: Path) -> dict[str, tuple[int, int]]:
    return {str(item.relative_to(path)): (item.stat().st_mtime_ns, item.stat().st_size)
            for item in (path, *path.rglob("*"))}


def test_missing_status_is_not_running_or_complete(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    view = _view(store.path)
    assert "讀不到：檔案不存在（status.json）" in _text(view)
    assert "現在物理" in _text(view) and store.identity.physics_identity in _text(view)
    assert "未判定" in _text(view)


def test_running_lock_is_observed_without_taking_or_writing_it(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    ledger.create_for(store)
    store.status_path.write_text(SearchStatus(timed_from_start=True).model_dump_json())
    descriptor = store.open_folder_lock()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = _snapshot(store.path)
        assert folder_process(store.path).held is True
        view = _view(store.path)
        assert "搜尋：進行中" in _text(view)
        assert "還沒有時間紀錄" in _text(view)
        assert _snapshot(store.path) == before
    finally:
        os.close(descriptor)


def test_running_without_process_is_interrupted(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    store.status_path.write_text(SearchStatus().model_dump_json())
    assert "上次中斷了，要接續請叫助理" in _text(_view(store.path))
    assert "搜尋：進行中" not in _text(_view(store.path))


@pytest.mark.parametrize("state,label", [("budget_exhausted", "因預算停止"),
                                          ("failed", "搜尋失敗"), ("interrupted", "搜尋中斷"),
                                          ("converged", "達到停止條件（暫行）"),
                                          ("user_stopped", "使用者停止")])
def test_search_stop_states_and_original_reason(tmp_path: Path, state: str, label: str) -> None:
    store, registry = make_store(tmp_path, budget=3)
    status = run(store, registry, FakeCompute(store))
    document = status.model_dump(mode="json") | {"state": state, "message": "原始原因：算到此處停止"}
    store.status_path.write_text(json.dumps(document))
    shown = _text(_view(store.path))
    assert label in shown and "原始原因：算到此處停止" in shown
    assert "篩選分數：" in shown and "公尺" in shown
    if state == "budget_exhausted":
        assert "本次預算內最佳" in shown
    assert "到最後一次存檔為止" in shown and "距今" in shown


@pytest.mark.parametrize("broken", ["{", '{"state":"running","future":true}'])
def test_bad_status_does_not_hide_ledger_or_identity(tmp_path: Path, broken: str) -> None:
    store, registry = make_store(tmp_path, budget=3)
    run(store, registry, FakeCompute(store))
    store.status_path.write_text(broken)
    shown = _text(_view(store.path))
    assert "讀不到：" in shown and "算完" in shown
    assert store.identity.physics_identity in shown
    assert "外圈結論：未判定" in shown


def test_partial_ledger_tail_is_visible_and_untouched(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path, budget=3)
    run(store, registry, FakeCompute(store))
    with store.ledger_path.open("ab") as handle:
        handle.write(b'{"trial_number":')
    before = _snapshot(store.path)
    shown = _text(_view(store.path))
    assert "末列尚未寫完，只顯示完整列" in shown
    assert "算完" in shown
    assert _snapshot(store.path) == before


@pytest.mark.parametrize("reason", ["stable", "refine_budget", "candidates_exhausted", "user_stopped"])
def test_refine_stop_and_outer_conclusion(tmp_path: Path, reason: str) -> None:
    store, _ = make_store(tmp_path)
    status = SearchStatus(state="budget_exhausted", refine=RefineStatus.model_validate({
        "state": "stopped", "stop_reason": reason, "best": "baseline", "best_total_cost": 0.123456789,
        "message": "細算原始訊息", "seconds": {1: 35.0}}))
    outer = OuterStatus(conclusion="refine_budget", message="細算用完上限，未完成", snapshot=snapshot_of(status))
    store.status_path.write_text(status.model_copy(update={"outer": outer}).model_dump_json())
    shown = _text(_view(store.path))
    assert "細算：已停" in shown and "細算原始訊息" in shown
    assert "原方案" in shown and "0.1235" in shown
    assert "細算用完上限，未完成" in shown
    assert "部分" in shown


def test_refine_running_without_lock_is_not_running(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    status = SearchStatus(state="converged", refine=RefineStatus(state="running"))
    store.status_path.write_text(status.model_dump_json())
    shown = _text(_view(store.path))
    assert "細算中斷" in shown and "細算：進行中" not in shown


def test_unreadable_locks_does_not_claim_process_or_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = make_store(tmp_path)
    store.status_path.write_text(SearchStatus().model_dump_json())
    def denied() -> str:
        raise PermissionError("不能讀行程鎖")
    monkeypatch.setattr("aosr.gui.search_view.read_proc_locks", denied)
    assert "判不出有沒有行程在跑" in _text(_view(store.path))
    assert "搜尋：進行中" not in _text(_view(store.path))


def test_counts_and_best_placement_come_from_ledger(tmp_path: Path) -> None:
    from aosr.search.store import candidate_name
    store, _ = make_store(tmp_path)
    book = ledger.create_for(store)
    params = {"front_distance": 1.234567891, "spacing": 2.345678912, "listening_distance": 3.456789123}
    scored = ledger.LedgerRow(batch_index=0, trial_number=0,
                               unit_params_hex={key: (0.5).hex() for key in params}, params_m=params,
                               outcome="scored", score=0.123456789123456, reason=None,
                               violation_m=None, seconds=0.5, result_file=candidate_name(0))
    book.append(scored)
    book.append(scored.model_copy(update={"trial_number": 1, "outcome": "illegal", "score": None,
                                          "reason": "cabinet_outside_room+wall_gap", "violation_m": 0.1, "result_file": None}))
    book.append(scored.model_copy(update={"trial_number": 2, "outcome": "excluded", "score": None,
                                          "reason": "eliminated", "result_file": candidate_name(2)}))
    status = SearchStatus(state="budget_exhausted", asked=3, best_trial=0, best_score=0.123456789123456)
    store.status_path.write_text(status.model_dump_json())
    view = build_search_view(store.path, server_physics="現在", server_program="現在")
    counts = next(block for block in view.blocks if block.key == "counts")
    best = next(block for block in view.blocks if block.key == "search-best")
    assert "算完 2 個；不合法 1 個" in counts.lines
    assert "不合法各原因數：箱體越界：1、離牆間隙不足：1" in counts.lines
    assert "被淘汰或排除 1 個；各區數：淘汰：1" in counts.lines
    assert "本次預算內最佳：試算 0；篩選分數：0.1235" in best.lines
    assert "喇叭離前牆：1.235 公尺" in best.lines
    assert "兩支喇叭間距：2.346 公尺" in best.lines
    assert "聆聽距離：3.457 公尺" in best.lines


def test_refinement_running_with_lock_and_partial_ledger(tmp_path: Path) -> None:
    from aosr.search.refine import RefineLedger, RefineRow
    from aosr.search.refine_run import header_for
    from aosr.search.store import refine_result_name
    store, _ = make_store(tmp_path)
    book = RefineLedger.create(store.refine_ledger_path, header_for(store))
    row = RefineRow(round=1, trial_number=None, result_file=refine_result_name(None),
                    outcome="scored", total_cost=0.5, seconds=1.0)
    book.append(row)
    with store.refine_ledger_path.open("ab") as handle:
        handle.write(b'{"round":')
    status = SearchStatus(state="converged", refine=RefineStatus(state="running", best="baseline", best_total_cost=0.5))
    store.status_path.write_text(status.model_dump_json())
    descriptor = store.open_folder_lock()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = _snapshot(store.path)
        shown = _text(_view(store.path))
        assert "細算：進行中" in shown
        assert "細算做了 1 個" in shown and "細算帳末列尚未寫完" in shown
        assert _snapshot(store.path) == before
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("part", ["refine", "outer"])
def test_new_nested_status_field_only_breaks_that_part(tmp_path: Path, part: str) -> None:
    store, registry = make_store(tmp_path, budget=3)
    status = run(store, registry, FakeCompute(store))
    document = status.model_dump(mode="json")
    document[part]["future_field"] = True
    store.status_path.write_text(json.dumps(document))
    view = build_search_view(store.path, server_physics="現在", server_program="現在")
    blocks = {block.key: block for block in view.blocks}
    assert "搜尋：因預算停止" in _text(_view(store.path))
    assert "future_field" in "；".join(blocks["stage"].lines)
    assert not blocks["search-best"].warning


@pytest.mark.parametrize("file", ["project.json", "identity.json", "settings.json", "purpose.json", "ledger.jsonl", "refine.jsonl"])
def test_unreadable_snapshot_or_ledger_does_not_blank_other_blocks(tmp_path: Path, file: str) -> None:
    store, registry = make_store(tmp_path, budget=3)
    run(store, registry, FakeCompute(store))
    (store.path / file).write_text("{")
    before = _snapshot(store.path)
    view = build_search_view(store.path, server_physics="現在", server_program="現在")
    blocks = {block.key: block for block in view.blocks}
    assert "搜尋：因預算停止" in "；".join(blocks["stage"].lines)
    assert "讀不到：" in _text(view.model_dump())
    assert "到最後一次存檔為止" in "；".join(blocks["timings"].lines)
    assert _snapshot(store.path) == before


def test_proc_lock_device_uses_hex_and_waiters_do_not_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = make_store(tmp_path)
    stat = store.path.stat()
    device = f"{os.major(stat.st_dev):x}:{os.minor(stat.st_dev):x}:{stat.st_ino}"
    monkeypatch.setattr("aosr.gui.search_view.read_proc_locks", lambda: f"2: -> FLOCK ADVISORY WRITE 123 {device} 0 EOF\n")
    assert folder_process(store.path).held is False
    monkeypatch.setattr("aosr.gui.search_view.read_proc_locks", lambda: f"2: FLOCK ADVISORY WRITE 123 {device} 0 EOF\n")
    assert folder_process(store.path).held is True
