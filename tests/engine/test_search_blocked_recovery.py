"""審查補考：中斷點掃完整段；答案只取同一題不中斷搜尋與細算的結果。"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest

from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError, furniture_problems
from aosr.search import run as search_run
from aosr.search.ledger import Ledger, LedgerRow
from aosr.search.refine import RefineLedger, refine_order
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.store import SearchStore
from tests.engine._search_blocked_cases import SavedFurnitureCompute, blocked_store
from tests.engine._furniture_cases import relative_item
from tests.engine._search_furniture_cases import enqueue_hand_placements
from tests.engine._search_refine_cases import RefineCompute, refine
from tests.engine._search_run_cases import Killed, next_params, rows, run
from tests.engine.test_search_baseline_furniture import resume


def intercept(monkeypatch: pytest.MonkeyPatch, store: SearchStore, source: Literal["append", "status"], *,
              kill_index: int | None = None, after: bool = False) -> list[LedgerRow | SearchStatus]:
    events: list[LedgerRow | SearchStatus] = []
    original_append, original_status = Ledger.append, search_run._write_status

    def checkpoint(value: LedgerRow | SearchStatus, *, written: bool) -> None:
        if written == after and len(events) - 1 == kill_index:
            raise Killed("在保存邊界被砍")

    def append(book: Ledger, row: LedgerRow) -> None:
        events.append(row)
        # 有比較身分的列落帳前，釘住編號必須已持久保存；否則中斷後會失去這次決定。
        if row.outcome == "scored" or row.reason == "eliminated":
            saved = SearchStatus.model_validate_json(store.status_path.read_bytes())
            assert saved.comparison_trial is not None
        checkpoint(row, written=False)
        original_append(book, row)
        checkpoint(row, written=True)

    def status(store: SearchStore, value: SearchStatus) -> SearchStatus:
        events.append(value)
        checkpoint(value, written=False)
        result = original_status(store, value)
        checkpoint(value, written=True)
        return result

    if source == "append":
        monkeypatch.setattr(Ledger, "append", append)
    else:
        monkeypatch.setattr(search_run, "_write_status", status)
    return events


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("source", ["append", "status"])
@pytest.mark.parametrize("missing_prefix", [0, 4, 7])
def test_blocked_kill_sweep_matches_whole_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             workers: int, source: Literal["append", "status"], missing_prefix: int) -> None:
    whole, registry = blocked_store(tmp_path / "whole", workers=workers)
    missing = frozenset(range(missing_prefix))
    with monkeypatch.context() as patch:
        events = intercept(patch, whole, source)
        expected = run(whole, registry, SavedFurnitureCompute(whole, missing=missing))
    assert expected.comparison_trial is not None
    assert expected.comparison_trial // whole.settings.batch_size == missing_prefix // whole.settings.batch_size
    for index in range(len(events)):
        for after in (False, True):
            store, other = blocked_store(tmp_path / f"cut-{index}-{after}", workers=workers)
            with monkeypatch.context() as patch:
                intercept(patch, store, source, kill_index=index, after=after)
                with pytest.raises(Killed):
                    run(store, other, SavedFurnitureCompute(store, missing=missing))
            saved = SearchStatus.model_validate_json(store.status_path.read_bytes()) if store.status_path.exists() else SearchStatus()
            continued = SavedFurnitureCompute(store, missing=missing)
            actual = resume(store, other, continued) if saved.state == "running" else saved
            assert actual == expected, (source, index, after)
            assert rows(store) == rows(whole), (source, index, after)
            assert next_params(store) == next_params(whole), (source, index, after)
            assert None not in continued.calls and not store.baseline_path.exists()


class ChangedPhysicsCompute(SavedFurnitureCompute):
    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in super().__call__(jobs, workers):
            if result.job.trial_number == 1:
                yield replace(result, identity=replace(result.identity, physics_identity="different"))
            else:
                yield result


@pytest.mark.parametrize("workers", [1, 4])
def test_unpinned_batch_physics_identity_change_interrupts(tmp_path: Path, workers: int) -> None:
    store, registry = blocked_store(tmp_path, workers=workers, budget=3)
    status = run(store, registry, ChangedPhysicsCompute(store))
    assert status.state == "interrupted" and "試算 1" in status.message and "physics_identity" in status.message
    assert 1 not in {row.trial_number for row in rows(store)}


def test_invalid_project_furniture_fails_before_compute(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path)
    assert store.project.furniture is not None
    item = store.project.furniture[0].model_dump() | {"placement": {"bottom_center_m": [6.1, 1.85, 1.1], "yaw_deg": 0}}
    project = Scheme.model_validate(store.project.model_dump() | {"furniture": [item]})
    (store.path / "project.json").write_text(project.model_dump_json())
    store = SearchStore.open(store.path)
    compute = SavedFurnitureCompute(store)
    with pytest.raises(SchemeValidationError) as caught:
        run(store, registry, compute)
    assert caught.value.problems == furniture_problems(project)
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).state == "failed"
    assert not compute.calls and not rows(store)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("missing_prefix", [0, 1, 2])
def test_refine_different_identities_kill_sweep_matches_whole(tmp_path: Path, workers: int, missing_prefix: int) -> None:
    whole, registry = blocked_store(tmp_path / "whole", workers=workers)
    run(whole, registry, SavedFurnitureCompute(whole))
    order = refine_order(rows(whole))
    missing, different = frozenset(order[:missing_prefix]), frozenset(order[1::2])
    complete = RefineCompute(whole, {}, missing=missing, different=different)
    expected = refine(whole, registry, complete)
    for index in range(len(complete.jobs)):
        store, other = blocked_store(tmp_path / f"cut-{index}", workers=workers)
        run(store, other, SavedFurnitureCompute(store))
        with pytest.raises(Killed):
            refine(store, other, RefineCompute(store, {}, missing=missing, different=different, kill_after=index))
        continued = RefineCompute(store, {}, missing=missing, different=different)
        assert refine(store, other, continued) == expected, index
        assert RefineLedger.read(store.refine_ledger_path)[1] == RefineLedger.read(whole.refine_ledger_path)[1], index
        assert all(job.trial_number is not None for job in continued.jobs)
        assert not store.refine_result_path(None).exists()


def test_comparison_restore_refuses_other_trial_result_and_scheme(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, SavedFurnitureCompute(store, fail_after=4, kill=True))
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.comparison_trial is not None
    other = next(row.trial_number for row in rows(store) if row.result_file is not None and row.trial_number != status.comparison_trial)
    target, source = store.candidate_path(status.comparison_trial), store.candidate_path(other)
    target.write_bytes(source.read_bytes())
    store.scheme_path_for(target).write_bytes(store.scheme_path_for(source).read_bytes())
    before = rows(store)
    compute = SavedFurnitureCompute(store)
    resumed = resume(store, registry, compute)
    assert resumed.state == "interrupted" and f"試算 {status.comparison_trial}" in resumed.message
    assert "候選代號" in resumed.message and not compute.calls and rows(store) == before


def test_unpinned_batch_kill_recomputes_completed_jobs(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path)
    compute = SavedFurnitureCompute(store, fail_after=1, kill=True)
    with pytest.raises(Killed):
        run(store, registry, compute)
    assert compute.calls and not rows(store)
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).comparison_trial is None
    continued = SavedFurnitureCompute(store)
    assert resume(store, registry, continued).state == "converged"
    assert set(compute.calls) <= set(continued.calls)


def test_illegal_row_before_pin_is_skipped_when_status_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不合法列排在釘住之前：狀態檔不見後重推要跳過它（它本來就沒有結果檔），照舊釘回同一個。"""
    from aosr.search import layout
    from aosr.search.sampler import SamplerAdapter
    # 下一批參數用不經 _adapter 的那一支取：_adapter 被換成會多排一點，經它取會重播不符。
    from tests.engine._search_furniture_cases import furnished_next_params

    probe, probe_registry = blocked_store(tmp_path / "probe")
    run(probe, probe_registry, SavedFurnitureCompute(probe))
    illegal_row = next(item for item in rows(probe) if item.outcome == "illegal")
    illegal = layout.LayoutParams(*(illegal_row.params_m[name] for name in layout.SEARCH_QUANTITIES))
    original = search_run._adapter

    def adapter(store: SearchStore) -> tuple[SamplerAdapter, bool]:
        result, enqueued = original(store)
        result.enqueue(layout.unit_from_params(illegal, store.settings.layout))
        return result, enqueued

    monkeypatch.setattr(search_run, "_adapter", adapter)
    missing = frozenset({0})
    whole, registry = blocked_store(tmp_path / "whole")
    expected = run(whole, registry, SavedFurnitureCompute(whole, missing=missing))
    outcomes = [(row.trial_number, row.outcome) for row in rows(whole)]
    assert outcomes[1] == (1, "illegal") and expected.comparison_trial == 2, outcomes
    store, other = blocked_store(tmp_path / "cut")
    with pytest.raises(Killed):
        run(store, other, SavedFurnitureCompute(store, missing=missing, fail_after=4, kill=True))
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).comparison_trial == 2
    store.status_path.unlink()
    resumed = resume(store, other, SavedFurnitureCompute(store, missing=missing))
    assert resumed.model_copy(update={"timed_from_start": expected.timed_from_start}) == expected, resumed.message
    assert rows(store) == rows(whole)
    assert furnished_next_params(store) == furnished_next_params(whole)


def test_eliminated_first_pin_is_restored_when_status_is_rebuilt(tmp_path: Path) -> None:
    """照編號第一個「定得出身分」的候選才是當初釘的，不是第一個有分數的：第一個被淘汰也照樣釘回它。"""
    from tests.engine.test_search_furniture_attachments import MatchingFurnitureCompute

    whole, registry = blocked_store(tmp_path / "whole")
    expected = run(whole, registry, MatchingFurnitureCompute(whole))
    assert expected.comparison_trial == 0
    assert {row.trial_number: row for row in rows(whole)}[0].reason == "eliminated"
    store, other = blocked_store(tmp_path / "cut")
    with pytest.raises(Killed):
        run(store, other, MatchingFurnitureCompute(store, fail_after=4, kill=True))
    store.status_path.unlink()
    actual = resume(store, other, MatchingFurnitureCompute(store))
    assert actual.comparison_trial == 0
    assert actual.model_copy(update={"timed_from_start": expected.timed_from_start}) == expected
    assert rows(store) == rows(whole)
    assert next_params(store) == next_params(whole)


@pytest.mark.parametrize("workers", [1, 4])
def test_blocked_candidate_in_unpinned_batch_never_reaches_compute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, workers: int,
) -> None:
    # 原方案被天雲擋住（_search_blocked_cases 的手算），這一批還沒釘住比較身分，走整批到齊那條路。
    # 另加一塊跟著主位走的小板（主位前 0.25 m、左 0.3 m、底高 1.0 m，0.1×0.1×0.5 m）。
    # 手定試算 0（前距 1.0、間距 1.2、聆聽距離 0.5）：左聲源 (1,1.4,1.25)、主位 (1.5,2,1.25)、面向 -x、左方 -y，
    # 小板中心 (1.25,1.7)、x=[1.2,1.3]、y=[1.65,1.75]、z=[1.0,1.5]；左聲源到主位在 x=1.25 時 y=1.7，穿過小板。
    # 手定試算 1（聆聽距離 1.3）：主位 (2.3,2)，小板 x=[2.0,2.1]，那一段直達 y 約 1.86–1.91，不擋；
    # 周圍點也一律要求，最貼近的是左側點 (2.3,1.9)，左聲源到它那一段 y 約 1.79–1.82，離板邊 y=1.75 仍有間隙，也不擋。
    board = relative_item(furniture_id="board", kind="desk", material="wood", width_m=0.1, depth_m=0.1, height_m=0.5,
                          placement={"forward_m": 0.25, "left_m": 0.3, "bottom_height_m": 1.0, "yaw_deg": 0})
    enqueue_hand_placements(monkeypatch, placements=((1.0, 0.5), (1.0, 1.3)))
    store, registry = blocked_store(tmp_path, workers=workers, batch=3, budget=3, extra_furniture=(board,))
    compute = SavedFurnitureCompute(store)
    status = run(store, registry, compute)
    recorded = {row.trial_number: row for row in rows(store)}
    assert status.baseline_outcome == "direct_path_blocked"
    assert recorded[0].outcome == "illegal" and recorded[0].reason == "direct_path_blocked"
    assert recorded[0].result_file is None and not store.candidate_path(0).exists()
    assert all(job_number != 0 for job_number in compute.calls)
    assert recorded[1].outcome == "scored" and status.comparison_trial == 1
