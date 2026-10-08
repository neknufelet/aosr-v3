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
