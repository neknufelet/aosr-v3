"""第二輪以後只細算剩下順序，第一名跨輪、停止與半批接續各自守住。"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.search.cli import main
from aosr.search.refine import RefineLedger
from aosr.search.refine_run import refinement_status
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.store import SearchStore, refine_result_name
from tests.engine._search_feedback_cases import prepared, resume, snapshot
from tests.engine._search_refine_cases import RefineCompute, SearchCompute, refine
from tests.engine._search_run_cases import Killed
from tests.engine.test_search_refine_cli import invoke
from tests.engine.test_search_refine_resume import clone


def second_round(tmp_path: Path, *, budget: int = 4, convergence: int = 50,
                 workers: int = 1, missing: frozenset[int | None] = frozenset()) -> tuple[SearchStore, Path, SearchStatus]:
    store, registry, first = prepared(tmp_path, workers=workers, refine_budget=budget,
                                      refine_convergence=convergence, anchor_number=1)
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 0
    # 新輪第一題嚴格較好、第二題同分：它們插在舊候選前面，其餘仍同分依編號。
    compute = SearchCompute(store, flat=True, persist_baseline=True,
                            values={first.asked: 0.5, first.asked + 1: 0.5}, missing=missing)
    stopped = resume(store, registry, compute)
    assert stopped.state == "converged" and stopped.round == 2
    return store, registry, stopped


class ObserveCompute(RefineCompute):
    """只在假計算邊界讀真狀態，並可在第一批後送細算停止記號。"""

    def __init__(self, store: SearchStore, values: dict[int | None, float], *, stop: bool = False) -> None:
        super().__init__(store, values)
        self.observed: list[SearchStatus] = []
        self.stop_requested = stop

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        self.observed.append(SearchStatus.model_validate_json(self.store.status_path.read_bytes()))
        yield from super().__call__(jobs, workers)
        if self.stop_requested:
            self.store.refine_stop_path.touch()


@pytest.mark.parametrize("search_state", ["converged", "budget_exhausted"])
def test_round_two_opens_resets_and_preserves_search(tmp_path: Path, search_state: str) -> None:
    store, registry, before = second_round(tmp_path)
    before = SearchStatus.model_validate(before.model_dump() | {"state": search_state})
    store.status_path.write_text(before.model_dump_json())
    accepted = refinement_status(store)
    assert accepted.refine.state == "running" and accepted.refine.round == before.round
    assert accepted.refine.stop_reason is None and accepted.refine.streak == 0
    assert accepted.refine.best == before.refine.best
    assert accepted.refine.best_total_cost == before.refine.best_total_cost
    compute = ObserveCompute(store, {})
    final = refine(store, registry, compute)
    opened = compute.observed[0]
    assert opened.refine.state == "running" and opened.refine.round == before.round
    assert opened.refine.stop_reason is None and opened.refine.streak == 0
    assert opened.refine.best == before.refine.best
    assert opened.refine.best_total_cost == before.refine.best_total_cost
    assert "第 2 輪" in opened.refine.message
    assert final.model_dump(exclude={"refine"}) == before.model_dump(exclude={"refine"})
    assert final.refine.round == before.round


@pytest.mark.parametrize("state", ["same_round", "search_running"])
def test_refine_rejects_same_round_and_running_search_without_writes(tmp_path: Path, state: str) -> None:
    store, registry, before = second_round(tmp_path)
    if state == "same_round":
        refine(store, registry, RefineCompute(store, {}))
    else:
        store.status_path.write_text(before.model_copy(update={"state": "running"}).model_dump_json())
    saved = snapshot(store)
    with pytest.raises(ValueError, match="這一輪已細算完|搜尋狀態"):
        refine(store, registry, RefineCompute(store, {}))
    assert snapshot(store) == saved


def test_round_two_order_skips_prior_rows_baseline_and_batches(tmp_path: Path) -> None:
    store, registry, before = second_round(tmp_path, budget=6)
    compute = RefineCompute(store, {})
    status = refine(store, registry, compute)
    rows = RefineLedger.read(store.refine_ledger_path)[1]
    # 這組固定種子的手排答案：兩題較好新題先走，再接尚未細算的舊題。
    assert [row.trial_number for row in rows if row.round == 2] == [8, 9, 6, 7, 10, 11]
    assert compute.batches == [(8, 9), (6, 7), (10, 11)]
    assert [job.trial_number for job in compute.jobs] == [8, 9, 6, 7, 10, 11]
    assert status.refine.best == before.refine.best
    assert rows[0].trial_number is None and rows[0].round == 1


@pytest.mark.parametrize("improved", [False, True])
def test_global_best_and_current_round_streak(tmp_path: Path, improved: bool) -> None:
    store, registry, before = second_round(tmp_path)
    values: dict[int | None, float] = {8: 0.05, 9: 0.05} if improved else {}
    status = refine(store, registry, RefineCompute(store, values)).refine
    if improved:
        assert status.best == 8 and status.best_total_cost == 0.05 / 4.0
        assert status.streak == status.refined - 1
    else:
        assert status.best == before.refine.best and status.best_total_cost == before.refine.best_total_cost
        assert status.streak == status.refined


@pytest.mark.parametrize("reason,budget,convergence,expected", [
    ("stable", 4, 3, [8, 9, 4, 5]),
    ("refine_budget", 4, 50, [8, 9, 4, 5]),
    ("candidates_exhausted", 30, 50, [8, 9, 10, 11, 12, 13, 14, 15]),
    ("user_stopped", 4, 50, [8, 9]),
])
def test_round_two_stop_conditions(tmp_path: Path, reason: str, budget: int,
                                   convergence: int, expected: list[int]) -> None:
    store, registry, _ = second_round(tmp_path, budget=budget, convergence=convergence)
    compute = ObserveCompute(store, {}, stop=reason == "user_stopped")
    status = refine(store, registry, compute).refine
    actual = [row.trial_number for row in RefineLedger.read(store.refine_ledger_path)[1] if row.round == 2]
    assert status.state == "stopped" and status.stop_reason == reason
    assert actual == expected and status.refined == len(expected)


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("partial", [False, True])
def test_round_two_killed_half_batch_matches_whole(tmp_path: Path, workers: int, partial: bool) -> None:
    store, registry, _ = second_round(tmp_path / "input", workers=workers)
    whole = clone(store, tmp_path / "whole")
    expected = refine(whole, registry, RefineCompute(whole, {}))
    with pytest.raises(Killed):
        refine(store, registry, RefineCompute(store, {}, kill_after=1))
    if partial:
        with store.refine_ledger_path.open("ab") as handle:
            handle.write(b'{"round":')
    compute = RefineCompute(store, {})
    actual = refine(store, registry, compute)
    assert compute.batches[0] == ((9,) if workers == 1 else (8, 9))
    assert store.refine_ledger_path.read_bytes() == whole.refine_ledger_path.read_bytes()
    assert store.status_path.read_bytes() == whole.status_path.read_bytes() and actual == expected


def test_round_two_single_multi_are_identical(tmp_path: Path) -> None:
    single, registry, _ = second_round(tmp_path / "single")
    multi, other, _ = second_round(tmp_path / "multi", workers=4)
    first = refine(single, registry, RefineCompute(single, {}))
    second = refine(multi, other, RefineCompute(multi, {}))
    assert RefineLedger.read(single.refine_ledger_path)[1] == RefineLedger.read(multi.refine_ledger_path)[1]
    assert first == second


@pytest.mark.parametrize("damage,reason", [
    ("prefix", "第 2 輪"), ("unknown", "有分數"),
    ("baseline_first", "第一列"), ("old_prefix", "第 1 輪"), ("unscored", "有分數"),
])
def test_round_two_invalid_ledger_interrupts_without_append(tmp_path: Path, damage: str, reason: str) -> None:
    store, registry, before = second_round(tmp_path, missing=frozenset({8}) if damage == "unscored" else frozenset())
    with pytest.raises(Killed):
        refine(store, registry, RefineCompute(store, {}, kill_after=1))
    lines = store.refine_ledger_path.read_bytes().splitlines()
    if damage in ("prefix", "unknown", "old_prefix", "unscored"):
        index = 2 if damage == "old_prefix" else -1
        document = json.loads(lines[index])
        number = 7 if damage in ("prefix", "old_prefix") else 8 if damage == "unscored" else before.asked + 100
        lines[index] = json.dumps(document | {"trial_number": number,
                                             "result_file": refine_result_name(number)}).encode()
    else:
        lines[1], lines[2] = lines[2], lines[1]
    store.refine_ledger_path.write_bytes(b"\n".join(lines) + b"\n")
    saved = store.refine_ledger_path.read_bytes()
    compute = RefineCompute(store, {})
    status = refine(store, registry, compute)
    assert status.refine.state == "interrupted" and not compute.jobs
    assert "細算帳" in status.refine.message and reason in status.refine.message
    assert store.refine_ledger_path.read_bytes() == saved


def test_round_two_baseline_cache_identity_is_checked(tmp_path: Path) -> None:
    store, registry, _ = second_round(tmp_path)
    path = store.refine_result_path(None)
    path.write_text(json.dumps(json.loads(path.read_bytes()) | {"program_fingerprint": "other"}))
    saved = store.refine_ledger_path.read_bytes()
    compute = RefineCompute(store, {})
    status = refine(store, registry, compute)
    assert status.refine.state == "interrupted" and "原方案" in status.refine.message
    assert not compute.jobs and store.refine_ledger_path.read_bytes() == saved


@pytest.mark.parametrize("damage", ["missing", "unreadable"])
def test_round_two_unavailable_baseline_interrupts_without_recompute(tmp_path: Path, damage: str) -> None:
    store, registry, _ = second_round(tmp_path)
    path = store.refine_result_path(None)
    if damage == "missing":
        path.unlink()
    else:
        path.write_text("{")
    saved = store.refine_ledger_path.read_bytes()
    compute = RefineCompute(store, {})
    status = refine(store, registry, compute)
    assert status.refine.state == "interrupted" and "原方案" in status.refine.message
    assert not compute.jobs and store.refine_ledger_path.read_bytes() == saved


def test_round_two_baseline_pins_comparison_identity(tmp_path: Path) -> None:
    store, registry, before = second_round(tmp_path)
    status = refine(store, registry, RefineCompute(store, {}, different=frozenset({8})))
    rows = RefineLedger.read(store.refine_ledger_path)[1]
    assert next(row for row in rows if row.trial_number == 8).outcome == "not_comparable"
    assert status.refine.best == before.refine.best


def test_round_two_half_batch_stable_finishes_original_batch(tmp_path: Path) -> None:
    store, registry, _ = second_round(tmp_path / "input", convergence=3)
    whole = clone(store, tmp_path / "whole")
    expected = refine(whole, registry, RefineCompute(whole, {}))
    with pytest.raises(Killed):
        refine(store, registry, RefineCompute(store, {}, kill_after=3))
    compute = RefineCompute(store, {})
    actual = refine(store, registry, compute)
    assert [job.trial_number for job in compute.jobs] == [5]
    assert actual == expected and actual.refine.stop_reason == "stable"
    assert store.refine_ledger_path.read_bytes() == whole.refine_ledger_path.read_bytes()
    assert store.status_path.read_bytes() == whole.status_path.read_bytes()


def test_round_three_excludes_every_earlier_round(tmp_path: Path) -> None:
    store, registry, _ = second_round(tmp_path)
    second = refine(store, registry, RefineCompute(store, {9: 0.05}))
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 0
    third = resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    assert third.round > second.round
    compute = ObserveCompute(store, {})
    final = refine(store, registry, compute)
    rows = RefineLedger.read(store.refine_ledger_path)[1]
    assert [row.trial_number for row in rows if row.round == third.round] == [6, 7, 10, 11]
    assert [job.trial_number for job in compute.jobs] == [6, 7, 10, 11]
    assert compute.observed[0].refine.best == second.refine.best
    assert final.refine.round == third.round and final.refine.best == second.refine.best
    assert final.refine.streak == final.refine.refined


@pytest.mark.parametrize("case", ["success", "same_round", "identity"])
def test_round_two_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    store, registry, _ = second_round(tmp_path)
    if case == "same_round":
        refine(store, registry, RefineCompute(store, {}))
    elif case == "identity":
        path = store.refine_result_path(None)
        path.write_text(json.dumps(json.loads(path.read_bytes()) | {"physics_identity": "other"}))
    compute = RefineCompute(store, {})
    saved = store.refine_ledger_path.read_bytes()
    exit_code = invoke(store, registry, compute, monkeypatch)
    if case == "success":
        assert exit_code == 0 and [job.trial_number for job in compute.jobs] == [8, 9, 4, 5]
    else:
        assert exit_code == (1 if case == "same_round" else 3)
        assert not compute.jobs and store.refine_ledger_path.read_bytes() == saved


def test_round_two_stale_stop_marker_is_cleared(tmp_path: Path) -> None:
    store, registry, _ = second_round(tmp_path)
    store.refine_stop_path.touch()
    store.stop_path.touch()
    status = refine(store, registry, RefineCompute(store, {}))
    assert status.refine.stop_reason == "refine_budget"
    assert not store.refine_stop_path.exists() and store.stop_path.exists()
    assert "殘留" in status.refine.message


def test_round_two_stop_message_names_round_and_this_rounds_feedback() -> None:
    """第 2 輪以後的停止訊息照實：寫第幾輪、連續數標本輪、完成與否交給外圈結論（第 5 支起）。"""
    from aosr.search.refine_run import _stop_message
    from aosr.search.run import RefineStatus

    second = RefineStatus(state="running", round=2, refined=4, best=30, best_total_cost=1.0, streak=4)
    message = _stop_message(second, "stable")
    assert message.startswith("細算已停：第 2 輪，") and "本輪連續 4 個" in message
    assert "完成與否看外圈結論" in message
    first = RefineStatus(state="running", round=1, refined=4, best=30, best_total_cost=1.0, streak=4)
    assert _stop_message(first, "stable") == "細算已停：細算第一名 30 號 連續 4 個沒被換掉（暫行設定，不代表細算完成——完成與否看外圈結論）"



def test_same_round_refusal_points_to_outer_conclusion_not_feedback(tmp_path: Path) -> None:
    """同一輪細算完再下 refine：訊息指去外圈結論，不再叫人一律「先回饋」（回饋不一定走得通）。"""
    store, registry, _ = second_round(tmp_path)
    refine(store, registry, RefineCompute(store, {}))
    with pytest.raises(ValueError) as refused:
        refine(store, registry, RefineCompute(store, {}))
    message = str(refused.value)
    assert "外圈結論" in message and "要先回饋" not in message
