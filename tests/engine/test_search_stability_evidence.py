"""補咬存活植錯：釘身分順序、計算交回、出處、入圍與周圍座位。"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.result import ResultOrigin, SchemeResult
from aosr.scoring.contract import CandidateEvaluation
from aosr.search import crossover_record, placement_stability_attach as module
from aosr.search.labels import STABILITY_CROSSOVERS
from aosr.search.outer_status import OuterStatus
from aosr.search.placement_stability_record import is_stale, read_summary
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.refine_run import header_for
from aosr.search.run import CandidateJob, ComputedCandidate, IdentityChanged, SearchStatus
from aosr.search.store import SearchStore, refine_result_name
from tests.engine._search_run_cases import RUN_DATE
from tests.engine._stability_attach_cases import ShiftCompute, attach, ready, refiner


def other_comparison(candidate: CandidateEvaluation, tag: str) -> CandidateEvaluation:
    return CandidateEvaluation.model_validate_json(candidate.model_dump_json().replace(
        '"settings_fingerprint":"', f'"settings_fingerprint":"{tag}-'))


def test_pin_starts_at_baseline_even_when_candidate_comparison_differs(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    for number in (7, 9):
        path = store.refine_result_path(number)
        result = SchemeResult.model_validate_json(path.read_bytes())
        path.write_text(result.model_copy(update={"candidate": other_comparison(result.candidate, str(number))}).model_dump_json())
    base = SchemeResult.model_validate_json(store.refine_result_path(None).read_bytes())
    judge = refiner(store, registry, status)
    judge.pin(base.candidate, base.scheme)
    actual = module._pinned(store, status, RefineLedger.read(store.refine_ledger_path)[1], load_quality_targets(registry), RUN_DATE)
    assert actual == judge.pinned
    compute = ShiftCompute(store)
    summary = attach(store, registry, status, compute)
    assert {p.outcome for p in summary.points if p.trial_number is None} == {"scored"}
    assert {p.outcome for p in summary.points if p.trial_number is not None} == {"not_comparable"}


def test_blocked_baseline_pin_uses_first_available_candidate_in_ledger_order(tmp_path: Path) -> None:
    from tests.engine._search_blocked_cases import blocked_store
    from tests.engine._crossover_cases import small_result
    store, registry = blocked_store(tmp_path)
    store.ensure_refine_dir()
    book = RefineLedger.create(store.refine_ledger_path, header_for(store))
    status = SearchStatus(baseline_outcome="direct_path_blocked", baseline_reason_codes=("direct_path_blocked",))
    expected = None
    for number in (9, 7, 11):
        result = small_result(store, number, 200, (1, 2), (220, 1000))
        candidate = result.candidate.model_copy(update={"evaluations": ()}) if number == 9 else other_comparison(result.candidate, str(number))
        result = result.model_copy(update={"candidate": candidate})
        store.refine_result_path(number).write_text(result.model_dump_json())
        store.scheme_path_for(store.refine_result_path(number)).write_text(result.scheme.model_dump_json())
        store.scheme_path_for(store.candidate_path(number)).write_text(result.scheme.model_dump_json())
        book.append(RefineRow(round=1, trial_number=number, result_file=refine_result_name(number),
                             outcome="not_evaluated" if number == 9 else "scored", total_cost=None if number == 9 else 1.0, seconds=0))
        if number == 7:
            judge = refiner(store, registry, status)
            judge.check_baseline()
            assert judge.baseline_blocked
            judge.pin(candidate, result.scheme)
            expected = judge.pinned
    assert expected is not None
    assert module._pinned(store, status, RefineLedger.read(store.refine_ledger_path)[1], load_quality_targets(registry), RUN_DATE) == expected


class FaultyCompute(ShiftCompute):
    def __init__(self, store: SearchStore, fault: str) -> None:
        super().__init__(store)
        self.fault = fault

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        if self.fault == "incomplete":
            yield from super().__call__(jobs[:1], workers)
            return
        for computed in super().__call__(jobs, workers):
            if self.fault == "identity":
                computed = replace(computed, identity=replace(computed.identity, physics_identity="changed"))
            elif self.fault == "content":
                computed = replace(computed, candidate=other_comparison(computed.candidate, "changed"))
            yield computed


def test_running_progress_counts_only_received_points(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    compute = ShiftCompute(store)
    observed: list[tuple[int, int]] = []
    def observing(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        received = 0
        for computed in compute(jobs, workers):
            saved = read_summary(store.path)
            assert saved is not None and saved.state == "running"
            observed.append((saved.computed_points, saved.total_points))
            assert saved.computed_points == received
            assert saved.total_points == len(jobs)
            yield computed
            received += 1
    summary = module.attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
        probe=lambda: store.identity, compute_factory=lambda root: observing)
    assert observed == [(i, len(compute.jobs)) for i in range(len(compute.jobs))]
    assert summary.computed_points == summary.total_points == len(compute.jobs)


@pytest.mark.parametrize("fault,exception,message", [("identity", IdentityChanged, "physics_identity"),
    ("content", ValueError, "保存的移位結果與計算交回的內容不同"), ("incomplete", ValueError, "計算沒有交回整批移位點")])
def test_computed_identity_saved_content_and_batch_completeness(tmp_path: Path, fault: str,
                                                             exception: type[Exception], message: str) -> None:
    store, registry, status = ready(tmp_path)
    compute = FaultyCompute(store, fault)
    with pytest.raises(exception, match=message):
        attach(store, registry, status, compute)
    saved = read_summary(store.path)
    assert saved is not None and saved.state == "failed" and not saved.completed
    assert saved.arithmetic is None and not saved.boundaries


@pytest.mark.parametrize("number", [None, 7])
def test_refined_origin_is_checked_before_shift_dispatch(tmp_path: Path, number: int | None) -> None:
    store, registry, status = ready(tmp_path)
    path = store.refine_result_path(number)
    result = SchemeResult.model_validate_json(path.read_bytes())
    path.write_text(result.model_copy(update={"origin": ResultOrigin(kind="run")}).model_dump_json())
    compute = ShiftCompute(store)
    with pytest.raises(ValueError, match="細算結果與出處"):
        attach(store, registry, status, compute)
    assert not compute.jobs


def test_usable_crossover_change_updates_winners_without_ledger_change(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    rows_before = store.refine_ledger_path.read_bytes()
    failed = crossover_record.fresh_summary(store, status).model_copy(update={"state": "failed", "reason_text": "交接失敗原文"})
    crossover_record.write_summary(store.path, failed)
    first = attach(store, registry, status, ShiftCompute(store))
    done = failed.model_copy(update={"state": "done", "completed": True, "reason_text": "", "official_best": 7,
        "variants": tuple(crossover_record.VariantRecord(key=key, label=key, basis="題目",
            ranking=(crossover_record.CostRow(trial_number=9, total_cost=0),)) for key in STABILITY_CROSSOVERS)})
    crossover_record.write_summary(store.path, done)
    compute = ShiftCompute(store)
    second = attach(store, registry, status, compute)
    assert all(w.winner is None for w in first.selection.crossover_winners)
    assert all(w.winner is not None and w.winner.trial_number == 9 for w in second.selection.crossover_winners)
    assert second.crossover_stamp != first.crossover_stamp and not compute.jobs
    assert store.refine_ledger_path.read_bytes() == rows_before


def test_staleness_includes_outer_conclusion_and_snapshot(tmp_path: Path) -> None:
    store, _, status = ready(tmp_path)
    first = module.fresh_summary(store, status)
    changed = status.model_copy(update={"outer": OuterStatus(conclusion="refine_budget")})
    assert is_stale(first, module.fresh_summary(store, changed))
    changed = status.model_copy(update={"asked": status.asked + 1})
    assert is_stale(first, module.fresh_summary(store, changed))
