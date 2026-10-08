"""原方案缺席時細算、附件、報告與選取皆照實降級。"""
from pathlib import Path

import pytest

from aosr.reporting.display import (
    FURNITURE_REASON, FURNITURE_MODEL_NOTE, FURNITURE_TRANSMISSION_NOTE, FURNITURE_REVERBERATION_NOTE,
)
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError, furniture_problems
from aosr.search.modal_record import role_inputs
from aosr.search.refine import RefineLedger, RefineRow, refine_order
from aosr.search.refine_run import header_for
from aosr.search.report import build_report, render_text
from aosr.search.run import SearchStatus
from aosr.search.select import select_refined
from aosr.search.store import check_project_furniture_layout, refine_result_name
from tests.engine._search_blocked_cases import SavedFurnitureCompute, blocked_store
from tests.engine._search_furniture_cases import furnished_store
from tests.engine._search_refine_cases import RefineCompute, refine
from tests.engine._search_run_cases import RUN_DATE, FakeCompute, Killed, rows, run


def test_blocked_baseline_refinement_completes_without_original(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path)
    run(store, registry, SavedFurnitureCompute(store))
    order = refine_order(rows(store))
    compute = RefineCompute(store, {})
    status = refine(store, registry, compute)
    recorded = RefineLedger.read(store.refine_ledger_path)[1]
    assert status.refine.state == "stopped" and status.refine.stop_reason == "candidates_exhausted"
    assert tuple(row.trial_number for row in recorded) == order
    assert {job.trial_number for job in compute.jobs} == set(order)
    assert not store.baseline_path.exists() and not store.refine_result_path(None).exists()
    assert status.refine.best != "baseline" and None not in {row.trial_number for row in recorded}


@pytest.mark.parametrize("workers,kill_after", [(1, 1), (4, 1), (4, 4)])
def test_blocked_baseline_refinement_resume_is_bit_identical(tmp_path: Path, workers: int, kill_after: int) -> None:
    whole, registry = blocked_store(tmp_path / "whole", workers=workers)
    interrupted, other = blocked_store(tmp_path / "interrupted", workers=workers)
    run(whole, registry, SavedFurnitureCompute(whole))
    run(interrupted, other, SavedFurnitureCompute(interrupted))
    expected = refine(whole, registry, RefineCompute(whole, {}))
    with pytest.raises(Killed):
        refine(interrupted, other, RefineCompute(interrupted, {}, kill_after=kill_after))
    compute = RefineCompute(interrupted, {})
    assert refine(interrupted, other, compute) == expected
    assert RefineLedger.read(interrupted.refine_ledger_path)[1] == RefineLedger.read(whole.refine_ledger_path)[1]
    assert all(job.trial_number is not None for job in compute.jobs)
    assert not interrupted.refine_result_path(None).exists()


@pytest.mark.parametrize("blocked", [False, True])
def test_refinement_rechecks_geometry_and_rejects_modified_baseline_status(tmp_path: Path, blocked: bool) -> None:
    store, registry = blocked_store(tmp_path, blocked=blocked)
    compute = SavedFurnitureCompute(store) if blocked else FakeCompute(store)
    status = run(store, registry, compute)
    changed = status.model_copy(update={"baseline_outcome": "scored" if blocked else "direct_path_blocked"})
    store.status_path.write_text(changed.model_dump_json())
    status = refine(store, registry, RefineCompute(store, {}))
    assert status.refine.state == "interrupted" and "不一致" in status.refine.message
    assert not store.refine_ledger_path.exists()


def test_blocked_baseline_refinement_ledger_cannot_contain_original(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path)
    run(store, registry, SavedFurnitureCompute(store))
    book = RefineLedger.create(store.refine_ledger_path, header_for(store))
    book.append(RefineRow(round=1, trial_number=None, result_file=refine_result_name(None),
                          outcome="not_evaluated", total_cost=None, seconds=0.0))
    status = refine(store, registry, RefineCompute(store, {}))
    assert status.refine.state == "interrupted" and "不准有原方案列" in status.refine.message


def test_report_uses_decision_words_and_lists_blocking_pairs(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path)
    status = run(store, registry, SavedFurnitureCompute(store))
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    text = render_text(report)
    assert "原方案不符合擺位要求" in report.ranks
    assert report.quality.original.message == "原方案不符合擺位要求"
    assert "原方案不符合擺位要求，不列" in text
    assert report.placement.original is None
    assert set(report.unassessed.items) == {"製作用途", "多人座位", "箱體反射"}
    assert all(problem.message in text for problem in furniture_problems(store.project))
    assert report.furniture_notes == (FURNITURE_REASON, FURNITURE_TRANSMISSION_NOTE, FURNITURE_REVERBERATION_NOTE)
    _, separator, after = text.partition(FURNITURE_REASON)
    assert separator == FURNITURE_REASON and FURNITURE_REASON not in after
    assert any(FURNITURE_MODEL_NOTE in line for line in report.ranks)
    role = next(item.record for item in role_inputs(store, status) if item.record.role == "baseline")
    assert role.state == "skipped" and role.reason_text == "原方案不符合擺位要求"
    with pytest.raises(ValueError, match="原方案不符合擺位要求"):
        select_refined(store, None, tmp_path / "selected")
    assert not (tmp_path / "selected").exists()


def test_furnished_unblocked_report_has_same_notes(tmp_path: Path) -> None:
    from tests.engine._search_run_cases import FakeCompute

    store, registry = furnished_store(tmp_path)
    run(store, registry, FakeCompute(store))
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert set(report.unassessed.items) == {"製作用途", "多人座位", "箱體反射"}
    assert report.furniture_notes == (FURNITURE_REASON, FURNITURE_TRANSMISSION_NOTE, FURNITURE_REVERBERATION_NOTE)
    assert any(FURNITURE_MODEL_NOTE in line for line in report.ranks)
    assert not report.placement.original_excluded


def test_project_furniture_layout_accepts_blocking_and_rejects_input_errors(tmp_path: Path) -> None:
    store, _ = blocked_store(tmp_path)
    check_project_furniture_layout(store.project)
    assert store.project.furniture is not None
    item = store.project.furniture[0].model_dump() | {"placement": {"bottom_center_m": [6.1, 1.85, 1.1], "yaw_deg": 0}}
    invalid = Scheme.model_validate(store.project.model_dump() | {"furniture": [item]})
    expected = furniture_problems(invalid)
    with pytest.raises(SchemeValidationError) as caught:
        check_project_furniture_layout(invalid)
    assert caught.value.problems == expected
    assert {problem.path for problem in expected} == {"furniture"}


def test_old_status_still_builds_report(tmp_path: Path) -> None:
    from tests.engine._search_run_cases import FakeCompute, make_store

    store, registry = make_store(tmp_path, budget=3)
    status = run(store, registry, FakeCompute(store))
    store.status_path.write_text(status.model_dump_json(exclude={"comparison_trial"}))
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert set(report.unassessed.items) == {"製作用途", "多人座位", "物件反射", "箱體反射"}
    assert not report.furniture_notes
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).comparison_trial is None
