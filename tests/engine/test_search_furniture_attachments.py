"""被擋原方案不進附件計算；其他候選照原本入口驗證。"""
import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeValidationError, furniture_problems
from aosr.search import crossover_sensitivity
from aosr.search.modal_attach import attach_modal
from aosr.search.modal_record import summary_lines
from aosr.search.outer import conclude
from aosr.search.outer_status import OuterConclusion
from aosr.search.refine import RefineLedger, refine_order
from aosr.search.run import CandidateJob, ComputedCandidate
from aosr.search.store import SearchStore, check_project_furniture_layout
from tests.engine._crossover_cases import evaluate, prepared, protected
from tests.engine._modal_cases import runner
from tests.engine._search_blocked_cases import SavedFurnitureCompute, blocked_store
from tests.engine._search_refine_cases import RefineCompute, refine
from tests.engine._search_review_cases import with_matching
from tests.engine._search_run_cases import rows, run
from tests.engine.test_scheme_furniture import _validation_document


class MatchingFurnitureCompute(SavedFurnitureCompute):
    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in super().__call__(jobs, workers):
            candidate = with_matching(result.candidate, breached=result.job.trial_number == 0)
            document = json.loads(result.job.result_path.read_bytes())
            result.job.result_path.write_text(json.dumps(document | {"candidate": candidate.model_dump(mode="json")}))
            yield replace(result, candidate=candidate)


def test_eliminated_first_candidate_still_pins_comparison_identity(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path, budget=3)
    status = run(store, registry, MatchingFurnitureCompute(store))
    recorded = {row.trial_number: row for row in rows(store)}
    assert status.comparison_trial == 0
    assert recorded[0].reason == "eliminated" and recorded[0].score is None
    assert recorded[1].outcome == "scored" and recorded[2].outcome == "scored"


def test_refinement_uses_first_available_identity_in_job_order(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path, workers=4)
    run(store, registry, SavedFurnitureCompute(store))
    order = refine_order(rows(store))
    compute = RefineCompute(store, {}, missing=frozenset({order[0]}), different=frozenset({order[1]}))
    status = refine(store, registry, compute)
    recorded = RefineLedger.read(store.refine_ledger_path)[1]
    assert tuple(row.trial_number for row in recorded) == order
    by_number = {row.trial_number: row for row in recorded}
    assert by_number[order[0]].outcome == "not_evaluated"
    assert by_number[order[1]].outcome == "scored"
    assert all(by_number[number].outcome == "not_comparable" for number in order[2:])
    assert status.refine.best == order[1]


def test_refinement_single_and_multi_worker_are_bit_identical(tmp_path: Path) -> None:
    single, registry = blocked_store(tmp_path / "single")
    multi, other = blocked_store(tmp_path / "multi", workers=4)
    run(single, registry, SavedFurnitureCompute(single))
    run(multi, other, SavedFurnitureCompute(multi))
    assert refine(single, registry, RefineCompute(single, {})) == refine(multi, other, RefineCompute(multi, {}))
    assert RefineLedger.read(single.refine_ledger_path)[1] == RefineLedger.read(multi.refine_ledger_path)[1]


@pytest.mark.parametrize("short", [False, True])
def test_crossover_skips_absent_original_and_scores_other_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short: bool) -> None:
    store, registry, status = prepared(tmp_path)
    header, original_rows = RefineLedger.read(store.refine_ledger_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header)
    for row in original_rows:
        if row.trial_number is not None and (not short or row.trial_number == 7):
            book.append(row)
    store.refine_result_path(None).unlink()
    status = status.model_copy(update={"baseline_outcome": "direct_path_blocked"})
    monkeypatch.setattr(crossover_sensitivity, "reevaluate", evaluate)
    before = protected(store)
    summary = crossover_sensitivity.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.state == "done" and summary.verdict == ("unverified" if short else "sensitive")
    assert "原方案不符合擺位要求" in summary.reason_text
    assert any(variant.tested for variant in summary.variants) != short
    assert all(row.trial_number is not None for variant in summary.variants for row in variant.ranking)
    assert protected(store) == before


@pytest.mark.parametrize("conclusion", ["complete", "search_failed"])
def test_modal_keeps_original_skipped_and_other_candidates_can_run(tmp_path: Path, conclusion: OuterConclusion) -> None:
    store, registry = blocked_store(tmp_path, budget=3)
    status = run(store, registry, SavedFurnitureCompute(store))
    for row in rows(store):
        path = store.candidate_path(row.trial_number)
        path.write_text(json.dumps(json.loads(path.read_bytes()) | {"scope": "stage_two_subset"}))
    status = conclude(store, status, conclusion)
    summary = attach_modal(store, status=status, cache_dir=tmp_path / "cache",
                            runner=runner(tmp_path / "runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))
    roles = {role.role: role for role in summary.roles}
    assert roles["baseline"].state == "skipped"
    assert roles["baseline"].reason_text == "原方案不符合擺位要求"
    assert "結果範圍讀不回" not in "\n".join(summary_lines(summary))
    assert roles["search_best"].state == ("not_computed" if conclusion == "complete" else "skipped")
    assert not store.baseline_path.exists()


@pytest.mark.parametrize("case", ["outside", "facing"])
def test_project_layout_errors_use_shared_validation_words(case: str) -> None:
    scheme = Scheme.model_validate(_validation_document(case))
    with pytest.raises(SchemeValidationError) as caught:
        check_project_furniture_layout(scheme)
    assert caught.value.problems == furniture_problems(scheme)


@pytest.mark.parametrize("case", ["valid", "main", "side", "both", "two_block"])
def test_project_layout_validation_allows_blocking_paths(case: str) -> None:
    check_project_furniture_layout(Scheme.model_validate(_validation_document(case)))


def test_no_furniture_project_does_not_read_geometry_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = blocked_store(tmp_path)
    plain = Scheme.model_validate(store.project.model_dump() | {"furniture": None})

    def forbidden(path: Path) -> float:
        raise AssertionError("沒有家具不得讀家具精度登記簿")

    monkeypatch.setattr("aosr.search.store.furniture_contact_rel", forbidden)
    check_project_furniture_layout(plain)
