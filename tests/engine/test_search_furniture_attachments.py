"""被擋原方案不進附件計算；其他候選照原本入口驗證。

沒家具 skipped 收尾答案出自主線 9f4c6d2c 的審查員實跑：
prepared(conclusion="search_failed") → attach_modal →
record_attachment_error(..., KeyboardInterrupt(), stopped=True)。
出處：本票樹外施工資料的 evidence 目錄內 review-downstream.json 第一條，
三角色皆 stopped、摘要同原因。
"""
import json
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.scheme import Scheme
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity
from aosr.reporting.validation import SchemeValidationError, furniture_problems
from aosr.search import crossover_record, crossover_sensitivity
from aosr.search.modal_attach import attach_modal, record_attachment_error
from aosr.search.modal_record import read_summary, summary_lines
from aosr.search.outer import conclude
from aosr.search.labels import BASELINE_BLOCKED_TEXT
from aosr.search.outer_status import OuterConclusion, attachment_skip_reason
from aosr.search.refine import RefineLedger, refine_order
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus
from aosr.search.store import SearchStore, check_project_furniture_layout
from tests.engine._crossover_cases import evaluate, prepared, protected
from tests.engine._modal_cases import runner
from tests.engine._search_modal_cases import prepared as modal_prepared
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


def _blocked_crossover(tmp_path: Path, *, swap: bool = True) -> tuple[SearchStore, Path, SearchStatus]:
    """拿掉細算表的原方案列與結果檔，狀態記原方案被擋；其餘列照 _crossover_cases.prepared。"""
    store, registry, status = prepared(tmp_path, swap=swap)
    header, original_rows = RefineLedger.read(store.refine_ledger_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header)
    for row in original_rows:
        if row.trial_number is not None:
            book.append(row)
    store.refine_result_path(None).unlink()
    return store, registry, status.model_copy(update={"baseline_outcome": "direct_path_blocked"})


def test_crossover_blocked_sentence_stands_alone_when_stable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # swap=False 這份資料主線判穩定（test_search_crossover_sensitivity 的 (200, False, "stable")）；
    # 拿掉原方案後兩列名次不變，本來沒有原因，這句單獨成一行。
    store, registry, status = _blocked_crossover(tmp_path, swap=False)
    monkeypatch.setattr(crossover_sensitivity, "reevaluate", evaluate)
    summary = crossover_sensitivity.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.state == "done" and summary.verdict == "stable"
    assert summary.reason_text == BASELINE_BLOCKED_TEXT
    assert BASELINE_BLOCKED_TEXT in crossover_record.summary_lines(summary)


def test_crossover_blocked_sentence_leads_skip_reason(tmp_path: Path) -> None:
    store, registry, status = _blocked_crossover(tmp_path)
    status = conclude(store, status, "user_stopped")
    summary = crossover_sensitivity.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.state == "skipped" and summary.completed
    assert summary.reason_text == f"{BASELINE_BLOCKED_TEXT}；{attachment_skip_reason('user_stopped')}"


@pytest.mark.parametrize("stopped", [False, True])
def test_crossover_failure_or_stop_keeps_its_own_reason(tmp_path: Path, stopped: bool) -> None:
    store, _, status = _blocked_crossover(tmp_path)
    error: BaseException = KeyboardInterrupt() if stopped else RuntimeError("boom")
    crossover_sensitivity.record_crossover_error(store, status, error)
    summary = crossover_record.read_summary(store.path)
    assert summary is not None and summary.state == ("stopped" if stopped else "failed")
    # 失敗字句照 record_crossover_error 的寫法；停止字句取寫入端常數。
    assert summary.reason_text == (crossover_record.STOPPED_REASON if stopped else "交接敏感度計算失敗：RuntimeError：boom")
    assert BASELINE_BLOCKED_TEXT not in summary.reason_text


@pytest.mark.parametrize("official", [False, True])
def test_crossover_missing_anchor_identity_names_trial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, official: bool) -> None:
    store, registry, status = prepared(tmp_path)
    header, recorded = RefineLedger.read(store.refine_ledger_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header)
    for row in recorded:
        if row.trial_number is not None:
            book.append(row)
    status = status.model_copy(update={"baseline_outcome": "direct_path_blocked"})
    monkeypatch.setattr(crossover_sensitivity, "reevaluate", evaluate)
    original = crossover_sensitivity.Evaluator.pin
    calls: list[SchemeResult] = []

    def pin(self: crossover_sensitivity.Evaluator, result: SchemeResult,
            candidate: CandidateEvaluation) -> tuple[ComparisonIdentity, ...] | None:
        if official or calls:
            return None
        calls.append(result)
        return original(self, result, candidate)

    monkeypatch.setattr(crossover_sensitivity.Evaluator, "pin", pin)
    summary = crossover_sensitivity.attach_crossover(store, status=status, quality_targets_path=registry)
    reasons = [summary.reason_text] if official else [variant.reason_text for variant in summary.variants]
    assert any("試算 7算不出比較身分" in reason for reason in reasons)
    assert all("原方案算不出比較身分" not in reason for reason in reasons)


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


def test_plain_skipped_roles_become_stopped_on_attachment_error(tmp_path: Path) -> None:
    store, _, status = modal_prepared(tmp_path, conclusion="search_failed")
    cache = tmp_path / "cache"
    summary = attach_modal(store, status=status, cache_dir=cache)
    assert {role.state for role in summary.roles} == {"skipped"}
    record_attachment_error(store, status, cache, KeyboardInterrupt(), stopped=True)
    saved = read_summary(store.path)
    assert saved is not None
    reason = "已停止，沒有算完；人手接續後會再試"
    assert {role.role: (role.state, role.reason_text) for role in saved.roles} == {
        role: ("stopped", reason) for role in ("baseline", "search_best", "refine_best")
    }
    assert saved.reason_text == reason


@pytest.mark.parametrize("stopped", [False, True])
def test_blocked_original_stays_skipped_on_attachment_error(tmp_path: Path, stopped: bool) -> None:
    store, registry = blocked_store(tmp_path, budget=3)
    status = conclude(store, run(store, registry, SavedFurnitureCompute(store)), "search_failed")
    cache = tmp_path / "cache"
    attach_modal(store, status=status, cache_dir=cache)
    record_attachment_error(store, status, cache, KeyboardInterrupt() if stopped else RuntimeError("摘要寫失敗"),
                            stopped=stopped)
    saved = read_summary(store.path)
    assert saved is not None
    roles = {role.role: role for role in saved.roles}
    assert roles["baseline"].state == "skipped"
    assert roles["baseline"].reason_text == "原方案不符合擺位要求"
    assert roles["search_best"].state == ("stopped" if stopped else "failed")


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
