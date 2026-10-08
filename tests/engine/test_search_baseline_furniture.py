"""原方案照 B4 跳過；候選依編號釘身分與逐位接續。"""
import json
from pathlib import Path

import pytest

from aosr.search import run as search_run
from aosr.search.ledger import Ledger
from aosr.search.report import build_report, render_text
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore
from tests.engine._search_blocked_cases import DelayedFurnitureCompute, SavedFurnitureCompute, blocked_store
from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, Killed, make_store, next_params, rows, run


def resume(store: SearchStore, registry: Path, compute: FakeCompute) -> SearchStatus:
    return search_run.resume_search(store, compute=compute, probe=lambda: store.identity,
                                    registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)


def test_blocked_baseline_search_reaches_stop_without_baseline_score(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path)
    compute = SavedFurnitureCompute(store, flat=True)
    status = run(store, registry, compute)
    assert status.state == "converged"
    assert status.baseline_outcome == "direct_path_blocked"
    assert status.baseline_reason_codes == ("direct_path_blocked",)
    assert status.comparison_trial is not None
    assert None not in compute.calls and not store.baseline_path.exists()
    assert status.best_trial is not None and status.best_score is not None
    assert "原方案不符合擺位要求" in render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))


def test_blocked_baseline_single_and_multi_worker_runs_are_bit_identical(tmp_path: Path) -> None:
    single, registry = blocked_store(tmp_path / "single")
    multi, other = blocked_store(tmp_path / "multi", workers=4)
    one, four = SavedFurnitureCompute(single), SavedFurnitureCompute(multi)
    assert run(single, registry, one) == run(multi, other, four)
    assert rows(single) == rows(multi)
    assert next_params(single) == next_params(multi)
    assert one.calls != four.calls
    assert None not in one.calls and None not in four.calls


@pytest.mark.parametrize("kill_after", [1, 4])
def test_blocked_baseline_search_resume_is_bit_identical(tmp_path: Path, kill_after: int) -> None:
    whole, registry = blocked_store(tmp_path / "whole")
    expected = run(whole, registry, SavedFurnitureCompute(whole))
    interrupted, other = blocked_store(tmp_path / "interrupted")
    with pytest.raises(Killed):
        run(interrupted, other, SavedFurnitureCompute(interrupted, fail_after=kill_after, kill=True))
    continued = SavedFurnitureCompute(interrupted)
    assert resume(interrupted, other, continued) == expected
    assert rows(interrupted) == rows(whole)
    assert next_params(interrupted) == next_params(whole)
    assert None not in continued.calls and not interrupted.baseline_path.exists()


@pytest.mark.parametrize("damage", ["deleted", "archived", "candidate", "physics_identity", "program_fingerprint", "purpose_settings"])
def test_pinning_result_unreadable_or_changed_interrupts_resume(tmp_path: Path, damage: str) -> None:
    store, registry = blocked_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, SavedFurnitureCompute(store, fail_after=4, kill=True))
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.comparison_trial is not None
    target = store.candidate_path(status.comparison_trial)
    if damage == "deleted":
        target.unlink()
    elif damage == "archived":
        target.rename(target.with_suffix(".archived"))
    else:
        document = json.loads(target.read_bytes())
        if damage == "candidate":
            document["candidate"]["candidate_id"] = "other-candidate"
        elif damage == "purpose_settings":
            document[damage]["fingerprint"] = "different"
        else:
            document[damage] = "different"
        target.write_text(json.dumps(document))
    before = rows(store)
    compute = SavedFurnitureCompute(store)
    assert resume(store, registry, compute).state == "interrupted"
    assert not compute.calls and rows(store) == before


def test_no_candidate_can_pin_stops_honestly_without_first_place(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path, budget=3)
    status = run(store, registry, SavedFurnitureCompute(store, missing=frozenset(range(store.settings.budget))))
    assert status.state == "budget_exhausted" and status.best_trial is None and status.best_score is None
    assert status.comparison_trial is None
    assert all(row.score is None for row in rows(store))
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert "沒有任何候選拿到分數" in text and "原方案不符合擺位要求" in text


def test_old_status_defaults_to_no_candidate_pin() -> None:
    assert SearchStatus.model_validate({"baseline_outcome": "scored"}).comparison_trial is None


def test_pin_follows_trial_number_when_first_finishes_later_in_processes(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path, workers=4, budget=3)
    compute = DelayedFurnitureCompute(store)
    status = run(store, registry, compute)
    assert compute.calls.index(1) < compute.calls.index(0)
    assert status.comparison_trial == 0
    recorded = {row.trial_number: row for row in rows(store)}
    assert recorded[0].outcome == "scored"
    assert recorded[1].score is None and recorded[1].reason == "incomparable"
    assert tuple(row.trial_number for row in Ledger.read(store.ledger_path)[1]) == tuple(sorted(recorded))


def test_candidate_before_pin_with_missing_identity_stays_unassessed(tmp_path: Path) -> None:
    store, registry = blocked_store(tmp_path, budget=3)
    status = run(store, registry, SavedFurnitureCompute(store, missing=frozenset({0})))
    recorded = {row.trial_number: row for row in rows(store)}
    assert recorded[0].reason == "unassessed" and recorded[0].score is None
    assert status.comparison_trial == 1


@pytest.mark.parametrize("furniture", [False, True])
def test_normal_baseline_keeps_per_completion_rows_when_killed(tmp_path: Path, furniture: bool) -> None:
    store, registry = blocked_store(tmp_path, blocked=False) if furniture else make_store(tmp_path)
    compute = FakeCompute(store, fail_after=1, kill=True)
    with pytest.raises(Killed):
        run(store, registry, compute)
    # 主線是完成一個就存一列；半批不等全批到齊。
    assert {row.trial_number for row in Ledger.read(store.ledger_path)[1]} == set(compute.calls) - {None}
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).comparison_trial is None
