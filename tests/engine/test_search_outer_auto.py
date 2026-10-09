"""自動外圈從各步與被砍處接續；只用假計算。"""

from pathlib import Path

import pytest

from aosr.search import ledger
from aosr.search.feedback import FeedbackLedger, feedback_search
from aosr.search.run import SearchStatus, _write_status
from aosr.search.store import SearchStore
from tests.engine._search_feedback_cases import prepared, resume
from tests.engine._search_outer_cases import OuterCompute, invoke
from tests.engine._search_refine_cases import RefineCompute, SearchCompute, refine
from tests.engine._search_run_cases import Killed, make_store
from tests.engine.test_search_refine_resume import clone


def test_auto_really_completes_stability_with_full_refinement_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.config.paths import config_path
    from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
    from aosr.search import cli
    from aosr.search.placement_stability import SHIFT_NAMES
    from aosr.search.placement_stability_record import read_summary
    from tests.engine._modal_cases import runner
    from tests.engine._search_outer_cases import CompleteOuterCompute
    from tests.engine._stability_attach_cases import ShiftCompute
    source, registry = seed(tmp_path / "input")
    store = SearchStore.create(tmp_path / "searches", project=source.project.model_copy(update={"source_model": "omnidirectional"}),
        settings=source.settings, identity=source.identity, versions=source.versions)
    ledger.create_for(store)
    _write_status(store, SearchStatus())
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    shifted = ShiftCompute(store)
    code = cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
        compute_factory=lambda *a: CompleteOuterCompute(store), stability_compute_factory=lambda root: shifted,
        modal_runner=runner(tmp_path / "modal-runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))
    assert code == 0
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "done" and summary.completed
    assert shifted.jobs and shifted.batches == [tuple(shifted.jobs)]
    assert store.scheme_path_for(store.refine_result_path(None)).is_file()
    assert {(j.trial_number, j.shift_name) for j in shifted.jobs} == {
        (f.trial_number, name) for f in summary.selection.finalists for name in SHIFT_NAMES}
    assert summary.total_points == summary.computed_points == len(shifted.jobs)
    assert summary.arithmetic is not None
    assert {b.trial_number for b in summary.boundaries} == {f.trial_number for f in summary.selection.finalists}
    assert {r.finalist.trial_number for r in summary.arithmetic.finalists} == {f.trial_number for f in summary.selection.finalists}


def seed(tmp_path: Path) -> tuple[SearchStore, Path]:
    store, registry = make_store(tmp_path, batch=2, budget=40, convergence=7,
                                 refine={"budget": 20, "convergence_run": 2}, feedback={"offset": 0.125})
    ledger.create_for(store)
    _write_status(store, SearchStatus())
    return store, registry


@pytest.mark.parametrize("checkpoint", ["running", "search_stopped", "refine_running", "refine_stopped", "feedback_written"])
def test_auto_each_checkpoint_matches_whole_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str) -> None:
    store, registry = seed(tmp_path / "seed")
    whole = clone(store, tmp_path / "whole")
    exit_code = invoke(whole, registry, monkeypatch)
    assert exit_code == 0
    expected = SearchStatus.model_validate_json(whole.status_path.read_bytes())
    assert expected.outer.conclusion == "complete"
    assert expected.round == 2 and expected.refine.round == 2
    if checkpoint != "running":
        resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    if checkpoint == "refine_running":
        with pytest.raises(Killed):
            refine(store, registry, RefineCompute(store, {None: 4.0, 1: 0.1}, kill_after=2))
    if checkpoint in ("refine_stopped", "feedback_written"):
        refine(store, registry, RefineCompute(store, {None: 4.0, 1: 0.1}))
    if checkpoint == "feedback_written":
        feedback_search(store)
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    for path in ("ledger_path", "refine_ledger_path", "feedback_path", "status_path"):
        assert getattr(store, path).read_bytes() == getattr(whole, path).read_bytes()


def test_auto_killed_at_every_computation_matches_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = seed(tmp_path / "seed")
    whole = clone(store, tmp_path / "whole")
    compute = OuterCompute(whole)
    exit_code = invoke(whole, registry, monkeypatch, compute)
    assert exit_code == 0
    for position in range(compute.completed):
        cut = clone(store, tmp_path / f"cut-{position}")
        with pytest.raises(Killed):
            invoke(cut, registry, monkeypatch, OuterCompute(cut, kill_after=position))
        assert SearchStatus.model_validate_json(cut.status_path.read_bytes()).outer.conclusion is None
        exit_code = invoke(cut, registry, monkeypatch)
        assert exit_code == 0
        for path in ("ledger_path", "refine_ledger_path", "feedback_path", "status_path"):
            assert getattr(cut, path).read_bytes() == getattr(whole, path).read_bytes()


@pytest.mark.parametrize("marker", ["stop_path", "refine_stop_path"])
def test_auto_keeps_stop_markers_and_other_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, marker: str) -> None:
    store, registry, before = prepared(tmp_path)
    path = getattr(store, marker)
    path.touch()
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.outer.conclusion == "user_stopped" and path.exists()
    assert after.model_dump(exclude={"outer"}) == before.model_dump(exclude={"outer"})


@pytest.mark.parametrize("search,refinement,want,code", [
    ("failed", "not_started", "search_failed", 1),
    ("interrupted", "not_started", "search_interrupted", 3),
    ("user_stopped", "not_started", "user_stopped", 0),
    ("converged", "failed", "refine_failed", 1),
    ("converged", "interrupted", "refine_interrupted", 3),
])
def test_auto_terminal_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                 search: str, refinement: str, want: str, code: int) -> None:
    store, registry, status = prepared(tmp_path)
    status = SearchStatus.model_validate(status.model_dump() | {"state": search,
        "refine": status.refine.model_dump() | {"state": refinement, "stop_reason": None}})
    _write_status(store, status)
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == code
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.outer.conclusion == want
    assert after.model_dump(exclude={"outer"}) == status.model_dump(exclude={"outer"})


@pytest.mark.parametrize("reason,search,best,want", [
    ("stable", "converged", "same", "complete"),
    ("stable", "budget_exhausted", "same", "stable_but_search_budget"),
    ("refine_budget", "converged", "changed", "refine_budget"),
    ("candidates_exhausted", "converged", "changed", "refine_candidates_exhausted"),
    ("stable", "budget_exhausted", "changed", "feedback_blocked_by_budget"),
    ("stable", "converged", "baseline", "baseline_first"),
    ("stable", "converged", None, "no_rankable_first"),
    ("user_stopped", "converged", "changed", "user_stopped"),
])
def test_auto_nonfailure_conclusions_preserve_both_states(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                          reason: str, search: str, best: str | None, want: str) -> None:
    store, registry, status = prepared(tmp_path)
    first = status.best_trial if best == "same" else status.refine.best if best == "changed" else best
    before = SearchStatus.model_validate(status.model_dump() | {"state": search,
        "refine": status.refine.model_dump() | {"stop_reason": reason, "best": first}})
    _write_status(store, before)
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.outer.conclusion == want
    assert after.model_dump(exclude={"outer"}) == before.model_dump(exclude={"outer"})


def test_auto_without_feedback_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path, feedback=False, refine_budget=4, refine_convergence=2, anchor_number=1)
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.outer.conclusion == "feedback_not_configured"
    assert after.model_dump(exclude={"outer"}) == status.model_dump(exclude={"outer"})


@pytest.mark.parametrize("phase", ["search_stop", "refine_stop", "feedback_state", "conclusion"])
def test_auto_killed_at_status_boundaries_replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    from aosr.search import feedback as feedback_module, outer, refine_run, run as run_module

    store, registry = seed(tmp_path / "seed")
    whole = clone(store, tmp_path / "whole")
    exit_code = invoke(whole, registry, monkeypatch)
    assert exit_code == 0
    module = {"search_stop": run_module, "refine_stop": refine_run,
              "feedback_state": feedback_module, "conclusion": outer}[phase]
    original = _write_status
    cut = False
    def killed(opened: SearchStore, status: SearchStatus) -> SearchStatus:
        nonlocal cut
        matches = ((phase == "search_stop" and status.state == "converged") or
                   (phase == "refine_stop" and status.refine.state == "stopped") or
                   phase in ("feedback_state", "conclusion"))
        if matches and not cut:
            cut = True
            raise Killed("寫狀態前被砍")
        return original(opened, status)
    with monkeypatch.context() as patch:
        patch.setattr(module, "_write_status", killed)
        with pytest.raises(Killed):
            invoke(store, registry, patch)
    assert cut
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).outer.conclusion is None
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    for path in ("ledger_path", "refine_ledger_path", "feedback_path", "status_path"):
        assert getattr(store, path).read_bytes() == getattr(whole, path).read_bytes()


@pytest.mark.parametrize("phase", ["search", "refine"])
@pytest.mark.parametrize("fault", ["failed", "interrupted"])
def test_auto_records_execution_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capsys: pytest.CaptureFixture[str], phase: str, fault: str) -> None:
    from collections.abc import Iterator, Sequence
    from dataclasses import replace
    from aosr.search.run import CandidateJob, ComputedCandidate

    store, registry = seed(tmp_path)
    if phase == "refine":
        resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    before = SearchStatus.model_validate_json(store.status_path.read_bytes())
    fake = OuterCompute(store)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        if fault == "failed":
            raise RuntimeError("計算原始錯誤")
        for result in fake(jobs, workers):
            yield replace(result, identity=replace(store.identity, physics_identity="changed"))
    exit_code = invoke(store, registry, monkeypatch, compute)
    expected_code = 1 if fault == "failed" else 3
    assert exit_code == expected_code
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.outer.conclusion == f"{phase}_{fault}"
    if phase == "refine":
        assert after.model_dump(exclude={"outer", "refine"}) == before.model_dump(exclude={"outer", "refine"})
    if fault == "failed":
        assert "計算原始錯誤" in capsys.readouterr().err


def test_stop_marker_precedes_simultaneous_compute_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Iterator, Sequence
    from aosr.search.run import CandidateJob, ComputedCandidate

    store, registry = seed(tmp_path)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        store.refine_stop_path.touch()
        raise RuntimeError("同時停止與失敗")
    exit_code = invoke(store, registry, monkeypatch, compute)
    assert exit_code == 0
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.outer.conclusion == "user_stopped"
    assert store.refine_stop_path.exists()
    assert after.state == "failed" and "同時停止與失敗" in after.message
