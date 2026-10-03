"""回饋中心、兩種無法回饋與完整性錯誤分清。"""

from pathlib import Path

import pytest

from aosr.search import cli, feedback as feedback_module, layout
from aosr.search.feedback import FeedbackEvent, FeedbackLedger, comparison_trial, feedback_search
from aosr.search.ledger import read_for
from aosr.search.run import SearchStatus, _write_status
from tests.engine._search_feedback_cases import prepared, resume, snapshot
from tests.engine._search_outer_cases import invoke
from tests.engine._search_refine_cases import RefineCompute, SearchCompute, refine
from tests.engine._search_run_cases import Killed, make_store, run
from tests.engine.test_search_refine_resume import clone


def test_round_one_reference_is_screening_first(tmp_path: Path) -> None:
    store, _, status = prepared(tmp_path)
    assert comparison_trial(store, status) == status.best_trial
    assert status.refine.best != status.best_trial


def test_round_two_reference_ignores_pending_next_event(tmp_path: Path) -> None:
    store, registry, first = prepared(tmp_path, refine_budget=4, refine_convergence=2, anchor_number=1)
    feedback_search(store)
    second = resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    changed = next(row.trial_number for row in read_for(store).rows
                   if row.outcome == "scored" and row.trial_number not in (first.refine.best, second.best_trial))
    assert comparison_trial(store, second) == first.refine.best
    future = FeedbackEvent(round=3, before_batch=second.asked // store.settings.batch_size, anchor_trial=changed,
                           points=({name: (0.5).hex() for name in layout.SEARCH_QUANTITIES},))
    FeedbackLedger.append(store.feedback_path, future)
    assert comparison_trial(store, second) == first.refine.best


def test_round_two_feedback_rejects_unchanged_center(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store, registry, first = prepared(tmp_path, refine_budget=4, refine_convergence=2, anchor_number=1)
    feedback_search(store)
    resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    second = refine(store, registry, RefineCompute(store, {None: 4.0, 1: 0.1}))
    assert second.refine.best == first.refine.best and second.refine.best != second.best_trial
    before = snapshot(store)
    exit_code = cli.main(["feedback", str(store.path)])
    assert exit_code == 1
    assert "第一名沒換" in capsys.readouterr().err
    assert snapshot(store) == before


def test_round_two_feedback_accepts_screening_first_when_center_changed(tmp_path: Path) -> None:
    store, registry, first = prepared(tmp_path, refine_budget=4, refine_convergence=2, anchor_number=1)
    feedback_search(store)
    second = resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True,
                    values={first.asked: 0.5, first.asked + 1: 0.5}))
    final = refine(store, registry, RefineCompute(store, {None: 4.0, second.best_trial: 0.01}))
    assert final.refine.best == second.best_trial and final.refine.best != first.refine.best
    exit_code = cli.main(["feedback", str(store.path)])
    assert exit_code == 0
    assert FeedbackLedger.read(store.feedback_path)[-1].anchor_trial == second.best_trial


def test_auto_no_remaining_feedback_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path, offset=1e-320, refine_budget=4, refine_convergence=2, anchor_number=1)
    assert status.refine.stop_reason == "stable"
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.outer.conclusion == "feedback_unavailable"
    assert after.model_dump(exclude={"outer"}) == status.model_dump(exclude={"outer"})
    assert not store.feedback_path.exists()


def test_auto_previous_feedback_points_not_finished(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = make_store(tmp_path, batch=2, budget=40, convergence=3,
                                 refine={"budget": 30, "convergence_run": 2}, feedback={"offset": 0.125})
    run(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    refine(store, registry, RefineCompute(store, {None: 4.0, 1: 0.1}))
    feedback_search(store)
    second = resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True,
                    values={4: 0.5, 5: 0.5}))
    event, = FeedbackLedger.read(store.feedback_path)
    assert second.asked - event.before_batch * store.settings.batch_size < len(event.points)
    final = refine(store, registry, RefineCompute(store, {None: 4.0, second.best_trial: 0.01}))
    assert final.refine.best != event.anchor_trial and final.refine.stop_reason == "stable"
    before = store.feedback_path.read_bytes()
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).outer.conclusion == "feedback_unavailable"
    assert store.feedback_path.read_bytes() == before


def test_auto_killed_between_feedback_event_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, _ = prepared(tmp_path, refine_budget=4, refine_convergence=2, anchor_number=1)
    whole = clone(store, tmp_path / "whole")
    exit_code = invoke(whole, registry, monkeypatch)
    assert exit_code == 0
    def killed(*args: object) -> None:
        raise Killed("被砍")
    with monkeypatch.context() as patch:
        patch.setattr(feedback_module, "_write_status", killed)
        with pytest.raises(Killed):
            invoke(store, registry, patch)
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).outer.conclusion is None
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    for path in ("ledger_path", "refine_ledger_path", "feedback_path", "status_path"):
        assert getattr(store, path).read_bytes() == getattr(whole, path).read_bytes()


@pytest.mark.parametrize("damage", ["ledger", "pending"])
def test_auto_feedback_integrity_error_does_not_write_conclusion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], damage: str) -> None:
    store, registry, status = prepared(tmp_path, refine_budget=4, refine_convergence=2, anchor_number=1)
    if damage == "ledger":
        _write_status(store, status.model_copy(update={"asked": status.asked + store.settings.batch_size}))
    else:
        event = FeedbackEvent(round=2, before_batch=status.asked // store.settings.batch_size, anchor_trial=1,
                              points=({name: (0.5).hex() for name in layout.SEARCH_QUANTITIES},))
        FeedbackLedger.append(store.feedback_path, event)
    before = snapshot(store)
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 1
    assert "對不上" in capsys.readouterr().err
    assert snapshot(store) == before


def test_auto_round_two_pending_event_uses_current_center_then_replays(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, first = prepared(tmp_path, refine_budget=4, refine_convergence=2, anchor_number=1)
    feedback_search(store)
    second = resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True,
                    values={first.asked: 0.5, first.asked + 1: 0.5}))
    final = refine(store, registry, RefineCompute(store, {None: 4.0, second.best_trial: 0.01}))
    assert final.refine.best != first.refine.best and final.refine.stop_reason == "stable"
    whole = clone(store, tmp_path / "whole")
    exit_code = invoke(whole, registry, monkeypatch)
    assert exit_code == 0
    def killed(*args: object) -> None:
        raise Killed("被砍")
    with monkeypatch.context() as patch:
        patch.setattr(feedback_module, "_write_status", killed)
        with pytest.raises(Killed):
            feedback_search(store)
    assert FeedbackLedger.read(store.feedback_path)[-1].round == final.round + 1
    exit_code = invoke(store, registry, monkeypatch)
    assert exit_code == 0
    after = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert after.round > final.round
    for path in ("ledger_path", "refine_ledger_path", "feedback_path", "status_path"):
        assert getattr(store, path).read_bytes() == getattr(whole, path).read_bytes()
