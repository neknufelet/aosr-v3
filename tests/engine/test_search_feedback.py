"""設定、點序、拒絕不改檔與手動回饋的契約考卷。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aosr.search import layout
from aosr.search.cli import main
from aosr.search.ledger import LedgerRow, read_for, row_from_outcome
from aosr.search.run import SearchStatus, _write_status
from aosr.search.sampler import Proposal, Scored
from aosr.search.settings import SearchSettings
from aosr.search.store import candidate_name
from tests.engine._search_feedback_cases import prepared, snapshot
from tests.engine._search_store_cases import settings_document


def test_optional_feedback_preserves_old_fingerprint_and_status(tmp_path: Path) -> None:
    document = settings_document()
    old = SearchSettings.model_validate(document)
    digest = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert old.fingerprint == digest
    assert "feedback" not in old.canonical()
    assert "feedback" not in json.loads(old.model_dump_json())
    changed = SearchSettings.model_validate(document | {"feedback": {"offset": 0.125}})
    assert changed.fingerprint != digest
    assert changed.canonical()["feedback"] == {"offset": 0.125}
    assert json.loads(changed.model_dump_json())["feedback"] == {"offset": 0.125}
    assert SearchSettings.model_validate(document | {"feedback": None}).fingerprint == digest
    path = tmp_path / "old-status"
    path.write_text('{"state":"converged","asked":6}')
    status = SearchStatus.model_validate_json(path.read_bytes())
    assert (status.round, status.round_start_trial, status.rounds) == (1, 0, ())


@pytest.mark.parametrize("feedback", [{}, {"offset": 0}, {"offset": -0.125}, {"offset": 0.5001},
                                      {"offset": float("nan")}, {"offset": float("inf")}])
def test_feedback_offset_is_required_and_bounded(tmp_path: Path, feedback: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SearchSettings.model_validate(settings_document() | {"feedback": feedback})
    assert tmp_path.is_dir()


@pytest.mark.parametrize("offset", [0.125, 0.5])
def test_feedback_accepts_offset_boundary(tmp_path: Path, offset: float) -> None:
    settings = SearchSettings.model_validate(settings_document() | {"feedback": {"offset": offset}})
    assert settings.canonical()["feedback"] == {"offset": offset}
    assert tmp_path.is_dir()


def _row(number: int, values: tuple[float, float, float]) -> LedgerRow:
    return row_from_outcome(batch_index=0, proposal=Proposal(number, dict(zip(layout.SEARCH_QUANTITIES, values, strict=True))),
                            params_m=dict.fromkeys(layout.SEARCH_QUANTITIES, 1.0),
                            outcome=Scored(float(number)), seconds=0.0, result_file=candidate_name(number))


def test_points_are_clipped_deduplicated_and_ordered(tmp_path: Path) -> None:
    from aosr.search.feedback import feedback_points

    center = _row(0, (0.125, 0.5, 0.875))
    expected = ((0.0, 0.5, 0.875), (0.375, 0.5, 0.875), (0.125, 0.25, 0.875),
                (0.125, 0.75, 0.875), (0.125, 0.5, 0.625), (0.125, 0.5, 1.0))
    points = feedback_points(center, (center,), 0.25)
    assert tuple(tuple(point[name] for name in layout.SEARCH_QUANTITIES) for point in points) == expected
    blocked = _row(1, expected[2])
    remaining = feedback_points(center, (center, blocked), 0.25)
    assert tuple(tuple(point[name] for name in layout.SEARCH_QUANTITIES) for point in remaining) == expected[:2] + expected[3:]
    edge = _row(0, (0.0, 0.5, 1.0))
    assert tuple(tuple(point[name] for name in layout.SEARCH_QUANTITIES)
                 for point in feedback_points(edge, (edge,), 0.25)) == (
                     (0.25, 0.5, 1.0), (0.0, 0.25, 1.0), (0.0, 0.75, 1.0), (0.0, 0.5, 0.75))
    with pytest.raises(ValueError, match="沒有可回饋"):
        feedback_points(center, (center, *(_row(i + 1, point) for i, point in enumerate(expected))), 0.25)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("case,reason", [
    ("budget_state", "搜尋預算已用完，回饋跑不了"), ("not_converged", "搜尋"),
    ("no_settings", "feedback"), ("refine_running", "細算"), ("refine_empty", "細算"),
    ("baseline", "細算第一名是原方案，沒有可回饋的點"),
    ("unchanged", "細算第一名跟篩選第一名相同，不需要回饋"),
    ("round", "輪次"), ("budget", "預算"), ("partial_boundary", "批界"),
])
def test_feedback_command_rejects_each_precondition_without_changes(
        tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str, reason: str) -> None:
    store, _, status = prepared(tmp_path, feedback=case != "no_settings")
    changes: dict[str, object] = {}
    refinement: dict[str, object] = {}
    if case == "budget_state":
        changes["state"] = "budget_exhausted"
    elif case == "not_converged":
        changes["state"] = "running"
    elif case == "refine_running":
        refinement.update(state="running", stop_reason=None)
    elif case == "refine_empty":
        refinement["best"] = None
    elif case == "baseline":
        refinement["best"] = "baseline"
    elif case == "unchanged":
        refinement["best"] = status.best_trial
    elif case == "round":
        refinement["round"] = status.refine.round + 1
    elif case == "budget":
        changes["asked"] = store.settings.budget
    elif case == "partial_boundary":
        changes["asked"] = status.asked + 1
    if refinement:
        changes["refine"] = status.refine.model_copy(update=refinement)
    _write_status(store, status.model_copy(update=changes))
    before = snapshot(store)
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 1
    assert reason in capsys.readouterr().err
    assert snapshot(store) == before


def test_success_records_event_and_round_without_changing_refinement(tmp_path: Path) -> None:
    from aosr.search.feedback import FeedbackLedger

    store, _, stopped = prepared(tmp_path)
    before = snapshot(store)
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 0
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    events = FeedbackLedger.read(store.feedback_path)
    assert [(event.round, event.before_batch, event.anchor_trial) for event in events] == [
        (2, stopped.asked // store.settings.batch_size, stopped.refine.best)]
    anchor = next(row for row in read_for(store).rows if row.trial_number == stopped.refine.best)
    x, y, z = (float.fromhex(anchor.unit_params_hex[name]) for name in layout.SEARCH_QUANTITIES)
    candidates = ((max(0.0, x - 0.125), y, z), (min(1.0, x + 0.125), y, z),
                  (x, max(0.0, y - 0.125), z), (x, min(1.0, y + 0.125), z),
                  (x, y, max(0.0, z - 0.125)), (x, y, min(1.0, z + 0.125)))
    existing = {tuple(row.unit_params_hex[name] for name in layout.SEARCH_QUANTITIES) for row in read_for(store).rows}
    expected = tuple(tuple(value.hex() for value in point) for point in candidates
                     if tuple(value.hex() for value in point) not in existing)
    assert tuple(tuple(point[name] for name in layout.SEARCH_QUANTITIES) for point in events[0].points) == expected
    assert status.state == "running" and status.round == 2
    assert status.round_start_trial == stopped.asked
    assert [(record.round, record.state, record.message, record.asked, record.best_trial, record.best_score)
            for record in status.rounds] == [
                (1, stopped.state, stopped.message, stopped.asked, stopped.best_trial, stopped.best_score)]
    assert "第 2 輪" in status.message and str(stopped.refine.best) in status.message
    assert str(len(events[0].points)) in status.message
    assert status.refine.model_dump_json() == stopped.refine.model_dump_json()
    after = snapshot(store)
    for path, content in before.items():
        if path != str(store.status_path.relative_to(store.path)):
            assert after[path] == content


def test_feedback_command_rejects_when_every_point_equals_center(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    store, _, _ = prepared(tmp_path, offset=1e-320)
    before = snapshot(store)
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 1
    assert "沒有可回饋" in capsys.readouterr().err
    assert snapshot(store) == before
