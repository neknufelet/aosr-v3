"""鎖定搜尋只要兩維題、三格公尺；完成順序與中斷位置不改答案。"""
from pathlib import Path

import pytest

from aosr.search import feedback, layout, ledger
from aosr.search.run import RoundRecord, SearchStatus, _write_status, resume_search
from tests.engine._seat_locked_cases import locked_store
from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, Killed, rows, run, next_params


def test_locked_single_and_multi_worker_runs_are_bit_identical(tmp_path: Path) -> None:
    single, registry = locked_store(tmp_path / "single", budget=41)
    multi, other = locked_store(tmp_path / "multi", workers=4, budget=41)
    one, four = FakeCompute(single), FakeCompute(multi)
    first, second = run(single, registry, one), run(multi, other, four)
    assert first == second
    assert rows(single) == rows(multi)
    assert next_params(single) == next_params(multi)
    assert one.calls != four.calls
    assert all(row.unit_params_hex.keys() == {"front_distance", "spacing"} for row in rows(single))
    assert all(row.params_m["listening_distance"].hex() == (3.2 - row.params_m["front_distance"]).hex()
               for row in rows(single))
    assert all(not {"seat_in_keep_out", "seat_outside_room"}.intersection((row.reason or "").split("+")) for row in rows(single))


@pytest.mark.parametrize("keep_baseline", [True, False])
def test_locked_resume_after_crash_continues_bit_identically(tmp_path: Path, keep_baseline: bool) -> None:
    whole, registry = locked_store(tmp_path / "whole", budget=41)
    split, other = locked_store(tmp_path / "split", budget=41)
    expected = run(whole, registry, FakeCompute(whole))
    with pytest.raises(Killed):
        run(split, other, FakeCompute(split, fail_after=5, kill=True, persist_baseline=True))
    saved = rows(split)
    if not keep_baseline:
        split.baseline_path.unlink()
    compute = FakeCompute(split)
    actual = resume_search(split, compute=compute, probe=lambda: split.identity,
        registry_path=other, run_date=RUN_DATE, engine_version=ENGINE)
    assert actual == expected
    assert rows(split) == rows(whole)
    assert next_params(split) == next_params(whole)
    assert not {row.trial_number for row in saved}.intersection(compute.calls)
    assert (None not in compute.calls) == keep_baseline


def test_locked_start_first_trial_is_hand_calculated_clamped_spacing(tmp_path: Path) -> None:
    store, registry = locked_store(tmp_path)
    status = run(store, registry, FakeCompute(store))
    # 原 F=1，L=2.2；60° 推 S=2.540...，夾至間距上限 2。
    assert status.start_enqueued
    assert rows(store)[0].unit_params_hex == {"front_distance": (3.0 / 7.0).hex(), "spacing": 1.0.hex()}
    assert rows(store)[0].params_m == {"front_distance": 1.0, "spacing": 2.0, "listening_distance": 2.2}


def test_locked_blocked_baseline_uses_two_dimensional_unpinned_batch(tmp_path: Path) -> None:
    from tests.engine._seat_locked_cases import LOCKED
    from tests.engine._search_blocked_cases import blocker
    from tests.engine._search_run_cases import make_store

    store, registry = make_store(tmp_path, budget=1, batch=1, layout_changes=LOCKED, furniture=[blocker()])
    compute = FakeCompute(store)
    status = run(store, registry, compute)
    assert status.baseline_outcome == "direct_path_blocked"
    assert not store.baseline_path.exists() and status.best_score is None
    first, = rows(store)
    assert first.unit_params_hex == {"front_distance": (3.0 / 7.0).hex(), "spacing": 1.0.hex()}
    assert first.params_m == {"front_distance": 1.0, "spacing": 2.0, "listening_distance": 2.2}
    assert first.reason == "direct_path_blocked"
    assert not compute.calls


@pytest.mark.parametrize("locked", [True, False])
def test_wrong_feedback_dimension_interrupts_without_appending(tmp_path: Path, locked: bool) -> None:
    from tests.engine._search_run_cases import make_store

    store, registry = (locked_store(tmp_path, budget=41, convergence=1, feedback={"offset": 0.25}) if locked
                       else make_store(tmp_path, budget=41, convergence=1, feedback={"offset": 0.25}))
    stopped = run(store, registry, FakeCompute(store, flat=True, persist_baseline=True))
    assert stopped.state == "converged"
    point = {name: 0.5.hex() for name in (layout.SEARCH_QUANTITIES if locked else ("front_distance", "spacing"))}
    event = feedback.FeedbackEvent(round=2, before_batch=stopped.asked // store.settings.batch_size,
        anchor_trial=0, points=(point,))
    feedback.FeedbackLedger.append(store.feedback_path, event)
    record = RoundRecord(**stopped.model_dump(include=set(RoundRecord.model_fields)))
    _write_status(store, stopped.model_copy(update={"state": "running", "round": 2,
        "round_start_trial": stopped.asked, "rounds": (record,)}))
    before = store.ledger_path.read_bytes()
    actual = resume_search(store, compute=FakeCompute(store), probe=lambda: store.identity,
        registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert actual.state == "interrupted"
    assert "回饋點的量名與搜尋空間不同" in actual.message
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).state == "interrupted"
    assert store.ledger_path.read_bytes() == before


def test_locked_feedback_replay_and_killed_resume_are_bit_identical(tmp_path: Path) -> None:
    from aosr.search.cli import main
    from tests.engine._seat_locked_cases import LOCKED
    from tests.engine._search_feedback_cases import prepared, resume
    from tests.engine._search_refine_cases import SearchCompute
    from tests.engine.test_search_refine_resume import clone

    single, registry, stopped = prepared(tmp_path / "single", layout_changes=LOCKED)
    multi, other, _ = prepared(tmp_path / "multi", workers=4, layout_changes=LOCKED)
    exit_code = main(["feedback", str(single.path)])
    assert exit_code == 0
    exit_code = main(["feedback", str(multi.path)])
    assert exit_code == 0
    event, = feedback.FeedbackLedger.read(single.feedback_path)
    assert all(point.keys() == {"front_distance", "spacing"} for point in event.points)
    split = clone(single, tmp_path / "split")
    expected = resume(single, registry, SearchCompute(single, flat=True, persist_baseline=True))
    actual = resume(multi, other, SearchCompute(multi, flat=True, persist_baseline=True))
    assert actual == expected
    assert rows(single) == rows(multi)
    assert single.feedback_path.read_bytes() == multi.feedback_path.read_bytes()
    assert [r.unit_params_hex for r in rows(single)[stopped.asked:stopped.asked + len(event.points)]] == list(event.points)
    with pytest.raises(Killed):
        resume(split, registry, SearchCompute(split, flat=True, persist_baseline=True, fail_after=1, kill=True))
    continued = resume(split, registry, SearchCompute(split, flat=True, persist_baseline=True))
    assert continued == expected
    assert single.ledger_path.read_bytes() == split.ledger_path.read_bytes()
    assert single.status_path.read_bytes() == split.status_path.read_bytes()
