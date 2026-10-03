"""回饋之後接續、跨批與被砍時逐位重播。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aosr.search.cli import main
from aosr.search.ledger import read_for
from aosr.search.run import SearchStatus, _progress
from aosr.search.report import build_report, render_text
from tests.engine._search_feedback_cases import prepared, resume
from tests.engine._search_refine_cases import SearchCompute
from tests.engine.test_search_refine_resume import clone
from tests.engine._search_run_cases import Killed, RUN_DATE
from tests.engine.test_search_feedback import _row


def test_resume_asks_feedback_first_and_retains_global_best(tmp_path: Path) -> None:
    from aosr.search.feedback import FeedbackLedger

    store, registry, stopped = prepared(tmp_path)
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 0
    event, = FeedbackLedger.read(store.feedback_path)
    assert len(event.points) > store.settings.batch_size
    final = resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    rows = sorted(read_for(store).rows, key=lambda row: row.trial_number)
    injected = rows[stopped.asked:stopped.asked + len(event.points)]
    assert [row.unit_params_hex for row in injected] == list(event.points)
    assert [row.trial_number for row in injected] == list(range(stopped.asked, stopped.asked + len(event.points)))
    assert final.asked > stopped.asked and final.round == 2 and final.state == "converged"
    assert final.best_trial == stopped.best_trial and final.best_score == stopped.best_score
    assert final.streak == sum(row.outcome != "illegal" for row in rows if row.trial_number >= stopped.asked)
    assert final.refine == stopped.refine
    assert [record.round for record in final.rounds] == [1]


@pytest.mark.parametrize("workers", [1, 4])
def test_killed_feedback_resume_matches_whole_run(tmp_path: Path, workers: int) -> None:
    store, registry, _ = prepared(tmp_path / "input", workers=workers)
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 0
    whole = clone(store, tmp_path / "whole")
    compute = SearchCompute(whole, flat=True, persist_baseline=True)
    expected = resume(whole, registry, compute)
    for kill_after in range(sum(number is not None for number in compute.calls)):
        killed = clone(store, tmp_path / f"cut-{kill_after}")
        with pytest.raises(Killed):
            resume(killed, registry, SearchCompute(killed, flat=True, persist_baseline=True,
                                                  fail_after=kill_after, kill=True))
        actual = resume(killed, registry, SearchCompute(killed, flat=True, persist_baseline=True))
        assert killed.ledger_path.read_bytes() == whole.ledger_path.read_bytes()
        assert killed.status_path.read_bytes() == whole.status_path.read_bytes()
        assert actual == expected


def test_single_and_multi_feedback_replay_match(tmp_path: Path) -> None:
    single, registry, _ = prepared(tmp_path / "single", workers=1)
    multi, other, _ = prepared(tmp_path / "multi", workers=4)
    exit_code = main(["feedback", str(single.path)])
    assert exit_code == 0
    exit_code = main(["feedback", str(multi.path)])
    assert exit_code == 0
    first = resume(single, registry, SearchCompute(single, flat=True, persist_baseline=True))
    second = resume(multi, other, SearchCompute(multi, flat=True, persist_baseline=True))
    assert sorted(read_for(single).rows, key=lambda row: row.trial_number) == sorted(read_for(multi).rows, key=lambda row: row.trial_number)
    assert single.feedback_path.read_bytes() == multi.feedback_path.read_bytes()
    assert first == second


@pytest.mark.parametrize("damage", ["bad_line", "jump", "fraction", "wrong_boundary", "future", "missing", "points",
                                    "status_round"])
def test_corrupt_feedback_interrupts_without_appending(tmp_path: Path, damage: str) -> None:
    store, registry, stopped = prepared(tmp_path)
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 0
    document = json.loads(store.feedback_path.read_bytes())
    if damage == "bad_line":
        store.feedback_path.write_bytes(b"bad\n" + store.feedback_path.read_bytes())
    elif damage == "status_round":
        # 事件與每輪紀錄都對，只有狀態檔的輪次數字被改亂：也要擋下。
        saved = json.loads(store.status_path.read_bytes())
        saved["round"] = saved["round"] + 1
        store.status_path.write_text(json.dumps(saved), encoding="utf-8")
    elif damage == "missing":
        store.feedback_path.unlink()
    else:
        if damage == "jump":
            document["round"] = 3
        elif damage == "fraction":
            document["before_batch"] = 0.5
        elif damage == "wrong_boundary":
            document["before_batch"] -= 1
        elif damage == "future":
            document["before_batch"] = stopped.asked
        elif damage == "points":
            document["points"][0]["spacing"] = "nan"
        store.feedback_path.write_text(json.dumps(document) + "\n")
    before = store.ledger_path.read_bytes()
    status = resume(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    assert status.state == "interrupted" and "回饋" in status.message
    assert store.ledger_path.read_bytes() == before


def test_progress_counts_only_round_but_best_is_global(tmp_path: Path) -> None:
    old = _row(0, (0.25, 0.25, 0.25)).model_copy(update={"score": 0.0})
    new = _row(1, (0.5, 0.5, 0.5)).model_copy(update={"score": 1.0})
    status = SearchStatus(round=2, round_start_trial=1)
    progress = _progress(status, (old, new))
    assert (progress.best_trial, progress.best_score, progress.streak) == (0, 0.0, 1)
    assert _progress(status, (old,)).streak == 0
    improved = new.model_copy(update={"score": -1.0})
    assert (_progress(status, (old, improved)).best_trial, _progress(status, (old, improved)).streak) == (1, 0)
    assert tmp_path.is_dir()


def test_report_shows_round_and_saved_stop_without_changing_refine_section(tmp_path: Path) -> None:
    store, registry, stopped = prepared(tmp_path)
    old = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    exit_code = main(["feedback", str(store.path)])
    assert exit_code == 0
    new = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert "目前第 2 輪" in new
    assert f"第 1 輪：達到停止條件；問過 {stopped.asked} 題；第一名 {stopped.best_trial}" in new
    assert old.split("\n\n")[1:] == new.split("\n\n")[1:]


def test_no_feedback_report_text_is_unchanged_by_default_round_fields(tmp_path: Path) -> None:
    store, registry, _ = prepared(tmp_path, feedback=False)
    expected = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    document = json.loads(store.status_path.read_bytes())
    for field in ("round", "round_start_trial", "rounds"):
        document.pop(field, None)
    store.status_path.write_text(json.dumps(document))
    assert render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE)) == expected
    assert "目前第" not in expected
