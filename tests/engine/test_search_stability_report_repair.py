"""第二輪修補：首次停止戳記、零點進度、接法未補原因與前 3 名之外的補入。"""
from pathlib import Path

import pytest

from aosr.search import crossover_record, crossover_sensitivity, placement_stability_attach as writer
from aosr.search.labels import STABILITY_CROSSOVERS, trial_label
from aosr.search.placement_stability_geometry import Legality
from aosr.search.placement_stability_record import STALE, STOPPED_REASON, read_summary, write_summary
from aosr.search.report_stability import stability_report, stability_text
from tests.engine._crossover_cases import evaluate
from tests.engine._stability_attach_cases import ShiftCompute, attach, ready
from tests.engine._stability_report_cases import add_backfill_finalist
from tests.engine.test_search_stability_reuse import change_rows


@pytest.mark.parametrize("crossing", ["done", "missing", "broken"])
def test_stopped_before_start_is_not_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crossing: str) -> None:
    store, registry, status = ready(tmp_path)
    if crossing == "done":
        monkeypatch.setattr(crossover_sensitivity, "reevaluate", evaluate)
        crossover_sensitivity.attach_crossover(store, status=status, quality_targets_path=registry)
    elif crossing == "broken":
        path = crossover_record.summary_path(store.path)
        path.parent.mkdir()
        path.write_text("{")
    assert read_summary(store.path) is None
    writer.record_stability_error(store, status, KeyboardInterrupt())
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "stopped"
    fresh = writer.fresh_summary(store, status)
    assert (summary.rows, summary.rows_fingerprint, summary.crossover_stamp) == (fresh.rows, fresh.rows_fingerprint, fresh.crossover_stamp)
    text = stability_text(stability_report(store, status))
    assert STOPPED_REASON in text and STALE not in text
    assert "已算" not in text
    change_rows(store)
    assert STALE in stability_text(stability_report(store, status))


@pytest.mark.parametrize("source", ["rows", "crossover"])
def test_error_stamps_keep_readable_source_if_other_read_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str) -> None:
    store, _, status = ready(tmp_path)
    current = writer.fresh_summary(store, status)
    def unreadable(*args: object, **kwargs: object) -> None:
        raise OSError("摘要戳記讀不到的原文")
    monkeypatch.setattr(writer, "read_refinement_rows" if source == "rows" else "_crossover", unreadable)
    writer.record_stability_error(store, status, RuntimeError("前段失敗原文"))
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "failed" and summary.reason_text == "前段失敗原文"
    if source == "rows":
        assert summary.rows == () and summary.rows_fingerprint == ""
        assert summary.crossover_stamp == current.crossover_stamp
    else:
        assert summary.crossover_stamp == ""
        assert (summary.rows, summary.rows_fingerprint) == (current.rows, current.rows_fingerprint)


@pytest.mark.parametrize("state", ["running", "done", "skipped", "failed", "stopped"])
def test_zero_total_points_never_prints_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    store, registry, status = ready(tmp_path)
    if state == "done":
        monkeypatch.setattr(writer, "check_shift", lambda *args, **kwargs: Legality("unplaceable"))
        summary = attach(store, registry, status, ShiftCompute(store))
    elif state == "skipped":
        from aosr.search.outer_status import OuterStatus
        status = status.model_copy(update={"outer": OuterStatus(conclusion="user_stopped")})
        summary = attach(store, registry, status, ShiftCompute(store))
    else:
        if state == "running":
            write_summary(store.path, writer.fresh_summary(store, status))
        else:
            writer.record_stability_error(store, status, KeyboardInterrupt() if state == "stopped" else RuntimeError("前段失敗原文"))
        saved = read_summary(store.path)
        assert saved is not None
        summary = saved
    assert summary.state == state and summary.total_points == 0
    report = stability_report(store, status)
    assert not any("已算" in line for line in report.lines)
    assert report.warning is (state == "failed")


@pytest.mark.parametrize("crossing", ["missing", "broken", "stale", "running", "failed", "stopped", "skipped"])
def test_unusable_crossover_lists_every_unfilled_reason(tmp_path: Path, crossing: str) -> None:
    store, registry, status = ready(tmp_path)
    if crossing == "broken":
        path = crossover_record.summary_path(store.path)
        path.parent.mkdir()
        path.write_text("{")
    elif crossing != "missing":
        current = crossover_record.fresh_summary(store, status)
        changes = ({"state": "done", "completed": True, "rows_fingerprint": "old"} if crossing == "stale" else
                   {"state": crossing, "completed": crossing == "skipped", "reason_text": "交接不能用的原文"})
        crossover_record.write_summary(store.path, current.model_copy(update=changes))
    summary = attach(store, registry, status, ShiftCompute(store))
    assert summary.state == "done" and summary.arithmetic is not None
    winners = summary.selection.crossover_winners
    assert {w.key for w in winners} == set(STABILITY_CROSSOVERS)
    assert all(w.winner is None and w.reason_text for w in winners)
    lines = stability_report(store, status).lines
    assert [line for line in lines if "第一名未補：" in line] == [
        f"{STABILITY_CROSSOVERS[w.key]}第一名未補：{w.reason_text}" for w in winners]


def test_backfill_marker_only_on_crossover_finalist_outside_top_three(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    number = add_backfill_finalist(store, status)
    summary = attach(store, registry, status, ShiftCompute(store))
    finalist = next(f for f in summary.selection.finalists if f.trial_number == number)
    assert finalist.top_rank_reason is None and finalist.refinement_rank > 3 and finalist.crossover_reasons
    top = [f for f in summary.selection.finalists if f.top_rank_reason is not None]
    assert top
    lines = stability_report(store, status).lines
    backfill = next(line for line in lines if line.startswith(trial_label(number) + "；入圍："))
    marker = "（由接法第一名補入）"
    assert marker in backfill
    assert [line for line in lines if marker in line] == [backfill]
    for entry in top:
        line = next(line for line in lines if line.startswith(trial_label(entry.trial_number) + "；入圍："))
        assert marker not in line
