"""唯讀擺位報告：數字取寫入端，壞摘要局部隔離，既有段落逐字保留。"""
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from aosr.search import placement_stability_attach as writer
from aosr.search.labels import (
    COUNT_REASONS, STABILITY_CROSSOVERS, STABILITY_EVENTS, STABILITY_FLAGS, STABILITY_OUTCOMES, STABILITY_SHIFTS,
    STABILITY_LIMITATION, STABILITY_MODEL_NOTE,
)
from aosr.search.outer_status import OuterStatus, attachment_skip_reason
from aosr.search.placement_stability import SHIFT_NAMES
from aosr.search.placement_stability_record import INCOMPLETE, STALE, TEMPORARY, summary_path, write_summary
from aosr.search.report import build_report, render_text
from aosr.search.report_stability import ABSENT, TITLE, stability_report, stability_text
from aosr.search.store import SearchStore
from tests.engine._search_run_cases import RUN_DATE
from tests.engine._stability_attach_cases import ShiftCompute, attach, ready
from tests.engine._stability_report_cases import report_case


def test_completed_report_uses_saved_arithmetic_and_every_shift(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch)
    arithmetic = summary.arithmetic
    assert arithmetic is not None and arithmetic.score_winner is not None and arithmetic.minimax_winner is not None
    assert arithmetic.score_winner != arithmetic.minimax_winner and arithmetic.minimax_incomplete
    report = stability_report(store, status)
    text = stability_text(report)
    assert not report.warning and "\n\n" not in text
    assert f"分數第一名：試算 {arithmetic.score_winner.trial_number}" in text
    assert f"最差情況最好：試算 {arithmetic.minimax_winner.trial_number}" in text
    for row in arithmetic.finalists:
        name = "原方案" if row.finalist.trial_number is None else f"試算 {row.finalist.trial_number}"
        line = next(line for line in report.lines if line.startswith(name + "；"))
        for value in (row.finalist.original_cost, row.best, row.worst):
            assert f"{value:.4f}" in line
        assert f"{row.scored_points}／{len(SHIFT_NAMES)} 點有分數" in line
        assert f"細算第 {row.finalist.refinement_rank} 名" in line
        for key in row.finalist.crossover_reasons:
            assert STABILITY_CROSSOVERS[key] in line
        for outcome, count in row.outcome_counts:
            if outcome != "scored" and count:
                assert f"{STABILITY_OUTCOMES[outcome]} {count} 點" in line
        if any(p.model_discontinuity for _, p in row.points):
            assert f"不含模型不連續的點：最佳 {row.continuous_best:.4f}；最差 {row.continuous_worst:.4f}" in line
    same = [w for w in arithmetic.shift_winners if w.winner == arithmetic.score_winner]
    assert f"其餘 {len(same)} 種移位第一名不變" in text
    for winner in arithmetic.shift_winners:
        if winner not in same:
            line = next(line for line in report.lines if line.startswith(f"移位 {STABILITY_SHIFTS[winner.name]}："))
            name = "沒有有分數的入圍" if winner.winner is None else "原方案" if winner.winner.trial_number is None else f"試算 {winner.winner.trial_number}"
            assert f"第一名 {name}" in line
        if winner.without_score:
            assert any(line.startswith(f"移位 {STABILITY_SHIFTS[winner.name]}：") and "沒有分數" in line for line in report.lines)
    for missing in arithmetic.minimax_incomplete:
        assert f"{missing.reason_text}：原方案；缺 {missing.missing_points} 點" in text
        for shift, point in missing.points:
            assert f"{STABILITY_SHIFTS[shift]}：{STABILITY_OUTCOMES[point.outcome]}" in text


def test_events_boundaries_illegal_reasons_and_point_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch)
    text = stability_text(stability_report(store, status))
    assert any(p.furniture_events for p in summary.points)
    for point in summary.points:
        for event in point.furniture_events:
            line = next(line for line in text.splitlines() if line.startswith("模型不連續：") and STABILITY_SHIFTS[point.shift_name] in line)
            assert event.furniture_id in line and STABILITY_EVENTS[event.change] in line
            from aosr.search.labels import FURNITURE_FACES, SPEAKERS
            assert event.receiver_id in line and f"主位：{'是' if event.is_primary else '否'}" in line
            assert FURNITURE_FACES[event.face] in line and SPEAKERS[event.speaker_id] in line
        if point.outcome in ("unplaceable", "placement_requirement_failed"):
            assert f"{STABILITY_SHIFTS[point.shift_name]}；{STABILITY_OUTCOMES[point.outcome]}" in text
            for violation in point.violations:
                assert COUNT_REASONS[violation.reason] in text and f"{violation.amount_m:g} 公尺" in text
            for problem in point.problems:
                assert problem.path in text
        for flag in ("out_of_spec", "outside_search", "search_range_not_checked"):
            if getattr(point, flag):
                assert any(STABILITY_SHIFTS[point.shift_name] in line and STABILITY_FLAGS[flag] in line for line in text.splitlines())
    for entry in summary.boundaries:
        for distance in entry.distances:
            assert f"{distance.edge_distance_m * 1000:.3f} 毫米" in text
            assert f"{distance.vertical_boundary_distance_deg:.3f} 度" in text
    assert "沒有家具反射" in text
    assert STABILITY_MODEL_NOTE in text
    assert STABILITY_LIMITATION in text


@pytest.mark.parametrize("scenario", ["running", "failed", "stopped", "temporary", "stale_rows", "stale_crossover", "stale_status"])
def test_progress_temporary_incomplete_and_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str) -> None:
    store, _, status, summary, compute = report_case(tmp_path, monkeypatch,
        fail_after=1 if scenario in ("failed", "stopped") else None, stop=scenario == "stopped")
    if scenario == "running":
        summary = compute.snapshots[1]
        write_summary(store.path, summary)
    elif scenario == "temporary":
        status = status.model_copy(update={"outer": OuterStatus(conclusion="refine_budget")})
    elif scenario == "stale_status":
        status = status.model_copy(update={"asked": status.asked + 1})
    elif scenario == "stale_crossover":
        from aosr.search import crossover_record
        crossing = crossover_record.read_summary(store.path)
        assert crossing is not None
        crossover_record.write_summary(store.path, crossing.model_copy(update={"reason_text": "已變"}))
    elif scenario == "stale_rows":
        from aosr.search.refine import RefineLedger
        header, rows = RefineLedger.read(store.refine_ledger_path)
        store.refine_ledger_path.unlink()
        book = RefineLedger.create(store.refine_ledger_path, header)
        for row in rows:
            book.append(row.model_copy(update={"total_cost": 20.0}))
    text = stability_text(stability_report(store, status))
    if scenario in ("running", "failed", "stopped"):
        assert INCOMPLETE in text
        assert f"已算 {summary.computed_points}／共 {summary.total_points} 點" in text
        assert summary.reason_text in text
        for finalist in summary.selection.finalists:
            name = "原方案" if finalist.trial_number is None else f"試算 {finalist.trial_number}"
            assert any(line.startswith(name + "；入圍：") and f"原分數 {finalist.original_cost:.4f}" in line
                       for line in text.splitlines())
    if scenario == "temporary":
        assert TEMPORARY in text
    if scenario.startswith("stale"):
        assert STALE in text
    assert stability_report(store, status).warning is (scenario == "failed")


@pytest.mark.parametrize("field", [None, "physics_identity", "program_fingerprint", "purpose_settings"])
def test_skip_reason_and_observed_identity(tmp_path: Path, field: str | None) -> None:
    store, registry, status = ready(tmp_path)
    if field is None:
        status = status.model_copy(update={"outer": OuterStatus(conclusion="user_stopped")})
        summary = attach(store, registry, status, ShiftCompute(store))
        expected = attachment_skip_reason(status.outer.conclusion)
    else:
        wrong = (replace(store.identity, purpose_settings=store.identity.purpose_settings.model_copy(update={"purpose": "different"}))
                 if field == "purpose_settings" else replace(store.identity, physics_identity="changed")
                 if field == "physics_identity" else replace(store.identity, program_fingerprint="changed"))
        summary = writer.attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
            probe=lambda: wrong, compute_factory=lambda root: ShiftCompute(store))
        expected = summary.reason_text
    report = stability_report(store, status)
    text = stability_text(report)
    assert expected in text and not report.warning
    if field:
        from aosr.search.labels import STABILITY_IDENTITIES
        assert STABILITY_IDENTITIES[field] in text
        assert summary.observed_identity is not None
    assert STALE not in text


def test_build_report_does_not_write_or_reevaluate_and_bad_summary_is_local(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status, _, _ = report_case(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in store.path.rglob("*") if p.is_file()}
    def forbidden(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("擺位報告不准重評、重算或寫檔")
    monkeypatch.setattr(writer, "report_arithmetic", forbidden)
    monkeypatch.setattr(writer, "screening_outcome", forbidden)
    monkeypatch.setattr(writer, "_read_shift", forbidden)
    monkeypatch.setattr(writer, "write_summary", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert {str(p): p.read_bytes() for p in store.path.rglob("*") if p.is_file()} == before
    with summary_path(store.path).open("w") as stream:
        stream.write("{")
    bad = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    original, changed = render_text(report).split("\n\n"), render_text(bad).split("\n\n")
    assert [s for s in original if not s.startswith(TITLE)] == [s for s in changed if not s.startswith(TITLE)]
    assert bad.stability.warning and "讀不到" in stability_text(bad.stability)
    assert bad.stability.lines == (next(line for line in bad.stability.lines if "讀不到" in line),)


def test_absent_and_retained_points_do_not_change_other_sections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status, summary, _ = report_case(tmp_path, monkeypatch)
    good = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    retained = summary.points[0].model_copy(update={"trial_number": 123456, "reason_text": "不應顯示的保留點"})
    write_summary(store.path, summary.model_copy(update={"retained_points": (retained,)}))
    assert stability_report(store, status) == good.stability
    summary_path(store.path).unlink()
    absent = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert absent.stability.lines == (ABSENT,) and not absent.stability.warning
    assert [s for s in render_text(good).split("\n\n") if not s.startswith(TITLE)] == [
        s for s in render_text(absent).split("\n\n") if not s.startswith(TITLE)]


def test_all_incomplete_and_shift_without_any_score(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    summary = attach(store, registry, status, ShiftCompute(store, different=True))
    arithmetic = summary.arithmetic
    assert arithmetic is not None and arithmetic.minimax_winner is None
    text = stability_text(stability_report(store, status))
    assert "最差情況最好：沒有（全部入圍的最差情況都不完整）" in text
    for row in arithmetic.minimax_incomplete:
        assert f"缺 {row.missing_points} 點" in text
    for shift in arithmetic.shift_winners:
        assert shift.winner is None and shift.without_score
        assert f"移位 {STABILITY_SHIFTS[shift.name]}：第一名 沒有有分數的入圍；沒有分數" in text
