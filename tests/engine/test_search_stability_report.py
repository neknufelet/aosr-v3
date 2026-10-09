"""唯讀擺位報告：數字取寫入端，壞摘要局部隔離，既有段落逐字保留。"""
from dataclasses import replace
from pathlib import Path
from typing import NoReturn

import pytest

from aosr.search import placement_stability_attach as writer
from aosr.search.labels import (
    COUNT_REASONS, FURNITURE_FACES, LISTENING_POINTS, PARAM_LABELS, SPEAKERS,
    STABILITY_CROSSOVERS, STABILITY_EVENTS, STABILITY_FLAGS, STABILITY_OUTCOMES, STABILITY_SHIFTS, trial_label,
    STABILITY_LIMITATION, STABILITY_MODEL_NOTE,
)
from aosr.search.outer_status import OuterStatus, attachment_skip_reason
from aosr.search.placement_stability import SHIFT_NAMES
from aosr.search.placement_stability_record import INCOMPLETE, STALE, TEMPORARY, StabilitySummary, summary_path, write_summary
from aosr.search.report import build_report, render_text
from aosr.search.report_stability import ABSENT, TITLE, StabilityReport, stability_report, stability_text
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
    headline = (f"分數第一名：{trial_label(arithmetic.score_winner.trial_number)}；"
                f"±2 公分內最差情況最好：{trial_label(arithmetic.minimax_winner.trial_number)}")
    assert report.lines[report.lines.index(next(line for line in report.lines if line.startswith("狀態："))) + 1] == headline
    score_lines = [line for line in report.lines if "；入圍：" in line]
    assert [line.split("；", 1)[0] for line in score_lines] == [trial_label(row.finalist.trial_number) for row in arithmetic.finalists]
    for row in arithmetic.finalists:
        name = trial_label(row.finalist.trial_number)
        line = next(line for line in report.lines if line.startswith(name + "；"))
        for value in (row.finalist.original_cost, row.best, row.worst):
            assert f"{value:.4f}" in line
        assert f"{row.scored_points}／{len(SHIFT_NAMES)} 點有分數" in line
        assert f"細算第 {row.finalist.refinement_rank} 名" in line
        for key in row.finalist.crossover_reasons:
            assert STABILITY_CROSSOVERS[key] in line
        for outcome, count in row.outcome_counts:
            if outcome != "scored" and count:
                assert f"{STABILITY_OUTCOMES[outcome]} {count} 點" not in line
        if any(p.model_discontinuity for _, p in row.points):
            assert f"不含模型不連續的點：最佳 {row.continuous_best:.4f}；最差 {row.continuous_worst:.4f}" in line
    for missing in arithmetic.minimax_incomplete:
        assert f"{trial_label(missing.finalist.trial_number)}（缺 {missing.missing_points} 點，見下）" in text
        for shift, point in missing.points:
            assert f"{STABILITY_SHIFTS[shift]} {STABILITY_OUTCOMES[point.outcome]}" in text


@pytest.mark.parametrize("missing_on_flip", [False, True])
def test_shift_ranking_line_counts_only_unlisted_shifts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       missing_on_flip: bool) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch, missing_on_flip=missing_on_flip)
    arithmetic = summary.arithmetic
    assert arithmetic is not None
    changed = [s for s in arithmetic.shift_winners if s.winner != arithmetic.score_winner]
    remaining = [s for s in arithmetic.shift_winners if s not in changed]
    missing = [s for s in remaining if s.without_score]
    assert changed and missing
    assert any(s.without_score for s in changed) is missing_on_flip
    lines = stability_report(store, status).lines
    line = next(line for line in lines if line.startswith("移位後第一名換人："))
    assert f"其餘 {len(remaining)} 種不變" in line
    assert f"其中 {len(missing)} 種有入圍沒分數：" in line
    for shift in changed:
        assert shift.winner is not None
        expected = f"{STABILITY_SHIFTS[shift.name]} → {trial_label(shift.winner.trial_number)}"
        if shift.without_score:
            expected += "（沒分數的是 " + "、".join(trial_label(n) for n in shift.without_score) + "）"
        assert expected in line
    missing_clause = line.split("種有入圍沒分數：", 1)[1]
    for shift in changed:
        assert STABILITY_SHIFTS[shift.name] not in missing_clause
    for shift in missing:
        assert STABILITY_SHIFTS[shift.name] in missing_clause
        for number in shift.without_score:
            assert trial_label(number) in missing_clause
    assert not any(line.startswith("移位 ") for line in lines)


def test_headlines_do_not_repeat_unscored_point_outcomes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch)
    arithmetic = summary.arithmetic
    assert arithmetic is not None and arithmetic.minimax_incomplete
    report = stability_report(store, status)
    headline = next(line for line in report.lines if line.startswith("分數第一名："))
    incomplete = next(line for line in report.lines if line.startswith("最差情況不完整、不參加比較："))
    expected = "最差情況不完整、不參加比較：" + "；".join(
        f"{trial_label(m.finalist.trial_number)}（缺 {m.missing_points} 點，見下）" for m in arithmetic.minimax_incomplete)
    assert incomplete == expected
    assert report.lines[report.lines.index(headline) + 1] == incomplete
    assert report.lines[report.lines.index(incomplete) + 1].startswith("移位後第一名換人：")
    for missing in arithmetic.minimax_incomplete:
        for shift, point in missing.points:
            for line in (headline, incomplete):
                assert STABILITY_SHIFTS[shift] not in line and STABILITY_OUTCOMES[point.outcome] not in line


def test_no_shift_changes_has_one_ranking_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch, flip=False)
    arithmetic = summary.arithmetic
    assert arithmetic is not None
    assert all(s.winner == arithmetic.score_winner for s in arithmetic.shift_winners)
    lines = stability_report(store, status).lines
    ranking = next(line for line in lines if line.startswith("每種移位的第一名都沒換人"))
    missing = [s for s in arithmetic.shift_winners if s.without_score]
    assert f"其中 {len(missing)} 種有入圍沒分數：" in ranking
    assert not any(line.startswith(("移位後第一名換人：", "移位 ", "其餘 ")) for line in lines)


def _finalist_groups(report: StabilityReport, summary: StabilitySummary) -> dict[int | None, tuple[str, ...]]:
    assert summary.arithmetic is not None
    prefixes = [trial_label(row.finalist.trial_number) + "；入圍：" for row in summary.arithmetic.finalists]
    starts = [next(i for i, line in enumerate(report.lines) if line.startswith(prefix)) for prefix in prefixes]
    end = next(i for i, line in enumerate(report.lines) if line in (STABILITY_MODEL_NOTE, STABILITY_LIMITATION))
    return {row.finalist.trial_number: report.lines[start:stop] for row, start, stop in
            zip(summary.arithmetic.finalists, starts, (*starts[1:], end), strict=True)}


def test_finalist_groups_keep_own_flags_in_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch)
    report = stability_report(store, status)
    groups = _finalist_groups(report, summary)
    assert [line for line in report.lines if line in (STABILITY_MODEL_NOTE, STABILITY_LIMITATION)] == list(report.lines[-2:])
    for number, group in groups.items():
        points = [p for p in summary.points if p.trial_number == number]
        flag_line = next(line for line in group if line.startswith("標記："))
        assert [STABILITY_SHIFTS[p.shift_name] for p in points if p.out_of_spec or p.outside_search or p.search_range_not_checked] == [
            label for label in STABILITY_SHIFTS.values() if label in flag_line]
        for point in points:
            for quantity in point.outside_search_quantities:
                assert f"{STABILITY_FLAGS['outside_search']} {STABILITY_SHIFTS[point.shift_name]}（{PARAM_LABELS[quantity]}）" in flag_line
        other_quantities = {q for p in summary.points if p.trial_number != number for q in p.outside_search_quantities}
        own_quantities = {q for p in points for q in p.outside_search_quantities}
        assert all(PARAM_LABELS[q] not in flag_line for q in other_quantities - own_quantities)
        assert group[-1].startswith("基準點離邊界：")
        assert all(group[-1].find(SPEAKERS[speaker]) >= 0 for speaker in store.project.speakers)
        assert group.index(flag_line) > 0
        for line in group:
            if line.startswith("沒分數的 "):
                assert group.index(flag_line) > group.index(line)
        for point in points:
            for violation in point.spec_violations:
                amount = "違反量〔弧長〕" if violation.reason == "base_angle_out_of_range" else "違反量"
                assert f"{COUNT_REASONS[violation.reason]}，{amount} {violation.amount_m:g} 公尺" in flag_line


def test_unscored_points_and_reasons_stay_in_finalist_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch)
    groups = _finalist_groups(stability_report(store, status), summary)
    assert summary.arithmetic is not None
    for missing in summary.arithmetic.minimax_incomplete:
        group = groups[missing.finalist.trial_number]
        line = next(line for line in group if line.startswith(f"沒分數的 {missing.missing_points} 點："))
        for point in summary.points:
            if point.trial_number != missing.finalist.trial_number or point.outcome is None or point.outcome == "scored":
                continue
            prefix = f"{STABILITY_SHIFTS[point.shift_name]} {STABILITY_OUTCOMES[point.outcome]}"
            assert prefix in line
            for violation in point.violations:
                reason = COUNT_REASONS[violation.reason].split("：")[-1]
                assert f"{prefix}（{reason}，違反量 {violation.amount_m:g} 公尺）" in line
            for problem in point.problems:
                assert f"{prefix}（問題路徑 {problem.path}）" in line


def test_events_and_boundaries_stay_in_finalist_group(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _, status, summary, _ = report_case(tmp_path, monkeypatch)
    report = stability_report(store, status)
    text = stability_text(report)
    groups = _finalist_groups(report, summary)
    assert any(p.furniture_events for p in summary.points)
    for point in summary.points:
        group = groups[point.trial_number]
        for event in point.furniture_events:
            line = next(line for line in group if line.startswith("模型不連續：") and STABILITY_SHIFTS[point.shift_name] in line)
            assert event.furniture_id in line and STABILITY_EVENTS[event.change] in line
            seat = f"主位（{event.receiver_id}）" if event.is_primary else f"周圍點 {LISTENING_POINTS[event.receiver_id]}（{event.receiver_id}）"
            assert seat in line
            assert FURNITURE_FACES[event.face] in line and SPEAKERS[event.speaker_id] in line
            assert group.index(line) > group.index(next(line for line in group if line.startswith("標記：")))
    for entry in summary.boundaries:
        line = groups[entry.trial_number][-1]
        for distance in entry.distances:
            assert f"{distance.edge_distance_m * 1000:.3f} 毫米" in line
            assert f"{distance.vertical_boundary_distance_deg:.3f} 度" in line
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
    assert summary.total_points == summary.computed_points
    assert "已算" not in text
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
    assert STABILITY_MODEL_NOTE not in text and text.splitlines()[-1] == STABILITY_LIMITATION
    for row in arithmetic.minimax_incomplete:
        assert f"缺 {row.missing_points} 點" in text
    for shift in arithmetic.shift_winners:
        assert shift.winner is None and shift.without_score
        assert f"{STABILITY_SHIFTS[shift.name]} → 沒有有分數的入圍（沒分數的是 " in text
