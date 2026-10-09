"""擺位摘要的唯讀投影；報告與網頁同文，不讀移位結果、不重評、不寫檔。"""
from __future__ import annotations

from pydantic import BaseModel, ValidationError

from aosr.search.constraints import Reason, Violation
from aosr.search.labels import (
    COUNT_REASONS, FURNITURE_FACES, LISTENING_POINTS, PARAM_LABELS, SPEAKERS,
    STABILITY_CROSSOVERS, STABILITY_EVENTS, STABILITY_FLAGS, STABILITY_IDENTITIES,
    STABILITY_FACING_MISMATCH, STABILITY_LIMITATION, STABILITY_MINIMAX_INCOMPLETE, STABILITY_MODEL_NOTE,
    STABILITY_OUTCOMES, STABILITY_SHIFTS, STABILITY_STATES, trial_label,
)
from aosr.search.placement_stability import Finalist, FinalistArithmetic, StabilityArithmetic
from aosr.search.placement_stability_attach import fresh_summary
from aosr.search.placement_stability_record import (
    ABSENT as ABSENT, INCOMPLETE, STALE, TEMPORARY, TITLE as TITLE,
    FinalistBoundaries, PointRecord, StabilitySummary, is_stale, read_summary,
)
from aosr.search.run import SearchStatus
from aosr.search.store import FROZEN, SearchStore

class StabilityReport(BaseModel):
    model_config = FROZEN
    lines: tuple[str, ...] = (ABSENT,)
    warning: bool = False


def _status_lines(summary: StabilitySummary, status: SearchStatus) -> tuple[str, ...]:
    temporary = TEMPORARY if summary.conclusion != "complete" or status.outer.conclusion != "complete" else ""
    line = f"狀態：{STABILITY_STATES[summary.state]}{temporary}"
    if summary.total_points:
        line += f"；已算 {summary.computed_points}／共 {summary.total_points} 點"
    if summary.reason_text:
        line += f"；{summary.reason_text}"
    lines = [line]
    if summary.observed_identity is not None:
        changed = [label for field, label in STABILITY_IDENTITIES.items()
                   if getattr(summary.observed_identity, field) != getattr(summary.identity, field)]
        if changed:
            lines.append("跟搜尋快照不同：" + "；".join(changed))
    if not summary.completed:
        lines.append(INCOMPLETE)
    lines.extend(f"{STABILITY_CROSSOVERS[w.key]}第一名未補：{w.reason_text}" for w in summary.selection.crossover_winners
                 if w.winner is None)
    return tuple(lines)


def _finalist_prefix(finalist: Finalist) -> str:
    reasons = [f"細算第 {finalist.refinement_rank} 名" + ("（由接法第一名補入）" if finalist.top_rank_reason is None else "")]
    reasons.extend(f"{STABILITY_CROSSOVERS[key]}第一名" for key in finalist.crossover_reasons)
    return f"{trial_label(finalist.trial_number)}；入圍：" + "；".join(reasons)


def _finalist_line(row: FinalistArithmetic) -> str:
    finalist = row.finalist
    parts = [_finalist_prefix(finalist),
             f"原分數 {finalist.original_cost:.4f}", f"含原點：最佳 {row.best:.4f}；最差 {row.worst:.4f}",
             f"{row.scored_points}／{len(row.points)} 點有分數"]
    if any(point.model_discontinuity for _, point in row.points):
        parts.append(f"不含模型不連續的點：最佳 {row.continuous_best:.4f}；最差 {row.continuous_worst:.4f}")
    return "；".join(parts)


def _ranking_lines(arithmetic: StabilityArithmetic) -> tuple[str, ...]:
    score, minimax = arithmetic.score_winner, arithmetic.minimax_winner
    lines = [f"分數第一名：{trial_label(score.trial_number) if score is not None else '沒有'}；"
             f"±2 公分內最差情況最好：{trial_label(minimax.trial_number) if minimax is not None else '沒有（全部入圍的最差情況都不完整）'}"]
    if arithmetic.minimax_incomplete:
        lines.append(f"{STABILITY_MINIMAX_INCOMPLETE}、不參加比較：" + "；".join(
            f"{trial_label(m.finalist.trial_number)}（缺 {m.missing_points} 點，見下）" for m in arithmetic.minimax_incomplete))
    unchanged, changed = [], []
    for shift in arithmetic.shift_winners:
        if shift.winner == score and score is not None:
            unchanged.append(shift)
            continue
        winner = trial_label(shift.winner.trial_number) if shift.winner is not None else "沒有有分數的入圍"
        line = f"{STABILITY_SHIFTS[shift.name]} → {winner}"
        if shift.without_score:
            line += "（沒分數的是 " + "、".join(trial_label(number) for number in shift.without_score) + "）"
        changed.append(line)
    ranking = "移位後第一名換人：" + "；".join(changed) if changed else "每種移位的第一名都沒換人"
    if changed and unchanged:
        ranking += f"；其餘 {len(unchanged)} 種不變"
    missing = [shift for shift in unchanged if shift.without_score]
    if missing:
        groups: dict[tuple[int | None, ...], list[str]] = {}
        for shift in missing:
            groups.setdefault(shift.without_score, []).append(STABILITY_SHIFTS[shift.name])
        details = "；".join("、".join(names) + "（沒分數的是 " + "、".join(trial_label(n) for n in numbers) + "）"
                          for numbers, names in groups.items())
        ranking += f"（其中 {len(missing)} 種有入圍沒分數：{details}）"
    lines.append(ranking)
    return tuple(lines)


def _violation_text(violation: Violation) -> str:
    amount = "違反量〔弧長〕" if violation.reason == Reason.BASE_ANGLE_OUT_OF_RANGE else "違反量"
    reason = COUNT_REASONS[violation.reason]
    if violation.reason == Reason.DIRECT_PATH_BLOCKED:
        reason = reason.split("：", 1)[-1]
    return f"{reason}，{amount} {violation.amount_m:g} 公尺"


def _unscored_line(points: tuple[PointRecord, ...], missing_points: int | None) -> str:
    parts = []
    for point in points:
        if point.outcome == "scored" or (point.outcome is None and not point.reason_text):
            continue
        outcome = STABILITY_OUTCOMES[point.outcome] if point.outcome is not None else point.reason_text
        details = [_violation_text(v) for v in point.violations]
        details.extend(f"問題路徑 {problem.path}" for problem in point.problems)
        if point.outcome is not None and point.reason_text:
            details.append(point.reason_text)
        line = f"{STABILITY_SHIFTS[point.shift_name]} {outcome}"
        parts.append(line + ("（" + "，".join(details) + "）" if details else ""))
    count = f" {missing_points} 點" if missing_points is not None else "點"
    return f"沒分數的{count}：" + "；".join(parts) if parts else ""


def _flags_line(points: tuple[PointRecord, ...]) -> str:
    parts = []
    for point in points:
        name = STABILITY_SHIFTS[point.shift_name]
        if point.out_of_spec:
            details = "，".join(_violation_text(v) for v in point.spec_violations)
            parts.append(f"{STABILITY_FLAGS['out_of_spec']} {name}" + (f"（{details}）" if details else ""))
        if point.outside_search:
            details = "、".join(PARAM_LABELS.get(key, key) for key in point.outside_search_quantities)
            parts.append(f"{STABILITY_FLAGS['outside_search']} {name}" + (f"（{details}）" if details else ""))
        if point.search_range_not_checked:
            parts.append(f"{STABILITY_FLAGS['search_range_not_checked']} {name}（{STABILITY_FACING_MISMATCH}）")
    return "標記：" + "；".join(parts) if parts else ""


def _event_lines(point: PointRecord) -> tuple[str, ...]:
    name = STABILITY_SHIFTS[point.shift_name]
    lines = []
    for event in point.furniture_events:
        speaker = SPEAKERS.get(event.speaker_id, event.speaker_id)
        seat = LISTENING_POINTS.get(event.receiver_id, event.receiver_id)
        receiver = f"{seat}（{event.receiver_id}）" if event.is_primary else f"周圍點 {seat}（{event.receiver_id}）"
        lines.append(f"模型不連續：{name}；家具 {event.furniture_id} {FURNITURE_FACES.get(event.face, event.face)} "
                     f"{STABILITY_EVENTS[event.change]}；{speaker}到{receiver}")
    if point.model_discontinuity and not point.furniture_events:
        lines.append(f"模型不連續：{name}；摘要沒有事件明細")
    return tuple(lines)


def _boundary_line(entry: FinalistBoundaries, store: SearchStore) -> str:
    parts = []
    for speaker in store.project.speakers:
        label = SPEAKERS.get(speaker, speaker)
        distances = [d for d in entry.distances if d.speaker_id == speaker]
        if not distances:
            parts.append(f"{label} 沒有家具反射")
        for distance in distances:
            parts.append(f"{label} {distance.furniture_id} {FURNITURE_FACES.get(distance.face, distance.face)} "
                         f"離面邊 {distance.edge_distance_m * 1000:.3f} 毫米、仰角離分區界線 {distance.vertical_boundary_distance_deg:.3f} 度")
    return "基準點離邊界：" + "；".join(parts)


def _finalist_lines(summary: StabilitySummary, store: SearchStore) -> tuple[str, ...]:
    rows = {r.finalist.trial_number: r for r in summary.arithmetic.finalists} if summary.arithmetic is not None else {}
    missing = {m.finalist.trial_number: m.missing_points for m in summary.arithmetic.minimax_incomplete} if summary.arithmetic is not None else {}
    boundaries = {entry.trial_number: entry for entry in summary.boundaries}
    lines = []
    for finalist in summary.selection.finalists:
        number = finalist.trial_number
        row = rows.get(number)
        lines.append(_finalist_line(row) if row is not None else
                     f"{_finalist_prefix(finalist)}；原分數 {finalist.original_cost:.4f}；最佳與最差尚未完成")
        points = tuple(p for p in summary.points if p.trial_number == number)
        lines.extend(line for line in (_unscored_line(points, missing.get(number)), _flags_line(points)) if line)
        lines.extend(line for point in points for line in _event_lines(point))
        if number in boundaries:
            lines.append(_boundary_line(boundaries[number], store))
    return tuple(lines)


def stability_report(store: SearchStore, status: SearchStatus) -> StabilityReport:
    try:
        summary = read_summary(store.path)
        if summary is None:
            return StabilityReport()
        current = fresh_summary(store, status)
        lines = list(_status_lines(summary, status))
        if is_stale(summary, current):
            lines.append(STALE)
        if summary.state == "done":
            if summary.arithmetic is None:
                raise ValueError("完成摘要缺報表算術")
            lines.extend(_ranking_lines(summary.arithmetic))
        lines.extend(_finalist_lines(summary, store))
        if any(p.model_discontinuity for p in summary.points):
            lines.append(STABILITY_MODEL_NOTE)
        lines.append(STABILITY_LIMITATION)
        return StabilityReport(lines=tuple(line.replace("\r", " ").replace("\n", "；") for line in lines),
                               warning=summary.state == "failed")
    except ValidationError:
        return StabilityReport(lines=("擺位穩定性摘要讀不到：JSON 資料損壞或欄位不完整",), warning=True)
    except (OSError, ValueError, KeyError) as error:
        reason = str(error).replace("\r", " ").replace("\n", "；")
        return StabilityReport(lines=(f"擺位穩定性摘要讀不到：{reason}",), warning=True)


def stability_text(report: StabilityReport) -> str:
    return "\n".join((TITLE, *(line.replace("\r", " ").replace("\n", "；") for line in report.lines)))
