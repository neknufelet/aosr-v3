"""擺位摘要的唯讀投影；報告與網頁同文，不讀移位結果、不重評、不寫檔。"""
from __future__ import annotations

from pydantic import BaseModel, ValidationError

from aosr.search.labels import (
    COUNT_REASONS, FURNITURE_FACES, LISTENING_POINTS, PARAM_LABELS, SPEAKERS,
    STABILITY_CROSSOVERS, STABILITY_EVENTS, STABILITY_FLAGS, STABILITY_IDENTITIES,
    STABILITY_LIMITATION, STABILITY_MODEL_NOTE, STABILITY_OUTCOMES, STABILITY_SHIFTS, STABILITY_STATES, trial_label,
)
from aosr.search.placement_stability import Finalist, FinalistArithmetic, StabilityArithmetic
from aosr.search.placement_stability_attach import fresh_summary
from aosr.search.placement_stability_record import (
    ABSENT as ABSENT, INCOMPLETE, STALE, TEMPORARY, TITLE as TITLE,
    PointRecord, StabilitySummary, is_stale, read_summary,
)
from aosr.search.run import SearchStatus
from aosr.search.store import FROZEN, SearchStore

class StabilityReport(BaseModel):
    model_config = FROZEN
    lines: tuple[str, ...] = (ABSENT,)
    warning: bool = False


def _status_lines(summary: StabilitySummary, status: SearchStatus) -> tuple[str, ...]:
    temporary = TEMPORARY if summary.conclusion != "complete" or status.outer.conclusion != "complete" else ""
    progress = f"已算 {summary.computed_points}／共 {summary.total_points} 點"
    line = f"狀態：{STABILITY_STATES[summary.state]}{temporary}；{progress}"
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
    outcomes = [f"{STABILITY_OUTCOMES[key]} {count} 點" for key, count in row.outcome_counts if key != "scored" and count]
    parts = [_finalist_prefix(finalist),
             f"原分數 {finalist.original_cost:.4f}", f"含原點：最佳 {row.best:.4f}；最差 {row.worst:.4f}",
             f"{row.scored_points}／{len(row.points)} 點有分數", *outcomes]
    if any(point.model_discontinuity for _, point in row.points):
        parts.append(f"不含模型不連續的點：最佳 {row.continuous_best:.4f}；最差 {row.continuous_worst:.4f}")
    return "；".join(parts)


def _ranking_lines(arithmetic: StabilityArithmetic) -> tuple[str, ...]:
    score, minimax = arithmetic.score_winner, arithmetic.minimax_winner
    lines = [f"分數第一名：{trial_label(score.trial_number) if score is not None else '沒有'}；"
             f"最差情況最好：{trial_label(minimax.trial_number) if minimax is not None else '沒有（全部入圍的最差情況都不完整）'}"]
    for missing in arithmetic.minimax_incomplete:
        outcomes = "；".join(f"{STABILITY_SHIFTS[name]}：{STABILITY_OUTCOMES[point.outcome]}" for name, point in missing.points)
        lines.append(f"{missing.reason_text}：{trial_label(missing.finalist.trial_number)}；缺 {missing.missing_points} 點；{outcomes}")
    unchanged = []
    for shift in arithmetic.shift_winners:
        if shift.winner == score and score is not None:
            unchanged.append(shift)
            if shift.without_score:
                lines.append(f"移位 {STABILITY_SHIFTS[shift.name]}：沒有分數：" + "；".join(trial_label(n) for n in shift.without_score))
            continue
        winner = trial_label(shift.winner.trial_number) if shift.winner is not None else "沒有有分數的入圍"
        line = f"移位 {STABILITY_SHIFTS[shift.name]}：第一名 {winner}"
        if shift.without_score:
            line += "；沒有分數：" + "；".join(trial_label(number) for number in shift.without_score)
        lines.append(line)
    if unchanged:
        lines.append(f"其餘 {len(unchanged)} 種移位第一名不變")
    return tuple(lines)


def _point_lines(point: PointRecord) -> tuple[str, ...]:
    name = f"{trial_label(point.trial_number)}；{STABILITY_SHIFTS[point.shift_name]}"
    lines = []
    for event in point.furniture_events:
        speaker = SPEAKERS.get(event.speaker_id, event.speaker_id)
        seat = LISTENING_POINTS.get(event.receiver_id, event.receiver_id)
        lines.append(f"模型不連續：{name}；家具 {event.furniture_id}；{FURNITURE_FACES.get(event.face, event.face)}；"
                     f"{STABILITY_EVENTS[event.change]}；{speaker}；座位 {seat}（{event.receiver_id}）；主位：{'是' if event.is_primary else '否'}")
    if point.model_discontinuity and not point.furniture_events:
        lines.append(f"模型不連續：{name}；摘要沒有事件明細")
    if point.outcome in ("unplaceable", "placement_requirement_failed"):
        reasons = [f"{COUNT_REASONS[v.reason]}；違反量 {v.amount_m:g} 公尺" for v in point.violations]
        reasons.extend(f"問題路徑：{problem.path}" for problem in point.problems)
        lines.append("；".join((name, STABILITY_OUTCOMES[point.outcome], *reasons)))
    if point.reason_text:
        lines.append(f"{name}；{point.reason_text}")
    flags = [label for key, label in STABILITY_FLAGS.items() if key != "model_discontinuity" and getattr(point, key)]
    if flags:
        flags.extend(f"{COUNT_REASONS[v.reason]}；違反量 {v.amount_m:g} 公尺" for v in point.spec_violations)
        flags.extend(PARAM_LABELS.get(key, key) for key in point.outside_search_quantities)
        lines.append("；".join((name, *flags)))
    return tuple(lines)


def _boundary_lines(summary: StabilitySummary, store: SearchStore) -> tuple[str, ...]:
    lines = []
    for entry in summary.boundaries:
        prefix = f"基準點離邊界：{trial_label(entry.trial_number)}；主位"
        for speaker in store.project.speakers:
            label = SPEAKERS.get(speaker, speaker)
            distances = [d for d in entry.distances if d.speaker_id == speaker]
            if not distances:
                lines.append(f"{prefix}；{label}；沒有家具反射")
            for distance in distances:
                lines.append(f"{prefix}；{label}；家具 {distance.furniture_id}；{FURNITURE_FACES.get(distance.face, distance.face)}；"
                             f"離面邊 {distance.edge_distance_m * 1000:.3f} 毫米；仰角離分區界線 {distance.vertical_boundary_distance_deg:.3f} 度")
    return tuple(lines)


def stability_report(store: SearchStore, status: SearchStatus) -> StabilityReport:
    try:
        summary = read_summary(store.path)
        if summary is None:
            return StabilityReport()
        lines = list(_status_lines(summary, status))
        if summary.state == "done":
            if summary.arithmetic is None:
                raise ValueError("完成摘要缺報表算術")
            lines.extend(_finalist_line(row) for row in summary.arithmetic.finalists)
            lines.extend(_ranking_lines(summary.arithmetic))
        else:
            lines.extend(f"{_finalist_prefix(f)}；原分數 {f.original_cost:.4f}；最佳與最差尚未完成"
                         for f in summary.selection.finalists)
        lines.extend(line for point in summary.points for line in _point_lines(point))
        if any(p.model_discontinuity for p in summary.points):
            lines.append(STABILITY_MODEL_NOTE)
        lines.extend(_boundary_lines(summary, store))
        lines.append(STABILITY_LIMITATION)
        current = fresh_summary(store, status)
        # 跳過寫入端用空字串表示交接缺席；正常判舊用 missing，兩者都是沒有摘要。
        if summary.state == "skipped" and summary.crossover_stamp == "" and current.crossover_stamp == "missing":
            current = current.model_copy(update={"crossover_stamp": ""})
        if is_stale(summary, current):
            lines.append(STALE)
        return StabilityReport(lines=tuple(line.replace("\r", " ").replace("\n", "；") for line in lines),
                               warning=summary.state == "failed")
    except ValidationError:
        return StabilityReport(lines=("擺位穩定性摘要讀不到：JSON 資料損壞或欄位不完整",), warning=True)
    except (OSError, ValueError, KeyError) as error:
        reason = str(error).replace("\r", " ").replace("\n", "；")
        return StabilityReport(lines=(f"擺位穩定性摘要讀不到：{reason}",), warning=True)


def stability_text(report: StabilityReport) -> str:
    return "\n".join((TITLE, *(line.replace("\r", " ").replace("\n", "；") for line in report.lines)))
