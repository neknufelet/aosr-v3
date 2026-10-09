"""擺位穩定性的入圍與報表算術；只吃呼叫端給的資料，不重評、不存檔。"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from aosr.search.crossover_record import CrossoverSummary
from aosr.search.labels import (
    IDENTICAL_CROSSOVER_REASON, STABILITY_CROSSOVERS, STABILITY_CROSSOVER_INCOMPLETE, STABILITY_MINIMAX_INCOMPLETE,
)
from aosr.search.refine import RefineRow
from aosr.search.report_comparison import scored_refinements

Outcome: TypeAlias = Literal["unplaceable", "placement_requirement_failed", "excluded", "not_evaluated", "not_comparable", "scored"]
OUTCOMES: tuple[Outcome, ...] = (
    "unplaceable", "placement_requirement_failed", "excluded", "not_evaluated", "not_comparable", "scored",
)
SHIFT_NAMES = (
    "speakers_forward", "speakers_backward", "speakers_outward", "speakers_inward",
    "seat_forward", "seat_backward", "seat_left", "seat_right",
    "ear_up", "ear_down", "acoustic_center_up", "acoustic_center_down",
)


@dataclass(frozen=True)
class Finalist:
    trial_number: int | None
    original_cost: float
    refinement_rank: int
    top_rank_reason: int | None = None
    crossover_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class CrossoverWinner:
    key: str
    winner: Finalist | None
    reason_text: str = ""


@dataclass(frozen=True)
class FinalistSelection:
    finalists: tuple[Finalist, ...]
    crossover_winners: tuple[CrossoverWinner, ...]


@dataclass(frozen=True)
class PointOutcome:
    outcome: Outcome
    total_cost: float | None = None
    model_discontinuity: bool = False
    out_of_spec: bool = False
    outside_search: bool = False

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES or (self.outcome == "scored") != (self.total_cost is not None):
            raise ValueError("只有有分數的點能帶總代價")
        if self.total_cost is not None and (not math.isfinite(self.total_cost) or self.total_cost < 0.0):
            raise ValueError("總代價必須是有限非負數")


@dataclass(frozen=True)
class FinalistArithmetic:
    finalist: Finalist
    best: float
    worst: float
    scored_points: int
    continuous_best: float
    continuous_worst: float
    outcome_counts: tuple[tuple[Outcome, int], ...]
    points: tuple[tuple[str, PointOutcome], ...]


@dataclass(frozen=True)
class ShiftWinner:
    name: str
    winner: Finalist | None
    without_score: tuple[int | None, ...]


@dataclass(frozen=True)
class MinimaxIncomplete:
    finalist: Finalist
    missing_points: int
    points: tuple[tuple[str, PointOutcome], ...]
    reason_text: str = STABILITY_MINIMAX_INCOMPLETE


@dataclass(frozen=True)
class StabilityArithmetic:
    finalists: tuple[FinalistArithmetic, ...]
    score_winner: Finalist | None
    minimax_winner: Finalist | None
    shift_winners: tuple[ShiftWinner, ...]
    minimax_incomplete: tuple[MinimaxIncomplete, ...] = ()


def select_finalists(rows: tuple[RefineRow, ...], crossover: CrossoverSummary | str) -> FinalistSelection:
    """字串代表呼叫端已判摘要不能用，內容就是原因。"""
    ranked = scored_refinements(rows)
    candidates = {row.trial_number: Finalist(row.trial_number, float(row.total_cost), rank,
                  rank if rank <= 3 else None) for rank, row in enumerate(ranked, 1) if row.total_cost is not None}
    selected = {row.trial_number for row in ranked[:3]}
    winners: list[CrossoverWinner] = []
    if isinstance(crossover, CrossoverSummary) and not crossover.completed:
        crossover = STABILITY_CROSSOVER_INCOMPLETE
    if isinstance(crossover, str):
        winners = [CrossoverWinner(key, None, crossover) for key in STABILITY_CROSSOVERS]
    else:
        variants = {variant.key: variant for variant in crossover.variants}
        for key in STABILITY_CROSSOVERS:
            if key not in variants:
                winners.append(CrossoverWinner(key, None, crossover.reason_text))
                continue
            variant = variants[key]
            number = (crossover.official_best if variant.reason_text == IDENTICAL_CROSSOVER_REASON
                      else variant.ranking[0].trial_number if variant.ranking else None)
            if variant.reason_text == IDENTICAL_CROSSOVER_REASON or variant.ranking:
                if number not in candidates:
                    raise ValueError("接法第一名不在有分數的細算列裡")
                finalist = candidates[number]
                candidates[number] = replace(finalist, crossover_reasons=(*finalist.crossover_reasons, variant.key))
                selected.add(number)
                winners.append(CrossoverWinner(variant.key, candidates[number]))
            else:
                winners.append(CrossoverWinner(variant.key, None, variant.reason_text))
    finalists = tuple(candidates[row.trial_number] for row in ranked if row.trial_number in selected)
    winners = [replace(w, winner=candidates[w.winner.trial_number]) if w.winner is not None else w for w in winners]
    return FinalistSelection(finalists, tuple(winners))


def _finalist_arithmetic(finalist: Finalist, points: Mapping[str, PointOutcome]) -> FinalistArithmetic:
    if points.keys() != set(SHIFT_NAMES):
        raise ValueError("每個入圍必須給齊具名移位的結局，缺分以結局表示")
    scored = tuple(point.total_cost for point in points.values() if point.total_cost is not None)
    continuous = tuple(point.total_cost for point in points.values()
                       if point.total_cost is not None and not point.model_discontinuity)
    values = (finalist.original_cost, *scored)
    continuous_values = (finalist.original_cost, *continuous)
    return FinalistArithmetic(finalist, min(values), max(values), len(scored),
        min(continuous_values), max(continuous_values),
        tuple((outcome, sum(point.outcome == outcome for point in points.values())) for outcome in OUTCOMES),
        tuple((name, points[name]) for name in SHIFT_NAMES))


def _shift_winner(name: str, finalists: tuple[Finalist, ...],
                  outcomes: Mapping[int | None, Mapping[str, PointOutcome]]) -> ShiftWinner:
    available = tuple((point.total_cost, finalist) for finalist in finalists
                      if (point := outcomes[finalist.trial_number][name]).total_cost is not None)
    winner = min(available, key=lambda pair: (pair[0], pair[1].refinement_rank))[1] if available else None
    missing = tuple(f.trial_number for f in finalists if outcomes[f.trial_number][name].total_cost is None)
    return ShiftWinner(name, winner, missing)


def report_arithmetic(finalists: tuple[Finalist, ...],
                      outcomes: Mapping[int | None, Mapping[str, PointOutcome]]) -> StabilityArithmetic:
    """原點納入兩組極值，缺分點只列結局；同分照細算名次。"""
    ordered = tuple(sorted(finalists, key=lambda finalist: finalist.refinement_rank))
    reports = tuple(_finalist_arithmetic(f, outcomes[f.trial_number]) for f in ordered)
    complete = tuple(report for report in reports if report.scored_points == len(SHIFT_NAMES))
    incomplete = tuple(MinimaxIncomplete(report.finalist, len(SHIFT_NAMES) - report.scored_points,
        tuple((name, point) for name, point in report.points if point.total_cost is None))
        for report in reports if report.scored_points != len(SHIFT_NAMES))
    minimax = min(complete, key=lambda report: (report.worst, report.finalist.refinement_rank)) if complete else None
    return StabilityArithmetic(reports, ordered[0] if ordered else None,
        minimax.finalist if minimax is not None else None,
        tuple(_shift_winner(name, ordered, outcomes) for name in SHIFT_NAMES), incomplete)
