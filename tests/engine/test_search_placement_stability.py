"""入圍與算術的手算答案：錯排序、假零分、排除標記點都必須紅。"""
from dataclasses import replace

import pytest

from aosr.search.crossover_record import CostRow, CrossoverSummary, VariantRecord
from aosr.search.placement_stability import (
    Finalist, Outcome, PointOutcome, SHIFT_NAMES, report_arithmetic, select_finalists,
)
from aosr.search.refine import RefineRow
from aosr.search.store import refine_result_name


def row(number: int | None, cost: float | None, outcome: str = "scored") -> RefineRow:
    return RefineRow.model_validate(dict(round=1, trial_number=number, result_file=refine_result_name(number),
                                        outcome=outcome, total_cost=cost, seconds=0.0))


def test_finalists_three_crossover_cases_include_original_and_deduplicate() -> None:
    rows = (row(9, 2.0), row(7, 1.0), row(8, 2.0), row(None, 5.0), row(20, None, "excluded"))
    summary = CrossoverSummary(official_best=7, variants=(
        VariantRecord(key="hard", label="", basis="", reason_text="與正式接法逐點權重相同，不算另一種比較"),
        VariantRecord(key="legacy", label="", basis="", tested=False,
                      ranking=(CostRow(trial_number=None, total_cost=0.5), CostRow(trial_number=9, total_cost=1.0)),
                      reason_text="可比較的列不足"),
        VariantRecord(key="wide", label="", basis="", reason_text="做不出交接帶"),
    ))
    selected = select_finalists(rows, summary)
    assert tuple(f.trial_number for f in selected.finalists) == (7, 9, 8, None)
    assert selected.finalists[0] == Finalist(7, 1.0, 1, 1, ("hard",))
    assert selected.finalists[-1] == Finalist(None, 5.0, 4, None, ("legacy",))
    winners = {w.key: w for w in selected.crossover_winners}
    assert winners["hard"].winner == selected.finalists[0]
    assert winners["legacy"].winner == selected.finalists[-1]
    assert winners["wide"].winner is None
    assert winners["wide"].reason_text == "做不出交接帶"


def test_top_three_ties_keep_ledger_order_and_crossover_reasons_accumulate() -> None:
    rows = (row(None, 1.0), row(9, 1.0), row(7, 1.0), row(8, 1.0))
    summary = CrossoverSummary(official_best=None, variants=tuple(
        VariantRecord(key=key, label="", basis="", ranking=(CostRow(trial_number=None, total_cost=1.0),))
        for key in ("hard", "legacy", "wide")))
    selected = select_finalists(rows, summary)
    assert tuple(f.trial_number for f in selected.finalists) == (None, 9, 7)
    assert selected.finalists[0].crossover_reasons == ("hard", "legacy", "wide")


@pytest.mark.parametrize("rows,expected", [
    ((), ()), ((row(4, 2.0),), (4,)),
    ((row(4, 2.0), row(None, 1.0), row(5, None, "not_evaluated"), row(6, None, "not_comparable")), (None, 4)),
])
def test_unusable_summary_and_fewer_than_three_scored_rows(
    rows: tuple[RefineRow, ...], expected: tuple[int | None, ...],
) -> None:
    selected = select_finalists(rows, "摘要舊了")
    assert tuple(f.trial_number for f in selected.finalists) == expected
    assert {w.key: w.reason_text for w in selected.crossover_winners} == {
        "hard": "摘要舊了", "legacy": "摘要舊了", "wide": "摘要舊了"}
    assert all(w.winner is None for w in selected.crossover_winners)


def test_summary_without_variants_keeps_each_crossover_unavailable_reason() -> None:
    summary = CrossoverSummary(completed=True, state="done", reason_text="可比較的方案不足")
    selected = select_finalists((row(4, 2.0),), summary)
    assert selected.finalists == (Finalist(4, 2.0, 1, 1),)
    assert {w.key: w.reason_text for w in selected.crossover_winners} == {
        "hard": "可比較的方案不足", "legacy": "可比較的方案不足", "wide": "可比較的方案不足"}
    assert all(w.winner is None for w in selected.crossover_winners)


def test_arithmetic_hand_scores_missing_points_flags_and_rank_ties() -> None:
    # 細算順序故意與輸入順序相反；甲最佳 0.5、最差 10，乙丙最差同為 5。
    a, b, c = Finalist(None, 1.0, 1), Finalist(4, 2.0, 2), Finalist(2, 2.0, 3)
    outcomes = {f.trial_number: dict.fromkeys(SHIFT_NAMES, PointOutcome("not_evaluated")) for f in (a, b, c)}
    outcomes[None].update({
        "speakers_forward": PointOutcome("scored", 0.5),
        "speakers_backward": PointOutcome("scored", 10.0, model_discontinuity=True),
        "seat_left": PointOutcome("unplaceable"),
        "seat_right": PointOutcome("placement_requirement_failed"),
        "ear_up": PointOutcome("excluded"), "ear_down": PointOutcome("not_comparable"),
    })
    outcomes[4].update({"speakers_forward": PointOutcome("scored", 2.0, out_of_spec=True, outside_search=True),
                        "speakers_backward": PointOutcome("scored", 5.0)})
    outcomes[2].update({"speakers_forward": PointOutcome("scored", 2.0),
                        "speakers_backward": PointOutcome("scored", 5.0)})
    report = report_arithmetic((c, a, b), outcomes)
    assert report.score_winner == a
    assert report.minimax_winner == b
    ra, rb, rc = report.finalists
    assert (ra.finalist.original_cost, ra.best, ra.worst, ra.scored_points) == (1.0, 0.5, 10.0, 2)
    assert (ra.continuous_best, ra.continuous_worst) == (0.5, 1.0)
    assert (rb.best, rb.worst, rc.best, rc.worst) == (2.0, 5.0, 2.0, 5.0)
    assert dict(ra.outcome_counts) == {"unplaceable": 1, "placement_requirement_failed": 1,
        "excluded": 1, "not_evaluated": 6, "not_comparable": 1, "scored": 2}
    assert dict(rb.points)["speakers_forward"].out_of_spec
    assert dict(rb.points)["speakers_forward"].outside_search
    winners = {w.name: w for w in report.shift_winners}
    assert winners["speakers_backward"].winner == b
    assert winners["seat_left"].winner is None
    assert winners["seat_left"].without_score == (None, 4, 2)
    # 同移位有人缺分：甲不可以假零分勝出；乙丙同分依細算名次。
    outcomes[None]["speakers_forward"] = PointOutcome("unplaceable")
    changed = report_arithmetic((c, a, b), outcomes)
    forward = next(w for w in changed.shift_winners if w.name == "speakers_forward")
    assert forward.winner == b
    assert forward.without_score == (None,)
    assert dict(changed.finalists[0].points)["speakers_forward"].total_cost is None


def test_original_always_included_even_when_every_shift_is_flagged_or_unscored() -> None:
    finalist = Finalist(1, 3.0, 1)
    points = dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 4.0, model_discontinuity=True))
    first = report_arithmetic((finalist,), {1: points}).finalists[0]
    assert (first.best, first.worst, first.continuous_best, first.continuous_worst) == (3.0, 4.0, 3.0, 3.0)
    points = dict.fromkeys(SHIFT_NAMES, PointOutcome("not_evaluated"))
    second = report_arithmetic((finalist,), {1: points}).finalists[0]
    assert (second.best, second.worst, second.scored_points) == (3.0, 3.0, 0)


@pytest.mark.parametrize("outcome,cost", [("unplaceable", 0.0), ("scored", None), ("scored", float("nan"))])
def test_missing_score_cannot_be_filled_or_nonfinite(outcome: Outcome, cost: float | None) -> None:
    with pytest.raises(ValueError):
        replace(PointOutcome("scored", 1.0), outcome=outcome, total_cost=cost)
