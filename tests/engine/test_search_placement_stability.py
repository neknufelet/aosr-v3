"""入圍與算術的手算答案：錯排序、假零分、排除標記點都必須紅。"""
from dataclasses import replace

import pytest

from aosr.search.crossover_record import CostRow, CrossoverSummary, VariantRecord
from aosr.search.crossover_sensitivity import VariantScores, _variant, stitchings
from aosr.search import labels
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
    summary = CrossoverSummary(completed=True, official_best=7, variants=(
        _variant(VariantScores(stitchings(340)[0]), 7),
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
    summary = CrossoverSummary(completed=True, official_best=None, variants=tuple(
        VariantRecord(key=key, label="", basis="", ranking=(CostRow(trial_number=None, total_cost=1.0),))
        for key in ("hard", "legacy", "wide")))
    selected = select_finalists(rows, summary)
    assert tuple(f.trial_number for f in selected.finalists) == (None, 9, 7)
    assert selected.finalists[0].crossover_reasons == ("hard", "legacy", "wide")
    assert all(w.winner == selected.finalists[0] for w in selected.crossover_winners)
    assert all(w.winner is not None and w.winner.crossover_reasons == ("hard", "legacy", "wide")
               for w in selected.crossover_winners)


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
    assert report.minimax_winner is None
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


@pytest.mark.parametrize("outcome,cost", [("unplaceable", 0.0), ("scored", None), ("scored", float("nan")), ("scored", -0.1)])
def test_missing_score_cannot_be_filled_or_nonfinite(outcome: Outcome, cost: float | None) -> None:
    with pytest.raises(ValueError):
        replace(PointOutcome("scored", 1.0), outcome=outcome, total_cost=cost)


def test_incomplete_summary_cannot_add_crossover_winners() -> None:
    rows = (row(1, 1.0), row(2, 1.1), row(3, 1.2), row(4, 1.3))
    summary = CrossoverSummary(completed=False, variants=tuple(
        VariantRecord(key=key, label="", basis="", ranking=(CostRow(trial_number=4, total_cost=0.5),))
        for key in ("hard", "legacy", "wide")))
    selected = select_finalists(rows, summary)
    assert tuple(f.trial_number for f in selected.finalists) == (1, 2, 3)
    assert all(w.winner is None and w.reason_text == labels.STABILITY_CROSSOVER_INCOMPLETE
               for w in selected.crossover_winners)


def test_minimax_only_competes_with_all_shifts_scored_and_lists_each_missing_outcome() -> None:
    a, b = Finalist(1, 1.3, 1), Finalist(2, 1.34, 2)
    complete = dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 1.387))
    incomplete = dict.fromkeys(SHIFT_NAMES, PointOutcome("excluded"))
    incomplete[SHIFT_NAMES[0]] = PointOutcome("scored", 1.35)
    report = report_arithmetic((b, a), {1: complete, 2: incomplete})
    assert report.minimax_winner == a
    (missing,) = report.minimax_incomplete
    assert missing.finalist == b
    assert missing.reason_text == labels.STABILITY_MINIMAX_INCOMPLETE
    assert missing.missing_points == len(SHIFT_NAMES) - 1
    assert dict(missing.points) == {name: PointOutcome("excluded") for name in SHIFT_NAMES[1:]}
    assert all(point.total_cost is None for _, point in missing.points)
    incomplete[SHIFT_NAMES[1]] = PointOutcome("unplaceable")
    all_incomplete = report_arithmetic((b, a), {1: incomplete, 2: incomplete})
    assert all_incomplete.minimax_winner is None
    assert tuple(entry.finalist for entry in all_incomplete.minimax_incomplete) == (a, b)
    assert all(dict(entry.points)[SHIFT_NAMES[1]].outcome == "unplaceable" for entry in all_incomplete.minimax_incomplete)


def test_minimax_complete_ties_follow_refinement_rank_and_include_original_and_discontinuity() -> None:
    a, b, c = Finalist(1, 3.0, 1), Finalist(2, 1.0, 2), Finalist(3, 2.0, 3)
    points: dict[int | None, dict[str, PointOutcome]] = {1: dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 2.0)),
              2: dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 2.5, model_discontinuity=True)),
              3: dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 2.5))}
    report = report_arithmetic((c, a, b), points)
    assert report.minimax_winner == b
    assert report.minimax_incomplete == ()


def test_minimax_compares_worst_instead_of_best_among_complete_finalists() -> None:
    a, b = Finalist(1, 1.0, 1), Finalist(2, 1.2, 2)
    first = dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 0.5))
    first["ear_down"] = PointOutcome("scored", 4.0)
    second = dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 2.0))
    report = report_arithmetic((a, b), {1: first, 2: second})
    assert report.score_winner == a
    assert report.minimax_winner == b


def test_continuous_extrema_only_remove_discontinuity_and_shift_winner_includes_it() -> None:
    a, b = Finalist(1, 2.0, 1), Finalist(2, 2.5, 2)
    first = dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 2.0))
    first.update({"ear_up": PointOutcome("scored", 0.5, model_discontinuity=True),
                  "seat_left": PointOutcome("scored", 0.8, out_of_spec=True),
                  "seat_right": PointOutcome("scored", 5.0, outside_search=True)})
    second = dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 1.0))
    report = report_arithmetic((a, b), {1: first, 2: second})
    assert (report.finalists[0].continuous_best, report.finalists[0].continuous_worst) == (0.8, 5.0)
    assert next(w for w in report.shift_winners if w.name == "ear_up").winner == a


def test_missing_shift_key_is_rejected() -> None:
    finalist = Finalist(1, 1.0, 1)
    points = dict.fromkeys(SHIFT_NAMES, PointOutcome("scored", 2.0))
    del points["seat_left"]
    with pytest.raises(ValueError, match="給齊具名移位"):
        report_arithmetic((finalist,), {1: points})


def test_stability_crossover_names_match_producer() -> None:
    assert labels.STABILITY_CROSSOVERS == {stitching.record.key: stitching.record.label for stitching in stitchings(200)}
