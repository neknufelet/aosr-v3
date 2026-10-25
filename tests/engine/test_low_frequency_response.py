"""#558 第二段 A：漏接低頻、除以二或讀錯能量都要紅；答案獨立從造曲線求。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualificationEntry, WeightTable, load_quality_targets
from aosr.scoring.contract import QualityCategory
from aosr.scoring.stage_one_score import score_stage_one, stage_one_score_settings
from tests.engine._stage_one_low_frequency_curves import (
    candidate, speaker, independent_own_line_answer, DIRECTIONS,
)
from tests.engine._placement import POINT_PLACEMENT
from tests.engine.test_ranking_recost import _PROVENANCE
from tests.engine.test_low_frequency_response_reload import saved as saved
from aosr.reporting.result import SchemeResult

TARGETS = config_path("quality_targets.toml")
PURPOSE = "dedicated_two_channel_listening_room"


def test_new_category_is_registered_as_optional() -> None:
    registry = load_quality_targets(TARGETS)
    category = next((item for item in QualityCategory if item.value == "low_frequency_response"), None)
    assert category is not None, "第二階段尚未新增低頻頻響類別"
    optional = registry.purpose(PURPOSE).entry("ranking.optional_categories")
    assert isinstance(optional, QualificationEntry) and isinstance(optional.value, tuple)
    assert category.value in optional.value


@pytest.mark.parametrize("axis", (LowFrequencyAxis.SEARCH_48, LowFrequencyAxis.VERIFICATION))
def test_two_channels_independent_formula_and_stage_one_components(axis: LowFrequencyAxis) -> None:
    from aosr.scoring.low_frequency_response import evaluate_low_frequency_response
    from aosr.scoring.low_frequency_response_cost import cost_evaluation
    registry = load_quality_targets(TARGETS)
    settings = stage_one_score_settings(registry, PURPOSE)
    frequencies = np.array(low_frequency_axis_frequencies(axis)[0])
    primary = [3 * np.sin(np.log2(frequencies) * 4), 5 * np.cos(np.log2(frequencies) * 3)]
    speakers = [speaker(y, {d: y + (index + 1) * (channel + 1) / 3
                            for index, d in enumerate(DIRECTIONS)},
                        frequencies=frequencies, speaker_id=name)
                for channel, (y, name) in enumerate(zip(primary, ("left", "right"), strict=True))]
    data = candidate(*speakers, axis=axis)
    measured = evaluate_low_frequency_response(data, registry=registry, purpose=PURPOSE,
        scene_fingerprint="a" * 64, placement=POINT_PLACEMENT, provenance=_PROVENANCE)
    costed = cost_evaluation(measured, registry.stage_one(PURPOSE), registry.fingerprint)
    own = math.fsum(independent_own_line_answer(frequencies, y, settings.scored_range_hz)[0]
                    for y in primary) / len(primary)
    # 六方向固定 dB 偏移；每方向 RMS=|偏移|，六方向平均，再兩聲道等權平均。
    seat = ((1 + 2 + 3 + 4 + 5 + 6) / 6 / 3) * ((1 + 2) / 2)
    expected = own + seat
    assert costed.category_cost is not None
    assert costed.category_cost.value == pytest.approx(expected, rel=5e-13)
    weights = registry.purpose(PURPOSE).entry("ranking.category_weights")
    assert isinstance(weights, WeightTable)
    weight = next(row.value for row in weights.item if row.name == QualityCategory.LOW_FREQUENCY_RESPONSE.value)
    assert weight * costed.category_cost.value == pytest.approx(0.25 * expected, rel=5e-13)
    payload = measured.payload
    from aosr.scoring.low_frequency_response_contract import LowFrequencyResponsePayload
    assert isinstance(payload, LowFrequencyResponsePayload)
    first = score_stage_one(payload.evaluation, settings)
    assert costed.category_cost.components == {item.name: item.weighted_cost for item in first.components}
    assert payload.worst_channel == first.worst_channel
    assert first.screening_total is not None
    assert costed.category_cost.value.hex() == first.screening_total.hex()
    assert {q.name: q.value for q in costed.raw_quantities} == {
        item.name: item.value_db for item in first.components}


def test_pipeline_produces_low_frequency_without_stage_one_scene_prerequisites() -> None:
    from tests.engine._stage_one_cases import scheme
    from tests.engine.test_scheme_pipeline import _run_control
    item = scheme().model_copy(update={"source_model": "product_default"})
    result = _run_control(item)
    category = next((e for e in result.candidate.evaluations
                     if e.category is QualityCategory.LOW_FREQUENCY_RESPONSE), None)
    assert category is not None, "evaluate_parts 漏接低頻頻響"
    assert category.payload is not None, category.reason_codes


def test_low_frequency_cost_flips_ranking_without_affecting_other_costs() -> None:
    from aosr.scoring.low_frequency_response import evaluate_low_frequency_response
    from tests.engine._ranking_fixtures import _candidate, _rank, _registry, _timbre
    registry = _registry()
    old = [_candidate(_timbre(name, tilt=0.5, residual=residual))
           for name, residual in (("a", 1.0), ("b", 2.0))]
    assert [r.candidate_id for r in _rank(*old).rankable] == ["a", "b"]
    # 舊總分甲=1/4=.25、乙=2/4=.5；低頻主位平坦、六座固定偏移甲 2 dB、乙 0 dB。
    # 新總分甲=.25+.25*2=.75、乙=.5；翻轉點類權重=(.5-.25)/(2-0)=.125。
    enriched = []
    for base, offset in zip(old, (2.0, 0.0), strict=True):
        y = np.zeros(len(low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH_48)[0]))
        curves = candidate(speaker(y, {d: y + offset for d in DIRECTIONS}))
        curves = curves.model_copy(update={"candidate_id": base.candidate_id})
        lf = evaluate_low_frequency_response(curves, registry=registry, purpose=PURPOSE,
            scene_fingerprint=base.scene_fingerprint, placement=POINT_PLACEMENT, provenance=_PROVENANCE)
        enriched.append(_candidate(*base.evaluations, lf))
    rows = _rank(*enriched, registry=registry).rankable
    from tests.engine._cross_cpu_tolerance import SCORING_REL, SCORING_ABS
    assert [r.candidate_id for r in rows] == ["b", "a"]
    assert {r.candidate_id: r.total_cost for r in rows} == pytest.approx(
        {"b": 0.5, "a": 0.75}, rel=SCORING_REL, abs=SCORING_ABS)
    for row in rows:
        timbre, = (c for c in row.categories if c.identity.category is QualityCategory.TIMBRE_BALANCE)
        assert timbre.category_cost == {"a": 0.25, "b": 0.5}[row.candidate_id]


@pytest.mark.parametrize("damage,expected", [(None, None), ("missing_direction", "low_frequency_seat_layout_invalid"),
    ("nonpositive", "non_positive_energy"), ("hole", "missing_declared_point"),
    ("wrong_axis", "frequency_axis_mismatch")])
def test_report_adapter_preserves_unavailable_reasons_and_optional_ranking(
    saved: SchemeResult, damage: str | None, expected: str | None,
) -> None:
    from aosr.reporting.low_frequency_response import evaluate_result_low_frequency
    from aosr.scoring.contract import EvaluationState
    from aosr.scoring.ranking import CandidateStatus, rank_candidates
    from tests.engine._stage_one_cases import scheme
    from tests.engine.test_scheme_pipeline import _run_control
    from tests.engine.test_low_frequency_response_reload import _context
    item = saved
    if damage == "missing_direction":
        receiver_points = item.scheme.receiver_set.points[:-1]
        changed = item.scheme.model_copy(update={"receiver_set": item.scheme.receiver_set.model_copy(update={"points": receiver_points})})
        pairs = tuple(p for p in item.pairs if p.receiver_id in {x.receiver_id for x in receiver_points})
        item = item.model_copy(update={"scheme": changed, "pairs": pairs})
    elif damage is not None:
        pair = item.pairs[-1]
        assert pair.report.points is not None
        points = list(pair.report.points)
        point = points[3]
        update = ({"fem_energy": 0.0} if damage == "nonpositive" else
                  {"fem_energy": None} if damage == "hole" else {"frequency_hz": point.frequency_hz + 0.01})
        points[3] = point.model_copy(update=update)
        pair = pair.model_copy(update={"report": pair.report.model_copy(update={"points": tuple(points)})})
        item = item.model_copy(update={"pairs": (*item.pairs[:-1], pair)})
    result = evaluate_result_low_frequency(item, TARGETS, item.candidate.scene_fingerprint)
    if expected is None:
        assert result.state is EvaluationState.MEASURED
    else:
        assert result.state is EvaluationState.UNAVAILABLE
        assert tuple(c.value for c in result.reason_codes) == (expected,)
        assert result.payload is None and not result.raw_quantities
        # 任一支不可估整類不湊數，作為選算類仍可排名、不淘汰。
        candidate_result = item.candidate.model_copy(update={"evaluations": tuple(
            result if e.category is QualityCategory.LOW_FREQUENCY_RESPONSE else e
            for e in item.candidate.evaluations)})
        ranked = rank_candidates([candidate_result], load_quality_targets(TARGETS), _context(item))
        assert ranked.status_of(item.scheme.scheme_id) is CandidateStatus.RANKABLE
        row, = ranked.rankable
        assert QualityCategory.LOW_FREQUENCY_RESPONSE in {u.category for u in row.uncovered}


def test_no_low_frequency_difference_keeps_head_control_costs_and_order() -> None:
    from aosr.scoring.ranking import rank_candidates
    from tests.engine.test_low_frequency_response_reload import _context
    from tests.engine.test_scheme_pipeline import _control_result
    from tests.engine._cross_cpu_tolerance import SCORING_REL, SCORING_ABS
    # 改動前 HEAD 7d966858 實跑 _control_result + rank_candidates 的逐格原值。
    # 兩座控制組新類不可估，沒有低頻項差異；其他五類每格及名次必須保留。
    expected = {
        "wall-1": (1, 4.249473949461246, {
            "channel_matching": 0.07616984431420677, "listening_area_stability": 0.14832994797740873,
            "reflections_and_echo": 0.15162092979796526, "reverberation": 0.015514454510611286,
            "timbre_balance": 3.8578387728610535}),
        "wall-2": (2, 4.95225064768086, {
            "channel_matching": 0.05175671961131201, "listening_area_stability": 0.14380477030239736,
            "reflections_and_echo": 0.3086342196906641, "reverberation": 0.015514454510611286,
            "timbre_balance": 4.432540483565876})}
    results = [_control_result(name) for name in expected]
    ranked = rank_candidates([x.candidate for x in results], load_quality_targets(TARGETS), _context(results[0]))
    for row in ranked.rankable:
        rank, total, costs = expected[row.candidate_id]
        assert row.rank == rank
        assert row.total_cost == pytest.approx(total, rel=SCORING_REL, abs=SCORING_ABS)
        assert {c.identity.category.value: c.category_cost for c in row.categories} == pytest.approx(
            costs, rel=SCORING_REL, abs=SCORING_ABS)


def test_common_input_rejects_other_seat_beyond_the_six_directions() -> None:
    from aosr.scoring.low_frequency_input import low_frequency_input
    from aosr.scoring.receiver_set import ReceiverRole
    from tests.engine._stage_one_cases import scheme
    item = scheme()
    extra = item.receiver_set.points[-1].model_copy(update={
        "receiver_id": "other", "role": ReceiverRole.OTHER_SEAT, "direction_relative_to_primary": None})
    receivers = item.receiver_set.model_copy(update={"points": (*item.receiver_set.points, extra)})
    axis = LowFrequencyAxis.SEARCH_48
    frequencies = low_frequency_axis_frequencies(axis)[0]
    energies = {(s, p.receiver_id): tuple(1.0 for _ in frequencies)
                for s in item.speakers for p in receivers.points}
    with pytest.raises(ValueError, match="主位|座位"):
        low_frequency_input(candidate_id=item.scheme_id, speaker_ids=tuple(item.speakers), receiver_set=receivers,
            axis=axis, energies=energies, frequency_hex=tuple(float(f).hex() for f in frequencies))
