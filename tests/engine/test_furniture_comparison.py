"""B1 只擴充反射比較支撐；同模型家具仍同表，其他類逐類可比。"""
from __future__ import annotations

import json
from datetime import date

import pytest

from aosr.config.quality_targets import load_quality_targets
from aosr.physics.report_io import ReportOutput
from aosr.scoring.contract import CandidateEvaluation, CategoryEvaluation, QualityCategory, Flag
from aosr.scoring.ranking import rank_candidates, RankingContext, RankingResult
from aosr.scoring.reflections import ReflectionInput
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from aosr.scoring.reflections_cost import comparison_support
from tests.engine import _scoring_source_model_control as control
from tests.engine import test_reflections as fixtures
from tests.engine.test_furniture_reflections import furniture_records
from tests.engine._report_cache import control_pair_cache as control_pair_cache


def test_furniture_comparison_support_adds_only_reflection_model_key() -> None:
    complete = fixtures._evaluate(fixtures._pair())
    approximate = fixtures._evaluate(furniture_records())
    assert isinstance(approximate.payload, ReflectionsAndEchoPayload)
    assert isinstance(complete.payload, ReflectionsAndEchoPayload)
    assert approximate.payload.furniture_model == "single_bounce_finite_size_v1"
    assert "furniture_model" not in complete.payload.model_dump(mode="json")
    original = json.loads(comparison_support(complete))
    assert json.loads(comparison_support(approximate)) == {
        **original, "reflection_model": "single_bounce_finite_size_v1"}


@pytest.mark.parametrize("coverage, model", [("approximate", None), ("complete", "single_bounce_finite_size_v1")])
def test_payload_furniture_model_iff_primary_approximate(coverage: str, model: str | None) -> None:
    evaluation = fixtures._evaluate(furniture_records()) if coverage == "approximate" else fixtures._evaluate(fixtures._pair())
    document = evaluation.model_dump(mode="python")
    document["payload"]["furniture_model"] = model
    with pytest.raises(ValueError, match="家具模型.*主位|主位.*家具模型"):
        CategoryEvaluation.model_validate(document)


@pytest.fixture
def candidate_sets(monkeypatch: pytest.MonkeyPatch) -> tuple[tuple[CandidateEvaluation, ...], tuple[CandidateEvaluation, ...]]:
    for module, name, stand_in in control.STAND_INS:
        monkeypatch.setattr(module, name, stand_in)
    complete, _ = control.run()
    original_solve = control._solve

    def supplied_furniture(candidate: str, speaker: str, receiver: str) -> tuple[ReportOutput, ReflectionInput]:
        _, record = original_solve(candidate, speaker, receiver)
        changed = furniture_records((record,), "desk" if candidate == complete[0].candidate_id else "coffee")[0]
        return changed.report, changed

    monkeypatch.setattr(control, "_solve", supplied_furniture)
    approximate, _ = control.run()
    return complete, approximate


def _rank(candidates: tuple[CandidateEvaluation, ...]) -> RankingResult:
    return rank_candidates(candidates, load_quality_targets(control.TARGETS), RankingContext(
        purpose=control.PURPOSE, receiver_set_fingerprint=control.receivers().fingerprint,
        channel_group_fingerprint=control.group().fingerprint, run_date=date(2026, 9, 27),
        engine_version="furniture-comparison-fixture"))


def test_furniture_changes_only_reflection_identity_and_preserves_other_support(
    candidate_sets: tuple[tuple[CandidateEvaluation, ...], tuple[CandidateEvaluation, ...]],
) -> None:
    complete, approximate = candidate_sets
    for original, changed in zip(complete, approximate, strict=True):
        by_category = {item.category: item for item in original.evaluations}
        for evaluation in changed.evaluations:
            assert evaluation.settings_fingerprint == by_category[evaluation.category].settings_fingerprint
            assert evaluation.evaluator_version == by_category[evaluation.category].evaluator_version
            if evaluation.category in {QualityCategory.TIMBRE_BALANCE, QualityCategory.LISTENING_AREA_STABILITY,
                                       QualityCategory.CHANNEL_MATCHING, QualityCategory.REFLECTIONS_AND_ECHO}:
                assert Flag.FURNITURE_MODEL_APPROXIMATE in evaluation.flags
            else:
                assert Flag.FURNITURE_MODEL_APPROXIMATE not in evaluation.flags
    original_identity = _rank(complete).header.main_table_identity
    changed_identity = _rank(approximate).header.main_table_identity
    assert {left.category for left, right in zip(original_identity, changed_identity, strict=True) if left != right} == {
        QualityCategory.REFLECTIONS_AND_ECHO}
    mixed = _rank((complete[0], approximate[1]))
    assert {row.candidate_id for row in mixed.rankable}.isdisjoint(
        {row.candidate_id for row in mixed.not_comparable.rows})
    assert mixed.not_comparable.rows
    assert not {complete[0].candidate_id, approximate[1].candidate_id} <= {
        row.candidate_id for row in mixed.rankable}
    assert not _rank(approximate).not_comparable.rows
    assert {row.candidate_id for row in _rank(approximate).rankable} == {
        candidate.candidate_id for candidate in approximate}
