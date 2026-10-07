"""第二輪複查：用途缺席、逐列沿用正式評估及單一接法缺席。"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from aosr.gui.search_view import build_search_view
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity
from aosr.search import crossover_sensitivity as module
from aosr.search.report import build_report, render_text
from aosr.search.report_comparison import read_refinement_rows
from aosr.search.sampler import Excluded, Scored
from tests.engine._crossover_cases import evaluate, prepared, protected, small_result
from tests.engine._search_run_cases import RUN_DATE


def test_renamed_purpose_is_completed_unverified_without_red_block(tmp_path: Path,
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    registry.write_text(registry.read_text().replace(store.project.purpose, store.project.purpose + "_renamed"))
    calls = []
    monkeypatch.setattr(module, "reevaluate", lambda *args, **kwargs: calls.append(args))
    before = protected(store)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.completed and summary.state == "done" and summary.verdict == "unverified"
    assert "評分設定與搜尋快照不同" in summary.reason_text and not calls
    block = next(b for b in build_search_view(store.path, server_physics="test", server_program="test").blocks if b.key == "crossover")
    assert not block.warning and "評分設定與搜尋快照不同" in "\n".join(block.lines)
    assert protected(store) == before


@pytest.mark.parametrize("f_s,different", [(300, ("legacy", "wide")), (340, ("legacy", "wide")),
                                        (150, ("hard", "legacy")), (120, ("hard", "legacy"))])
def test_equal_weights_reuse_official_evaluation_and_baseline_pin(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, f_s: float, different: tuple[str, ...]) -> None:
    store, registry, status = prepared(tmp_path, f_s=f_s, frequencies=(240, 260, 293, 1000))
    calls, pins, scores = [], [], []
    original_pin, original_score = module.Evaluator.pin, module.Evaluator.score
    def counted(result: SchemeResult, **kwargs: object) -> CandidateEvaluation:
        calls.append(result.origin.trial_number)
        return evaluate(result)
    def pin(self: module.Evaluator, result: SchemeResult,
            candidate: CandidateEvaluation) -> tuple[ComparisonIdentity, ...] | None:
        pins.append(result.origin.trial_number)
        return original_pin(self, result, candidate)
    def score(self: module.Evaluator, result: SchemeResult, candidate: CandidateEvaluation,
              pinned: tuple[ComparisonIdentity, ...]) -> Scored | Excluded:
        scores.append(result.origin.trial_number)
        return original_score(self, result, candidate, pinned)
    monkeypatch.setattr(module, "reevaluate", counted)
    monkeypatch.setattr(module.Evaluator, "pin", pin)
    monkeypatch.setattr(module.Evaluator, "score", score)
    before = protected(store)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    rows = read_refinement_rows(store)
    expected = Counter(row.trial_number for row in rows for _ in ("official", *different))
    assert Counter(calls) == expected
    assert Counter(pins) == Counter(None for _ in ("official", *different))
    assert Counter(scores) == Counter(row.trial_number for row in rows for _ in ("official", "hard", "legacy", "wide"))
    assert {v.key for v in summary.variants if v.tested} == set(different)
    assert protected(store) == before


def test_baseline_equal_weights_do_not_skip_later_different_axis(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path, f_s=340, swap=False)
    for number, energies in ((7, (1, 2)), (9, (3, 4))):
        result = small_result(store, number, 340, energies, (260, 293, 1000))
        store.refine_result_path(number).write_text(result.model_dump_json())
    calls = []
    def counted(result: SchemeResult, **kwargs: object) -> CandidateEvaluation:
        calls.append(result.origin.trial_number)
        return evaluate(result)
    monkeypatch.setattr(module, "reevaluate", counted)
    before = protected(store)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    expected = Counter(n for n in (None, 7, 9) for _ in ("official", "wide"))
    expected.update((7, 9))  # 原方案的上一代權重相同，候選自己的頻率軸卻有換接差異。
    assert Counter(calls) == expected
    legacy = next(v for v in summary.variants if v.key == "legacy")
    assert legacy.tested and {r.trial_number for r in legacy.ranking} == {None, 7, 9}
    assert protected(store) == before


def test_one_missing_variant_keeps_other_stable_comparisons_and_reason_once(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path, swap=False)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    original_pin = module.Evaluator.pin
    def pin(self: module.Evaluator, result: SchemeResult,
            candidate: CandidateEvaluation) -> tuple[ComparisonIdentity, ...] | None:
        points = result.pairs[0].report.points
        assert points
        if result.origin.trial_number is None and points[0].w_geo == 0:
            return None
        return original_pin(self, result, candidate)
    monkeypatch.setattr(module.Evaluator, "pin", pin)
    before = protected(store)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.completed and summary.verdict == "stable" and not summary.reason_text
    hard = next(v for v in summary.variants if v.key == "hard")
    assert not hard.tested and hard.reason_text == "此接法的原方案算不出比較身分"
    assert {v.key for v in summary.variants if v.tested} == {"legacy", "wide"}
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    rendered = render_text(report)
    assert "在已測接法下，排名穩定" in rendered
    assert [line for line in report.crossover.lines if hard.reason_text in line] == [f"{hard.label}：{hard.reason_text}"]
    assert protected(store) == before


def test_hard_room_observed_flip_does_not_add_whole_reason_for_missing_variant(tmp_path: Path,
                                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path, f_s=340, frequencies=(260, 293, 1000))
    monkeypatch.setattr(module, "reevaluate", evaluate)
    original_pin = module.Evaluator.pin
    legacy = next(s for s in module.stitchings(340) if s.record.key == "legacy")
    def pin(self: module.Evaluator, result: SchemeResult,
            candidate: CandidateEvaluation) -> tuple[ComparisonIdentity, ...] | None:
        points = result.pairs[0].report.points
        assert points
        if result.origin.trial_number is None and points[0].w_geo == legacy.geo_weight(points[0].frequency_hz):
            return None
        return original_pin(self, result, candidate)
    monkeypatch.setattr(module.Evaluator, "pin", pin)
    before = protected(store)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.verdict == "sensitive" and not summary.reason_text
    assert next(v for v in summary.variants if v.key == "legacy").reason_text == "此接法的原方案算不出比較身分"
    assert next(v for v in summary.variants if v.key == "wide").tested
    assert protected(store) == before


def test_no_available_variant_is_unverified_with_only_general_whole_reason(tmp_path: Path,
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    original_pin = module.Evaluator.pin
    official = SchemeResult.model_validate_json(store.refine_result_path(None).read_bytes()).pairs[0].report.points
    assert official
    official_weight = official[0].w_geo
    def pin(self: module.Evaluator, result: SchemeResult,
            candidate: CandidateEvaluation) -> tuple[ComparisonIdentity, ...] | None:
        points = result.pairs[0].report.points
        assert points
        if points[0].w_geo != official_weight:
            return None
        return original_pin(self, result, candidate)
    monkeypatch.setattr(module.Evaluator, "pin", pin)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.completed and summary.verdict == "unverified"
    assert summary.reason_text == "做不出比較；各接法原因列在下面"
    assert all(not v.tested and v.reason_text == "此接法的原方案算不出比較身分" for v in summary.variants)
