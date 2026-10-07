"""審查修補：逐列記憶體、比較排除、同分與有效完成摘要。"""
from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.gui.search_view import build_search_view
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity
from aosr.search import crossover_sensitivity as module
from aosr.search.crossover_record import read_summary, summary_path, write_summary
from aosr.search.outer_status import OuterStatus
from aosr.search.report import build_report
from aosr.search.report_comparison import read_refinement_rows
from aosr.search.sampler import Excluded, RankingZone, Scored
from tests.engine._crossover_cases import evaluate, prepared, protected
from tests.engine._search_run_cases import RUN_DATE


def test_streaming_keeps_at_most_current_and_restitch_and_reads_once(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    live: weakref.WeakValueDictionary[int, SchemeResult] = weakref.WeakValueDictionary()
    peak, reads = [], []
    original_read = Path.read_bytes
    original_validate = SchemeResult.model_validate_json
    original_copy = SchemeResult.model_copy
    def track(result: SchemeResult) -> SchemeResult:
        live[id(result)] = result
        peak.append(len(live))
        return result
    def validate(data: str | bytes | bytearray) -> SchemeResult:
        return track(original_validate(data))
    def copy(result: SchemeResult, *, update: Mapping[str, object] | None = None, deep: bool = False) -> SchemeResult:
        return track(original_copy(result, update=update, deep=deep))
    def read(path: Path) -> bytes:
        if path.parent == store.refine_dir:
            reads.append(path)
        return original_read(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(SchemeResult, "model_validate_json", validate)
    monkeypatch.setattr(SchemeResult, "model_copy", copy)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.completed and summary.verdict == "sensitive"
    expected = Counter(store.refine_result_path(r.trial_number) for r in read_refinement_rows(store))
    assert Counter(reads) == expected and max(peak) <= 2
    assert not live


@pytest.mark.parametrize("few", [False, True])
@pytest.mark.parametrize("f_s", [200, 340])
def test_one_variant_excludes_official_is_unverified_without_distance(tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch, few: bool, f_s: float) -> None:
    store, registry, status = prepared(tmp_path, f_s=f_s, swap=False, frequencies=(260, 1000))
    monkeypatch.setattr(module, "reevaluate", evaluate)
    original_score = module.Evaluator.score
    stitching = next(s for s in module.stitchings(f_s) if s.record.key == "legacy")
    original = SchemeResult.model_validate_json(store.refine_result_path(7).read_bytes())
    weights = tuple(p.w_geo for p in module.restitch(original, stitching).pairs[0].report.points or ())
    def score(self: module.Evaluator, result: SchemeResult, candidate: CandidateEvaluation,
              pinned: tuple[ComparisonIdentity, ...]) -> Scored | Excluded:
        actual = tuple(p.w_geo for p in result.pairs[0].report.points or ())
        if actual == weights and result.origin.trial_number in ((7, 9) if few else (7,)):
            return Excluded(RankingZone.ELIMINATED)
        return original_score(self, result, candidate, pinned)
    monkeypatch.setattr(module.Evaluator, "score", score)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    legacy = next(v for v in summary.variants if v.key == "legacy")
    assert summary.verdict == "unverified"
    assert "上一代接法" in summary.reason_text and "淘汰" in summary.reason_text
    assert "正式第一名在此接法被淘汰" in legacy.reason_text
    assert ("此接法可比較的列少於兩列" in legacy.reason_text) is few
    assert not any(v.speaker_distance_cm or v.primary_distance_cm is not None for v in summary.variants)
    if f_s >= 300:
        assert "300 Hz 到 f_s" in summary.reason_text


def test_alternative_only_tie_prefers_official_over_earlier_ledger_row(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    original_score = module.Evaluator.score
    weights = {n: tuple(p.w_geo for p in SchemeResult.model_validate_json(
        store.refine_result_path(n).read_bytes()).pairs[0].report.points or ()) for n in (None, 7, 9)}
    def score(self: module.Evaluator, result: SchemeResult, candidate: CandidateEvaluation,
              pinned: tuple[ComparisonIdentity, ...]) -> Scored | Excluded:
        if tuple(p.w_geo for p in result.pairs[0].report.points or ()) != weights[result.origin.trial_number]:
            return Scored(0.0)
        return original_score(self, result, candidate, pinned)
    monkeypatch.setattr(module.Evaluator, "score", score)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.official_best == 7 and summary.verdict == "stable"
    for variant in summary.variants:
        assert variant.tested and variant.ranking[0].trial_number == 7 and variant.official_rank == 1
        assert not variant.speaker_distance_cm and variant.primary_distance_cm is None


@pytest.mark.parametrize("error", [KeyboardInterrupt(), KeyError("f_s_hz"), AssertionError()])
def test_error_record_preserves_fresh_completion_byte_for_byte(tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch, error: BaseException) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    module.attach_crossover(store, status=status, quality_targets_path=registry)
    before = summary_path(store.path).read_bytes()
    module.record_crossover_error(store, status, error)
    assert summary_path(store.path).read_bytes() == before


@pytest.mark.parametrize("change", ["conclusion", "snapshot"])
def test_changed_conclusion_or_snapshot_really_reevaluates(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    store, registry, status = prepared(tmp_path)
    calls = []
    def counted(result: SchemeResult, **kwargs: object) -> object:
        calls.append(result.scheme.scheme_id)
        return evaluate(result)
    monkeypatch.setattr(module, "reevaluate", counted)
    first = module.attach_crossover(store, status=status, quality_targets_path=registry)
    previous = tuple(calls)
    changed = status.model_copy(update={"outer": OuterStatus(conclusion="refine_budget")}) if change == "conclusion" else status.model_copy(
        update={"refine": status.refine.model_copy(update={"refined": status.refine.refined + 1})})
    current = module.attach_crossover(store, status=changed, quality_targets_path=registry)
    assert tuple(calls) != previous and current.completed
    assert current.conclusion != first.conclusion if change == "conclusion" else current.snapshot != first.snapshot


@pytest.mark.parametrize("missing", ["primary", "speakers"])
def test_inconsistent_distances_only_break_crossover_block(tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    import json
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    module.attach_crossover(store, status=status, quality_targets_path=registry)
    view = build_search_view(store.path, server_physics="test", server_program="test")
    before = protected(store)
    data = json.loads(summary_path(store.path).read_bytes())
    variant = next(v for v in data["variants"] if v["speaker_distance_cm"])
    variant["primary_distance_cm" if missing == "primary" else "speaker_distance_cm"] = None if missing == "primary" else {}
    summary_path(store.path).write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        read_summary(store.path)
    current = build_search_view(store.path, server_physics="test", server_program="test")
    assert [b for b in current.blocks if b.key != "crossover"] == [b for b in view.blocks if b.key != "crossover"]
    block = next(b for b in current.blocks if b.key == "crossover")
    assert block.warning and "交接敏感度摘要讀不到" in "\n".join(block.lines)
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert report.crossover.warning and "讀不到" in "\n".join(report.crossover.lines)
    assert protected(store) == before


def test_observed_flip_stays_sensitive_when_another_variant_excludes_official(tmp_path: Path,
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    """一種接法把正式第一名排除、另一種接法看到換人：看到的事實優先，判敏感，排除那一種照列原因。"""
    store, registry, status = prepared(tmp_path, f_s=200)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    original_score = module.Evaluator.score
    stitching = next(s for s in module.stitchings(200) if s.record.key == "legacy")
    original = SchemeResult.model_validate_json(store.refine_result_path(7).read_bytes())
    weights = tuple(p.w_geo for p in module.restitch(original, stitching).pairs[0].report.points or ())

    def score(self: module.Evaluator, result: SchemeResult, candidate: CandidateEvaluation,
              pinned: tuple[ComparisonIdentity, ...]) -> Scored | Excluded:
        actual = tuple(p.w_geo for p in result.pairs[0].report.points or ())
        if actual == weights and result.origin.trial_number == 7:
            return Excluded(RankingZone.ELIMINATED)
        return original_score(self, result, candidate, pinned)

    monkeypatch.setattr(module.Evaluator, "score", score)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    legacy = next(v for v in summary.variants if v.key == "legacy")
    wide = next(v for v in summary.variants if v.key == "wide")
    assert summary.verdict == "sensitive"
    assert "正式第一名在此接法被淘汰" in legacy.reason_text and "上一代接法" in summary.reason_text
    assert wide.tested and wide.ranking[0].trial_number != summary.official_best
    assert wide.primary_distance_cm is not None
