"""合成細算表的交接提醒：只換能量權重、不改正式排序與任何搜尋位元組。"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from aosr.reporting.result import SchemeResult
from aosr.search import crossover_sensitivity as module
from aosr.search.crossover_record import VERDICTS, read_summary, summary_lines
from aosr.search import labels
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.store import SearchStore
from aosr.search.report_comparison import read_refinement_rows
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity, RankingContext, comparison_identity_of
from aosr.config.quality_targets import QualityTargets
from tests.engine._crossover_cases import evaluate, prepared, protected


def _rewrite_rows(store: SearchStore, rows: tuple[RefineRow, ...]) -> None:
    header, _ = RefineLedger.read(store.refine_ledger_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header)
    for row in rows:
        book.append(row)


@pytest.mark.parametrize("f_s,swap,verdict", [(200, True, "sensitive"), (200, False, "stable"),
    (340, False, "unverified"), (340, True, "sensitive")])
def test_synthetic_rankings_and_distances(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                         f_s: float, swap: bool, verdict: str) -> None:
    store, registry, status = prepared(tmp_path, f_s=f_s, swap=swap)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    before = protected(store)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.completed and summary.verdict == verdict
    assert read_summary(store.path) == summary and protected(store) == before
    assert VERDICTS[verdict] in summary_lines(summary)
    if f_s >= 300 and not swap:
        assert "300 Hz 到" in summary.reason_text and "沒有有限元素" in summary.reason_text
    changed = [v for v in summary.variants if v.tested and v.ranking[0].trial_number != summary.official_best]
    if swap:
        assert changed
        original = SchemeResult.model_validate_json(store.refine_result_path(summary.official_best).read_bytes()).scheme
        for variant in changed:
            other = SchemeResult.model_validate_json(store.refine_result_path(variant.ranking[0].trial_number).read_bytes()).scheme
            expected = {key: math.dist(p.as_tuple(), other.speakers[key].as_tuple()) * 100
                        for key, p in original.speakers.items()}
            assert variant.speaker_distance_cm == expected
            assert variant.primary_distance_cm == math.dist(original.receiver_set.primary.position_m,
                other.receiver_set.primary.position_m) * 100
            assert variant.official_rank is not None and variant.official_rank > 1
    else:
        assert not changed


def test_all_variants_equal_official_are_not_comparisons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path, frequencies=(20, 1000))
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.verdict == "unverified" and not any(v.tested for v in summary.variants)
    assert all(v.reason_text == labels.IDENTICAL_CROSSOVER_REASON for v in summary.variants)


@pytest.mark.parametrize("change,reason", [("short", "少於兩列"), ("unreadable", "試算 9：結果讀不回"),
    ("physics_identity", "身分不合"), ("program_fingerprint", "身分不合"), ("purpose_settings", "身分不合"),
    ("settings", "評分設定與搜尋快照不同"), ("ulp", "正式接法自檢不等：試算 9"),
    ("fs", "f_s 不同"), ("baseline", "缺原方案"), ("header", "細算帳身分")])
def test_unverified_defences(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str, reason: str) -> None:
    import json
    from aosr.reporting.evaluation import purpose_settings
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    rows = read_refinement_rows(store)
    path = store.refine_result_path(9)
    if change in ("short", "baseline", "ulp"):
        selected = rows[:1] if change == "short" else tuple(r for r in rows if r.trial_number is not None)
        if change == "ulp":
            selected = tuple(r.model_copy(update={"total_cost": math.nextafter(r.total_cost, math.inf)})
                if r.trial_number == 9 and r.total_cost is not None else r for r in rows)
        _rewrite_rows(store, selected)
    elif change in ("settings", "purpose_settings"):
        registry.write_text(registry.read_text().replace("value = -10.0", "value = -11.0", 1))
        if change == "purpose_settings":
            result = SchemeResult.model_validate_json(path.read_bytes())
            result = result.model_copy(update={"purpose_settings": purpose_settings(registry, result.scheme.purpose)})
            path.write_text(result.model_dump_json())
            registry.write_text(registry.read_text().replace("value = -11.0", "value = -10.0", 1))
    elif change == "unreadable":
        path.write_text("{")
    elif change == "header":
        text = store.refine_ledger_path.read_text().splitlines()
        text[0] = json.dumps(json.loads(text[0]) | {"physics_identity": "other"})
        store.refine_ledger_path.write_text("\n".join(text) + "\n")
    else:
        result = SchemeResult.model_validate_json(path.read_bytes())
        if change == "fs":
            result = result.model_copy(update={"pairs": tuple(p.model_copy(update={"report": p.report.model_copy(update={
                "top": p.report.top.model_copy(update={"f_s_hz": 201})})}) for p in result.pairs)})
        else:
            prefix = "phys-v1:" if change == "physics_identity" else "calc-v1:"
            result = result.model_copy(update={change: prefix + "c" * 64})
        path.write_text(result.model_dump_json())
    before = protected(store)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.verdict == "unverified" and reason in summary.reason_text
    assert protected(store) == before


def test_missing_comparison_identity_is_explained_per_stitching(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    real_pin = comparison_identity_of
    calls = []
    def pin(candidate: CandidateEvaluation, registry: QualityTargets, context: RankingContext) -> tuple[ComparisonIdentity, ...] | None:
        calls.append(candidate)
        return real_pin(candidate, registry, context) if len(calls) == 1 else None
    monkeypatch.setattr(module, "comparison_identity_of", pin)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.verdict == "unverified"
    assert all(not v.tested and "原方案算不出比較身分" in v.reason_text for v in summary.variants)


@pytest.mark.parametrize("f_s,key", [(340, "hard"), (150, "wide"), (100, "wide")])
def test_official_weights_define_identical_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                   f_s: float, key: str) -> None:
    store, registry, status = prepared(tmp_path, f_s=f_s)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    variant = next(v for v in summary.variants if v.key == key)
    assert not variant.tested and variant.reason_text == labels.IDENTICAL_CROSSOVER_REASON


def test_identical_reason_uses_shared_producer_value(monkeypatch: pytest.MonkeyPatch) -> None:
    reason = labels.IDENTICAL_CROSSOVER_REASON + "（來源替換探針）"
    monkeypatch.setattr(module, "IDENTICAL_CROSSOVER_REASON", reason, raising=False)
    variant = module._variant(module.VariantScores(module.stitchings(340)[0]), 7)
    assert variant.reason_text == reason


@pytest.mark.parametrize("f_s,truncated,unavailable", [(157, False, False), (340, True, False),
    (424.27, False, True), (500, False, True)])
def test_legacy_logarithmic_band_and_clipping(tmp_path: Path, f_s: float, truncated: bool, unavailable: bool) -> None:
    from tests.engine._crossover_cases import small_result
    store, _, _ = prepared(tmp_path)
    stitching = next(s for s in module.stitchings(f_s) if s.record.key == "legacy")
    assert stitching.record.truncated is truncated
    assert bool(stitching.record.reason_text) is unavailable
    if unavailable:
        from aosr.search.crossover_record import _variant_lines
        assert "不適用" in _variant_lines(stitching.record)[0]
    if not unavailable:
        low, high = f_s / math.sqrt(2), min(f_s * math.sqrt(2), 300)
        frequencies = (low, math.sqrt(low * high), high, 1000)
        result = small_result(store, None, f_s, (2, 4), frequencies)
        changed = module.restitch(result, stitching)
        points = changed.pairs[0].report.points
        assert points is not None
        assert tuple(p.w_geo for p in points) == pytest.approx((0, 0.5, 1, 1))
        assert points[-1].fem_energy is None and points[-1].total_energy == 4
        assert changed.pairs[0].report.top == result.pairs[0].report.top
        for old, new in zip(result.pairs[0].report.points or (), points, strict=True):
            assert old.model_dump(exclude={"w_fem", "w_geo", "total_energy"}) == new.model_dump(exclude={"w_fem", "w_geo", "total_energy"})


def test_cached_completion_reads_results_once_and_retries_changed_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    calls, reads = [], []
    def counted(result: SchemeResult, **kwargs: object) -> object:
        calls.append(result.scheme.scheme_id)
        return evaluate(result)
    original_read = Path.read_bytes
    def read(path: Path) -> bytes:
        if path.parent == store.refine_dir:
            reads.append(path)
        return original_read(path)
    monkeypatch.setattr(module, "reevaluate", counted)
    monkeypatch.setattr(Path, "read_bytes", read)
    first = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert first.completed
    assert reads == [store.refine_result_path(r.trial_number) for r in read_refinement_rows(store)]
    previous_calls = tuple(calls)
    again = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert again == first and tuple(calls) == previous_calls
    rows = read_refinement_rows(store)
    _rewrite_rows(store, tuple(r.model_copy(update={"total_cost": math.nextafter(r.total_cost, math.inf)})
        if r.trial_number == 9 and r.total_cost is not None else r for r in rows))
    changed = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert changed.rows_fingerprint != first.rows_fingerprint and tuple(calls) != previous_calls


def test_incomparable_changed_candidate_is_listed_and_not_ranked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    def differing(result: SchemeResult, **kwargs: object) -> object:
        candidate = evaluate(result)
        official = SchemeResult.model_validate_json(store.refine_result_path(9).read_bytes())
        if result.origin.trial_number == 9 and not module.same_weights(result, official):
            candidate = candidate.model_copy(update={"evaluations": tuple(e.model_copy(update={"evaluator_version": "other"})
                for e in candidate.evaluations)})
        return candidate
    monkeypatch.setattr(module, "reevaluate", differing)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.variants
    for variant in summary.variants:
        assert all(r.trial_number != 9 for r in variant.ranking)
        assert any(r.trial_number == 9 and r.reason_text == "不能同表" for r in variant.excluded)


def test_official_pin_missing_is_unverified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    monkeypatch.setattr(module, "reevaluate", evaluate)
    monkeypatch.setattr(module, "comparison_identity_of", lambda *args: None)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.verdict == "unverified" and "正式接法的原方案算不出比較身分" in summary.reason_text


def test_official_excluded_under_all_alternatives_is_unverified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path)
    original = SchemeResult.model_validate_json(store.refine_result_path(7).read_bytes())
    def excluded(result: SchemeResult, **kwargs: object) -> CandidateEvaluation:
        candidate = evaluate(result)
        if result.origin.trial_number == 7 and not module.same_weights(original, result):
            return candidate.model_copy(update={"evaluations": tuple(e.model_copy(update={"evaluator_version": "other"})
                for e in candidate.evaluations)})
        return candidate
    monkeypatch.setattr(module, "reevaluate", excluded)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.verdict == "unverified"
    assert all(not v.tested and v.official_rank is None and "正式第一名在此接法不能同表" in v.reason_text for v in summary.variants)


def test_missing_fem_below_cap_still_uses_geometric_only(tmp_path: Path) -> None:
    store, _, _ = prepared(tmp_path)
    result = SchemeResult.model_validate_json(store.refine_result_path(None).read_bytes())
    pair = result.pairs[0]
    assert pair.report.points
    missing = pair.report.points[0].model_copy(update={"fem_energy": None})
    pair = pair.model_copy(update={"report": pair.report.model_copy(update={"points": (missing, *pair.report.points[1:])})})
    result = result.model_copy(update={"pairs": (pair, *result.pairs[1:])})
    for stitching in module.stitchings(200):
        changed = module.restitch(result, stitching)
        points = changed.pairs[0].report.points
        assert points
        assert points[0].w_geo == 1 and points[0].w_fem == 0
        assert points[0].total_energy == points[0].geometric_energy


def test_ties_preserve_ledger_order_and_do_not_trigger_sensitivity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.engine._crossover_cases import small_result
    from aosr.config.quality_targets import load_quality_targets
    from aosr.search.scoring import screening_outcome
    from aosr.search.sampler import Scored
    store, registry, status = prepared(tmp_path)
    result = small_result(store, 9, 200, (1, 2), (220, 1000))
    outcome, _ = screening_outcome(result.candidate, result.scheme, registry=load_quality_targets(registry),
        run_date=result.run_date, engine_version=store.identity.program_fingerprint, pinned=None)
    assert isinstance(outcome, Scored)
    store.refine_result_path(9).write_text(result.model_dump_json())
    rows = read_refinement_rows(store)
    _rewrite_rows(store, (rows[0], rows[2].model_copy(update={"total_cost": outcome.value}), rows[1]))
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    assert summary.verdict == "stable" and summary.official_best == 9
    assert all(v.ranking[0].trial_number == 9 and v.official_rank == 1 for v in summary.variants if v.tested)


def test_alternative_winner_can_be_original_scheme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.engine._crossover_cases import small_result
    from aosr.config.quality_targets import load_quality_targets
    from aosr.search.scoring import screening_outcome
    from aosr.search.sampler import Scored
    store, registry, status = prepared(tmp_path)
    result = small_result(store, None, 200, (0.9, 3), (220, 1000))
    outcome, _ = screening_outcome(result.candidate, result.scheme, registry=load_quality_targets(registry),
        run_date=result.run_date, engine_version=store.identity.program_fingerprint, pinned=None)
    assert isinstance(outcome, Scored)
    store.refine_result_path(None).write_text(result.model_dump_json())
    rows = read_refinement_rows(store)
    _rewrite_rows(store, tuple(r.model_copy(update={"total_cost": outcome.value}) if r.trial_number is None else r for r in rows))
    monkeypatch.setattr(module, "reevaluate", evaluate)
    summary = module.attach_crossover(store, status=status, quality_targets_path=registry)
    hard = next(v for v in summary.variants if v.key == "hard")
    assert summary.verdict == "sensitive" and hard.ranking[0].trial_number is None
    assert "300 Hz 硬切第一名：原方案" in "\n".join(summary_lines(summary))


def test_reminder_modules_are_outside_both_identity_closures() -> None:
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.paths import config_path
    from aosr.reporting.physics_identity import physics_identity_parts
    from aosr.reporting.modal_diagnosis import modal_import_closure
    physics = physics_identity_parts(capabilities=load_capabilities(config_path("capabilities.toml")),
        directivity=load_directivity_defaults(config_path("directivity_defaults.toml")))
    changed = {"aosr.config.crossover_sensitivity", "aosr.search.crossover_sensitivity",
        "aosr.search.crossover_record", "aosr.search.report_crossover"}
    assert not changed.intersection(physics.closure) and not changed.intersection(modal_import_closure().modules)
