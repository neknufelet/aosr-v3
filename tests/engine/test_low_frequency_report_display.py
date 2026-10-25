"""低頻搜尋報告：分類代價與第一階段尺的校準狀態，不把現在的尺稱為搜尋快照。"""
import json
from pathlib import Path

import pytest

from aosr.config.quality_targets import QualityPurpose, load_quality_targets
from aosr.reporting.compare import compare_results
from aosr.reporting.display import LABELS
from aosr.reporting.result import SchemeResult
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import EvaluationState, QualityCategory
from aosr.scoring.contract_base import ReasonCode
from aosr.scoring.ranking import rank_candidates
from aosr.scoring.ranking_models import CandidateStatus, ExternalAcceptance, ExternalFloors, RankingContext
from aosr.scoring.stage_one_score import SCORED_RANGE_KEY
from aosr.search.report import _candidate_quality, build_report, render_text
from aosr.search.report_calibration import calibration_lines, calibration_progress
from aosr.search.run import SearchStatus
from tests.engine._search_run_cases import RUN_DATE, make_store
from tests.engine._stage_one_low_frequency_curves import DIRECTIONS, speaker
from tests.engine.test_low_frequency_response_display import (
    TARGETS, ZEROS, assert_details, expected_description, low_frequency, measured,
)
from tests.engine.test_low_frequency_response_reload import saved as saved

LF = QualityCategory.LOW_FREQUENCY_RESPONSE


def _calibrated(section: QualityPurpose, baseline: str) -> QualityPurpose:
    """考卷只改記憶體內的狀態；每次留一格基線，其餘已校準。"""
    def status(key: str) -> str:
        return "baseline" if key == baseline else "calibrated"
    return section.model_copy(update={
        "setting": tuple(item.model_copy(update={"status": status(item.key)}) for item in section.setting),
        "target": tuple(item.model_copy(update={"status": status(item.key)}) for item in section.target),
        "weight": tuple(table.model_copy(update={"item": tuple(item.model_copy(update={
            "status": status(f"{table.key}.{item.name}")}) for item in table.item)}) for table in section.weight)})


def test_calibration_counts_stage_one_sources(saved: SchemeResult) -> None:
    registry = load_quality_targets(TARGETS)
    purpose = registry.purpose(saved.scheme.purpose)
    section = registry.stage_one(purpose.name)
    sources = CATEGORY_REGISTRY[LF].registry_sources(section)
    progress = calibration_progress(purpose, stage_one=section)
    low = next(item for item in progress.categories if item.category == LF.value)
    assert low.total == sum(1 for _ in sources)
    assert low.calibrated == sum(status == "calibrated" for _, status in sources)
    assert low.judged and not low.uses_all_calibrated
    assert "低頻頻響採現在登記簿的第一階段尺；搜尋快照未保存該段" in calibration_lines(progress)[0]
    for key in ("", *(key for key, _ in sources)):
        promoted = calibration_progress(purpose, stage_one=_calibrated(section, key))
        item = next(item for item in promoted.categories if item.category == LF.value)
        assert item.uses_all_calibrated == (key == "")
        assert ("低頻頻響" in calibration_lines(promoted)[2]) == (key == "")


def test_report_prints_low_frequency_cost_and_details(saved: SchemeResult, tmp_path: Path) -> None:
    store, path = make_store(tmp_path)
    registry = load_quality_targets(path)
    store.status_path.write_text(SearchStatus(best_trial=0).model_dump_json())
    # 第一名的低頻跟原方案不同，兩段各自要印自己的代價與詳情，誤用另一份或漏一段都會紅（審查補）。
    # 改名走存檔文字整份換代號（評估裡的候選代號也跟著換），報告讀得回。
    renamed = SchemeResult.model_validate_json(
        saved.model_dump_json().replace(json.dumps(saved.scheme.scheme_id), json.dumps("other")))
    best = measured(renamed, speaker(ZEROS, {name: ZEROS + 6.0 for name in DIRECTIONS}))
    SchemeResult.model_validate_json(best.model_dump_json())
    ranked = compare_results((saved, best), quality_targets=registry, run_date=RUN_DATE)
    costs = {row.candidate_id: next(item.category_cost for item in row.categories if item.identity.category is LF)
             for row in ranked.rankable}
    assert costs[saved.scheme.scheme_id] != costs[best.scheme.scheme_id]
    quality = _candidate_quality(saved, ranked, "第一名結果已讀回")
    assert f"{LABELS[LF.value]}：代價 {costs[saved.scheme.scheme_id]:.3f}" in quality.category_lines
    assert_details("\n".join(quality.category_lines), low_frequency(saved))
    store.baseline_path.write_text(saved.model_dump_json())
    store.candidate_path(0).write_text(best.model_dump_json())
    text = render_text(build_report(store, quality_targets_path=path, run_date=RUN_DATE))
    for label, result in (("原方案", saved), ("第一名", best)):
        own = "\n".join(line for line in text.splitlines() if line.startswith(label))
        assert f"{label}低頻頻響：代價 {costs[result.scheme.scheme_id]:.3f}" in own
        assert_details(own, low_frequency(result))
    assert expected_description(registry, saved.scheme.purpose) in text
    assert "低頻頻響採現在登記簿的第一階段尺" in text


def test_eliminated_candidate_keeps_its_low_frequency_cost(saved: SchemeResult) -> None:
    # 淘汰是先合分才判底線，淘汰列留下的低頻代價照印，不寫成「未計代價」；答案取同一份輸入在可排名時的代價（審查補）。
    registry = load_quality_targets(TARGETS)
    normal = compare_results((saved,), quality_targets=registry, run_date=RUN_DATE)
    expected = next(item.category_cost for item in normal.rankable[0].categories if item.identity.category is LF)
    context = RankingContext(purpose=saved.scheme.purpose,
                             receiver_set_fingerprint=saved.scheme.receiver_set.fingerprint,
                             channel_group_fingerprint=saved.scheme.channel_group.fingerprint,
                             run_date=RUN_DATE, engine_version=saved.program_fingerprint)
    floors = ExternalFloors(declared_by="考卷", verdicts={saved.candidate.candidate_id: ExternalAcceptance.FAILED})
    ranked = rank_candidates([saved.candidate], registry, context, floors)
    assert ranked.status_of(saved.candidate.candidate_id) is CandidateStatus.ELIMINATED
    lines = _candidate_quality(saved, ranked, "已讀回").category_lines
    assert f"{LABELS[LF.value]}：代價 {expected:.3f}" in lines


def test_registry_without_stage_one_section_keeps_the_report(saved: SchemeResult, tmp_path: Path) -> None:
    # 現在的登記簿沒有這個用途的第一階段那一段：進度照實說沒數、說明從缺；
    # 可估的低頻讓排名照原文報「排名失敗」，報告不整份當掉（審查補）。
    store, path = make_store(tmp_path / "search")
    head, marker, rest = path.read_text(encoding="utf-8").partition("\n[[stage_one]]\n")
    _, purposes, tail = rest.partition("\n[[purpose]]\n")
    assert marker and purposes
    bare = tmp_path / "bare.toml"
    bare.write_text(head + purposes + tail, encoding="utf-8")
    # 排名層自己把缺段報成設定錯（ValueError），網頁與搜尋報告都照「排名失敗」接。
    with pytest.raises(ValueError, match="第一階段那一段"):
        compare_results((saved,), quality_targets=load_quality_targets(bare), run_date=RUN_DATE)
    store.status_path.write_text(SearchStatus().model_dump_json())
    unavailable = low_frequency(saved).model_copy(update={"state": EvaluationState.UNAVAILABLE,
        "payload": None, "raw_quantities": (), "category_cost": None,
        "reason_codes": (ReasonCode.SCORING_RANGE_GAP,)})
    quiet = saved.model_copy(update={"candidate": saved.candidate.model_copy(update={"evaluations": tuple(
        unavailable if item.category is LF else item for item in saved.candidate.evaluations)})})
    for original, answer in ((quiet, "低頻頻響的範圍與類權重說明從缺"), (saved, "排名失敗，這份報告不重排")):
        store.baseline_path.write_text(original.model_dump_json())
        text = render_text(build_report(store, quality_targets_path=bare, run_date=RUN_DATE))
        assert "低頻頻響沒數：現在的登記簿沒有這個用途的第一階段尺" in text
        assert answer in text


def test_stage_one_section_missing_a_source_keeps_the_full_report(saved: SchemeResult, tmp_path: Path) -> None:
    # 第一階段那一段在、但少計分範圍那一格：可估的低頻讓排名丟錯，整份報告照原文寫「排名失敗」，
    # 校準進度照實說判不出，不整份當掉（複查補）。
    store, path = make_store(tmp_path / "search")
    blocks = path.read_text(encoding="utf-8").split("\n[[")
    kept = [block for block in blocks if f'key = "{SCORED_RANGE_KEY}"' not in block]
    assert len(kept) == len(blocks) - 1
    short = tmp_path / "short.toml"
    short.write_text("\n[[".join(kept), encoding="utf-8")
    store.status_path.write_text(SearchStatus().model_dump_json())
    store.baseline_path.write_text(saved.model_dump_json())
    text = render_text(build_report(store, quality_targets_path=short, run_date=RUN_DATE))
    assert "排名失敗，這份報告不重排" in text
    assert "這次搜尋快照裡的登記簿跟現在的評分程式（評估器與排名層）對不上（" in text
    assert SCORED_RANGE_KEY in text


def test_stage_one_section_missing_a_source_says_unjudged(saved: SchemeResult) -> None:
    # 第一階段那一段在、但少一格來源：校準進度照實說判不出，不丟錯（審查補）。
    registry = load_quality_targets(TARGETS)
    purpose = registry.purpose(saved.scheme.purpose)
    section = registry.stage_one(purpose.name)
    short = section.model_copy(update={"setting": tuple(item for item in section.setting if item.key != SCORED_RANGE_KEY)})
    progress = calibration_progress(purpose, stage_one=short)
    assert progress.dependency_error is not None and SCORED_RANGE_KEY in progress.dependency_error
    assert not any(item.uses_all_calibrated for item in progress.categories)


def test_report_unavailable_low_frequency_keeps_chinese_reason(saved: SchemeResult) -> None:
    unavailable = low_frequency(saved).model_copy(update={"state": EvaluationState.UNAVAILABLE,
        "payload": None, "raw_quantities": (), "category_cost": None,
        "reason_codes": (ReasonCode.SCORING_RANGE_GAP,)})
    changed = saved.model_copy(update={"candidate": saved.candidate.model_copy(update={"evaluations": tuple(
        unavailable if item.category is LF else item for item in saved.candidate.evaluations)})})
    ranked = compare_results((changed,), quality_targets=load_quality_targets(TARGETS), run_date=RUN_DATE)
    lines = _candidate_quality(changed, ranked, "已讀回").category_lines
    assert "低頻頻響：未計代價" in lines
    assert "不可估：低頻計分範圍有缺段" in lines
