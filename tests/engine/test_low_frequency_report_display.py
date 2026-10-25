"""低頻搜尋報告：分類代價與第一階段尺的校準狀態，不把現在的尺稱為搜尋快照。"""
import json
from pathlib import Path

from aosr.config.quality_targets import QualityPurpose, load_quality_targets
from aosr.reporting.display import LABELS
from aosr.reporting.result import SchemeResult
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import EvaluationState, QualityCategory
from aosr.scoring.contract_base import ReasonCode
from aosr.search.report import _candidate_quality, build_report, render_text
from aosr.search.report_calibration import calibration_lines, calibration_progress
from aosr.search.run import SearchStatus
from tests.engine._search_run_cases import RUN_DATE, make_store
from tests.engine.test_low_frequency_response_display import TARGETS, assert_details, low_frequency
from tests.engine.test_low_frequency_response_reload import saved as saved
from aosr.reporting.compare import compare_results

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
    assert "低頻頻響採本次排名讀取的第一階段尺；搜尋快照未保存該段" in calibration_lines(progress)[0]
    for key in ("", *(key for key, _ in sources)):
        promoted = calibration_progress(purpose, stage_one=_calibrated(section, key))
        item = next(item for item in promoted.categories if item.category == LF.value)
        assert item.uses_all_calibrated == (key == "")
        assert ("低頻頻響" in calibration_lines(promoted)[2]) == (key == "")


def test_report_prints_low_frequency_cost_and_details(saved: SchemeResult, tmp_path: Path) -> None:
    store, path = make_store(tmp_path)
    registry = load_quality_targets(path)
    store.status_path.write_text(SearchStatus(best_trial=0).model_dump_json())
    ranked = compare_results((saved,), quality_targets=registry, run_date=RUN_DATE)
    line = next(item for item in ranked.rankable[0].categories if item.identity.category is LF)
    quality = _candidate_quality(saved, ranked, "第一名結果已讀回")
    assert f"{LABELS[LF.value]}：代價 {line.category_cost:.3f}" in quality.category_lines
    assert_details("\n".join(quality.category_lines), low_frequency(saved))
    store.baseline_path.write_text(saved.model_dump_json())
    renamed = saved.model_dump_json().replace(json.dumps(saved.scheme.scheme_id), json.dumps("other"))
    SchemeResult.model_validate_json(renamed)
    store.candidate_path(0).write_text(renamed)
    text = render_text(build_report(store, quality_targets_path=path, run_date=RUN_DATE))
    assert f"第一名低頻頻響：代價 {line.category_cost:.3f}" in text
    assert_details(text, low_frequency(saved))
    assert "低頻頻響採本次排名讀取的第一階段尺" in text


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
