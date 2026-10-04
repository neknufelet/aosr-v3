"""品質段的尺校準進度（#585 第 3 題）：答案從登記簿的 records 另算，不手打數字、不照抄被測的分類程式。"""

from pathlib import Path

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.search.report import build_report, render_text
from aosr.search.report_calibration import (
    CATEGORY_LABELS, CalibrationProgress, CategoryProgress, calibration_lines, calibration_progress,
)
from aosr.search.run import SearchStatus
from tests.engine._search_run_cases import RUN_DATE, make_store
from tests.engine.test_search_report import _sections

PURPOSE = "dedicated_two_channel_listening_room"


def test_totals_match_every_record_with_a_status() -> None:
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose(PURPOSE)
    progress = calibration_progress(purpose)
    assert progress.total == len(purpose.records)
    assert progress.calibrated == sum(record.status == "calibrated" for record in purpose.records)
    assert sum(item.total for item in progress.categories) == progress.total


def test_weight_items_count_under_their_table_category() -> None:
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose(PURPOSE)
    by_category = {item.category: item for item in calibration_progress(purpose).categories}
    for table in purpose.weight:
        category = table.key.split(".", 1)[0]
        assert category in by_category
    expected_ranking = (sum(1 for entry in purpose.qualification if entry.key.startswith("ranking."))
                        + sum(len(table.item) for table in purpose.weight if table.key.startswith("ranking.")))
    assert by_category["ranking"].total == expected_ranking


def test_lines_say_no_category_is_fully_calibrated() -> None:
    progress = CalibrationProgress(calibrated=1, total=5, categories=(
        CategoryProgress(category="reverberation", calibrated=1, total=3),
        CategoryProgress(category="ranking", calibrated=0, total=2),
    ))
    assert calibration_lines(progress) == (
        "尺的校準進度：共 5 條，已校準 1 條、未校準 4 條",
        "各類已校準／共：" + CATEGORY_LABELS["reverberation"] + " 1／3、" + CATEGORY_LABELS["ranking"] + " 0／2",
        "沒有任何一類的尺全部校準完，所以還沒有「已校準項目通過幾項」可以報（等 #358）",
    )


def test_lines_name_fully_calibrated_categories() -> None:
    progress = CalibrationProgress(calibrated=4, total=6, categories=(
        CategoryProgress(category="reverberation", calibrated=3, total=3),
        CategoryProgress(category="made_up_category", calibrated=1, total=3),
    ))
    lines = calibration_lines(progress)
    assert lines[1].endswith("made_up_category 1／3")
    assert lines[2] == "全部校準完的類別：" + CATEGORY_LABELS["reverberation"] + "；這幾類過不過還沒接進報告（等 #358）"


def test_quality_section_lists_progress_and_keeps_verdict_undetermined(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path)
    store.status_path.write_text(SearchStatus().model_dump_json())
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    purpose = load_quality_targets(registry).purpose(store.settings.purpose)
    section = _sections(render_text(report))["品質合不合格"].splitlines()
    assert report.quality.verdict == "未判定合格"
    # _sections 去掉段標題：第 0 行是判定、接著原因與依據，校準進度三行緊跟在後。
    assert section[0] == "判定：未判定合格"
    assert section[3:6] == list(calibration_lines(calibration_progress(purpose)))
