"""交接敏感度標示只依鎖定模式，不改數值或排名。"""
from pathlib import Path

import pytest

from aosr.search.crossover_record import CrossoverSummary, VariantRecord, summary_lines


def test_crossover_seat_distance_marks_locked_primary(tmp_path: Path) -> None:
    variant = VariantRecord(key="test", label="測試", basis="手算", speaker_distance_cm={"left": 2.0, "right": 3.0},
        primary_distance_cm=0.0)
    summary = CrossoverSummary(variants=(variant,))
    unlocked = summary_lines(summary)
    locked = summary_lines(summary, seat_locked=True)
    expected = "與正式第一名的距離：左聲道喇叭相距 2.0 公分；右聲道喇叭相距 3.0 公分；主位相距 0.0 公分"
    assert expected in unlocked
    assert expected + "（座位鎖定，主位不動）" in locked
    assert tmp_path.is_dir()


def test_model_listening_range_reason_has_unambiguous_chinese(tmp_path: Path) -> None:
    from aosr.search.labels import COUNT_REASONS

    assert COUNT_REASONS["listening_distance_out_of_range"] == "型號適用聆聽距離超出範圍"
    assert tmp_path.is_dir()


@pytest.mark.parametrize("locked", [True, False])
def test_crossover_report_passes_seat_lock_from_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                        locked: bool) -> None:
    # 報告與搜尋頁都經 crossover_report；鎖定與否要從搜尋設定接過去，不只靠 summary_lines 的參數。
    from aosr.search import report_crossover
    from aosr.search.run import SearchStatus
    from tests.engine._seat_locked_cases import locked_store
    from tests.engine._search_run_cases import make_store

    store, _ = locked_store(tmp_path) if locked else make_store(tmp_path)
    variant = VariantRecord(key="test", label="測試", basis="手算", speaker_distance_cm={"left": 2.0, "right": 3.0},
        primary_distance_cm=0.0)
    monkeypatch.setattr(report_crossover, "read_summary", lambda path: CrossoverSummary(variants=(variant,)))
    lines = report_crossover.crossover_report(store, SearchStatus()).lines
    expected = "與正式第一名的距離：左聲道喇叭相距 2.0 公分；右聲道喇叭相距 3.0 公分；主位相距 0.0 公分"
    assert (expected + "（座位鎖定，主位不動）" if locked else expected) in lines
