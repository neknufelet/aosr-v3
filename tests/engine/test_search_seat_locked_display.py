"""交接敏感度標示只依鎖定模式，不改數值或排名。"""
from pathlib import Path

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
