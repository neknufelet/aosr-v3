"""原方案被擋只列結構路徑 pairs 的直達問題；新顯示支援實際型號旗標。"""
from pathlib import Path

import pytest

from aosr.reporting.validation import SchemeProblem
from aosr.search import report_comparison
from aosr.search.labels import speaker_setup_text
from aosr.search.report import build_report, render_text
from tests.engine._search_blocked_cases import SavedFurnitureCompute, blocked_store
from tests.engine._search_run_cases import RUN_DATE, run
from tests.engine._search_speaker_setup_cases import flow_project


def test_blocked_original_report_filters_problem_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = blocked_store(tmp_path)
    run(store, registry, SavedFurnitureCompute(store))
    monkeypatch.setattr(report_comparison, "furniture_problems", lambda project: (
        SchemeProblem("speakers.left.z", "高度輸入問題不准列在直達段"),
        SchemeProblem("pairs.left.main", "直達被家具擋住"),
        # 訊息刻意含 pairs.；只認結構欄位。
        SchemeProblem("furniture", "pairs. 是訊息而不是路徑")))
    text = render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE))
    assert "直達被家具擋住" in text
    assert "高度輸入問題不准列在直達段" not in text
    assert "pairs. 是訊息而不是路徑" not in text


def test_actual_model_notice_has_exact_text() -> None:
    setup = flow_project("stand").speaker_setup
    assert setup is not None
    assert speaker_setup_text(setup.model_copy(update={"representative": False})) == (
        "喇叭：書架喇叭、放腳架（實際型號）；箱體 寬 0.21 × 深 0.28 × 高 0.355 m，聲學中心離箱底 0.205 m")
