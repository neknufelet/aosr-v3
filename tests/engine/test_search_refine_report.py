"""細算報告只讀狀態，不執行搜尋或細算。"""

import json
from pathlib import Path

import pytest

from aosr.search.report import build_report, render_text
from aosr.search.run import SearchStatus
from tests.engine._search_run_cases import RUN_DATE, make_store


def refinement_section(text: str) -> str:
    return next(section for section in text.strip().split("\n\n") if section.startswith("細算做完沒\n"))


def test_default_refine_report_preserves_previous_text(tmp_path: Path) -> None:
    store, registry = make_store(tmp_path)
    store.status_path.write_text(SearchStatus().model_dump_json(), encoding="utf-8")
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert report.refinement.state == "not_started"
    assert refinement_section(render_text(report)) == "細算做完沒\n細算未開始：還沒有任何候選用驗證軸細算"


@pytest.mark.parametrize("state,label,reason", [
    ("running", "進行中", None), ("stopped", "已停", "用完細算預算"),
    ("failed", "失敗", None), ("interrupted", "中斷", None),
])
def test_refine_report_reads_saved_state_and_message(tmp_path: Path, state: str, label: str, reason: str | None) -> None:
    store, registry = make_store(tmp_path)
    document = SearchStatus(state="budget_exhausted", message="搜尋已用完預算").model_dump(mode="json")
    document["refine"] = {"state": state, "message": "這是細算自己的訊息", "stop_reason": reason}
    store.status_path.write_text(json.dumps(document), encoding="utf-8")
    before = store.status_path.read_bytes()
    report = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    section = refinement_section(render_text(report))
    assert report.refinement.state == state
    assert report.refinement.message == "這是細算自己的訊息"
    assert f"狀態：{label}" in section
    assert "這是細算自己的訊息" in section
    if reason is not None:
        assert reason in section
    assert report.search.state == "budget_exhausted"
    assert report.search.message == "搜尋已用完預算"
    assert store.status_path.read_bytes() == before
