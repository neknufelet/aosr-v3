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
    # 第 5 支起細算段最後多一行外圈結論（設計紙第五節第 5 條）；前兩行逐字照舊。
    assert refinement_section(render_text(report)) == "細算做完沒\n細算未開始：還沒有任何候選用驗證軸細算\n外圈結論：未判定"


@pytest.mark.parametrize("state,label,reason,shown", [
    ("running", "進行中", None, None), ("stopped", "已停", "refine_budget", "用完細算上限"),
    ("stopped", "已停", "stable", "細算第一名連續一段沒被換掉"),
    ("stopped", "已停", "candidates_exhausted", "沒有候選可以再細算"),
    ("stopped", "已停", "user_stopped", "使用者停止"),
    ("failed", "失敗", None, None), ("interrupted", "中斷", None, None),
])
def test_refine_report_reads_saved_state_and_message(
    tmp_path: Path, state: str, label: str, reason: str | None, shown: str | None,
) -> None:
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
    if shown is not None and reason is not None:
        assert f"狀態：{label}（{shown}）" in section and reason not in section
    assert report.search.state == "budget_exhausted"
    assert report.search.message == "搜尋已用完預算"
    assert store.status_path.read_bytes() == before


def test_stop_section_carries_no_refine_state(tmp_path: Path) -> None:
    """三件事分開：搜尋停止那一段的資料不帶細算狀態，細算改成進行中時停止段逐字不變。"""
    from aosr.search.report import SearchStopReport

    assert "refine" not in SearchStopReport.model_fields
    store, registry = make_store(tmp_path)
    document = SearchStatus(state="converged", message="達到停止條件").model_dump(mode="json")
    store.status_path.write_text(json.dumps(document), encoding="utf-8")
    before = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    document["refine"] = {"state": "running", "message": "細算進行中"}
    store.status_path.write_text(json.dumps(document), encoding="utf-8")
    after = build_report(store, quality_targets_path=registry, run_date=RUN_DATE)
    assert after.search == before.search
    stop = next(section for section in render_text(after).split("\n\n") if section.startswith("搜尋停了沒\n"))
    assert "細算" not in stop


def test_every_stop_reason_code_has_chinese(tmp_path: Path) -> None:
    """每個停止原因代碼都有中文、報告不印英文代碼；以後加代碼忘了補中文，這題會紅。"""
    from typing import get_args

    from aosr.search.run import RefineStopReason

    store, registry = make_store(tmp_path)
    for code in get_args(RefineStopReason):
        document = SearchStatus(state="converged", message="達到停止條件").model_dump(mode="json")
        document["refine"] = {"state": "stopped", "stop_reason": code, "message": "細算已停"}
        store.status_path.write_text(json.dumps(document), encoding="utf-8")
        section = refinement_section(render_text(build_report(store, quality_targets_path=registry, run_date=RUN_DATE)))
        assert section.startswith("細算做完沒\n狀態：已停（") and code not in section

