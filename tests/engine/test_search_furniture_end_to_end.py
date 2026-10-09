"""命令列 start 起跑：連 create 接線一起驗，再接續、細算、外圈、報告與搜尋頁。

答案取不中斷整段實跑。只放過 status.search_seconds、status.refine.seconds，
報告的「各輪搜尋花的時間」「各輪細算花的時間」「搜尋＋細算合計」分鐘數，
搜尋頁的 timings、updated 兩個區塊、fetched_text（讀取時刻），以及 best_versions 每一格 version 裡結果檔修改時間那一段；
搜尋頁其餘欄位（代號、名稱、階段、其他區塊、預設最佳、最佳版本的其他格與 version 其餘各段）逐格比。帳本秒數仍逐位比。
"""
from __future__ import annotations

import logging
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

from aosr.config.paths import config_path
from aosr.gui.search_view import SearchView, build_search_view
FURNITURE_MODEL_NOTE = "家具模型：近似"  # 決策紙第 13 條原文。
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.scheme import Scheme, absolute_furniture
from aosr.search import cli, crossover_record, layout, ledger, modal_record
from aosr.search.refine import RefineLedger
from aosr.search.report import build_report, render_text
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore
from tests.engine._search_blocked_cases import blocked_store
from tests.engine._search_furniture_cases import FurnitureFlowCompute, enqueue_hand_placements, furnished_store
from tests.engine._modal_cases import runner
from tests.engine._search_run_cases import Killed, RUN_DATE
from tests.engine._stability_attach_cases import ShiftCompute
from tests.engine._seat_locked_cases import LOCKED, SEAT_LINE


@dataclass
class Flow:
    store: SearchStore
    registry: Path
    compute: FurnitureFlowCompute
    args: list[str]
    cache_dir: Path


def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str) -> Flow:
    # 同一代號只用在各自獨立的 tmp_path，讓帳本、候選包與報告全文能直接比。
    monkeypatch.setattr("aosr.search.store.uuid.uuid4", lambda: UUID("55900000-0000-0000-0000-000000000003"))
    monkeypatch.setattr("aosr.search.modal_attach.uuid4", lambda: UUID("55900000-0000-0000-0000-000000000003"))
    source, registry = (blocked_store(tmp_path / "input") if scenario == "A"
                        else furnished_store(tmp_path / "input"))
    settings = source.settings.model_dump(mode="json")
    settings["refine"] = {"budget": 50, "convergence_run": 50}
    if scenario == "C":
        settings["layout"]["front_wall"] = "y0"
    if scenario == "D":
        settings["layout"].update(LOCKED)
        settings["budget"] = 17
    project_path, settings_path = tmp_path / "project.json", tmp_path / "settings.json"
    project_path.write_text(source.project.model_dump_json())
    settings_path.write_text(json.dumps(settings))
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: source.identity)
    args = ["start", "--project", str(project_path), "--settings", str(settings_path),
            "--root", str(tmp_path / "searches"), "--engine-commit", "fixture"]
    # 計算替身必須用命令列 create 交來的新 store，不能沿用 input 那個資料夾。
    return Flow(source, registry, FurnitureFlowCompute(source), args, tmp_path.parent / "flow-modal-cache")


def start(flow: Flow, *, kill_at: int | None = None) -> None:
    def factory(store: SearchStore, capabilities: Path, commit: str) -> FurnitureFlowCompute:
        flow.store, flow.compute = store, FurnitureFlowCompute(store)
        if kill_at is not None:
            # 在第 k 個候選計算前被砍：最後一個也確實留下未完成的候選。
            flow.compute.search.fail_after = kill_at - 1
            flow.compute.search.kill = True
        return flow.compute
    code = cli.main(flow.args, compute_factory=factory)
    assert code == 0


def finish(flow: Flow, monkeypatch: pytest.MonkeyPatch) -> tuple[str, SearchView]:
    code = cli.main(["refine", str(flow.store.path), "--engine-commit", "fixture"],
                    compute_factory=lambda store, capabilities, commit: flow.compute)
    assert code == 0
    # 照 _search_outer_cases.invoke 注入計算與低頻 runner；各比對回合共用同一個快取路徑。
    code = cli.main(["auto", str(flow.store.path), "--engine-commit", "fixture",
                     "--modal-cache-dir", str(flow.cache_dir)],
                    compute_factory=lambda store, capabilities, commit: flow.compute,
                    stability_compute_factory=lambda root: ShiftCompute(flow.store),
                    modal_runner=runner(flow.store.path.parent / "modal-runner",
                                        ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))
    assert code == 0
    report = build_report(flow.store, quality_targets_path=flow.registry, run_date=RUN_DATE)
    view = build_search_view(flow.store.path, server_physics=flow.store.identity.physics_identity,
                             server_program=flow.store.identity.program_fingerprint)
    assert all(FURNITURE_MODEL_NOTE in block.lines for block in view.blocks
               if block.key in ("search-best", "refine-best"))
    return render_text(report), view


def status_without_seconds(store: SearchStore) -> dict[str, object]:
    return SearchStatus.model_validate_json(store.status_path.read_bytes()).model_dump(
        exclude={"search_seconds": True, "refine": {"seconds"}})


def resume(flow: Flow) -> None:
    code = cli.main(["resume", str(flow.store.path), "--engine-commit", "fixture"],
                    compute_factory=lambda store, capabilities, commit: flow.compute)
    assert code == 0


def report_without_minutes(text: str) -> str:
    return "\n".join(re.sub(r"\d+\.\d+ 分", "<時間> 分", line)
                     if line.startswith(("各輪搜尋花的時間：", "各輪細算花的時間：", "搜尋＋細算合計 ")) else line
                     for line in text.splitlines())


def view_without_times(view: SearchView) -> dict[str, object]:
    document = view.model_dump(exclude={"fetched_text": True})
    # version 是 which:candidate:filename:mtime_ns:size:purpose（search_view._best_version），只遮修改時間那一段。
    for best in document["best_versions"].values():
        if best["version"]:
            parts = best["version"].split(":")
            best["version"] = ":".join((*parts[:3], "<mtime>", *parts[4:]))
    document["blocks"] = [block for block in document["blocks"] if block["key"] not in ("timings", "updated")]
    return document


def assert_same(whole: Flow, resumed: Flow, expected: tuple[str, SearchView], actual: tuple[str, SearchView]) -> None:
    assert ledger.Ledger.read(whole.store.ledger_path) == ledger.Ledger.read(resumed.store.ledger_path)
    assert RefineLedger.read(whole.store.refine_ledger_path) == RefineLedger.read(resumed.store.refine_ledger_path)
    assert status_without_seconds(whole.store) == status_without_seconds(resumed.store)
    assert report_without_minutes(expected[0]) == report_without_minutes(actual[0])
    assert view_without_times(expected[1]) == view_without_times(actual[1])
    assert crossover_record.read_summary(whole.store.path) == crossover_record.read_summary(resumed.store.path)
    one, other = modal_record.read_summary(whole.store.path), modal_record.read_summary(resumed.store.path)
    assert one is not None and other is not None
    assert one == other


def assert_no_original_score(flow: Flow, observed: tuple[str, SearchView]) -> None:
    status = SearchStatus.model_validate_json(flow.store.status_path.read_bytes())
    assert status.baseline_outcome == "direct_path_blocked"
    assert status.baseline_reason_codes == ("direct_path_blocked",)
    assert status.refine.best != "baseline"
    assert not flow.store.baseline_path.exists() and not flow.store.refine_result_path(None).exists()
    assert all(job.trial_number is not None for job in flow.compute.jobs)
    assert all(row.trial_number is not None for row in RefineLedger.read(flow.store.refine_ledger_path)[1])
    crossover = crossover_record.read_summary(flow.store.path)
    modal = modal_record.read_summary(flow.store.path)
    assert crossover is not None and modal is not None
    assert all(row.trial_number is not None for row in crossover.rows)
    assert all(row.trial_number is not None for variant in crossover.variants for row in variant.ranking)
    baseline = next(role for role in modal.roles if role.role == "baseline")
    assert baseline.state == "skipped" and baseline.reason_text == "原方案不符合擺位要求"
    assert baseline.diagnosis_file is not None
    diagnosis = json.loads((flow.store.path / "modal-diagnosis" / baseline.diagnosis_file).read_bytes())
    assert diagnosis["state"] == "not_computed"
    assert diagnosis["reason_text"] == "原方案不符合擺位要求"
    report = build_report(flow.store, quality_targets_path=flow.registry, run_date=RUN_DATE)
    assert report.placement.original is None and report.quality.original.zone is None
    for line in (*observed[0].splitlines(), *(line for block in observed[1].blocks for line in block.lines)):
        assert re.search(r"原方案[^\n]*(?:分數|代價)[:：]\s*[+-]?\d", line) is None
    assert "原方案不符合擺位要求" in observed[0]


def assert_attachments(flow: Flow, *, blocked: bool) -> None:
    crossover = crossover_record.read_summary(flow.store.path)
    modal = modal_record.read_summary(flow.store.path)
    assert crossover is not None and modal is not None
    # 替身只存候選包；真正附件讀完整結果的降級答案由整段不中斷實跑取得。
    assert crossover.state == "done" and crossover.completed and crossover.verdict == "unverified"
    assert crossover.reason_text
    assert modal.completed
    for role in modal.roles:
        if role.role == "baseline" and blocked:
            assert role.state == "skipped"
        else:
            assert role.state == "failed" and "結果缺少 scope（範圍標記）" in role.reason_text


def test_blocked_baseline_whole_and_first_middle_last_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = time.perf_counter()
    whole = inputs(tmp_path / "whole", monkeypatch, "A")
    start(whole)
    computations = len(whole.compute.search.calls)
    checkpoints = {"第一個": 1, "中間": (computations + 1) // 2, "最後": computations}
    expected = finish(whole, monkeypatch)
    assert_no_original_score(whole, expected)
    assert_attachments(whole, blocked=True)
    for label, k in checkpoints.items():
        resumed = inputs(tmp_path / label.encode().hex(), monkeypatch, "A")
        with pytest.raises(Killed):
            start(resumed, kill_at=k)
        assert SearchStatus.model_validate_json(resumed.store.status_path.read_bytes()).state == "running"
        resumed.compute = FurnitureFlowCompute(resumed.store)
        resume(resumed)
        actual = finish(resumed, monkeypatch)
        assert_same(whole, resumed, expected, actual)
        assert_no_original_score(resumed, actual)
    logging.getLogger(__name__).warning("情境 A：計算 %s，k=%s；%.6f 秒", computations, checkpoints, time.perf_counter() - started)


def test_unblocked_baseline_filters_candidates_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_hand_placements(monkeypatch)
    whole = inputs(tmp_path / "whole", monkeypatch, "B")
    start(whole)
    expected = finish(whole, monkeypatch)
    recorded = {row.trial_number: row for row in ledger.Ledger.read(whole.store.ledger_path)[1]}
    # 周圍點的穿盒長度 sqrt(0.5²+0.5²)×0.5；後座椅最大 x=6.25，越界 0.25 m。
    for number, reason, violation in ((0, "direct_path_blocked", math.sqrt(0.5) / 2),
                                      (1, "furniture_placement_invalid", 0.25)):
        row = recorded[number]
        assert row.reason == reason
        assert row.violation_m == pytest.approx(violation)
        assert row.outcome == "illegal" and row.score is None and row.result_file is None and row.seconds == 0.0
        assert not whole.store.candidate_path(number).exists()
        assert all(job.trial_number != number for job in whole.compute.jobs)
        assert all(item.trial_number != number for item in RefineLedger.read(whole.store.refine_ledger_path)[1])
        text = {0: "不符合擺位要求：直達路徑被家具擋住", 1: "家具擺放不合法"}[number]
        assert text in expected[0]
        assert any(text in line for block in expected[1].blocks for line in block.lines)
    assert recorded[2].outcome == "scored"
    assert SearchStatus.model_validate_json(whole.store.status_path.read_bytes()).baseline_outcome == "scored"
    assert_attachments(whole, blocked=False)
    resumed = inputs(tmp_path / "resumed", monkeypatch, "B")
    with pytest.raises(Killed):
        start(resumed, kill_at=1)
    resumed.compute = FurnitureFlowCompute(resumed.store)
    resume(resumed)
    assert_same(whole, resumed, expected, finish(resumed, monkeypatch))


def test_changed_front_wall_turns_listener_furniture_through_public_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    flow = inputs(tmp_path, monkeypatch, "C")
    enqueue_hand_placements(monkeypatch)
    start(flow)
    observed = finish(flow, monkeypatch)
    recorded = {row.trial_number: row for row in ledger.Ledger.read(flow.store.ledger_path)[1]}
    # 換牆後喇叭、座位、周圍點與跟著主位走的家具一起轉 90°，相對幾何跟 B 一樣：
    # 試算 0 照樣被桌板擋、穿盒長度同 B 的手算；試算 2 照樣合法（家具不跟著轉的話試算 2 會被擋）。
    assert recorded[0].reason == "direct_path_blocked" and recorded[0].violation_m == pytest.approx(math.sqrt(0.5) / 2)
    assert recorded[2].outcome == "scored"
    jobs = [job for job in flow.compute.jobs if job.trial_number is not None and job.result_path.parent == flow.store.candidate_path(0).parent]
    assert jobs
    for job in jobs:
        x, y, _ = job.scheme.receiver_set.primary.position_m
        furniture = absolute_furniture(job.scheme)
        assert furniture is not None
        items = {item.furniture_id: item for item in furniture}
        # 原案面向 -x；換 y0 面向 -y，左方為 +x。
        # 桌子 forward=0.25、left=0.4，底高=1：中心 (x+0.4,y-0.25,1)，角度 180°。
        assert items["desk"].bottom_center_m == pytest.approx((x + 0.4, y - 0.25, 1.0))
        assert items["desk"].yaw_deg == 180.0
        assert job.scheme.furniture == flow.store.project.furniture
    assert "家具模型：近似" in observed[0]


def test_locked_furniture_cli_report_view_and_first_middle_last_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    whole = inputs(tmp_path / "whole", monkeypatch, "D")
    start(whole)
    expected = finish(whole, monkeypatch)
    assert SEAT_LINE in expected[0].splitlines()
    stage = next(block for block in expected[1].blocks if block.key == "stage")
    best = next(block for block in expected[1].blocks if block.key == "search-best")
    assert stage.lines[-1] == SEAT_LINE
    assert any("聆聽距離（由座位推出）：" in line for line in best.lines)
    assert {block.key for block in expected[1].blocks} == {
        "stage", "counts", "timings", "updated", "search-best", "refine-best", "crossover", "stability", "reasons", "modal", "identity"}
    for job in whole.compute.jobs:
        assert job.scheme.receiver_set == whole.store.project.receiver_set
        assert absolute_furniture(job.scheme) == absolute_furniture(whole.store.project)
    computations = sum(number is not None for number in whole.compute.search.calls)
    for label, k in {"first": 1, "middle": (computations + 1) // 2, "last": computations}.items():
        resumed = inputs(tmp_path / label, monkeypatch, "D")
        with pytest.raises(Killed):
            start(resumed, kill_at=k)
        resumed.compute = FurnitureFlowCompute(resumed.store)
        resume(resumed)
        actual = finish(resumed, monkeypatch)
        assert_same(whole, resumed, expected, actual)
        assert SEAT_LINE in actual[0].splitlines()
        assert all(job.scheme.receiver_set == resumed.store.project.receiver_set for job in resumed.compute.jobs)
