"""三種擺法從 CLI（命令列）補值、手定候選、接續一路到報告與搜尋頁。"""
import json
import math
from pathlib import Path
from uuid import UUID

import pytest

from aosr.config.paths import config_path
from aosr.reporting.scheme import Scheme
from aosr.search import cli, ledger
from aosr.search.store import SearchStore
from tests.engine._furniture_cases import relative_item
from tests.engine._search_furniture_cases import FurnitureFlowCompute, enqueue_hand_placements
from tests.engine._search_run_cases import Killed
from tests.engine._search_speaker_setup_cases import flow_project, store_for
from tests.engine.test_search_furniture_end_to_end import Flow, assert_same, finish, resume, start


def speaker_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount: str) -> Flow:
    monkeypatch.setattr("aosr.search.store.uuid.uuid4", lambda: UUID("55900000-0000-0000-0000-000000000005"))
    monkeypatch.setattr("aosr.search.modal_attach.uuid4", lambda: UUID("55900000-0000-0000-0000-000000000005"))
    project = flow_project(mount)
    changes: dict[str, object] = {}
    if mount == "stand":
        sofa = relative_item(furniture_id="rear-seat", placement={
            "forward_m": -1.0, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0})
        project = Scheme.model_validate(project.model_dump() | {"furniture": [
            *(item.model_dump() for item in project.furniture or ()), sofa]})
        changes["keep_out"] = [{"x": {"low": 4.6, "high": 6.0}, "y": {"low": 0.0, "high": 4.0},
                                "z": {"low": 0.0, "high": 2.0}}]
    source, registry = store_for(tmp_path / "input", project, layout_changes=changes, budget=3)
    settings = source.settings.model_dump(mode="json")
    settings["refine"] = {"budget": 3, "convergence_run": 1}
    del settings["layout"]["speaker_height_m"]
    del settings["layout"]["cabinet"]
    project_path, settings_path = tmp_path / "project.json", tmp_path / "settings.json"
    project_path.write_text(project.model_dump_json())
    settings_path.write_text(json.dumps(settings))
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: source.identity)
    args = ["start", "--project", str(project_path), "--settings", str(settings_path),
            "--root", str(tmp_path / "searches"), "--engine-commit", "fixture"]
    return Flow(source, registry, FurnitureFlowCompute(source), args, tmp_path.parent / "flow-modal-cache")


@pytest.mark.parametrize("mount", ["stand", "desk", "floor"])
def test_mount_cli_hand_candidates_resume_report_and_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mount: str) -> None:
    placements = {"stand": [(1.0, 0.85), (1.0, 2.5), (1.0, 1.5)],
                  "desk": [(1.0, 0.5), (1.0, 1.5), (1.0, 0.85)],
                  "floor": [(1.0, 0.5), (1.0, 0.85), (1.0, 1.5)]}[mount]
    enqueue_hand_placements(monkeypatch, placements=placements)
    whole = speaker_inputs(tmp_path / "whole", monkeypatch, mount)
    start(whole)
    # 左聲源朝主位 f=(l/r,.6/r)，箱體最小 x=1-.28*l/r-.105*.6/r。
    r85, r50, r15 = math.hypot(0.85, 0.6), math.hypot(0.5, 0.6), math.hypot(1.5, 0.6)
    overlap85 = 1.15 - (1.0 - 0.28 * 0.85 / r85 - 0.105 * 0.6 / r85)
    expected = {
        "stand": [("stand_space_occupied", overlap85, "腳架下方有家具"),
                  ("furniture_in_keep_out", 4.75 - 4.6, "家具進入禁區")],
        "desk": [("cabinet_off_table", 1.0 + 0.105 * 0.6 / r50 - 0.8, "箱體超出桌面"),
                 ("cabinet_off_table", 1.2 - (1.0 - 0.28 * 1.5 / r15 - 0.105 * 0.6 / r15), "箱體超出桌面")],
        "floor": [("cabinet_in_furniture", 0.8 - (1.0 - 0.28 * 0.5 / r50 - 0.105 * 0.6 / r50), "箱體穿入家具"),
                  ("cabinet_in_furniture", overlap85, "箱體穿入家具")],
    }[mount]
    recorded = {row.trial_number: row for row in ledger.read_for(whole.store).rows}
    for number, (reason, amount, _) in enumerate(expected):
        row = recorded[number]
        assert row.reason == reason and row.violation_m == pytest.approx(amount)
        assert row.outcome == "illegal" and row.result_file is None and row.score is None
        assert not whole.store.candidate_path(number).exists()
        assert all(job.trial_number != number for job in whole.compute.jobs)
    assert recorded[2].outcome == "scored"
    heights = {"stand": 1.2, "desk": 0.935, "floor": 0.8}
    assert whole.compute.jobs
    assert {point.z for job in whole.compute.jobs for point in job.scheme.speakers.values()} == {heights[mount]}
    stored = json.loads((whole.store.path / "settings.json").read_text())["settings"]["layout"]
    assert stored["speaker_height_m"] == heights[mount]
    assert whole.store.project.speaker_setup is not None
    assert stored["cabinet"] == whole.store.project.speaker_setup.cabinet.model_dump()
    observed = finish(whole, monkeypatch)
    kind, position, height, center = {"stand": ("書架喇叭", "腳架", "0.355", "0.205"),
        "desk": ("書架喇叭", "桌面", "0.355", "0.205"), "floor": ("落地喇叭", "地面", "1.105", "0.8")}[mount]
    text = f"喇叭：{kind}、放{position}（代表模型，非實際型號）；箱體 寬 0.21 × 深 0.28 × 高 {height} m，聲學中心離箱底 {center} m"
    assert text in observed[0].splitlines()
    assert "禁區：未限制" in observed[0].splitlines() or mount == "stand"
    assert text in next(block for block in observed[1].blocks if block.key == "stage").lines
    assert {block.key for block in observed[1].blocks} == {
        "stage", "counts", "timings", "updated", "search-best", "refine-best", "crossover", "reasons", "modal", "identity"}
    for _, _, label in expected:
        assert label in observed[0]
        assert any(label in line for block in observed[1].blocks for line in block.lines)
    resumed = speaker_inputs(tmp_path / "resumed", monkeypatch, mount)
    with pytest.raises(Killed):
        start(resumed, kill_at=1)
    resumed.compute = FurnitureFlowCompute(resumed.store)
    resume(resumed)
    assert_same(whole, resumed, observed, finish(resumed, monkeypatch))


@pytest.mark.parametrize("field,value", [("speaker_height_m", 1.25), ("cabinet", None)])
def test_cli_preserves_written_settings_and_rejects_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str], field: str, value: object) -> None:
    flow = speaker_inputs(tmp_path, monkeypatch, "stand")
    path = Path(flow.args[flow.args.index("--settings") + 1])
    doc = json.loads(path.read_text())
    doc["layout"][field] = value if field == "speaker_height_m" else {
        **flow.store.settings.layout.cabinet.model_dump(), "width_m": 0.22}
    path.write_text(json.dumps(doc))
    code = cli.main(flow.args, compute_factory=lambda store, capabilities, commit: FurnitureFlowCompute(store))
    assert code == 1
    assert not (tmp_path / "searches").exists()
    expected = "搜尋設定的喇叭高度 1.25 m 跟方案的 1.2 m 不同" if field == "speaker_height_m" else "搜尋設定的箱體 width_m 0.22 m 跟方案的 0.21 m 不同"
    assert expected in capsys.readouterr().err


def test_changed_mount_and_height_do_not_reuse_fem_or_modal_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    from collections.abc import Mapping
    from typing import cast
    from aosr.geometry.shoebox import Point
    from aosr.reporting import fem_slices
    from aosr.reporting.modal_lookup import placement_digest
    from tests.engine._directivity import DIRECTIVITY
    from tests.engine._furniture_cases import CAPABILITIES
    desk = flow_project("desk")
    doc = desk.model_dump()
    doc["speaker_setup"]["mount"] = "stand"
    for point in doc["speakers"].values():
        point["z"] = 1.2
    stand = Scheme.model_validate(doc)
    def energies(**values: object) -> dict[str, dict[tuple[str, str], tuple[float, ...]]]:
        # 只替換求解能量；驗方案、座標、候選紀錄與重用核對皆走產品入口。
        candidates = cast(Mapping[str, tuple[Mapping[str, Point], Mapping[str, Point]]], values["candidates"])
        indices = cast(tuple[int, ...], values["frequency_indices"])
        return {name: {(speaker, receiver): tuple(0.0 for _ in indices)
                       for speaker in sources for receiver in receivers}
                for name, (sources, receivers) in candidates.items()}
    monkeypatch.setattr(fem_slices, "solve_fem_energies_many", energies)
    shards = [fem_slices.solve_slice((scheme,), capabilities=CAPABILITIES, directivity=DIRECTIVITY,
        slices=1, slice_index=0, physics_identity="fixture") for scheme in (desk, stand)]
    assert shards[0].candidates[desk.scheme_id] != shards[1].candidates[stand.scheme_id]
    assert placement_digest(desk) != placement_digest(stand)
    with pytest.raises(ValueError, match="分片喇叭或座位座標與方案不同"):
        fem_slices.energies_from_shards(shards[:1], scheme=stand, capabilities=CAPABILITIES,
                                      directivity=DIRECTIVITY, physics_identity="fixture")


def test_blocked_baseline_unpinned_candidates_also_apply_new_constraints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.run import start_search
    from tests.engine._furniture_cases import cloud_item
    from tests.engine._search_run_cases import ENGINE, RUN_DATE
    project = flow_project("stand")
    cloud = cloud_item(width_m=2.0, depth_m=0.2, height_m=0.2,
        placement={"bottom_center_m": [2.1, 2.0, 1.1], "yaw_deg": 90})
    project = Scheme.model_validate(project.model_dump() | {"furniture": [
        *(item.model_dump() for item in project.furniture or ()), cloud]})
    store, registry = store_for(tmp_path, project, budget=2)
    enqueue_hand_placements(monkeypatch, placements=[(1.0, 0.85), (0.4, 1.5)])
    compute = FurnitureFlowCompute(store)
    status = start_search(store, compute=compute, probe=lambda: store.identity,
                         registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.baseline_outcome == "direct_path_blocked"
    rows = {row.trial_number: row for row in ledger.read_for(store).rows}
    assert rows[0].reason == "stand_space_occupied" and rows[0].outcome == "illegal"
    assert rows[1].outcome == "scored"
    assert status.comparison_trial == 1
    assert compute.search.calls == [1]
