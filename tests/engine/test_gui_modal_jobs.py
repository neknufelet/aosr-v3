"""獨立模態工作：只查快取、按鈕補算、同房接回、停止及結果排名不變。"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine._modal_cases import runner, sample
from tests.engine.test_gui_browser import RUN_ID, _save
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def poll(client: TestClient, job: dict[str, object], run_id: str = RUN_ID) -> dict[str, object]:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        body = client.get(f"/api/results/{run_id}/modal", params={"job_id": job["run_id"]}).json()
        if not body.get("job") or body["job"]["status"] != "running":
            return cast(dict[str, object], body)
        time.sleep(0.02)
    pytest.fail("模態替身沒有結束")


def test_auto_cache_lookup_then_explicit_calculation(tmp_path: Path, result: SchemeResult) -> None:
    _save(tmp_path, result)
    missing = ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)
    command = runner(tmp_path, missing)
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path, modal_runner=command)),
                    base_url="http://localhost") as client:
        before = client.get(f"/api/results/{RUN_ID}").json()
        started = client.get(f"/api/results/{RUN_ID}/modal").json()
        shown = poll(client, started["job"])
        shown_view = cast(dict[str, object], shown["view"])
        assert shown_view["state_text"] == "未計算" and shown_view["can_calculate"]
        args = json.loads((tmp_path / "modal-args.json").read_text())
        assert "--cache-only" in args and args[args.index("--cache-dir") + 1] == str(tmp_path / "modal-cache")
        started = client.post(f"/api/results/{RUN_ID}/modal", json={}).json()
        poll(client, started["job"])
        assert "--cache-only" not in json.loads((tmp_path / "modal-args.json").read_text())
        assert client.get(f"/api/results/{RUN_ID}").json() == before
        assert client.get("/api/runs").json()["running"] == []


def test_matching_placement_is_immediate_without_job(tmp_path: Path, result: SchemeResult) -> None:
    from aosr.reporting.modal_lookup import placement_path, save_diagnosis
    _save(tmp_path, result)
    diagnosis = sample(result.scheme)[0]
    path = placement_path(result.scheme, cache_dir=tmp_path / "modal-cache")
    assert path is not None
    save_diagnosis(diagnosis, path)
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path,
                                          modal_runner=("must-not-run",))), base_url="http://localhost") as client:
        body = client.get(f"/api/results/{RUN_ID}/modal").json()
        assert body["view"]["state_text"] == "已診斷不計分" and body.get("job") is None


def test_same_room_attaches_without_freezing_name_or_hiding_results(tmp_path: Path, result: SchemeResult) -> None:
    _save(tmp_path, result)
    command = runner(tmp_path, sample(result.scheme)[0], wait=True)
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path,
                                          modal_runner=command)), base_url="http://localhost") as client:
        endpoint = f"/api/results/{RUN_ID}/modal"
        first = client.get(endpoint).json()
        second = client.post(endpoint, json={}).json()
        assert first["job"]["run_id"] == second["job"]["run_id"]
        assert client.get(f"/api/results/{RUN_ID}").status_code == 200
        assert client.get("/api/runs").json()["running"] == []
        # 去掉原結果後，只有模態工作存在；它不能凍結名字。
        (tmp_path / "results" / f"{RUN_ID}.json").unlink()
        value = result.scheme.model_dump(mode="json")
        value["scene"]["impedance_pa_s_per_m_by_wall"]["floor"] *= 1.1
        assert client.put(f"/api/schemes/{result.scheme.scheme_id}", json=value).status_code == 200
        stopped = client.post(f"/api/modal-jobs/{first['job']['run_id']}/stop", json={})
        assert stopped.status_code == 200 and stopped.json()["status"] == "stopped"


def test_auto_room_hit_publishes_shared_placement(tmp_path: Path, result: SchemeResult) -> None:
    from aosr.reporting.modal_lookup import read_placement
    _save(tmp_path, result)
    command = runner(tmp_path, sample(result.scheme)[0])
    with TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path,
                                          modal_runner=command)), base_url="http://localhost") as client:
        started = client.get(f"/api/results/{RUN_ID}/modal").json()
        shown = poll(client, started["job"])
        assert cast(dict[str, object], shown["view"])["state_text"] == "已診斷不計分"
        assert "--cache-only" in json.loads((tmp_path / "modal-args.json").read_text())
        assert read_placement(result.scheme, cache_dir=tmp_path / "modal-cache") is not None


def test_diagnosis_keeps_ranking_and_cost_bits_unchanged(tmp_path: Path, result: SchemeResult) -> None:
    from aosr.config.paths import config_path
    from aosr.config.quality_targets import load_quality_targets
    from aosr.reporting.compare import compare_results
    from aosr.reporting.modal_lookup import placement_path, save_diagnosis
    from aosr.scoring.contract import QualityCategory
    other = result.model_copy(update={"scheme": result.scheme.model_copy(update={"scheme_id": "second"}),
        "candidate": result.candidate.model_copy(update={"candidate_id": "second"})})
    results = (result, other)
    targets = load_quality_targets(config_path("quality_targets.toml"))
    before = compare_results(results, quality_targets=targets, run_date=result.run_date)
    frozen = tuple(value.model_dump_json() for value in results)
    for value in results:
        path = placement_path(value.scheme, cache_dir=tmp_path)
        assert path is not None
        save_diagnosis(sample(value.scheme)[0], path)
    after = compare_results(results, quality_targets=targets, run_date=result.run_date)
    assert before.rankable and [(r.candidate_id, r.total_cost.hex()) for r in after.rankable] == [
        (r.candidate_id, r.total_cost.hex()) for r in before.rankable]
    assert after.model_dump_json() == before.model_dump_json()
    assert tuple(value.model_dump_json() for value in results) == frozen
    assert all(e.category is not QualityCategory.LOW_FREQUENCY_DECAY for value in results for e in value.candidate.evaluations)


def test_other_placement_attaches_then_gets_own_lookup(tmp_path: Path, result: SchemeResult) -> None:
    from aosr.geometry.shoebox import Point
    from aosr.gui.modal_jobs import ModalJobs
    from aosr.config.paths import config_path
    from aosr.reporting.modal_lookup import read_placement
    original = result.scheme
    name = next(iter(original.speakers))
    point = original.speakers[name]
    moved = original.model_copy(update={"speakers": {**original.speakers, name: Point(point.x + 0.01, point.y, point.z)}})
    command = runner(tmp_path, sample(original)[0], wait=True)
    jobs = ModalJobs(tmp_path, command, "a" * 40, config_path("capabilities.toml"))
    first = cast(dict[str, object], jobs.lookup(original, result_id=RUN_ID)["job"])
    try:
        attached = cast(dict[str, object], jobs.lookup(moved, result_id="b" * 32)["job"])
        assert attached["run_id"] == first["run_id"]
        assert original.scheme_id in str(attached["display_text"]) and str(attached["scheme_id"]) not in str(attached["display_text"])
        (tmp_path / "modal-release").touch()
        deadline = time.monotonic() + 20
        while jobs.manager.get(str(first["run_id"]))["status"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        jobs.lookup(original, result_id=RUN_ID, job_id=str(first["run_id"]))
        runner(tmp_path, sample(moved)[0])
        next_job = cast(dict[str, object], jobs.lookup(moved, result_id="b" * 32, job_id=str(first["run_id"]))["job"])
        assert next_job["run_id"] != first["run_id"] and next_job["cache_only"]
        while jobs.manager.get(str(next_job["run_id"]))["status"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.02)
        jobs.lookup(moved, result_id="b" * 32, job_id=str(next_job["run_id"]))
        assert read_placement(moved, cache_dir=tmp_path / "modal-cache") is not None
    finally:
        for active in jobs.manager.processes:
            jobs.manager.stop(active)
