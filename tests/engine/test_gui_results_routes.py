"""結果清單、讀回拒收與同方案重算的端點考卷。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import GuiSettings, create_app
from aosr.gui.jobs import JobManager
from aosr.reporting.result import SchemeResult, save_result
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def _client(tmp_path: Path) -> TestClient:
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)),
                      base_url="http://localhost")


def _files(tmp_path: Path, result: SchemeResult) -> str:
    run_id = "b" * 32
    (tmp_path / "schemes" / "wall-1.json").write_text(result.scheme.model_dump_json())
    save_result(result, tmp_path / "results" / f"{run_id}.json")
    (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps({
        "run_id": run_id, "scheme_id": result.scheme.scheme_id,
        "status": "done", "started_at": 0, "pid": 0, "exit_code": 0,
        "result_path": str(tmp_path / "results" / f"{run_id}.json"),
        "stderr_path": str(tmp_path / "runs" / f"{run_id}.stderr"),
    }))
    return run_id


def test_results_list_and_detail_return_json(tmp_path: Path, result: SchemeResult) -> None:
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        listed = client.get("/api/results")
        assert listed.status_code == 200
        assert {item["run_id"] for item in listed.json()["results"]} == {run_id}
        assert listed.json()["results"][0]["scheme_id"] == result.scheme.scheme_id
        detail = client.get(f"/api/results/{run_id}")
        assert detail.status_code == 200
        assert detail.json()["scheme_id"] == result.scheme.scheme_id
        assert client.get(f"/results/{run_id}").status_code == 200


def test_rejected_result_returns_only_reason_and_rerun(tmp_path: Path,
                                                       result: SchemeResult,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        monkeypatch.setattr("aosr.gui.app.load_result", lambda *args, **kwargs:
                            (_ for _ in ()).throw(ValueError("舊評估器")))
        rejected = client.get(f"/api/results/{run_id}")
        assert rejected.status_code == 409
        assert rejected.json()["reason"] == "舊評估器"
        assert "frequency_responses" not in rejected.json()
        assert rejected.json()["rerun_url"].endswith("/rerun")


def test_rerun_uses_saved_scheme_and_jobs(tmp_path: Path, result: SchemeResult,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        called: list[Path] = []
        def fake_start(self: JobManager, path: Path) -> dict[str, object]:
            called.append(path)
            return {"run_id": "c" * 32, "status": "running"}
        monkeypatch.setattr("aosr.gui.jobs.JobManager.start", fake_start)
        response = client.post(f"/api/results/{run_id}/rerun", json={})
        assert response.status_code == 200
        assert response.json()["run_id"] == "c" * 32
        assert called == [tmp_path / "schemes" / "wall-1.json"]
