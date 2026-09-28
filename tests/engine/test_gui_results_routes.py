"""結果清單、讀回拒收與同方案重算的端點考卷。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import RUN_ID, GuiHandlers, GuiSettings, _result_paths, create_app
from aosr.gui.jobs import JobManager
from aosr.reporting.result import SchemeResult, save_result
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def _client(tmp_path: Path, *, raise_server_exceptions: bool = True) -> TestClient:
    # 500 那一類要看伺服器自己的處理器回什麼，就得叫 TestClient 別把例外直接拋回考卷。
    return TestClient(create_app(GuiSettings(engine_commit="a" * 40, data_dir=tmp_path)),
                      base_url="http://localhost", raise_server_exceptions=raise_server_exceptions)


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


@pytest.mark.parametrize("saved_change", ["changed", "deleted"])
def test_rerun_uses_result_scheme_even_if_saved_scheme_changes(tmp_path: Path, result: SchemeResult,
                                          monkeypatch: pytest.MonkeyPatch,
                                          saved_change: str) -> None:
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        saved = tmp_path / "schemes" / "wall-1.json"
        if saved_change == "deleted":
            saved.unlink()
        else:
            document = json.loads(saved.read_text())
            document["speakers"]["left"]["x"] += 0.1
            saved.write_text(json.dumps(document))
        called: list[Path] = []
        def fake_start(self: JobManager, path: Path, run_id: str) -> dict[str, object]:
            called.append(path)
            assert json.loads(path.read_text()) == result.scheme.model_dump(mode="json")
            assert path.parent.name == run_id
            return {"run_id": run_id, "status": "running"}
        monkeypatch.setattr("aosr.gui.jobs.JobManager._start", fake_start)
        response = client.post(f"/api/results/{run_id}/rerun", json={})
        assert response.status_code == 200
        assert RUN_ID.fullmatch(response.json()["run_id"])
        assert called == [tmp_path / "runs" / response.json()["run_id"] / "scheme.json"]


def test_bad_run_id_and_result_sorting(tmp_path: Path, result: SchemeResult) -> None:
    import os

    with _client(tmp_path) as client:
        first = _files(tmp_path, result)
        second = "c" * 32
        second_path = tmp_path / "results" / f"{second}.json"
        save_result(result, second_path)
        os.utime(second_path, (1, 1))
        assert [item["run_id"] for item in client.get("/api/results").json()["results"]] == [first, second]
        bad = client.get("/api/results/xyz")
        assert bad.status_code == 400 and "rerun_url" not in bad.json()


def test_rejected_validation_error_has_only_field_paths(tmp_path: Path,
                                                       result: SchemeResult) -> None:
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        path = tmp_path / "results" / f"{run_id}.json"
        document = json.loads(path.read_text())
        document["schema_version"] = "aosr.scheme_result.v1"
        path.write_text(json.dumps(document))
        response = client.get(f"/api/results/{run_id}")
        assert response.status_code == 409
        reason = response.json()["reason"]
        assert "結果檔格式是舊版" in reason and "schema_version" in reason
        assert "input_value" not in reason and "http" not in reason


def test_result_error_class_is_visible(tmp_path: Path, result: SchemeResult,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, raise_server_exceptions=False) as client:
        run_id = _files(tmp_path, result)
        monkeypatch.setattr("aosr.gui.app.build_result_view", lambda *args, **kwargs:
                            (_ for _ in ()).throw(AssertionError()))
        response = client.get(f"/api/results/{run_id}")
        assert response.status_code == 500
        assert "AssertionError" in response.json()["error"]


def test_zero_importance_unavailable_result_returns_page(tmp_path: Path, result: SchemeResult,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.scoring.contract import EvaluationState, QualityCategory, ReasonCode
    from aosr.scoring.receiver_set import ReceiverRole

    points = tuple(point.model_copy(update={"importance": 0.0})
                   if point.role is ReceiverRole.SURROUNDING else point
                   for point in result.scheme.receiver_set.points)
    receivers = result.scheme.receiver_set.model_copy(update={"points": points})
    scheme = result.scheme.model_copy(update={"receiver_set": receivers})
    evaluations = tuple(item.model_copy(update={"state": EvaluationState.UNAVAILABLE,
                                         "payload": None,
                                         "reason_codes": (ReasonCode.ZERO_TOTAL_IMPORTANCE,)})
                        if item.category in (QualityCategory.LISTENING_AREA_STABILITY,
                                             QualityCategory.REFLECTIONS_AND_ECHO) else item
                        for item in result.candidate.evaluations)
    candidate = result.candidate.model_copy(update={"evaluations": evaluations})
    altered = result.model_copy(update={"scheme": scheme, "candidate": candidate})
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        monkeypatch.setattr("aosr.gui.app.load_result", lambda *args, **kwargs: altered)
        response = client.get(f"/api/results/{run_id}")
        assert response.status_code == 200
        assert response.json()["listening_area"]["state"] == "unavailable"
        assert response.json()["reflections"][0]["state"] == "unavailable"


def test_snapshot_job_keeps_scheme_in_its_run_directory(tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    manager = JobManager(tmp_path, ("unused",), "a" * 40, tmp_path / "capabilities.toml")
    seen: list[Path] = []
    def fake_start(self: JobManager, path: Path, run_id: str) -> dict[str, object]:
        seen.append(path)
        assert path.parent.name == run_id
        return {"run_id": run_id}
    monkeypatch.setattr(JobManager, "_start", fake_start)
    state = manager.start_snapshot('{"scheme_id":"original"}')
    assert seen == [tmp_path / "runs" / str(state["run_id"]) / "scheme.json"]
    assert json.loads(seen[0].read_text()) == {"scheme_id": "original"}


@pytest.mark.parametrize("scheme_id", ["../../results/" + "9" * 32, "/tmp/outside-snapshot", "../x"])
def test_rerun_snapshot_ignores_tampered_scheme_id(tmp_path: Path, result: SchemeResult,
                                                   monkeypatch: pytest.MonkeyPatch, scheme_id: str) -> None:
    """結果檔裡的方案代號被動過（../、絕對路徑）：快照照樣只寫在新計算代號自己的目錄，資料夾外一個檔都不多。"""
    seen: list[Path] = []
    def fake_start(self: JobManager, path: Path, run_id: str) -> dict[str, object]:
        seen.append(path)
        return {"run_id": run_id, "status": "running"}
    monkeypatch.setattr(JobManager, "_start", fake_start)
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        path = tmp_path / "results" / f"{run_id}.json"
        document = json.loads(path.read_text())
        document["scheme"]["scheme_id"] = scheme_id
        path.write_text(json.dumps(document))
        before = {item for item in tmp_path.parent.rglob("*") if tmp_path not in item.parents and item != tmp_path}
        response = client.post(f"/api/results/{run_id}/rerun", json={})
        after = {item for item in tmp_path.parent.rglob("*") if tmp_path not in item.parents and item != tmp_path}
    assert after == before
    assert not Path("/tmp/outside-snapshot.json").exists()
    if response.status_code == 200:
        assert seen and seen[0].parent.parent == tmp_path / "runs" and seen[0].name == "scheme.json"


def test_rerun_route_is_wired_to_result_snapshot(tmp_path: Path) -> None:
    import inspect

    source = inspect.getsource(GuiHandlers.rerun_result)
    assert 'document["scheme"]' in source and "start_snapshot" in source
    assert "_read_scheme(" not in source
    assert tmp_path.is_dir()


def test_result_paths_use_modification_time(tmp_path: Path) -> None:
    import os

    results = tmp_path / "results"
    results.mkdir()
    newer = results / f"{'a' * 32}.json"
    older = results / f"{'f' * 32}.json"
    newer.write_text("{}")
    older.write_text("{}")
    os.utime(older, (1, 1))
    assert _result_paths(tmp_path) == [newer, older]


def test_job_status_has_public_read_method(tmp_path: Path) -> None:
    manager = JobManager(tmp_path, ("unused",), "a" * 40, tmp_path / "capabilities.toml")
    run_id = "a" * 32
    (tmp_path / "runs" / f"{run_id}.json").write_text('{"scheme_id":"demo"}')
    assert manager.read_state(run_id)["scheme_id"] == "demo"
