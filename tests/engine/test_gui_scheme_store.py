"""方案存檔、結果清單與工作找回的端點契約。"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time
from http import HTTPStatus
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from aosr.gui.app import STATIC, GuiSettings, create_app
from aosr.gui.result_list import summarize_result
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
    return run_id


def _changed(document: dict[str, object]) -> dict[str, object]:
    changed = cast(dict[str, object], json.loads(json.dumps(document)))
    speakers = cast(dict[str, dict[str, float]], changed["speakers"])
    speakers["left"]["x"] += 0.1
    return changed


def test_save_as_collision_and_same_content(tmp_path: Path) -> None:
    with _client(tmp_path) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        url = "/api/schemes/demo"
        assert client.put(url, json=document).status_code == HTTPStatus.OK
        before = (tmp_path / "schemes" / "demo.json").read_bytes()
        collision = client.put(url, json=_changed(document), headers={"If-None-Match": "*"})
        assert collision.status_code == HTTPStatus.CONFLICT
        assert "已經有叫" in collision.json()["error"]
        assert (tmp_path / "schemes" / "demo.json").read_bytes() == before
        same = client.put(url, json=document, headers={"If-None-Match": "*"})
        assert same.status_code == HTTPStatus.OK
        assert same.json()["message"] == "方案沒有變動"
        assert client.put(url, json=_changed(document)).status_code == HTTPStatus.OK


def test_save_as_does_not_overwrite_file_written_meanwhile(tmp_path: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    # 「不存在才建立」要一步完成：查的時候還沒有、寫暫存檔那一瞬間別人先存了同名，照樣 409、不蓋掉。
    with _client(tmp_path) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        path = tmp_path / "schemes" / "demo.json"
        racer = json.dumps(_changed(document))
        real_mkstemp = tempfile.mkstemp

        def mkstemp_after_racer(dir: Path, prefix: str, suffix: str) -> tuple[int, str]:
            path.write_text(racer)
            return real_mkstemp(dir=dir, prefix=prefix, suffix=suffix)

        monkeypatch.setattr("aosr.gui.app.tempfile.mkstemp", mkstemp_after_racer)
        response = client.put("/api/schemes/demo", json=document, headers={"If-None-Match": "*"})
        assert response.status_code == HTTPStatus.CONFLICT
        assert path.read_text() == racer


def test_summary_names_same_or_different_engine_and_registry(tmp_path: Path,
                                                             result: SchemeResult) -> None:
    path = tmp_path / "result.json"
    save_result(result, path)
    same = summarize_result(path, result.engine_commit, result.quality_targets_fingerprint)
    assert "同版" in same.engine_text
    assert "不同版" not in same.engine_text
    assert "相同" in same.registry_text
    other = summarize_result(path, "a" * 40, "另一份登記簿的指紋")
    assert "不同版" in other.engine_text
    assert "已換" in other.registry_text


def test_result_freezes_scheme_and_validation_precedes_conflict(tmp_path: Path,
                                                               result: SchemeResult) -> None:
    with _client(tmp_path) as client:
        _files(tmp_path, result)
        path = tmp_path / "schemes" / "wall-1.json"
        before = path.read_bytes()
        document = result.scheme.model_dump(mode="json")
        assert client.put("/api/schemes/wall-1", json=document).status_code == HTTPStatus.OK
        frozen = client.put("/api/schemes/wall-1", json=_changed(document))
        assert frozen.status_code == HTTPStatus.CONFLICT
        assert "另存新名字" in frozen.json()["error"]
        assert path.read_bytes() == before
        bad = _changed(document)
        scene = cast(dict[str, object], bad["scene"])
        walls = cast(dict[str, float], scene["impedance_pa_s_per_m_by_wall"])
        walls["floor"] = -1
        assert client.put("/api/schemes/wall-1", json=bad,
                          headers={"If-None-Match": "*"}).status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_results_summary_bad_file_and_changed_cache(tmp_path: Path, result: SchemeResult) -> None:
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        listed = client.get("/api/results")
        assert listed.status_code == HTTPStatus.OK
        row = next(item for item in listed.json()["results"] if item["run_id"] == run_id)
        assert all(isinstance(value, str) for value in row.values())
        assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", row["finished_text"])
        assert "不同版" in row["engine_text"]
        assert "相同" in row["registry_text"]
        bad_id = "c" * 32
        (tmp_path / "results" / f"{bad_id}.json").write_text("ok")
        rows = client.get("/api/results").json()["results"]
        broken = next(item for item in rows if item["run_id"] == bad_id)
        assert all(isinstance(value, str) for value in broken.values())
        assert "讀不出" in broken["scheme_id"]
        # 換成一樣長的代號、明寫新的修改時間：大小不變，只有修改時間變，快取不看修改時間就讀到舊的。
        path = tmp_path / "results" / f"{run_id}.json"
        raw = path.read_bytes()
        original = f'"{result.scheme.scheme_id}"'.encode()
        replaced = f'"{"n" * len(result.scheme.scheme_id)}"'.encode()
        mtime_ns = path.stat().st_mtime_ns
        path.write_bytes(raw.replace(original, replaced))
        os.utime(path, ns=(mtime_ns + 10**9, mtime_ns + 10**9))
        updated = client.get("/api/results").json()["results"]
        assert next(item for item in updated
                    if item["run_id"] == run_id)["scheme_id"] == "n" * len(result.scheme.scheme_id)


def test_runs_list_and_finished_elapsed_is_fixed(tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path) as client:
        run_id = "d" * 32
        state = {"run_id": run_id, "scheme_id": "demo", "status": "running",
                 "started_at": time.time() - 4, "pid": 123,
                 "stderr_path": str(tmp_path / "runs" / "missing.stderr"),
                 "result_path": str(tmp_path / "results" / f"{run_id}.json")}
        (tmp_path / "runs" / "missing.stderr").write_text("")
        (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps(state))
        monkeypatch.setattr("aosr.gui.jobs.JobManager._group_alive", lambda self, pid: True)
        running = client.get("/api/runs")
        assert running.status_code == HTTPStatus.OK
        assert {item["run_id"] for item in running.json()["running"]} == {run_id}
        assert running.json()["running"][0]["scheme_id"] == "demo"
        state["status"] = "done"
        state["finished_at"] = float(str(state["started_at"])) + 4
        (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps(state))
        first = client.get(f"/api/runs/{run_id}").json()["elapsed_s"]
        time.sleep(0.02)
        assert client.get(f"/api/runs/{run_id}").json()["elapsed_s"] == first
        assert client.get("/api/runs").json()["recent"]["run_id"] == run_id


def test_finished_elapsed_uses_result_time_and_old_state_file_time(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 一小時前開始、五秒就算完，沒人開網頁、現在才第一次查：要記五秒，不是一小時。
    # 記 finished_at 以前就結束的舊狀態檔用它最後寫入的時間，也不准一直長。
    with _client(tmp_path) as client:
        started = time.time() - 3600
        done_id, old_id = "e" * 32, "f" * 32
        for run_id, status in ((done_id, "running"), (old_id, "done")):
            (tmp_path / "runs" / f"{run_id}.stderr").write_text("")
            state = {"run_id": run_id, "scheme_id": "demo", "status": status,
                     "started_at": started, "pid": 123,
                     "stderr_path": str(tmp_path / "runs" / f"{run_id}.stderr"),
                     "result_path": str(tmp_path / "results" / f"{run_id}.json")}
            (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps(state))
        (tmp_path / "results" / f"{done_id}.json").write_text("{}")
        os.utime(tmp_path / "results" / f"{done_id}.json", (started + 5, started + 5))
        os.utime(tmp_path / "runs" / f"{old_id}.json", (started + 7, started + 7))
        monkeypatch.setattr("aosr.gui.jobs.JobManager._group_alive", lambda self, pid: False)
        done = client.get(f"/api/runs/{done_id}").json()
        assert (done["status"], done["elapsed_s"]) == ("done", 5.0)
        assert client.get(f"/api/runs/{old_id}").json()["elapsed_s"] == 7.0


def test_page_uses_safe_dom_and_finds_all_lists() -> None:
    script = (STATIC / "app.js").read_text()
    assert "innerHTML" not in script
    assert {"If-None-Match", "/api/schemes", "/api/results", "/api/runs"} <= set(
        re.findall(r"If-None-Match|/api/(?:schemes|results|runs)", script))
