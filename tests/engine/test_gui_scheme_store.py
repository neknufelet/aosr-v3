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

from aosr.config.paths import config_path
from aosr.gui.app import STATIC, GuiSettings, create_app
from aosr.gui.jobs import ResultStatus
from aosr.gui.result_list import ResultList, summarize_result
from aosr.reporting.calculation_fingerprint import short_fingerprint
from aosr.reporting.evaluation import quality_targets_fingerprint
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


def _summary_file(tmp_path: Path, run_id: str = "b" * 32) -> Path:
    path = tmp_path / "results" / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "aosr.scheme_result.v3",
        "scheme": {"scheme_id": "wall-1"},
        "timings": {"total_s": 1.25},
        "engine_commit": "a" * 40,
        "calculation_fingerprint": "calc-v1:" + "0" * 64,
        "quality_targets_fingerprint": quality_targets_fingerprint(
            config_path("quality_targets.toml")),
    }))
    return path


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


def test_summary_names_same_or_different_calculation_and_registry(tmp_path: Path) -> None:
    path = _summary_file(tmp_path)
    same = summarize_result(path, "calc-v1:" + "0" * 64,
                            quality_targets_fingerprint(config_path("quality_targets.toml")))
    # 「計算版本」格只講白話：跟現在的程式同不同、不同就要重算；不寫計算指紋、提交代號、「引擎」這些行話。
    assert same.calculation_text == "跟現在的程式相同"
    assert "相同" in same.registry_text
    other = summarize_result(path, "calc-v1:" + "1" * 64, "另一份登記簿的指紋")
    assert other.calculation_text == "跟現在的程式不同，要重算"
    assert "已換" in other.registry_text
    for summary in (same, other):
        assert not re.search(r"[0-9a-f]{7}|指紋|引擎|提交", summary.calculation_text), summary.calculation_text
        # 提交代號與指紋前幾碼收在滑鼠停留的說明（calculation_detail）。
        assert "a" * 7 in summary.calculation_detail and "計算指紋" in summary.calculation_detail
    assert short_fingerprint("calc-v1:" + "0" * 64) in same.calculation_detail


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
        # 另存撞到有結果的那一份：開舊方案是死路（打開會洗掉剛改的、打開了也改不得），訊息直接叫他換名字。
        taken = client.put("/api/schemes/wall-1", json=_changed(document),
                           headers={"If-None-Match": "*"})
        assert taken.status_code == HTTPStatus.CONFLICT
        error = taken.json()["error"]
        assert "已經有叫" in error
        assert "已經有算好的結果" in error
        assert "另存新名字" in error
        assert "開舊方案" not in error
        assert path.read_bytes() == before
        bad = _changed(document)
        scene = cast(dict[str, object], bad["scene"])
        walls = cast(dict[str, float], scene["impedance_pa_s_per_m_by_wall"])
        walls["floor"] = -1
        assert client.put("/api/schemes/wall-1", json=bad,
                          headers={"If-None-Match": "*"}).status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_v2_result_keeps_scheme_name_frozen(tmp_path: Path, result: SchemeResult) -> None:
    with _client(tmp_path) as client:
        run_id = _files(tmp_path, result)
        path = tmp_path / "results" / (run_id + ".json")
        document = json.loads(path.read_text())
        document["schema_version"] = "aosr.scheme_result.v2"
        document.pop("calculation_fingerprint")
        path.write_text(json.dumps(document))
        listed = client.get("/api/results").json()["results"]
        assert listed[0]["scheme_id"] == result.scheme.scheme_id
        assert "舊格式" in listed[0]["calculation_text"]
        # 主畫面不露內部版本代號（v2、v3）：老闆只要知道這份是程式更新前算的、要重算。
        assert not re.search(r"v\d", listed[0]["calculation_text"]), listed[0]["calculation_text"]
        assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", listed[0]["finished_text"])
        assert listed[0]["duration_text"] == "舊格式不顯示"
        assert listed[0]["registry_text"] == "舊格式不顯示"
        response = client.put("/api/schemes/wall-1", json=_changed(result.scheme.model_dump(mode="json")))
        assert response.status_code == HTTPStatus.CONFLICT
        assert "已經有算好的結果" in response.json()["error"]


def test_malformed_version_field_still_keeps_readable_scheme_id(tmp_path: Path) -> None:
    path = _summary_file(tmp_path)
    document = json.loads(path.read_text())
    document["calculation_fingerprint"] = "broken"
    path.write_text(json.dumps(document))
    summary = summarize_result(path, "current", "registry")
    assert summary.scheme_id == "wall-1"
    assert summary.calculation_text == "讀不出"
    assert summary.finished_text == "讀不出"
    assert summary.duration_text == "讀不出"
    assert summary.registry_text == "讀不出"


@pytest.mark.parametrize("version", [None, "aosr.scheme_result.v4", 3])
def test_unknown_version_is_not_labelled_v2(tmp_path: Path, version: object) -> None:
    """版本欄缺、認不得或比現在新，不准冒充成舊格式 v2（那一列看起來會像正常的舊檔）。"""
    path = _summary_file(tmp_path)
    document = json.loads(path.read_text())
    if version is None:
        del document["schema_version"]
    else:
        document["schema_version"] = version
    path.write_text(json.dumps(document))
    summary = summarize_result(path, "current", "registry")
    assert summary.scheme_id == "wall-1"
    assert "認不得" in summary.calculation_text and "v2" not in summary.calculation_text
    # 欄位名（schema_version）是內部名字，只放在滑鼠停留的說明；缺欄時說明也不印 Python 的 None。
    assert "schema_version" not in summary.calculation_text
    assert "schema_version" in summary.calculation_detail and "None" not in summary.calculation_detail
    assert summary.finished_text == "讀不出"


def test_running_scheme_is_frozen_and_result_without_file_too(
        tmp_path: Path, result: SchemeResult, monkeypatch: pytest.MonkeyPatch) -> None:
    # 算的那幾分鐘正是最可能改表單按儲存的時候；改了，結果裡存的設定就跟方案檔對不上。
    with _client(tmp_path) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        assert client.put("/api/schemes/demo", json=document).status_code == HTTPStatus.OK
        path = tmp_path / "schemes" / "demo.json"
        before = path.read_bytes()
        run_id = "d" * 32
        (tmp_path / "runs" / f"{run_id}.stderr").write_text("")
        (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps({
            "run_id": run_id, "scheme_id": "demo", "status": "running", "started_at": time.time(),
            "pid": 123, "stderr_path": str(tmp_path / "runs" / f"{run_id}.stderr"),
            "result_path": str(tmp_path / "results" / f"{run_id}.json")}))
        monkeypatch.setattr("aosr.gui.jobs.JobManager._group_alive", lambda self, pid: True)
        busy = client.put("/api/schemes/demo", json=_changed(document))
        assert busy.status_code == HTTPStatus.CONFLICT
        assert "正在計算" in busy.json()["error"]
        assert path.read_bytes() == before
        assert client.put("/api/schemes/demo", json=document).status_code == HTTPStatus.OK
        # 有結果、方案檔卻不在：照樣不准用這個代號存另一份內容，帶不帶標頭都一樣。
        _files(tmp_path, result)
        (tmp_path / "schemes" / "wall-1.json").unlink()
        changed = _changed(result.scheme.model_dump(mode="json"))
        for headers in ({}, {"If-None-Match": "*"}):
            response = client.put("/api/schemes/wall-1", json=changed, headers=headers)
            assert response.status_code == HTTPStatus.CONFLICT
            assert "已經有算好的結果" in response.json()["error"]
        # 他剛用的就是另存：訊息要叫他換名字，不是再叫他去用另存。
        assert "填一個新名字" in response.json()["error"]
        assert not (tmp_path / "schemes" / "wall-1.json").exists()


def test_old_format_scheme_file_does_not_block_saving(tmp_path: Path) -> None:
    # 舊格式或壞掉的舊檔當成「內容不同」：沒結果就照舊覆蓋，另存撞名照樣 409，不回舊檔的 422。
    with _client(tmp_path) as client:
        document = client.get("/api/example").json()["scheme"]
        document["scheme_id"] = "demo"
        (tmp_path / "schemes" / "demo.json").write_text('{"legacy_field": 1}')
        assert client.put("/api/schemes/demo", json=document).status_code == HTTPStatus.OK
        assert json.loads((tmp_path / "schemes" / "demo.json").read_text())["scheme_id"] == "demo"
        document["scheme_id"] = "other"
        (tmp_path / "schemes" / "other.json").write_text("ok")
        taken = client.put("/api/schemes/other", json=document, headers={"If-None-Match": "*"})
        assert taken.status_code == HTTPStatus.CONFLICT
        assert "已經有叫" in taken.json()["error"]
        # 叫他填的是「新名字」那一格（輸入頁那一列的字），不是那顆「另存新名字」按鈕。
        assert "「新名字」這一格" in taken.json()["error"]
        assert (tmp_path / "schemes" / "other.json").read_text() == "ok"


def test_results_summary_bad_file_and_changed_cache(tmp_path: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("aosr.gui.result_list.calculation_fingerprint",
                        lambda **kwargs: "calc-v1:" + "1" * 64)
    with _client(tmp_path) as client:
        run_id = "b" * 32
        _summary_file(tmp_path, run_id)
        listed = client.get("/api/results")
        assert listed.status_code == HTTPStatus.OK
        row = next(item for item in listed.json()["results"] if item["run_id"] == run_id)
        assert all(isinstance(value, str) for value in row.values())
        assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d", row["finished_text"])
        assert "不同" in row["calculation_text"]
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
        original = b'"wall-1"'
        replaced = b'"nnnnnn"'
        mtime_ns = path.stat().st_mtime_ns
        path.write_bytes(raw.replace(original, replaced))
        os.utime(path, ns=(mtime_ns + 10**9, mtime_ns + 10**9))
        updated = client.get("/api/results").json()["results"]
        assert next(item for item in updated
                    if item["run_id"] == run_id)["scheme_id"] == "nnnnnn"


def test_results_list_recalculates_current_fingerprint_each_request(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    current = ["calc-v1:" + "0" * 64]
    monkeypatch.setattr("aosr.gui.result_list.calculation_fingerprint",
                        lambda **kwargs: current[0])
    with _client(tmp_path) as client:
        run_id = "b" * 32
        _summary_file(tmp_path, run_id)
        def row() -> dict[str, str]:
            return next(item for item in client.get("/api/results").json()["results"]
                        if item["run_id"] == run_id)

        assert "相同" in row()["calculation_text"]
        current[0] = "calc-v1:" + "1" * 64
        assert "不同" in row()["calculation_text"]


def test_result_list_refreshes_summary_when_current_fingerprint_changes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _summary_file(tmp_path)
    current = ["calc-v1:" + "0" * 64]
    monkeypatch.setattr("aosr.gui.result_list.calculation_fingerprint",
                        lambda **kwargs: current[0])
    summaries = ResultList(config_path("capabilities.toml"),
                           config_path("quality_targets.toml"))
    assert "相同" in summaries.list([path], lambda _: ResultStatus())[0].calculation_text
    current[0] = "calc-v1:" + "1" * 64
    assert "不同" in summaries.list([path], lambda _: ResultStatus())[0].calculation_text


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
        assert "「demo」" in running.json()["running"][0]["display_text"]
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
        done_id, old_id, failed_id = "e" * 32, "f" * 32, "0" * 32
        for run_id, status in ((done_id, "running"), (old_id, "done"), (failed_id, "running")):
            (tmp_path / "runs" / f"{run_id}.stderr").write_text("")
            state = {"run_id": run_id, "scheme_id": "demo", "status": status,
                     "started_at": started, "pid": 123,
                     "stderr_path": str(tmp_path / "runs" / f"{run_id}.stderr"),
                     "result_path": str(tmp_path / "results" / f"{run_id}.json")}
            (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps(state))
        (tmp_path / "results" / f"{done_id}.json").write_text("{}")
        os.utime(tmp_path / "results" / f"{done_id}.json", (started + 5, started + 5))
        os.utime(tmp_path / "runs" / f"{old_id}.json", (started + 7, started + 7))
        # 失敗沒有結果檔：最後寫進 stderr 的是錯誤訊息，用它的時間。
        os.utime(tmp_path / "runs" / f"{failed_id}.stderr", (started + 6, started + 6))
        monkeypatch.setattr("aosr.gui.jobs.JobManager._group_alive", lambda self, pid: False)
        done = client.get(f"/api/runs/{done_id}").json()
        assert (done["status"], done["elapsed_s"]) == ("done", 5.0)
        assert client.get(f"/api/runs/{old_id}").json()["elapsed_s"] == 7.0
        failed = client.get(f"/api/runs/{failed_id}").json()
        assert (failed["status"], failed["elapsed_s"]) == ("failed", 6.0)


def test_silently_killed_run_counts_until_last_seen_alive(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 計算只在開頭寫 stderr；被記憶體不夠或 SIGKILL 殺掉時不留錯誤訊息。一小時後還看到它活著、
    # 下一次查才發現死了：秒數要算到約一小時，不是 stderr 最後寫入的六秒。
    with _client(tmp_path) as client:
        started = time.time() - 3600
        run_id = "1" * 32
        stderr = tmp_path / "runs" / f"{run_id}.stderr"
        stderr.write_text("")
        os.utime(stderr, (started + 6, started + 6))
        (tmp_path / "runs" / f"{run_id}.json").write_text(json.dumps({
            "run_id": run_id, "scheme_id": "demo", "status": "running", "started_at": started,
            "pid": 123, "stderr_path": str(stderr),
            "result_path": str(tmp_path / "results" / f"{run_id}.json")}))
        alive = {"now": True}
        monkeypatch.setattr("aosr.gui.jobs.JobManager._group_alive", lambda self, pid: alive["now"])
        assert client.get(f"/api/runs/{run_id}").json()["status"] == "running"
        alive["now"] = False
        failed = client.get(f"/api/runs/{run_id}").json()
        assert failed["status"] == "failed"
        assert failed["elapsed_s"] > 3000


def test_page_uses_safe_dom_and_finds_all_lists() -> None:
    script = (STATIC / "app.js").read_text()
    assert "innerHTML" not in script
    assert {"If-None-Match", "/api/schemes", "/api/results", "/api/runs"} <= set(
        re.findall(r"If-None-Match|/api/(?:schemes|results|runs)", script))
