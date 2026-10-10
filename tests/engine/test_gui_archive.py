"""封存與搬回保住結果、附屬檔及方案名字的身分。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.reporting.result import SchemeResult, save_result
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_compare_routes import pair
from tests.engine.test_gui_failed_results import _client, _path, _runner, _start, _wait_file, _wait_state


def _bundle(root: Path, result: SchemeResult, run_id: str, status: str = "done") -> None:
    for folder in ("results", "runs"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    save_result(result, _path(root, "results", run_id))
    stderr = (root / "runs" / run_id).with_suffix(".stderr")
    stderr.write_text("診斷輸出", encoding="utf-8")
    if status != "none":
        _path(root, "runs", run_id).write_text(json.dumps({
            "run_id": run_id, "scheme_id": result.scheme.scheme_id, "status": status,
            "started_at": 0, "finished_at": 1, "exit_code": 3 if status == "failed" else 0,
            "result_path": str(_path(root, "results", run_id)), "stderr_path": str(stderr),
        }), encoding="utf-8")
    snapshot = root / "runs" / run_id
    snapshot.mkdir()
    (snapshot / "scheme").with_suffix(".json").write_text(result.scheme.model_dump_json())


def _contents(root: Path, run_id: str) -> dict[Path, bytes]:
    paths = [_path(root, "results", run_id), _path(root, "runs", run_id),
             (root / "runs" / run_id).with_suffix(".stderr")]
    paths.extend((root / "runs" / run_id).rglob("*"))
    return {path.relative_to(root): path.read_bytes() for path in paths if path.is_file()}


def _ids(client: TestClient, url: str) -> set[str]:
    return {item["run_id"] for item in client.get(url).json()["results"]}


def test_archive_and_restore_move_all_artifacts_and_lists(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, pair[0], run_id)
        before = _contents(tmp_path, run_id)
        response = client.post(f"/api/results/{run_id}/archive", json={})
        assert response.status_code == 200, response.text
        assert pair[0].scheme.scheme_id in response.json()["message"]
        assert "已封存" in response.json()["message"] and "搬回" in response.json()["message"]
        archive = tmp_path / "archive"
        assert _contents(archive, run_id) == before
        assert not _contents(tmp_path, run_id) and not (tmp_path / "runs" / run_id).exists()
        assert run_id not in _ids(client, "/api/results") and run_id in _ids(client, "/api/archive")
        row = next(item for item in client.get("/api/archive").json()["results"]
                   if item["run_id"] == run_id)
        assert row["result_url"] == "" and row["run_status"] == "done"
        assert client.get(f"/api/results/{run_id}").status_code == 404
        restored = client.post(f"/api/archive/{run_id}/restore", json={})
        assert restored.status_code == 200, restored.text
        assert "搬回" in restored.json()["message"]
        assert _contents(tmp_path, run_id) == before
        assert not _contents(archive, run_id) and not (archive / "runs" / run_id).exists()
        assert run_id in _ids(client, "/api/results") and run_id not in _ids(client, "/api/archive")
        assert client.get(f"/api/results/{run_id}").status_code == 200
        assert client.get(f"/results/{run_id}").status_code == 200


def test_result_without_optional_artifacts_can_be_archived_and_restored(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        path = _path(tmp_path, "results", run_id)
        save_result(pair[0], path)
        content, modified = path.read_bytes(), path.stat().st_mtime_ns
        assert client.post(f"/api/results/{run_id}/archive", json={}).status_code == 200
        archived = _path(tmp_path / "archive", "results", run_id)
        assert archived.read_bytes() == content and archived.stat().st_mtime_ns == modified
        assert client.post(f"/api/archive/{run_id}/restore", json={}).status_code == 200
        assert path.read_bytes() == content and path.stat().st_mtime_ns == modified
        assert _contents(tmp_path, run_id) == {path.relative_to(tmp_path): content}


def test_running_result_cannot_be_archived(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    result = pair[0]
    with _client(tmp_path, _runner(tmp_path, {result.scheme.scheme_id: (result, 0, True)})) as client:
        run_id = _start(client, result)
        try:
            _wait_file(_path(tmp_path, "results", run_id))
            snapshot = tmp_path / "runs" / run_id
            snapshot.mkdir()
            (snapshot / "scheme").write_text(result.scheme.model_dump_json())
            before = _contents(tmp_path, run_id)
            response = client.post(f"/api/results/{run_id}/archive", json={})
            assert response.status_code == 409
            assert "正在計算，不能封存" in response.json()["error"]
            assert _contents(tmp_path, run_id) == before
            assert not _contents(tmp_path / "archive", run_id)
        finally:
            (tmp_path / "release" / result.scheme.scheme_id).touch()
            _wait_state(client, run_id, "done")


@pytest.mark.parametrize("action", ["archive", "restore"])
@pytest.mark.parametrize("run_id, expected", [("invalid", 400), ("b" * 32, 404)])
def test_invalid_and_missing_ids(tmp_path: Path, action: str, run_id: str, expected: int) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        url = (f"/api/results/{run_id}/archive" if action == "archive"
               else f"/api/archive/{run_id}/restore")
        assert client.post(url, json={}).status_code == expected


@pytest.mark.parametrize("action", ["archive", "restore"])
@pytest.mark.parametrize("collision", ["result", "state", "stderr", "snapshot"])
def test_conflicts_leave_both_sides_untouched(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], action: str, collision: str) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        source, target = ((tmp_path, tmp_path / "archive") if action == "archive"
                          else (tmp_path / "archive", tmp_path))
        _bundle(source, pair[0], run_id)
        if collision == "snapshot":
            folder = target / "runs" / run_id
            folder.mkdir()
            (folder / "keep").write_text("保留")
        else:
            path = (_path(target, "results" if collision == "result" else "runs", run_id)
                    if collision != "stderr" else (target / "runs" / run_id).with_suffix(".stderr"))
            path.write_text("保留", encoding="utf-8")
        before = (_contents(source, run_id), _contents(target, run_id))
        url = (f"/api/results/{run_id}/archive" if action == "archive"
               else f"/api/archive/{run_id}/restore")
        response = client.post(url, json={})
        assert response.status_code == 409
        place = "封存區" if action == "archive" else "結果或計算資料夾"
        assert f"{place}已經有同代號" in response.json()["error"]
        assert (_contents(source, run_id), _contents(target, run_id)) == before


@pytest.mark.parametrize("action", ["archive", "restore"])
def test_second_move_failure_rolls_back_first(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        source, target = ((tmp_path, tmp_path / "archive") if action == "archive"
                          else (tmp_path / "archive", tmp_path))
        _bundle(source, pair[0], run_id)
        before = _contents(source, run_id)
        replace = os.replace
        failed_source = _path(source, "runs", run_id)
        moved: list[Path] = []

        def fail_state(src: Path, dst: Path) -> None:
            if src == failed_source:
                raise OSError("第二個搬動失敗")
            moved.append(src)
            replace(src, dst)

        monkeypatch.setattr("aosr.gui.jobs.os.replace", fail_state)
        url = (f"/api/results/{run_id}/archive" if action == "archive"
               else f"/api/archive/{run_id}/restore")
        response = client.post(url, json={})
        assert response.status_code == 500, response.text
        assert "已搬回原處" in response.json()["error"]
        assert "OSError" not in response.json()["error"]
        assert "請助理檢查資料夾權限" in response.json()["error"]
        assert _path(source, "results", run_id) in moved
        assert _path(target, "results", run_id) in moved
        assert _contents(source, run_id) == before and not _contents(target, run_id)


@pytest.mark.parametrize("action", ["archive", "restore"])
def test_rollback_failure_reports_split_artifacts(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        source, target = ((tmp_path, tmp_path / "archive") if action == "archive"
                          else (tmp_path / "archive", tmp_path))
        _bundle(source, pair[0], run_id)
        before = _contents(source, run_id)
        replace, moved_once = os.replace, False

        def fail_after_first(src: Path, dst: Path) -> None:
            nonlocal moved_once
            if moved_once:
                raise OSError("後續搬動與復原都被拒絕")
            replace(src, dst)
            moved_once = True

        monkeypatch.setattr("aosr.gui.jobs.os.replace", fail_after_first)
        url = (f"/api/results/{run_id}/archive" if action == "archive"
               else f"/api/archive/{run_id}/restore")
        response = client.post(url, json={})
        assert response.status_code == 500, response.text
        assert "搬回都失敗" in response.json()["error"]
        assert "已搬回原處" not in response.json()["error"]
        assert "OSError" not in response.json()["error"]
        assert not _path(source, "results", run_id).exists()
        assert _path(target, "results", run_id).is_file()
        assert _contents(source, run_id) | _contents(target, run_id) == before


@pytest.mark.parametrize("action", ["archive", "restore"])
def test_first_move_failure_does_not_attempt_rollback(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult],
        monkeypatch: pytest.MonkeyPatch, action: str) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        source, target = ((tmp_path, tmp_path / "archive") if action == "archive"
                          else (tmp_path / "archive", tmp_path))
        _bundle(source, pair[0], run_id)
        before = _contents(source, run_id)
        attempted: list[Path] = []

        def refuse_move(src: Path, dst: Path) -> None:
            attempted.append(src)
            raise OSError("第一個搬動被拒絕")

        monkeypatch.setattr("aosr.gui.jobs.os.replace", refuse_move)
        url = (f"/api/results/{run_id}/archive" if action == "archive"
               else f"/api/archive/{run_id}/restore")
        response = client.post(url, json={})
        assert response.status_code == 500, response.text
        assert attempted and all(src == _path(source, "results", run_id) for src in attempted)
        assert _contents(source, run_id) == before and not _contents(target, run_id)


def _row(client: TestClient, url: str, run_id: str) -> dict[str, object]:
    return next(item for item in client.get(url).json()["results"] if item["run_id"] == run_id)


@pytest.mark.parametrize("status", ["done", "none", "failed", "stopped"])
def test_archive_releases_name_and_restore_marks_changed_scheme(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], status: str) -> None:
    # #755（老闆選 A）：封存後放開名字，同名方案存得進去；搬回後正常完成的那筆又鎖住名字，
    # 而且它帶的方案副本跟現在存著的不一樣，代號底下標「改之前的方案算的」。
    run_id, result = "a" * 32, pair[0]
    name = result.scheme.scheme_id
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, result, run_id, status)
        (tmp_path / "schemes").mkdir(exist_ok=True)
        (tmp_path / "schemes" / f"{name}.json").write_text(result.scheme.model_dump_json(), encoding="utf-8")
        # 對照：存著的方案跟結果帶的一樣，不標。
        assert _row(client, "/api/results", run_id)["scheme_text"] == ""
        changed = result.scheme.model_dump(mode="json")
        changed["scene"]["room_m"]["Lx"] += 0.1
        frozen = 409 if status in {"done", "none"} else 200
        assert client.put(f"/api/schemes/{name}", json=changed).status_code == frozen
        assert client.post(f"/api/results/{run_id}/archive", json={}).status_code == 200
        assert client.put(f"/api/schemes/{name}", json=changed).status_code == 200
        assert _row(client, "/api/archive", run_id)["scheme_text"] == "改之前的方案算的"
        assert client.post(f"/api/archive/{run_id}/restore", json={}).status_code == 200
        assert _row(client, "/api/results", run_id)["scheme_text"] == "改之前的方案算的"
        changed["scene"]["room_m"]["Lx"] += 0.1
        assert client.put(f"/api/schemes/{name}", json=changed).status_code == frozen


def test_archive_uses_its_own_status_and_modification_order(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    failed, done = "a" * 32, "b" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, pair[0], failed, "failed")
        _bundle(tmp_path, pair[1], done)
        os.utime(_path(tmp_path, "results", done), (1, 1))
        for run_id in (failed, done):
            assert client.post(f"/api/results/{run_id}/archive", json={}).status_code == 200
        rows = client.get("/api/archive").json()["results"]
        assert [item["run_id"] for item in rows] == [failed, done]
        assert rows[0]["run_status"] == "failed"
        assert rows[0]["status_text"] == "失敗：內容可能不完整"
        # 主目錄裡的另一份狀態不能冒充封存區的狀態。
        _path(tmp_path, "runs", failed).write_text('{"status":"done"}')
        row = next(item for item in client.get("/api/archive").json()["results"]
                   if item["run_id"] == failed)
        assert row["run_status"] == "failed"
