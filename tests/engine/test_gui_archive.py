"""封存與搬回保住結果、附屬檔及方案名字的身分。"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import cast

import pytest
from starlette.testclient import TestClient

from aosr.reporting.result import SchemeResult, save_result
from aosr.gui.jobs import JobManager
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


def _saved(root: Path, result: SchemeResult) -> Path:
    path = root / "schemes" / f"{result.scheme.scheme_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.scheme.model_dump_json(), encoding="utf-8")
    return path


def _tree(root: Path) -> dict[Path, bytes]:
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _archive(client: TestClient, name: str) -> dict[str, object]:
    response = client.post(f"/api/schemes/{name}/archive", json={})
    assert response.status_code == 200, response.text
    return cast(dict[str, object], response.json())


def _restore(client: TestClient, package: dict[str, object]) -> None:
    response = client.post(f"/api/archive/packages/{package['package_id']}/restore", json={})
    assert response.status_code == 200, response.text


def test_archive_and_restore_move_all_artifacts_and_lists(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    result, other = pair
    name = result.scheme.scheme_id
    ids = {"a" * 32, "b" * 32}
    with _client(tmp_path, (sys.executable,)) as client:
        for run_id in ids:
            _bundle(tmp_path, result, run_id)
        _bundle(tmp_path, other, "c" * 32)
        scheme = _saved(tmp_path, result)
        before = {key: value for run_id in ids for key, value in _contents(tmp_path, run_id).items()}
        before[scheme.relative_to(tmp_path)] = scheme.read_bytes()
        package = _archive(client, name)
        folder = tmp_path / "archive" / "packages" / str(package["package_id"])
        assert {key: value for key, value in _tree(folder).items() if key != Path("manifest.json")} == before
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["scheme_id"] == name and set(manifest["result_ids"]) == ids
        assert manifest["archived_at"] and manifest["package_id"] == package["package_id"]
        assert all(not (tmp_path / key).exists() for key in before)
        assert name not in client.get("/api/schemes").json()["schemes"]
        assert ids.isdisjoint(_ids(client, "/api/results"))
        assert "c" * 32 in _ids(client, "/api/results")
        listing = client.get("/api/archive").json()
        assert {row["package_id"] for row in listing["packages"]} == {package["package_id"]}
        assert listing["results"] == []
        _restore(client, package)
        assert {key: (tmp_path / key).read_bytes() for key in before} == before
        assert name in client.get("/api/schemes").json()["schemes"]
        assert ids <= _ids(client, "/api/results")
        assert client.get("/api/archive").json()["packages"] == []
        assert all(not (folder / key).exists() for key in before)


def test_same_name_archived_twice_restores_each_original(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    original = pair[0]
    changed = original.model_copy(update={"scheme": original.scheme.model_copy(update={"purpose": "mixing"})})
    packages, contents = [], []
    with _client(tmp_path, (sys.executable,)) as client:
        for run_id, result in (("a" * 32, original), ("b" * 32, changed)):
            _bundle(tmp_path, result, run_id)
            scheme = _saved(tmp_path, result)
            contents.append((_contents(tmp_path, run_id), scheme.read_bytes()))
            packages.append(_archive(client, result.scheme.scheme_id))
        assert packages[0]["package_id"] != packages[1]["package_id"]
        assert {row["package_id"] for row in client.get("/api/archive").json()["packages"]} == {
            package["package_id"] for package in packages}
        for index, run_id in enumerate(("a" * 32, "b" * 32)):
            _restore(client, packages[index])
            assert _contents(tmp_path, run_id) == contents[index][0]
            assert (tmp_path / "schemes" / f"{original.scheme.scheme_id}.json").read_bytes() == contents[index][1]
            if index == 0:
                _archive(client, original.scheme.scheme_id)


@pytest.mark.parametrize("collision", ["scheme", "result", "state", "stderr", "snapshot"])
def test_package_restore_collision_moves_nothing(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], collision: str) -> None:
    run_id, result = "a" * 32, pair[0]
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, result, run_id)
        _saved(tmp_path, result)
        package = _archive(client, result.scheme.scheme_id)
        paths = {"scheme": Path("schemes") / f"{result.scheme.scheme_id}.json",
                 "result": Path("results") / f"{run_id}.json",
                 "state": Path("runs") / f"{run_id}.json",
                 "stderr": Path("runs") / f"{run_id}.stderr", "snapshot": Path("runs") / run_id}
        path = tmp_path / paths[collision]
        if collision == "snapshot":
            path.mkdir()
            (path / "keep").write_text("現在那份")
        else:
            path.write_text("現在那份", encoding="utf-8")
        before = _tree(tmp_path)
        response = client.post(f"/api/archive/packages/{package['package_id']}/restore", json={})
        assert response.status_code == 409, response.text
        assert "先把現在那份封存或改名" in response.json()["error"]
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("only", ["scheme", "result"])
def test_scheme_only_or_result_only_can_archive_and_restore(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], only: str) -> None:
    result, run_id = pair[0], "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        if only == "scheme":
            path = _saved(tmp_path, result)
        else:
            path = _path(tmp_path, "results", run_id)
            save_result(result, path)
        content, modified = path.read_bytes(), path.stat().st_mtime_ns
        package = _archive(client, result.scheme.scheme_id)
        assert not path.exists()
        assert package["result_ids"] == ([run_id] if only == "result" else [])
        _restore(client, package)
        assert path.read_bytes() == content and path.stat().st_mtime_ns == modified
        if only == "result":
            assert not (tmp_path / "schemes" / f"{result.scheme.scheme_id}.json").exists()


def test_running_scheme_blocks_even_when_archiving_older_result(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    result, old_id = pair[0], "a" * 32
    with _client(tmp_path, _runner(tmp_path, {result.scheme.scheme_id: (result, 0, True)})) as client:
        run_id = _start(client, result)
        try:
            _wait_file(_path(tmp_path, "results", run_id))
            _bundle(tmp_path, result, old_id)
            before = _tree(tmp_path)
            response = client.post(f"/api/results/{old_id}/archive", json={})
            assert response.status_code == 409
            assert "正在計算，不能封存" in response.json()["error"]
            assert _tree(tmp_path) == before
        finally:
            (tmp_path / "release" / result.scheme.scheme_id).touch()
            _wait_state(client, run_id, "done")


@pytest.mark.parametrize("action", ["archive", "restore"])
@pytest.mark.parametrize("failure", ["second_result", "manifest"])
def test_package_move_failure_rolls_back_every_member(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch,
        action: str, failure: str) -> None:
    result = pair[0]
    with _client(tmp_path, (sys.executable,)) as client:
        for run_id in ("a" * 32, "b" * 32):
            _bundle(tmp_path, result, run_id)
        _saved(tmp_path, result)
        package = _archive(client, result.scheme.scheme_id) if action == "restore" else None
        before = _tree(tmp_path)
        replace, unlink = os.replace, Path.unlink

        def fail_move(src: Path, dst: Path) -> None:
            if src.name == f"{'b' * 32}.json" and src.parent.name == "results":
                raise OSError("第二筆結果搬動失敗")
            replace(src, dst)

        def fail_manifest(path: Path, missing_ok: bool = False) -> None:
            if path.name == "manifest.json":
                raise OSError("封存清單更新失敗")
            unlink(path, missing_ok=missing_ok)

        if failure == "second_result":
            monkeypatch.setattr("aosr.gui.jobs.os.replace", fail_move)
        elif action == "restore":
            monkeypatch.setattr(Path, "unlink", fail_manifest)
        else:
            original = Path.write_text

            def fail_write(path: Path, data: str, *args: object, **kwargs: object) -> int:
                if path.name == "manifest.json":
                    raise OSError("封存清單更新失敗")
                return original(path, data)

            monkeypatch.setattr(Path, "write_text", fail_write)
        url = (f"/api/archive/packages/{package['package_id']}/restore" if package else
               f"/api/schemes/{result.scheme.scheme_id}/archive")
        response = client.post(url, json={})
        assert response.status_code == 500, response.text
        assert "已搬回原處" in response.json()["error"]
        assert "OSError" not in response.json()["error"]
        assert _tree(tmp_path) == before


def test_package_rollback_failure_says_files_are_split(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    result = pair[0]
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, result, "a" * 32)
        _saved(tmp_path, result)
        before, replace = _tree(tmp_path), os.replace
        moved = False

        def fail_after_first(src: Path, dst: Path) -> None:
            nonlocal moved
            if moved:
                raise OSError("後續搬動與復原都被拒絕")
            replace(src, dst)
            moved = True

        monkeypatch.setattr("aosr.gui.jobs.os.replace", fail_after_first)
        response = client.post(f"/api/schemes/{result.scheme.scheme_id}/archive", json={})
        assert response.status_code == 500
        assert "搬回都失敗" in response.json()["error"]
        assert "已搬回原處" not in response.json()["error"]
        after = _tree(tmp_path)
        assert all(value in after.values() for value in before.values())
        assert not (tmp_path / "schemes" / f"{result.scheme.scheme_id}.json").exists()


def test_legacy_single_result_stays_separate_and_restores(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    result, run_id = pair[0], "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path / "archive", result, run_id, "failed")
        before = _contents(tmp_path / "archive", run_id)
        _saved(tmp_path, result)
        package = _archive(client, result.scheme.scheme_id)
        listing = client.get("/api/archive").json()
        assert {row["package_id"] for row in listing["packages"]} == {package["package_id"]}
        assert {row["run_id"] for row in listing["results"]} == {run_id}
        assert listing["results"][0]["run_status"] == "failed"
        assert _contents(tmp_path / "archive", run_id) == before
        response = client.post(f"/api/archive/{run_id}/restore", json={})
        assert response.status_code == 200, response.text
        assert _contents(tmp_path, run_id) == before
        assert client.get("/api/archive").json()["results"] == []


@pytest.mark.parametrize("status", ["done", "none", "failed", "stopped"])
def test_archive_releases_name_for_new_save_as(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], status: str) -> None:
    result, run_id = pair[0], "a" * 32
    name = result.scheme.scheme_id
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, result, run_id, status)
        _saved(tmp_path, result)
        package = _archive(client, name)
        changed = result.scheme.model_dump(mode="json")
        changed["scene"]["room_m"]["Lx"] += 0.1
        response = client.put(f"/api/schemes/{name}", json=changed, headers={"If-None-Match": "*"})
        assert response.status_code == 200, response.text
        assert client.get(f"/api/schemes/{name}").json()["scheme"] == changed
        before = _tree(tmp_path)
        assert client.post(f"/api/archive/packages/{package['package_id']}/restore", json={}).status_code == 409
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("action", ["archive", "restore"])
@pytest.mark.parametrize("identity, expected", [("missing", 404)])
def test_invalid_and_missing_names(tmp_path: Path, action: str, identity: str, expected: int) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        url = (f"/api/schemes/{identity}/archive" if action == "archive" else
               f"/api/archive/packages/{identity}/restore")
        assert client.post(url, json={}).status_code == expected



def test_result_button_endpoint_archives_all_matching_results(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    result, ids = pair[0], {"a" * 32, "b" * 32}
    with _client(tmp_path, (sys.executable,)) as client:
        _saved(tmp_path, result)
        for run_id in ids:
            _bundle(tmp_path, result, run_id)
        response = client.post(f"/api/results/{'a' * 32}/archive", json={})
        assert response.status_code == 200, response.text
        assert set(response.json()["result_ids"]) == ids
        assert ids.isdisjoint(_ids(client, "/api/results"))
        assert result.scheme.scheme_id not in client.get("/api/schemes").json()["schemes"]


def test_existing_package_id_is_never_overwritten(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    result = pair[0]
    with _client(tmp_path, (sys.executable,)) as client:
        _saved(tmp_path, result)
        first = _archive(client, result.scheme.scheme_id)
        _saved(tmp_path, result)
        before = _tree(tmp_path)
        import uuid
        monkeypatch.setattr("aosr.gui.scheme_archive.uuid.uuid4", lambda: uuid.UUID(str(first["package_id"])))
        response = client.post(f"/api/schemes/{result.scheme.scheme_id}/archive", json={})
        assert response.status_code == 409
        assert "已存在，沒有搬動任何檔案" in response.json()["error"]
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("collision", ["results", "runs", "stderr", "snapshot"])
def test_legacy_restore_conflict_keeps_both_sides(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], collision: str) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path / "archive", pair[0], run_id)
        if collision == "snapshot":
            path = tmp_path / "runs" / run_id
            path.mkdir()
            (path / "keep").write_text("不能蓋")
        else:
            path = (_path(tmp_path, collision, run_id) if collision != "stderr" else
                    (tmp_path / "runs" / run_id).with_suffix(".stderr"))
            path.write_text("不能蓋")
        before = _tree(tmp_path)
        response = client.post(f"/api/archive/{run_id}/restore", json={})
        assert response.status_code == 409
        assert _tree(tmp_path) == before


def test_legacy_restore_move_failure_rolls_back(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path / "archive", pair[0], run_id)
        before, replace = _tree(tmp_path), os.replace

        def fail_state(src: Path, dst: Path) -> None:
            if src == _path(tmp_path / "archive", "runs", run_id):
                raise OSError("第二個搬動失敗")
            replace(src, dst)

        monkeypatch.setattr("aosr.gui.jobs.os.replace", fail_state)
        response = client.post(f"/api/archive/{run_id}/restore", json={})
        assert response.status_code == 500
        assert "已搬回原處" in response.json()["error"]
        assert _tree(tmp_path) == before


def test_running_scheme_with_no_result_yet_is_not_archived(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    result = pair[0]
    script = tmp_path / "hold"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    with _client(tmp_path, (sys.executable, str(script))) as client:
        run_id = _start(client, result)
        try:
            before = _tree(tmp_path)
            response = client.post(f"/api/schemes/{result.scheme.scheme_id}/archive", json={})
            assert response.status_code == 409
            assert "正在計算，不能封存" in response.json()["error"]
            assert _tree(tmp_path) == before
            assert not _path(tmp_path, "results", run_id).exists()
        finally:
            assert client.post(f"/api/runs/{run_id}/stop", json={}).status_code == 200


def test_restore_freezes_only_completed_visible_results(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    result = pair[0]
    name, run_id = result.scheme.scheme_id, "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _saved(tmp_path, result)
        _bundle(tmp_path, result, run_id)
        package = _archive(client, name)
        _restore(client, package)
        changed = result.scheme.model_dump(mode="json")
        changed["scene"]["room_m"]["Lx"] += 0.1
        assert client.put(f"/api/schemes/{name}", json=changed).status_code == 409
        _path(tmp_path, "runs", run_id).write_text('{"status":"failed"}')
        assert client.put(f"/api/schemes/{name}", json=changed).status_code == 200
        row = next(item for item in client.get("/api/results").json()["results"] if item["run_id"] == run_id)
        assert row["scheme_text"] == "改之前的方案算的"


@pytest.mark.parametrize("with_result", [False, True])
def test_running_scheme_is_blocked_when_status_display_cannot_be_read(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch,
        with_result: bool) -> None:
    result = pair[0]
    script = tmp_path / "hold"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    with _client(tmp_path, (sys.executable, str(script))) as client:
        run_id = _start(client, result)
        try:
            if with_result:
                save_result(result, _path(tmp_path, "results", run_id))
            before = _tree(tmp_path)
            get = JobManager.get

            def unreadable(manager: JobManager, identity: str) -> dict[str, object]:
                if identity == run_id:
                    raise OSError("計算進度讀取失敗")
                return get(manager, identity)

            with monkeypatch.context() as patch:
                patch.setattr(JobManager, "get", unreadable)
                response = client.post(f"/api/schemes/{result.scheme.scheme_id}/archive", json={})
            assert response.status_code == 409, response.text
            assert "正在計算，不能封存" in response.json()["error"]
            assert _tree(tmp_path) == before
        finally:
            # 紅燈時檔案可能已被錯搬：先從暫存包找回紀錄，再確實停掉本題自己的行程。
            if not _path(tmp_path, "runs", run_id).is_file():
                for path in (tmp_path / "archive" / "packages").rglob(f"{run_id}.json"):
                    if path.parent.name == "runs":
                        path.replace(_path(tmp_path, "runs", run_id))
            assert client.post(f"/api/runs/{run_id}/stop", json={}).status_code == 200
