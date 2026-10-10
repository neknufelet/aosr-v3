"""整包封存中斷、壞紀錄與被改過的清單，都必須留得下可救的資料。"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import cast

import pytest

from aosr.gui.jobs import JobManager
from aosr.gui.scheme_archive import ArchivePackage, SchemeArchive
from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_archive import _archive, _bundle, _restore, _saved, _tree
from tests.engine.test_gui_compare_routes import pair
from tests.engine.test_gui_failed_results import _client, _path, _start


class Interrupted(BaseException):
    """像行程被砍，不經過一般例外的反向復原。"""


def test_manifest_write_failure_moves_nothing(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        scheme = _saved(tmp_path, pair[0])
        _bundle(tmp_path, pair[0], "a" * 32)
        before = _tree(tmp_path)

        def fail(archive: SchemeArchive, folder: Path, package: ArchivePackage) -> None:
            raise OSError("清單尚未寫出")

        monkeypatch.setattr(SchemeArchive, "_write_manifest", fail)
        response = client.post(f"/api/schemes/{scheme.stem}/archive", json={})
        assert response.status_code == 500, response.text
        assert "檔案沒有搬動" in response.json()["error"]
        assert _tree(tmp_path) == before
        assert client.get("/api/archive").json()["packages"] == []


def test_completion_write_and_rollback_failure_keeps_incomplete_package(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        scheme = _saved(tmp_path, pair[0])
        _bundle(tmp_path, pair[0], "a" * 32)
        before, write, replace = _tree(tmp_path), SchemeArchive._write_manifest, Path.replace

        def fail_complete(archive: SchemeArchive, folder: Path, package: ArchivePackage) -> None:
            if package.status == "complete":
                raise OSError("搬完後更新清單失敗")
            write(archive, folder, package)

        def fail_rollback(origin: Path, destination: Path) -> Path:
            if origin.is_relative_to(tmp_path / "archive" / "packages") and destination.name != "manifest.json":
                raise OSError("復原也失敗")
            return replace(origin, destination)

        with monkeypatch.context() as patch:
            patch.setattr(SchemeArchive, "_write_manifest", fail_complete)
            patch.setattr(Path, "replace", fail_rollback)
            response = client.post(f"/api/schemes/{scheme.stem}/archive", json={})
        assert response.status_code == 500, response.text
        assert "搬回都失敗" in response.json()["error"]
        package = client.get("/api/archive").json()["packages"][0]
        assert package["status"] == "incomplete"
        assert package["restore_available"] is True
        _restore(client, package)
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("stop_at", ["scheme", "state", "snapshot"])
def test_interrupted_archive_is_visible_and_restores_bytes(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch,
        stop_at: str) -> None:
    result, run_id = pair[0], "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, result, run_id)
        scheme = _saved(tmp_path, result)
        before, replace = _tree(tmp_path), Path.replace
        stops = {"scheme": scheme, "state": _path(tmp_path, "runs", run_id),
                 "snapshot": tmp_path / "runs" / run_id}

        def interrupt(origin: Path, destination: Path) -> Path:
            moved = replace(origin, destination)
            if origin == stops[stop_at]:
                raise Interrupted()
            return moved

        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", interrupt)
            with pytest.raises(Interrupted):
                SchemeArchive(JobManager(tmp_path, (), "a" * 40, tmp_path / "unused")).archive(
                    result.scheme.scheme_id, [run_id])
        listing = client.get("/api/archive")
        assert listing.status_code == 200, listing.text
        package = listing.json()["packages"][0]
        assert package["status"] == "incomplete"
        assert package["status_text"] == "封存沒做完，請助理檢查"
        assert set(package["paths"]) == {f"schemes/{result.scheme.scheme_id}.json",
                                         f"results/{run_id}.json", f"runs/{run_id}.json",
                                         f"runs/{run_id}.stderr", f"runs/{run_id}"}
        _restore(client, package)
        assert _tree(tmp_path) == before
        assert client.get("/api/archive").json()["packages"] == []


def test_incomplete_archive_duplicate_moves_nothing(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    result, run_id = pair[0], "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, result, run_id)
        scheme = _saved(tmp_path, result)
        replace = Path.replace

        def interrupt(origin: Path, destination: Path) -> Path:
            moved = replace(origin, destination)
            if origin == scheme:
                raise Interrupted()
            return moved

        with monkeypatch.context() as patch:
            patch.setattr(Path, "replace", interrupt)
            with pytest.raises(Interrupted):
                SchemeArchive(JobManager(tmp_path, (), "a" * 40, tmp_path / "unused")).archive(
                    result.scheme.scheme_id, [run_id])
        package = client.get("/api/archive").json()["packages"][0]
        scheme.write_text("原位另有一份，不能蓋", encoding="utf-8")
        before = _tree(tmp_path)
        response = client.post(f"/api/archive/packages/{package['package_id']}/restore", json={})
        assert response.status_code == 409, response.text
        assert "對不上" in response.json()["error"]
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("damage", ["broken", "missing", "false_finished"])
def test_live_process_and_damaged_state_block_archive(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], damage: str) -> None:
    result = pair[0]
    script = tmp_path / "hold"
    script.write_text("import time\nwhile True: time.sleep(0.05)\n")
    with _client(tmp_path, (sys.executable, str(script))) as client:
        run_id = _start(client, result)
        path = _path(tmp_path, "runs", run_id)
        original = path.read_bytes()
        try:
            if damage == "missing":
                path.unlink()
            elif damage == "broken":
                path.write_text("{", encoding="utf-8")
            else:
                state = json.loads(original)
                state["status"] = "failed"
                path.write_text(json.dumps(state))
            before = _tree(tmp_path)
            response = client.post(f"/api/schemes/{result.scheme.scheme_id}/archive", json={})
            assert response.status_code == 409, response.text
            assert "沒有搬動任何檔案" in response.json()["error"]
            assert _tree(tmp_path) == before
            assert client.get("/api/archive").json()["packages"] == []
        finally:
            path.write_bytes(original)
            assert client.post(f"/api/runs/{run_id}/stop", json={}).status_code == 200


def test_unreadable_state_without_process_blocks_all_archive(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        scheme = _saved(tmp_path, pair[0])
        _path(tmp_path, "runs", "a" * 32).write_text("{", encoding="utf-8")
        before = _tree(tmp_path)
        response = client.post(f"/api/schemes/{scheme.stem}/archive", json={})
        assert response.status_code == 409, response.text
        assert "計算紀錄讀不出" in response.json()["error"]
        assert "可能正在計算" in response.json()["error"]
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("identity_source", ["state", "snapshot"])
def test_broken_result_members_and_counts_use_all_identity_sources(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], identity_source: str) -> None:
    result, run_id, artifact_id = pair[0], "a" * 32, "b" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        scheme = _saved(tmp_path, result)
        _bundle(tmp_path, result, run_id, "failed" if identity_source == "state" else "none")
        _path(tmp_path, "results", run_id).write_bytes(b"{\xff")
        if identity_source == "state":
            (tmp_path / "runs" / run_id / "scheme.json").unlink()
        _path(tmp_path, "runs", artifact_id).write_text(json.dumps(
            {"scheme_id": scheme.stem, "status": "failed"}))
        # 手寫答案：壞結果一筆，另有只有紀錄的成員；快照資料夾仍一起搬。
        expected_paths = {f"schemes/{scheme.stem}.json", f"results/{run_id}.json",
                          f"runs/{run_id}.stderr", f"runs/{run_id}", f"runs/{artifact_id}.json"}
        if identity_source == "state":
            expected_paths.add(f"runs/{run_id}.json")
        expected_results = [run_id]
        before = _tree(tmp_path)
        preview_response = client.get(f"/api/schemes/{scheme.stem}/archive-preview")
        assert preview_response.status_code == 200, preview_response.text
        preview = preview_response.json()
        assert _tree(tmp_path) == before
        assert set(preview["paths"]) == expected_paths
        assert preview["result_ids"] == expected_results
        assert preview["result_count"] == len(expected_results)
        package = _archive(client, scheme.stem)
        assert set(cast(list[str], package["paths"])) == expected_paths
        assert set(cast(list[str], package["artifact_ids"])) == {run_id, artifact_id}
        assert package["result_ids"] == expected_results
        assert package["result_count"] == preview["result_count"]
        assert f"{len(expected_results)} 筆結果" in str(package["message"])
        folder = tmp_path / "archive" / "packages" / str(package["package_id"])
        expected_files = {key: value for key, value in before.items()
                          if str(key) in expected_paths or Path(f"runs/{run_id}") in key.parents}
        assert {key: value for key, value in _tree(folder).items() if key != Path("manifest.json")} == expected_files
        listing = client.get("/api/archive").json()["packages"][0]
        assert listing["result_count"] == preview["result_count"]
        _restore(client, package)
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("tamper", ["paths", "scheme_id", "extra_file"])
def test_tampered_package_restores_nothing(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], tamper: str) -> None:
    result, run_id = pair[0], "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, result, run_id)
        if tamper != "scheme_id":
            _saved(tmp_path, result)
        package = _archive(client, result.scheme.scheme_id)
        folder = tmp_path / "archive" / "packages" / str(package["package_id"])
        manifest = folder / "manifest.json"
        document = json.loads(manifest.read_text())
        if tamper == "paths":
            document["paths"] = [f"schemes/{result.scheme.scheme_id}.json"]
        elif tamper == "scheme_id":
            document["scheme_id"] = "other-scheme"
        else:
            (folder / "runs" / "extra.json").write_text("清單沒列到")
        manifest.write_text(json.dumps(document))
        before = _tree(tmp_path)
        response = client.post(f"/api/archive/packages/{package['package_id']}/restore", json={})
        assert response.status_code == 409, response.text
        assert "對不上" in response.json()["error"]
        assert _tree(tmp_path) == before
        assert client.get("/api/archive").json()["packages"]


def test_corrupt_manifest_keeps_healthy_packages_and_legacy_visible(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        _saved(tmp_path, pair[0])
        healthy = _archive(client, pair[0].scheme.scheme_id)
        _bundle(tmp_path / "archive", pair[1], "a" * 32)
        broken_id = "b" * 32
        folder = tmp_path / "archive" / "packages" / broken_id
        folder.mkdir()
        (folder / "manifest.json").write_text("{")
        before = _tree(tmp_path)
        response = client.get("/api/archive")
        assert response.status_code == 200, response.text
        listing = response.json()
        rows = {item["package_id"]: item for item in listing["packages"]}
        assert set(rows) == {healthy["package_id"], broken_id}
        assert rows[broken_id]["status_text"] == "這一包清單讀不出，請助理檢查"
        assert rows[broken_id]["restore_available"] is False
        assert rows[healthy["package_id"]]["restore_available"] is True
        assert {item["run_id"] for item in listing["results"]} == {"a" * 32}
        assert _tree(tmp_path) == before
        _restore(client, healthy)
        assert client.post(f"/api/archive/{'a' * 32}/restore", json={}).status_code == 200
