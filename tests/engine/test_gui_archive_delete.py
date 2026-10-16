"""永久刪除只動確認的封存成員，失敗留清單供重試。"""
from __future__ import annotations

import errno
import json
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.gui.jobs import JobManager
from aosr.gui.scheme_archive import ArchivePackage, SchemeArchive
from aosr.reporting.result import SchemeResult
from tests.engine._gui_cache import gui_startup_identity_memo
from tests.engine.test_gui_archive import _archive, _bundle, _saved, _tree
from tests.engine.test_gui_compare_routes import pair
from tests.engine.test_gui_failed_results import _client


def _package(client: TestClient, root: Path, result: SchemeResult, run_id: str = "a" * 32) -> Path:
    _saved(root, result)
    _bundle(root, result, run_id)
    package = _archive(client, result.scheme.scheme_id)
    return root / "archive" / "packages" / str(package["package_id"])


def _damage(folder: Path, kind: str) -> None:
    manifest = folder / "manifest.json"
    if kind == "broken":
        manifest.write_text("{")
    elif kind == "extra":
        (folder / "extra").write_text("不在清單")
    else:
        doc = json.loads(manifest.read_text())
        doc["status"] = "incomplete"
        manifest.write_text(json.dumps(doc))


def test_delete_package_preserves_every_other_byte(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        other = _package(client, tmp_path, pair[1], "b" * 32)
        _bundle(tmp_path / "archive", pair[0], "c" * 32)
        _saved(tmp_path, pair[0])
        _bundle(tmp_path, pair[0], "d" * 32)
        (tmp_path / "searches").mkdir(exist_ok=True)
        (tmp_path / "searches" / "keep").write_bytes(b"\x00\xff")
        before = {p: data for p, data in _tree(tmp_path).items() if not (tmp_path / p).is_relative_to(folder)}
        response = client.post(f"/api/archive/packages/{folder.name}/delete", json={})
        assert response.status_code == 200, response.text
        assert "已永久刪除封存的" in response.json()["message"]
        assert not folder.exists()
        assert _tree(tmp_path) == before
        assert {row["package_id"] for row in client.get("/api/archive").json()["packages"]} == {other.name}


@pytest.mark.parametrize("kind", ["incomplete", "broken", "extra"])
def test_unsafe_package_changes_nothing(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], kind: str) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        _damage(folder, kind)
        before = _tree(tmp_path)
        response = client.post(f"/api/archive/packages/{folder.name}/delete", json={})
        assert response.status_code == 409, response.text
        assert "沒有刪任何檔案" in response.json()["error"]
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("identity", ["A" * 32, "a" * 31, "..", "aa%2Fbb"])
@pytest.mark.parametrize("kind", ["packages/", ""])
def test_invalid_delete_ids_touch_nothing(tmp_path: Path, identity: str, kind: str) -> None:
    (tmp_path / "keep").write_bytes(b"keep")
    with _client(tmp_path, (sys.executable,)) as client:
        before = _tree(tmp_path)
        response = client.post(f"/api/archive/{kind}{identity}/delete", json={})
        assert not response.is_success
        assert _tree(tmp_path) == before


@pytest.mark.parametrize("link_kind", ["member", "snapshot", "package"])
def test_symlinks_never_delete_targets(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], link_kind: str) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        target = tmp_path / "own-target"
        if link_kind == "package":
            folder.rename(target)
            folder.symlink_to(target, target_is_directory=True)
        else:
            member = folder / "results" / f"{'a' * 32}.json"
            target.write_bytes(member.read_bytes())
            if link_kind == "member":
                member.unlink()
                member.symlink_to(target)
            else:
                (folder / "runs" / ("a" * 32) / "link").symlink_to(target)
        before = _tree(target) if target.is_dir() else target.read_bytes()
        response = client.post(f"/api/archive/packages/{folder.name}/delete", json={})
        assert response.status_code == (409 if link_kind == "package" else 200), response.text
        assert (_tree(target) if target.is_dir() else target.read_bytes()) == before
        assert folder.is_symlink() if link_kind == "package" else not folder.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_partial_failure_keeps_listing_then_retry_deletes(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch, legacy: bool) -> None:
    run_id = "a" * 32
    with _client(tmp_path, (sys.executable,)) as client:
        if legacy:
            _bundle(tmp_path / "archive", pair[0], run_id)
            folder = tmp_path / "archive"
            url = f"/api/archive/{run_id}"
        else:
            folder = _package(client, tmp_path, pair[0])
            url = f"/api/archive/packages/{folder.name}"
        unlink = Path.unlink
        deleted: list[Path] = []

        def fail_second(path: Path, missing_ok: bool = False) -> None:
            if path.is_relative_to(folder):
                if deleted:
                    raise PermissionError(errno.EACCES, "第二個刪不掉")
                deleted.append(path)
            unlink(path, missing_ok=missing_ok)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail_second)
            response = client.post(f"{url}/delete", json={})
        assert response.status_code == 500, response.text
        assert "刪除" in response.json()["error"] and "搬動" not in response.json()["error"]
        assert "還剩" in response.json()["error"] and "可以再刪" in response.json()["error"]
        listing = client.get("/api/archive").json()
        if legacy:
            assert {row["run_id"] for row in listing["results"]} == {run_id}
        else:
            row = next(row for row in listing["packages"] if row["package_id"] == folder.name)
            assert row["restore_available"] is False and row["delete_available"] is True
            assert client.post(f"{url}/restore", json={}).status_code == 404
        assert client.post(f"{url}/delete", json={}).status_code == 200
        assert client.get("/api/archive").json()["results" if legacy else "packages"] == []


def test_legacy_delete_four_members_and_missing(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        _bundle(tmp_path / "archive", pair[0], run_id)
        response = client.post(f"/api/archive/{run_id}/delete", json={})
        assert response.status_code == 200, response.text
        assert "舊的單筆封存" in response.json()["message"]
        assert _tree(tmp_path / "archive") == {}
        response = client.post(f"/api/archive/{run_id}/delete", json={})
        assert response.status_code == 404 and response.json()["error"] == "封存的結果找不到"


def test_preview_and_clear_only_confirmed_ids(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        chosen = _package(client, tmp_path, pair[0])
        incomplete = _package(client, tmp_path, pair[1], "b" * 32)
        _damage(incomplete, "incomplete")
        _bundle(tmp_path / "archive", pair[0], "c" * 32)
        listing = client.get("/api/archive").json()
        preview = listing["clear_preview"]
        assert set(preview["package_ids"]) == {row["package_id"] for row in listing["packages"] if row["delete_available"]}
        assert set(preview["run_ids"]) == {row["run_id"] for row in listing["results"]}
        assert all(row["size_text"] for row in listing["packages"] + listing["results"])
        assert preview["size_text"] and preview["skipped"]
        later = _package(client, tmp_path, pair[0], "d" * 32)
        response = client.post("/api/archive/clear", json={
            "package_ids": preview["package_ids"] + [incomplete.name, "f" * 32], "run_ids": preview["run_ids"]})
        assert response.status_code == 200, response.text
        data = response.json()
        assert set(data["deleted_packages"]) == {chosen.name}
        assert set(data["deleted_results"]) == {"c" * 32}
        assert {row["id"] for row in data["skipped"]} == {incomplete.name, "f" * 32}
        assert data["failed"] == []
        assert later.exists() and incomplete.exists() and not chosen.exists()


@pytest.mark.parametrize("body", [None, [], {}, {"package_ids": [], "run_ids": [1]},
    {"package_ids": ["a" * 32, "a" * 32], "run_ids": []}, {"package_ids": [], "run_ids": [".."]}])
def test_clear_bad_body_deletes_nothing(
        tmp_path: Path, body: object, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        if isinstance(body, dict) and isinstance(body.get("package_ids"), list):
            body = {**body, "package_ids": [folder.name for _ in body["package_ids"]]}
        before = _tree(tmp_path)
        assert client.post("/api/archive/clear", content=json.dumps(body), headers={"content-type": "application/json"}).status_code == 400
        assert _tree(tmp_path) == before


def test_clear_continues_after_failure_and_deletes_under_shared_lock(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        first = _package(client, tmp_path, pair[0])
        second = _package(client, tmp_path, pair[1], "b" * 32)
        _bundle(tmp_path / "archive", pair[0], "c" * 32)
        lock = JobManager(tmp_path, (), "a" * 40, tmp_path / "unused")._lock
        # 從這個伺服器的封存物件捕捉實際共用的鎖。
        original_load = SchemeArchive._load
        unlink = Path.unlink

        def load(archive: SchemeArchive, package_id: str) -> ArchivePackage:
            nonlocal lock
            lock = archive.jobs._lock
            return original_load(archive, package_id)

        def guarded(path: Path, missing_ok: bool = False) -> None:
            assert getattr(lock, "_is_owned")()
            if path.is_relative_to(first) and path.parent.name == "results":
                raise PermissionError(errno.EACCES, "這一筆被拒")
            unlink(path, missing_ok=missing_ok)

        monkeypatch.setattr(SchemeArchive, "_load", load)
        monkeypatch.setattr(Path, "unlink", guarded)
        response = client.post("/api/archive/clear", json={"package_ids": [first.name, second.name], "run_ids": ["c" * 32]})
        assert response.status_code == 200, response.text
        data = response.json()
        assert {row["id"] for row in data["failed"]} == {first.name}
        assert set(data["deleted_packages"]) == {second.name}
        assert set(data["deleted_results"]) == {"c" * 32}
        assert first.exists() and not second.exists()


@pytest.mark.parametrize("legacy", [False, True])
def test_snapshot_directory_link_is_unlinked_without_traversal(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], legacy: bool) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        if legacy:
            _bundle(tmp_path / "archive", pair[0], run_id)
            folder, url = tmp_path / "archive", f"/api/archive/{run_id}/delete"
        else:
            folder = _package(client, tmp_path, pair[0])
            url = f"/api/archive/packages/{folder.name}/delete"
        snapshot = folder / "runs" / run_id
        target = tmp_path / "own-snapshot-target"
        snapshot.rename(target)
        snapshot.symlink_to(target, target_is_directory=True)
        before = _tree(target)
        assert client.post(url, json={}).status_code == 200
        assert not snapshot.is_symlink() and _tree(target) == before


@pytest.mark.parametrize("parent_name", ["schemes", "results", "runs"])
def test_unlisted_parent_link_is_rejected_before_any_deletion(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], parent_name: str) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        origin = folder / parent_name
        target = tmp_path / "own-parent-target"
        origin.rename(target)
        origin.symlink_to(target, target_is_directory=True)
        before, target_before = _tree(tmp_path), _tree(target)
        response = client.post(f"/api/archive/packages/{folder.name}/delete", json={})
        assert response.status_code == 409, response.text
        assert _tree(tmp_path) == before and _tree(target) == target_before


@pytest.mark.parametrize("marker_failure", [True, False])
def test_last_cleanup_failure_keeps_manifest_for_retry(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch,
        marker_failure: bool) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        marker = folder / "manifest.json"
        unlink, rmdir = Path.unlink, Path.rmdir

        def fail_marker(path: Path, missing_ok: bool = False) -> None:
            if path == marker:
                raise PermissionError(errno.EACCES, "清單刪不掉")
            unlink(path, missing_ok=missing_ok)

        def fail_folder(path: Path) -> None:
            if path == folder:
                raise PermissionError(errno.EACCES, "空資料夾刪不掉")
            rmdir(path)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail_marker if marker_failure else unlink)
            patch.setattr(Path, "rmdir", rmdir if marker_failure else fail_folder)
            response = client.post(f"/api/archive/packages/{folder.name}/delete", json={})
        assert response.status_code == 500 and marker.is_file()
        row = next(row for row in client.get("/api/archive").json()["packages"] if row["package_id"] == folder.name)
        assert row["restore_available"] is False and row["delete_available"] is True
        assert client.post(f"/api/archive/packages/{folder.name}/delete", json={}).status_code == 200
        assert not folder.exists()


def test_clear_rejects_missing_json_and_malformed_json(tmp_path: Path) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        before = _tree(tmp_path)
        assert client.post("/api/archive/clear").status_code == 400
        assert client.post("/api/archive/clear", content="{", headers={"content-type": "application/json"}).status_code == 400
        assert _tree(tmp_path) == before


def test_delete_does_not_read_through_result_link(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    import aosr.gui.scheme_archive as archive_module

    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        member = folder / "results" / f"{'a' * 32}.json"
        target = tmp_path / "own-target.json"
        member.rename(target)
        member.symlink_to(target)
        identify = archive_module._scheme_id

        def guarded(path: Path, *, result: bool = False) -> str | None:
            assert not path.is_symlink(), "刪除不准跟連結去讀目標"
            return identify(path, result=result)

        monkeypatch.setattr(archive_module, "_scheme_id", guarded)
        before = target.read_bytes()
        assert client.post(f"/api/archive/packages/{folder.name}/delete", json={}).status_code == 200
        assert target.read_bytes() == before
