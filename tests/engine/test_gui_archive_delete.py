"""永久刪除只動確認的封存成員，失敗留清單供重試。"""
from __future__ import annotations

import errno
import json
import re
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aosr.gui.archive_delete import ArchiveDelete
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


@pytest.mark.parametrize("identity", ["A" * 32, "a" * 31])
@pytest.mark.parametrize("kind", ["packages/", ""])
def test_invalid_delete_ids_touch_nothing(tmp_path: Path, identity: str, kind: str) -> None:
    (tmp_path / "keep").write_bytes(b"keep")
    with _client(tmp_path, (sys.executable,)) as client:
        before = _tree(tmp_path)
        response = client.post(f"/api/archive/{kind}{identity}/delete", json={})
        assert response.status_code == 404
        assert response.json()["error"] == ("封存包找不到" if kind else "封存的結果找不到")
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
        assert re.search(r"已刪 \d+ 項，還剩 \d+ 項", response.json()["error"]), response.json()["error"]
        listing = client.get("/api/archive").json()
        if legacy:
            row = next(row for row in listing["results"] if row["run_id"] == run_id)
            assert row["restore_available"] is False
            assert row["status_text"] == "這一筆刪到一半，不能搬回；可以按永久刪除，或請助理檢查"
            assert f"results/{run_id}.json" in response.json()["error"]
            assert f"deleting/{run_id}" in response.json()["error"]
            before = _tree(tmp_path)
            restored = client.post(f"{url}/restore", json={})
            assert restored.status_code == 409 and restored.json()["error"] == row["status_text"]
            assert _tree(tmp_path) == before
        else:
            row = next(row for row in listing["packages"] if row["package_id"] == folder.name)
            assert row["restore_available"] is False and row["delete_available"] is True
            assert client.post(f"{url}/restore", json={}).status_code == 404
        assert client.post(f"{url}/delete", json={}).status_code == 200
        assert client.get("/api/archive").json()["results" if legacy else "packages"] == []
        assert not (tmp_path / "archive" / "deleting").exists()


def test_legacy_delete_four_members_and_missing(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        _bundle(tmp_path / "archive", pair[0], run_id)
        _bundle(tmp_path / "archive", pair[0], "b" * 32)
        _package(client, tmp_path, pair[1], "c" * 32)
        _bundle(tmp_path, pair[0], run_id)
        relatives = [Path("archive/results") / f"{run_id}.json", Path("archive/runs") / f"{run_id}.json",
                     Path("archive/runs") / f"{run_id}.stderr", Path("archive/runs") / run_id]
        before = {p: data for p, data in _tree(tmp_path).items()
                  if not any(p == r or r in p.parents for r in relatives)}
        response = client.post(f"/api/archive/{run_id}/delete", json={})
        assert response.status_code == 200, response.text
        assert "舊的單筆封存" in response.json()["message"]
        assert _tree(tmp_path) == before
        assert not (tmp_path / "archive" / "runs" / run_id).exists()
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


@pytest.mark.parametrize("body", [None, [], {}, {"package_ids": ["a" * 32], "run_ids": [1]},
    {"package_ids": ["a" * 32, "a" * 32], "run_ids": []}, {"package_ids": ["a" * 32], "run_ids": [".."]},
    {"package_ids": ["a" * 32], "run_ids": [], "extra": True}, {"package_ids": "a" * 32, "run_ids": []}])
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
        assert client.post("/api/archive/clear").status_code == 415
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


@pytest.mark.parametrize("identity", ["..", "aa/bb"])
@pytest.mark.parametrize("legacy", [False, True])
def test_invalid_direct_delete_ids(tmp_path: Path, identity: str, legacy: bool) -> None:
    deletion = ArchiveDelete(SchemeArchive(JobManager(tmp_path, (), "a" * 40, tmp_path / "unused")))
    (tmp_path / "keep").write_bytes(b"keep")
    before = _tree(tmp_path)
    with pytest.raises(FileNotFoundError, match="封存的結果找不到" if legacy else "封存包找不到"):
        deletion.result(identity, []) if legacy else deletion.package(identity)
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("parent_name", ["archive", "archive/results", "archive/runs", "archive/packages", "archive/deleting"])
@pytest.mark.parametrize("legacy", [False, True])
def test_archive_parent_links_reject_both_deletes(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], parent_name: str, legacy: bool) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        run_id = "b" * 32
        _bundle(tmp_path / "archive", pair[1], run_id)
        (tmp_path / "archive/deleting").mkdir()
        origin, target = tmp_path / parent_name, tmp_path / "own-parent-target"
        origin.rename(target)
        origin.symlink_to(target, target_is_directory=True)
        before, target_before = _tree(tmp_path), _tree(target)
        url = f"/api/archive/{run_id}" if legacy else f"/api/archive/packages/{folder.name}"
        response = client.post(f"{url}/delete", json={})
        assert response.status_code == 409, response.text
        assert _tree(tmp_path) == before and _tree(target) == target_before


def test_tampered_scheme_id_deletes_nothing(tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        _bundle(tmp_path, pair[0], "a" * 32)
        package = _archive(client, pair[0].scheme.scheme_id)
        manifest = tmp_path / "archive/packages" / str(package["package_id"]) / "manifest.json"
        doc = json.loads(manifest.read_text())
        doc["scheme_id"] = "other-scheme"
        manifest.write_text(json.dumps(doc))
        before = _tree(tmp_path)
        response = client.post(f"/api/archive/packages/{package['package_id']}/delete", json={})
        assert response.status_code == 409, response.text
        assert "對不上" in response.json()["error"]
        assert _tree(tmp_path) == before


def test_clear_skips_restored_legacy_and_keeps_unconfirmed_legacy(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        chosen = _package(client, tmp_path, pair[0])
        run_id, later = "c" * 32, "d" * 32
        _bundle(tmp_path / "archive", pair[1], run_id)
        preview = client.get("/api/archive").json()["clear_preview"]
        assert client.post(f"/api/archive/{run_id}/restore", json={}).status_code == 200
        _bundle(tmp_path / "archive", pair[1], later)
        before = {p: data for p, data in _tree(tmp_path).items() if not (tmp_path / p).is_relative_to(chosen)}
        response = client.post("/api/archive/clear", json={"package_ids": preview["package_ids"], "run_ids": preview["run_ids"]})
        assert response.status_code == 200, response.text
        assert {row["id"] for row in response.json()["skipped"]} == {run_id}
        assert _tree(tmp_path) == before
        assert (tmp_path / "archive/results" / f"{later}.json").is_file()


@pytest.mark.parametrize("legacy", [False, True])
def test_single_delete_unlink_and_rmdir_hold_shared_lock(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch, legacy: bool) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        folder = _package(client, tmp_path, pair[0])
        if legacy:
            _bundle(tmp_path / "archive", pair[1], run_id)
        locks: list[object] = []
        original = ArchiveDelete.result if legacy else ArchiveDelete.package
        unlink, rmdir = Path.unlink, Path.rmdir
        operations: set[str] = set()

        def capture(deletion: ArchiveDelete, identity: str, *args: list[dict[str, object]]) -> dict[str, object]:
            locks.append(deletion.jobs._lock)
            return original(deletion, identity, *args)

        def guarded_unlink(path: Path, missing_ok: bool = False) -> None:
            if path.is_relative_to(tmp_path / "archive"):
                assert getattr(locks[0], "_is_owned")()
                operations.add("unlink")
            unlink(path, missing_ok=missing_ok)

        def guarded_rmdir(path: Path) -> None:
            if path.is_relative_to(tmp_path / "archive"):
                assert getattr(locks[0], "_is_owned")()
                operations.add("rmdir")
            rmdir(path)

        with monkeypatch.context() as patch:
            patch.setattr(ArchiveDelete, "result" if legacy else "package", capture)
            patch.setattr(Path, "unlink", guarded_unlink)
            patch.setattr(Path, "rmdir", guarded_rmdir)
            url = f"/api/archive/{run_id}" if legacy else f"/api/archive/packages/{folder.name}"
            response = client.post(f"{url}/delete", json={})
        assert response.status_code == 200, response.text
        assert operations == {"unlink", "rmdir"}


@pytest.mark.parametrize("size, expected", [(0, "不到 0.1 MB"), (99_999, "不到 0.1 MB"),
    (100_000, "約 0.1 MB"), (278_400_000, "約 278.4 MB")])
def test_size_text_boundaries(size: int, expected: str) -> None:
    from aosr.gui.archive_delete import size_text
    assert size_text(size) == expected


def test_size_bytes_counts_link_itself(tmp_path: Path) -> None:
    from aosr.gui.archive_delete import size_bytes
    target, link = tmp_path / "large", tmp_path / "link"
    target.write_bytes(b"large" * 100_000)
    link.symlink_to(target)
    assert size_bytes([link]) == link.lstat().st_size


def test_unreadable_package_size_keeps_other_rows(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult]) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        other = _package(client, tmp_path, pair[1], "b" * 32)
        blocked = folder / "runs" / ("a" * 32)
        blocked.chmod(0)
        try:
            response = client.get("/api/archive")
            assert response.status_code == 200, response.text
            rows = {row["package_id"]: row for row in response.json()["packages"]}
            assert set(rows) == {folder.name, other.name}
            assert rows[folder.name]["size_text"] == "大小讀不出"
            assert rows[other.name]["size_text"] != "大小讀不出"
            preview = response.json()["clear_preview"]
            assert preview["size_text"].startswith("至少約 ") and "（1 項大小讀不出）" in preview["size_text"]
        finally:
            blocked.chmod(0o700)


@pytest.mark.parametrize("failure", ["load", "legacy_size"])
def test_listing_read_failure_is_local_to_one_row(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    import aosr.gui.archive_delete as module
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        _bundle(tmp_path / "archive", pair[1], "b" * 32)
        load, measure = SchemeArchive._load, module.size_bytes
        loaded: list[str] = []

        def fail_load(archive: SchemeArchive, identity: str) -> ArchivePackage:
            if identity in loaded:
                raise ValueError("第二次清單讀不出")
            loaded.append(identity)
            return load(archive, identity)

        def fail_size(paths: list[Path]) -> int:
            if any(path.parent == tmp_path / "archive/results" for path in paths):
                raise OSError("舊單筆大小讀不出")
            return measure(paths)

        monkeypatch.setattr(SchemeArchive, "_load", fail_load if failure == "load" else load)
        monkeypatch.setattr(module, "size_bytes", fail_size if failure == "legacy_size" else measure)
        response = client.get("/api/archive")
        assert response.status_code == 200, response.text
        data = response.json()
        row = data["packages"][0] if failure == "load" else data["results"][0]
        assert row["size_text"] == "大小讀不出"
        assert data["packages"][0]["package_id"] == folder.name
        assert data["results"][0]["run_id"] == "b" * 32
        assert "（1 項大小讀不出）" in data["clear_preview"]["size_text"]


def test_legacy_marker_write_failure_deletes_nothing(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        _bundle(tmp_path / "archive", pair[0], run_id)
        marker = tmp_path / "archive/deleting" / run_id
        marker.parent.mkdir()
        before = _tree(tmp_path)
        touch = Path.touch

        def fail(path: Path, mode: int = 0o666, exist_ok: bool = True) -> None:
            if path == marker:
                raise PermissionError(errno.EACCES, "標記寫不出")
            touch(path, mode=mode, exist_ok=exist_ok)

        monkeypatch.setattr(Path, "touch", fail)
        response = client.post(f"/api/archive/{run_id}/delete", json={})
        assert response.status_code == 500, response.text
        assert _tree(tmp_path) == before


def test_failed_manifest_rewrite_reports_disappeared_row(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        rmdir, write = Path.rmdir, Path.write_bytes

        def fail_rmdir(path: Path) -> None:
            if path == folder:
                raise PermissionError(errno.EACCES, "資料夾刪不掉")
            rmdir(path)

        def fail_write(path: Path, data: bytes) -> int:
            if path == folder / "manifest.json":
                raise OSError(errno.ENOSPC, "清單寫不回")
            return write(path, data)

        monkeypatch.setattr(Path, "rmdir", fail_rmdir)
        monkeypatch.setattr(Path, "write_bytes", fail_write)
        response = client.post(f"/api/archive/packages/{folder.name}/delete", json={})
        assert response.status_code == 500, response.text
        assert response.json()["error"] == "清單已刪、資料夾沒清掉，畫面不會再列這一包；請助理檢查"
        assert folder.is_dir() and not (folder / "manifest.json").exists()
        assert client.get("/api/archive").json()["packages"] == []


def test_errno_missing_package_reports_already_gone(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        read = Path.read_bytes

        def gone(path: Path) -> bytes:
            if path == folder / "manifest.json":
                raise FileNotFoundError(errno.ENOENT, "鎖外先刪了")
            return read(path)

        before = _tree(tmp_path)
        monkeypatch.setattr(Path, "read_bytes", gone)
        response = client.post(f"/api/archive/packages/{folder.name}/delete", json={})
        assert response.status_code == 404 and response.json()["error"] == "封存項目已經不在"
        monkeypatch.undo()
        assert _tree(tmp_path) == before


def test_legacy_marker_precedes_deletion_and_is_removed_last(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        _bundle(tmp_path / "archive", pair[0], run_id)
        root = tmp_path / "archive"
        marker, result = root / "deleting" / run_id, root / "results" / f"{run_id}.json"
        unlink, rmdir = Path.unlink, Path.rmdir
        deleted: list[Path] = []

        def guard_unlink(path: Path, missing_ok: bool = False) -> None:
            if path.is_relative_to(root):
                assert marker.is_file()
                if path != marker:
                    assert result.is_file()
                else:
                    assert not result.exists()
                deleted.append(path)
            unlink(path, missing_ok=missing_ok)

        def guard_rmdir(path: Path) -> None:
            if path.is_relative_to(root / "runs"):
                assert marker.is_file() and result.is_file()
            rmdir(path)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", guard_unlink)
            patch.setattr(Path, "rmdir", guard_rmdir)
            response = client.post(f"/api/archive/{run_id}/delete", json={})
        assert response.status_code == 200, response.text
        assert deleted[-2:] == [result, marker]
        assert not marker.parent.exists()


@pytest.mark.parametrize("identity", ["..", "aa/bb"])
def test_invalid_legacy_id_with_decoys_deletes_nothing(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], identity: str) -> None:
    """代號檢查拿掉時，「..」會把 runs/.. 也就是整個封存區列進要刪的清單；放誘餌讓它真的刪得到。"""
    with _client(tmp_path, (sys.executable,)) as client:
        _package(client, tmp_path, pair[0])
        _bundle(tmp_path / "archive", pair[1], "b" * 32)
    decoy = tmp_path / "archive/results" / f"{identity}.json"
    decoy.parent.mkdir(parents=True, exist_ok=True)
    decoy.write_bytes(b"{}")
    deletion = ArchiveDelete(SchemeArchive(JobManager(tmp_path, (), "a" * 40, tmp_path / "unused")))
    before = _tree(tmp_path)
    with pytest.raises(FileNotFoundError, match="封存的結果找不到"):
        deletion.result(identity, [])
    assert _tree(tmp_path) == before


def test_leftover_empty_deleting_folder_still_reports_deleted(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    """全部刪完、只剩空的 deleting 資料夾刪不掉時，照實報已刪，不報未完成。"""
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        _bundle(tmp_path / "archive", pair[0], run_id)
        parent = tmp_path / "archive/deleting"
        rmdir = Path.rmdir

        def fail(path: Path) -> None:
            if path == parent:
                raise PermissionError(errno.EACCES, "空資料夾刪不掉")
            rmdir(path)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "rmdir", fail)
            response = client.post(f"/api/archive/{run_id}/delete", json={})
        assert response.status_code == 200, response.text
        assert "已永久刪除" in response.json()["message"]
        assert parent.is_dir() and not any(parent.iterdir())
        assert client.get("/api/archive").json()["results"] == []


def test_orphan_marker_failure_says_already_deleted(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    """結果檔已刪、只剩標記刪不掉：那一列已經不在，不准說「還列在封存區，可以再刪」。"""
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        _bundle(tmp_path / "archive", pair[0], run_id)
        marker = tmp_path / "archive/deleting" / run_id
        unlink = Path.unlink

        def fail(path: Path, missing_ok: bool = False) -> None:
            if path == marker:
                raise PermissionError(errno.EACCES, "標記刪不掉")
            unlink(path, missing_ok=missing_ok)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail)
            response = client.post(f"/api/archive/{run_id}/delete", json={})
        assert response.status_code == 500, response.text
        assert response.json()["error"] == f"已刪除；只剩刪除標記 deleting/{run_id} 沒清掉，不影響清單；請助理檢查"
        assert marker.is_file() and not (tmp_path / "archive/results" / f"{run_id}.json").exists()
        assert client.get("/api/archive").json()["results"] == []


def test_clear_skipped_reason_is_plain_chinese(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    """全部清除跳過的理由跟單刪一樣白話，不帶系統訊息與完整路徑。"""
    with _client(tmp_path, (sys.executable,)) as client:
        folder = _package(client, tmp_path, pair[0])
        preview = client.get("/api/archive").json()["clear_preview"]
        read = Path.read_bytes

        def gone(path: Path) -> bytes:
            if path == folder / "manifest.json":
                raise FileNotFoundError(errno.ENOENT, "鎖外先刪了", str(path))
            return read(path)

        before = _tree(tmp_path)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", gone)
            response = client.post("/api/archive/clear",
                                   json={"package_ids": preview["package_ids"], "run_ids": preview["run_ids"]})
        assert response.status_code == 200, response.text
        assert response.json()["skipped"] == [{"kind": "package", "id": folder.name, "reason": "封存項目已經不在"}]
        assert _tree(tmp_path) == before


def test_unreadable_deleting_marker_blocks_restore_without_breaking_listing(
        tmp_path: Path, pair: tuple[SchemeResult, SchemeResult], monkeypatch: pytest.MonkeyPatch) -> None:
    """刪除標記讀不出時，清單照列、那一筆不給搬回，搬回端點也擋下，不回系統訊息。"""
    with _client(tmp_path, (sys.executable,)) as client:
        run_id = "a" * 32
        _bundle(tmp_path / "archive", pair[0], run_id)
        deleting = tmp_path / "archive/deleting"
        exists = Path.exists

        def blocked(path: Path, *, follow_symlinks: bool = True) -> bool:
            if path.is_relative_to(deleting):
                raise PermissionError(errno.EACCES, "讀不進去", str(path))
            return exists(path, follow_symlinks=follow_symlinks)

        before = _tree(tmp_path)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "exists", blocked)
            listing = client.get("/api/archive")
            restored = client.post(f"/api/archive/{run_id}/restore", json={})
        assert listing.status_code == 200, listing.text
        row = next(row for row in listing.json()["results"] if row["run_id"] == run_id)
        assert row["restore_available"] is False
        assert row["status_text"] == "這一筆刪到一半，不能搬回；可以按永久刪除，或請助理檢查"
        assert restored.status_code == 409 and restored.json()["error"] == row["status_text"]
        assert _tree(tmp_path) == before
