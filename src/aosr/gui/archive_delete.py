"""封存永久刪除：先核對成員、不跟連結、最後才刪列舉用的清單。"""
from __future__ import annotations

import stat
from collections.abc import Callable
from pathlib import Path
from typing import cast

from aosr.gui.jobs import LEGACY_DELETING_TEXT, ResultMoveConflict
from aosr.gui.scheme_archive import ArchivePackage, IDENTITY, SchemeArchive, _present, result_relatives


class DeleteConflict(ValueError):
    """尚未開始刪除，整筆跳過。"""


def _entries(path: Path) -> list[Path]:
    """先列後刪；lstat（只看連結自己）避免走進連結指的資料夾。"""
    if not _present(path):
        return []
    mode = path.lstat().st_mode
    if not stat.S_ISDIR(mode):
        return [path]
    return [item for child in sorted(path.iterdir()) for item in _entries(child)] + [path]


def size_bytes(paths: list[Path]) -> int:
    return sum(item.lstat().st_size for path in paths for item in _entries(path))


def size_text(size: int) -> str:
    mb = size / 1_000_000
    return "不到 0.1 MB" if mb < 0.1 else f"約 {mb:.1f} MB"


def _remove_entry(path: Path) -> None:
    if stat.S_ISDIR(path.lstat().st_mode):
        path.rmdir()
    else:
        path.unlink()


def _delete_entries(entries: list[Path], marker: Path, label: str) -> None:
    """保留列舉依據；檔案失敗不回滾已刪內容，照實列出剩下的路徑。"""
    deleted: list[Path] = []
    try:
        for path in entries:
            _remove_entry(path)
            deleted.append(path)
        marker.unlink()
    except OSError as exc:
        base = marker.parent if marker.name == "manifest.json" else marker.parent.parent
        removed = "、".join(str(path.relative_to(base)) for path in deleted) or "無"
        remaining = [str(path.relative_to(base)) for path in [*entries, marker] if _present(path)]
        raise OSError(f"刪除未完成：已刪 {len(deleted)} 項，還剩 {len(remaining)} 項；"
                      f"已刪：{removed}；未刪：{'、'.join(remaining) or marker.name}。{label}還列在封存區，可以再刪；請助理檢查") from exc


def _safe_parents(root: Path, folders: list[Path]) -> None:
    for folder in folders:
        if folder.is_symlink() or not folder.resolve().is_relative_to(root.resolve()):
            raise DeleteConflict("封存資料夾位置不安全，不能刪除；沒有刪任何檔案，請助理檢查")


class ArchiveDelete:
    def __init__(self, archive: SchemeArchive) -> None:
        self.archive = archive
        self.jobs = archive.jobs
        self.root = self.jobs.data_dir / "archive"

    def _safe_root(self) -> None:
        _safe_parents(self.root, [self.root, *(self.root / name for name in ("results", "runs", "packages", "deleting"))])

    def _package(self, package_id: str) -> tuple[Path, ArchivePackage, list[Path]]:
        if not IDENTITY.fullmatch(package_id):
            raise FileNotFoundError("封存包找不到")
        self._safe_root()
        folder = self.archive.packages_dir / package_id
        if not _present(folder):
            raise FileNotFoundError("封存包找不到")
        _safe_parents(self.root, [self.root, self.archive.packages_dir, folder])
        if not folder.is_dir():
            raise DeleteConflict("這一包不是資料夾，不能刪除；沒有刪任何檔案，請助理檢查")
        try:
            package = self.archive._load(package_id)
        except FileNotFoundError:
            raise
        except (OSError, ValueError) as exc:
            raise DeleteConflict("這一包清單讀不出，不能刪；請助理檢查，沒有刪任何檔案") from exc
        if package.status != "complete":
            raise DeleteConflict("這一包封存沒做完，不能刪；請助理檢查，沒有刪任何檔案")
        entries = self._package_entries(folder, package)
        try:
            self.archive._validate_contents(folder, package, follow_links=False)
        except (ResultMoveConflict, OSError) as exc:
            raise DeleteConflict("這一包成員與清單對不上，不能刪除；請助理檢查，沒有刪任何檔案") from exc
        return folder, package, entries

    def _package_entries(self, folder: Path, package: ArchivePackage) -> list[Path]:
        listed = {folder / path for path in package.paths}
        parents = {parent for path in listed for parent in path.parents if parent.is_relative_to(folder) and parent != folder}
        entries = [entry for child in sorted(folder.iterdir()) for entry in _entries(child)
                   if entry != folder / "manifest.json"]
        snapshots = {folder / "runs" / run_id for run_id in package.artifact_ids if f"runs/{run_id}" in package.paths}
        for path in entries:
            if path not in listed and not any(parent in snapshots for parent in path.parents):
                if path not in parents or path.is_symlink() or not path.is_dir():
                    raise DeleteConflict("這一包成員與清單對不上，不能刪除；請助理檢查，沒有刪任何檔案")
        return entries

    def package(self, package_id: str) -> dict[str, object]:
        with self.jobs._lock:
            folder, package, entries = self._package(package_id)
            marker = folder / "manifest.json"
            manifest = marker.read_bytes()
            _delete_entries(entries, marker, "這一包")
            try:
                folder.rmdir()
            except OSError as exc:
                try:
                    marker.write_bytes(manifest)
                except OSError as rewrite:
                    raise OSError("清單已刪、資料夾沒清掉，畫面不會再列這一包；請助理檢查") from rewrite
                raise OSError("刪除未完成：已刪全部成員，還剩空資料夾；這一包還列在封存區，可以再刪；請助理檢查") from exc
            row = package.summary()
            return {"message": f"已永久刪除封存的「{package.scheme_id}」（{row['archived_text']} 封存、"
                               f"{len(package.result_ids)} 筆結果）"}

    def result(self, run_id: str, rows: list[dict[str, object]]) -> dict[str, object]:
        with self.jobs._lock:
            if not IDENTITY.fullmatch(run_id):
                raise FileNotFoundError("封存的結果找不到")
            self._safe_root()
            marker = self.root / "results" / f"{run_id}.json"
            if not marker.is_file():
                raise FileNotFoundError("封存的結果找不到")
            row = next((row for row in rows if row["run_id"] == run_id), {})
            relatives = result_relatives(run_id)
            # 快照、錯誤輸出、紀錄先刪，結果檔是清單依據，最後才刪。
            paths = [self.root / path for path in reversed(relatives[1:])]
            entries = [entry for path in paths for entry in _entries(path)]
            deleting = self.root / "deleting" / run_id
            if deleting.is_symlink() or (_present(deleting) and not deleting.is_file()):
                raise DeleteConflict("刪除標記位置不安全，不能刪除；沒有刪任何檔案，請助理檢查")
            deleting.parent.mkdir(exist_ok=True)
            deleting.touch(exist_ok=True)
            try:
                _delete_entries([*entries, marker], deleting, "這一筆")
            except OSError as exc:
                if not _present(marker):
                    raise OSError(f"已刪除；只剩刪除標記 deleting/{run_id} 沒清掉，不影響清單；請助理檢查") from exc
                raise
            try:
                if not any(deleting.parent.iterdir()):
                    deleting.parent.rmdir()
            except OSError:
                pass  # 空的 deleting 資料夾留著無害，下一次刪除會再清。
            return {"message": f"已永久刪除舊的單筆封存「{row.get('scheme_id', run_id)}」"
                               f"（{row.get('finished_text', '讀不出')} 算完）"}

    def listing(self, results: list[dict[str, object]]) -> dict[str, object]:
        """呼叫端持共用鎖；表格與確認預覽用同一份名單與大小。"""
        packages = self.archive.list_packages()
        total, unknown = 0, 0
        for row in packages:
            folder = self.archive.packages_dir / str(row["package_id"])
            row["delete_available"] = row.get("status") == "complete"
            size = self._package_size(row, folder)
            if row["delete_available"]:
                total += size or 0
                unknown += size is None
        for row in results:
            run_id = str(row["run_id"])
            size = _row_size(row, [self.root / path for path in result_relatives(run_id)])
            total += size or 0
            unknown += size is None
            try:
                row["restore_available"] = not _present(self.root / "deleting" / run_id)
            except OSError:
                row["restore_available"] = False  # 標記讀不出，當成可能刪到一半。
            if not row["restore_available"]:
                row["status_text"] = LEGACY_DELETING_TEXT
        deletable = [row for row in packages if row["delete_available"]]
        skipped = [{"package_id": row["package_id"], "scheme_id": row["scheme_id"],
                    "archived_text": row["archived_text"], "reason": row["status_text"]}
                   for row in packages if not row["delete_available"]]
        return {"packages": packages, "results": results, "clear_preview": {
            "package_ids": [row["package_id"] for row in deletable], "run_ids": [row["run_id"] for row in results],
            "package_count": len(deletable), "result_count": sum(cast(int, row["result_count"]) for row in deletable),
            "legacy_count": len(results), "size_text": (
                f"至少約 {total / 1_000_000:.1f} MB（{unknown} 項大小讀不出）" if unknown else size_text(total)), "skipped": skipped}}

    def _package_size(self, row: dict[str, object], folder: Path) -> int | None:
        size = _row_size(row, [folder])
        if not row["delete_available"]:
            return size
        try:
            package = self.archive._load(str(row["package_id"]))
            if any(not _present(folder / path) for path in package.paths):
                row["restore_available"] = False
                row["status_text"] = "這一包缺少檔案，不能搬回；可以按永久刪除，或請助理檢查"
        except (OSError, ValueError):
            row["size_text"] = "大小讀不出"
            return None
        return size

    def clear(self, body: object, rows: list[dict[str, object]]) -> dict[str, object]:
        packages, results = clear_ids(body)
        deleted_packages: list[str] = []
        deleted_results: list[str] = []
        skipped: list[dict[str, object]] = []
        failed: list[dict[str, object]] = []
        with self.jobs._lock:
            for kind, identities, deleted in (("package", packages, deleted_packages), ("result", results, deleted_results)):
                for identity in identities:
                    try:
                        self.package(identity) if kind == "package" else self.result(identity, rows)
                        deleted.append(identity)
                    except (DeleteConflict, FileNotFoundError) as exc:
                        skipped.append({"kind": kind, "id": identity, "reason": _error_message(exc)})
                    except OSError as exc:
                        failed.append({"kind": kind, "id": identity, "reason": _error_message(exc)})
        message = f"已永久刪除 {len(deleted_packages)} 包封存方案、{len(deleted_results)} 筆舊的單筆封存。"
        if skipped or failed:
            message += f"\n跳過 {len(skipped)} 筆，刪除失敗 {len(failed)} 筆；請助理檢查。"
        return {"deleted_packages": deleted_packages, "deleted_results": deleted_results,
                "skipped": skipped, "failed": failed, "message": message}


def clear_ids(body: object) -> tuple[list[str], list[str]]:
    if not isinstance(body, dict) or set(body) != {"package_ids", "run_ids"}:
        raise ValueError("全部清除需要 package_ids、run_ids 兩個字串清單；沒有刪任何檔案")
    lists: list[list[str]] = []
    for key in ("package_ids", "run_ids"):
        ids = body[key]
        if not isinstance(ids, list) or any(not isinstance(item, str) or not IDENTITY.fullmatch(item) for item in ids):
            raise ValueError("清除名單的代號格式不對；沒有刪任何檔案")
        if len(set(ids)) != len(ids):
            raise ValueError("清除名單有重複代號；沒有刪任何檔案")
        lists.append(ids)
    return lists[0], lists[1]


def _row_size(row: dict[str, object], paths: list[Path]) -> int | None:
    try:
        size = size_bytes(paths)
        row["size_text"] = size_text(size)
        return size
    except (OSError, ValueError):
        row["size_text"] = "大小讀不出"
        return None


def _error_message(exc: ValueError | OSError) -> str:
    if isinstance(exc, FileNotFoundError) and exc.errno is not None:
        return "封存項目已經不在"
    if isinstance(exc, OSError) and exc.errno is not None:
        return "刪除未完成，封存項目仍保留；請助理檢查資料夾權限後再刪"
    return str(exc)


def delete_response(work: Callable[[], dict[str, object]]) -> tuple[dict[str, object], int]:
    try:
        return work(), 200
    except (ValueError, OSError) as exc:
        code = (409 if isinstance(exc, DeleteConflict) else 404 if isinstance(exc, FileNotFoundError)
                else 500 if isinstance(exc, OSError) else 400)
        return {"error": _error_message(exc)}, code
