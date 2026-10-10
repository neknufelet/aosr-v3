"""方案整包封存；每包有自己的清單，不改動舊的單筆封存。"""
from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from aosr.gui.jobs import JobManager, ResultMoveConflict


NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,199}\Z")
IDENTITY = re.compile(r"[0-9a-f]{32}\Z")


def result_relatives(run_id: str) -> list[Path]:
    return [Path("results") / f"{run_id}.json", Path("runs") / f"{run_id}.json",
            Path("runs") / f"{run_id}.stderr", Path("runs") / run_id]


class ArchivePackage(BaseModel):
    """持久清單；路徑只准是此方案與這些計算的原位置。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["aosr.gui_scheme_archive.v1"] = "aosr.gui_scheme_archive.v1"
    package_id: str
    scheme_id: str
    archived_at: datetime
    result_ids: list[str]
    artifact_ids: list[str]
    paths: list[str]
    # 舊包沒有這一格，視為已搬完；新包在第一個成員搬動前就記為未完成。
    status: Literal["incomplete", "complete"] = "complete"

    @model_validator(mode="after")
    def valid_members(self) -> ArchivePackage:
        if not NAME.fullmatch(self.scheme_id) or not IDENTITY.fullmatch(self.package_id):
            raise ValueError("封存清單的代號無效")
        if any(not IDENTITY.fullmatch(item) for item in self.artifact_ids + self.result_ids):
            raise ValueError("封存清單的結果代號無效")
        allowed = {str(path) for path in self.reserved_paths()}
        if not self.paths or not set(self.paths) <= allowed or len(self.paths) != len(set(self.paths)):
            raise ValueError("封存清單的檔案路徑無效")
        if not set(self.result_ids) <= set(self.artifact_ids):
            raise ValueError("封存清單缺少結果成員")
        if self.archived_at.tzinfo is None:
            raise ValueError("封存清單缺少時區")
        return self

    def reserved_paths(self) -> list[Path]:
        return [Path("schemes") / f"{self.scheme_id}.json",
                *(path for run_id in self.artifact_ids for path in result_relatives(run_id))]

    def summary(self) -> dict[str, object]:
        return {**self.model_dump(mode="json"),
                "archived_text": self.archived_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
                "result_count": len(self.result_ids), "restore_available": True,
                "status_text": "封存沒做完，請助理檢查" if self.status == "incomplete" else ""}


class ArchivePlan(BaseModel):
    """唯讀成員計畫；預覽回覆與封存清單取同一份欄位。"""

    scheme_id: str
    result_ids: list[str]
    artifact_ids: list[str]
    paths: list[str]

    def summary(self) -> dict[str, object]:
        return {**self.model_dump(), "result_count": len(self.result_ids)}


def _scheme_id(path: Path, *, result: bool = False) -> str | None:
    """只讀成員身分；壞結果仍可由計算紀錄或重算快照認領。"""
    try:
        document: object = json.loads(path.read_text(encoding="utf-8"))
        if result and isinstance(document, dict):
            document = document.get("scheme")
        found = document.get("scheme_id") if isinstance(document, dict) else None
        return found if isinstance(found, str) else None
    except (ValueError, OSError):
        return None


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _package_files(folder: Path) -> list[Path]:
    """清單以外的實際檔案，包含快照底下的全部檔案與符號連結。"""
    return [path.relative_to(folder) for path in folder.rglob("*")
            if (path.is_file() or path.is_symlink()) and path != folder / "manifest.json"]


class SchemeArchive:
    def __init__(self, jobs: JobManager) -> None:
        self.jobs = jobs
        self.data_dir = jobs.data_dir
        self.packages_dir = self.data_dir / "archive" / "packages"

    def _load(self, package_id: str) -> ArchivePackage:
        if not IDENTITY.fullmatch(package_id):
            raise FileNotFoundError("封存包找不到")
        manifest = self.packages_dir / package_id / "manifest.json"
        try:
            package = ArchivePackage.model_validate_json(manifest.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError("封存包找不到") from exc
        except OSError as exc:
            raise OSError("封存清單讀不出，請助理檢查資料夾") from exc
        except ValueError as exc:
            raise ValueError("封存清單讀不出，請助理檢查資料夾") from exc
        if package.package_id != package_id:
            raise ValueError("封存清單與資料夾的代號不同，請助理檢查資料夾")
        return package

    def list_packages(self) -> list[dict[str, object]]:
        with self.jobs._lock:
            packages: list[ArchivePackage] = []
            damaged: list[dict[str, object]] = []
            for folder in self.packages_dir.glob("*"):
                if not folder.is_dir() or not (folder / "manifest.json").is_file():
                    continue
                try:
                    packages.append(self._load(folder.name))
                except (ValueError, OSError):
                    damaged.append({"package_id": folder.name, "scheme_id": "（方案名字讀不出）",
                                    "archived_text": "讀不出", "result_count": None,
                                    "status_text": "這一包清單讀不出，請助理檢查", "restore_available": False})
            return [item.summary() for item in sorted(packages, key=lambda item: item.archived_at, reverse=True)] + damaged

    def _members(self, name: str, result_ids: list[str]) -> set[str]:
        members = set(result_ids)
        for path in (self.data_dir / "results").glob("*.json"):
            if IDENTITY.fullmatch(path.stem) and _scheme_id(path, result=True) == name:
                members.add(path.stem)
        for path in (self.data_dir / "runs").glob("*.json"):
            try:
                state = self.jobs.read_state(path.stem)
            except (ValueError, OSError) as exc:
                raise ResultMoveConflict("計算紀錄讀不出，可能正在計算，不能封存；沒有搬動任何檔案") from exc
            if IDENTITY.fullmatch(path.stem) and state.get("scheme_id") == name:
                if state.get("status") == "running":
                    raise ResultMoveConflict(f"「{name}」正在計算，不能封存；沒有搬動任何檔案")
                members.add(path.stem)
        for folder in (self.data_dir / "runs").glob("*"):
            run_id = folder.name
            if (IDENTITY.fullmatch(run_id) and folder.is_dir()
                    and _scheme_id(self.data_dir / "results" / f"{run_id}.json", result=True) is None
                    and _scheme_id(folder / "scheme.json") == name):
                members.add(run_id)
        for run_id, process in tuple(self.jobs.processes.items()):
            if process.poll() is None and (self.jobs.process_schemes.get(run_id) == name or run_id in members):
                raise ResultMoveConflict(f"「{name}」正在計算，不能封存；沒有搬動任何檔案")
        return members

    def preview(self, name: str, result_ids: list[str]) -> ArchivePlan:
        """預覽與實際封存共用成員計畫，筆數只算真的存在的結果檔。呼叫端持鎖。"""
        if not NAME.fullmatch(name):
            raise ValueError("方案代號只收英數字、底線與連字號")
        members = self._members(name, result_ids)
        relatives = [Path("schemes") / f"{name}.json",
                     *(path for run_id in sorted(members) for path in result_relatives(run_id))]
        present = [str(path) for path in relatives if _present(self.data_dir / path)]
        if not present:
            raise FileNotFoundError("方案設定檔與結果都找不到")
        results = sorted(run_id for run_id in members if (self.data_dir / "results" / f"{run_id}.json").is_file())
        return ArchivePlan(scheme_id=name, result_ids=results, artifact_ids=sorted(members), paths=present)

    def archive(self, name: str, result_ids: list[str]) -> ArchivePackage:
        with self.jobs._lock:
            plan = self.preview(name, result_ids)
            package = ArchivePackage(package_id=uuid.uuid4().hex, scheme_id=name,
                                     archived_at=datetime.now(timezone.utc), status="incomplete",
                                     result_ids=plan.result_ids, artifact_ids=plan.artifact_ids, paths=plan.paths)
            return self._archive_files(package)

    def _write_manifest(self, folder: Path, package: ArchivePackage) -> None:
        """完整寫出再換清單，更新中斷仍保留前一份；暫存檔不放在成員包內。"""
        handle, name = tempfile.mkstemp(dir=self.packages_dir, suffix=".tmp")
        temporary = Path(name)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(package.model_dump_json())
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(folder / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)

    def _archive_files(self, package: ArchivePackage) -> ArchivePackage:
        folder = self.packages_dir / package.package_id
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        try:
            folder.mkdir()
        except FileExistsError as exc:
            raise ResultMoveConflict("封存包代號已存在，沒有搬動任何檔案；請再試一次") from exc
        manifest = folder / "manifest.json"
        complete = package.model_copy(update={"status": "complete"})
        try:
            self._write_manifest(folder, package)
            self.jobs._move_files([(self.data_dir / path, folder / path) for path in package.paths],
                                  lambda: self._write_manifest(folder, complete))
        except OSError as exc:
            if not _package_files(folder) and not any(_present(folder / path) for path in package.paths):
                manifest.unlink(missing_ok=True)
            if not manifest.exists():
                raise OSError("封存失敗，檔案沒有搬動或已搬回原處；請助理檢查資料夾") from exc
            raise
        return complete

    def _validate_contents(self, folder: Path, package: ArchivePackage) -> None:
        listed = [Path(path) for path in package.paths]
        snapshots = {Path("runs") / run_id for run_id in package.artifact_ids}
        for path in _package_files(folder):
            if not any(path == item or (item in snapshots and item in path.parents) for item in listed):
                raise ResultMoveConflict("包裡的檔案與封存清單對不上，沒有搬動任何檔案；請助理檢查")
        for path in (folder / "results").glob("*.json"):
            found = _scheme_id(path, result=True)
            if found is not None and found != package.scheme_id:
                raise ResultMoveConflict("結果的方案代號與封存清單對不上，沒有搬動任何檔案；請助理檢查")

    def _finish_restore(self, folder: Path, package: ArchivePackage) -> None:
        if _package_files(folder) or any(_present(folder / path) for path in package.paths):
            raise ResultMoveConflict("包裡仍有成員檔與清單對不上，請助理檢查")
        (folder / "manifest.json").unlink()

    def restore(self, package_id: str) -> ArchivePackage:
        with self.jobs._lock:
            package = self._load(package_id)
            folder = self.packages_dir / package.package_id
            self._validate_contents(folder, package)
            present = [Path(path) for path in package.paths if _present(folder / path)]
            conflicts = package.reserved_paths() if package.status == "complete" else present
            if any(_present(self.data_dir / path) for path in conflicts):
                if package.status == "incomplete":
                    raise ResultMoveConflict("原位和包裡都有同一個檔，與封存清單對不上，沒有搬動任何檔案；請助理檢查")
                raise ResultMoveConflict("原位置已有同名方案或同代號結果；請先把現在那份封存或改名，沒有搬動任何檔案")
            if package.status == "complete" and any(not _present(folder / path) for path in package.paths):
                raise FileNotFoundError("封存包缺少檔案，沒有搬動任何檔案；請助理檢查資料夾")
            self.jobs._move_files([(folder / path, self.data_dir / path) for path in present],
                                  lambda: self._finish_restore(folder, package))
            return package
