"""方案整包封存；每包有自己的清單，不改動舊的單筆封存。"""
from __future__ import annotations

import re
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
                "result_count": len(self.result_ids)}


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
            packages = [self._load(folder.name) for folder in self.packages_dir.glob("*")
                        if folder.is_dir() and (folder / "manifest.json").is_file()]
            return [item.summary() for item in sorted(packages, key=lambda item: item.archived_at, reverse=True)]

    def archive(self, name: str, result_ids: list[str]) -> ArchivePackage:
        with self.jobs._lock:
            if not NAME.fullmatch(name):
                raise ValueError("方案代號只收英數字、底線與連字號")
            if any(self.jobs.result_status(run_id).status == "running" for run_id in result_ids):
                raise ResultMoveConflict(f"「{name}」正在計算，不能封存；沒有搬動任何檔案")
            members = set(result_ids)
            for path in (self.data_dir / "runs").glob("*.json"):
                if not IDENTITY.fullmatch(path.stem):
                    continue
                try:
                    state = self.jobs.read_state(path.stem)
                except (ValueError, OSError):
                    continue
                if state.get("scheme_id") == name:
                    if self.jobs.result_status(path.stem).status == "running":
                        raise ResultMoveConflict(f"「{name}」正在計算，不能封存；沒有搬動任何檔案")
                    members.add(path.stem)
            relatives = [Path("schemes") / f"{name}.json",
                         *(path for run_id in sorted(members) for path in result_relatives(run_id))]
            present = [path for path in relatives if (self.data_dir / path).exists()
                       or (self.data_dir / path).is_symlink()]
            if not present:
                raise FileNotFoundError("方案設定檔與結果都找不到")
            package = ArchivePackage(package_id=uuid.uuid4().hex, scheme_id=name,
                                     archived_at=datetime.now(timezone.utc), result_ids=sorted(result_ids),
                                     artifact_ids=sorted(members), paths=[str(path) for path in present])
            return self._archive_files(package)

    def _archive_files(self, package: ArchivePackage) -> ArchivePackage:
        folder = self.packages_dir / package.package_id
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        try:
            folder.mkdir()
        except FileExistsError as exc:
            raise ResultMoveConflict("封存包代號已存在，沒有搬動任何檔案；請再試一次") from exc
        manifest = folder / "manifest.json"
        try:
            self.jobs._move_files([(self.data_dir / path, folder / path) for path in package.paths],
                                  lambda: manifest.write_text(package.model_dump_json(), encoding="utf-8"))
        except OSError:
            # 復原成功才清掉可能只寫了一半的清單；復原失敗時保留現場，錯誤訊息會明說。
            if all((self.data_dir / path).exists() or (self.data_dir / path).is_symlink()
                   for path in package.paths):
                manifest.unlink(missing_ok=True)
            raise
        return package

    def restore(self, package_id: str) -> ArchivePackage:
        with self.jobs._lock:
            package = self._load(package_id)
            folder = self.packages_dir / package.package_id
            if any((self.data_dir / path).exists() or (self.data_dir / path).is_symlink()
                   for path in package.reserved_paths()):
                raise ResultMoveConflict("原位置已有同名方案或同代號結果；請先把現在那份封存或改名，沒有搬動任何檔案")
            if any(not (folder / path).exists() and not (folder / path).is_symlink() for path in package.paths):
                raise FileNotFoundError("封存包缺少檔案，沒有搬動任何檔案；請助理檢查資料夾")
            self.jobs._move_files([(folder / path, self.data_dir / path) for path in package.paths],
                                  lambda: (folder / "manifest.json").unlink())
            return package
