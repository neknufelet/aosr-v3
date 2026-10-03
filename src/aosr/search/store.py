"""每次搜尋自己的資料夾：固定原方案、搜尋設定、評分設定與身分版本。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.reporting.result import PurposeSettings
from aosr.reporting.scheme import Scheme
from aosr.search.settings import SearchSettings


SEARCH_STORE_VERSION: Final = "aosr.search_store.v1"
JSON_SUFFIX = ".json"
JSONL_SUFFIX = ".jsonl"
STDERR_SUFFIX = ".stderr"
PROJECT_FILE = f"project{JSON_SUFFIX}"
SETTINGS_FILE = f"settings{JSON_SUFFIX}"
PURPOSE_FILE = f"purpose{JSON_SUFFIX}"
IDENTITY_FILE = f"identity{JSON_SUFFIX}"
LEDGER_FILE = f"ledger{JSONL_SUFFIX}"
CANDIDATES_DIR = "candidates"
REFINE_DIR = "verification"
REFINE_LEDGER_FILE = f"refine{JSONL_SUFFIX}"
FEM_DIR = "fem-parts"
REQUIRED_VERSIONS = frozenset(("python", "optuna", "numpy"))
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


@dataclass(frozen=True)
class SearchIdentity:
    """設計紙第四節的物理身分、程式指紋與當次用途評分設定。"""

    physics_identity: str
    program_fingerprint: str
    purpose_settings: PurposeSettings


class _SettingsSnapshot(BaseModel):
    """搜尋總表及它的內容摘要，讀回不能接受失配。"""

    model_config = FROZEN
    settings: SearchSettings
    fingerprint: str

    @model_validator(mode="after")
    def _matches(self) -> Self:
        if self.fingerprint != self.settings.fingerprint:
            raise ValueError("settings fingerprint mismatch")
        return self


class _IdentitySnapshot(BaseModel):
    """有版號、可驗型別的身分與呼叫端提供的版本快照。"""

    model_config = FROZEN
    store_version: Literal["aosr.search_store.v1"]
    search_id: str = Field(min_length=1, strict=True)
    physics_identity: str = Field(min_length=1, strict=True)
    program_fingerprint: str = Field(min_length=1, strict=True)
    versions: dict[StrictStr, StrictStr]

    @model_validator(mode="after")
    def _required_versions(self) -> Self:
        missing = REQUIRED_VERSIONS - self.versions.keys()
        if missing:
            raise ValueError(f"missing versions: {sorted(missing)}")
        return self


def candidate_name(trial_number: int) -> str:
    """試算編號 → 候選結果在搜尋資料夾裡的相對名字；帳本列的 result_file 與 candidate_path 共用這一支。"""
    if type(trial_number) is not int or trial_number < 0:
        raise ValueError("trial_number must be an integer >= 0")
    return f"{CANDIDATES_DIR}/trial-{trial_number:06d}{JSON_SUFFIX}"


def refine_result_name(trial_number: int | None) -> str:
    """細算結果相對搜尋資料夾的檔名；None 是原方案，候選沿用篩選的編號格式。帳列與路徑都用它，不各寫一份。"""
    name = f"baseline{JSON_SUFFIX}" if trial_number is None else Path(candidate_name(trial_number)).name
    return f"{REFINE_DIR}/{name}"


def refine_scheme_id(search_id: str, trial_number: int | None) -> str:
    """細算方案在搜尋代號後加 verification（驗證）後綴，與篩選方案分開。"""
    name = "baseline" if trial_number is None else Path(candidate_name(trial_number)).stem
    return f"{search_id}-{name}-verification"


def _write_snapshot(path: Path, document: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(document), ensure_ascii=False, indent=1, allow_nan=False) + "\n")


def _check_purpose(project: Scheme, settings: SearchSettings, purpose: PurposeSettings) -> None:
    if project.purpose != settings.purpose or purpose.purpose != settings.purpose:
        raise ValueError("project, search settings and purpose settings must have the same purpose")


def check_search_axis(project: Scheme) -> None:
    """第六節第 6 條：驗證軸只給細算的候選；候選沿用專案場景，專案寫了驗證軸整場就會用 1 Hz 軸跑。"""
    if project.scene.low_frequency_axis not in (None, LowFrequencyAxis.SEARCH):
        raise ValueError("searches run on the search axis; the verification axis is only for refining candidates")


class SearchStore:
    """新搜尋用新代號；只提供候選結果路徑，不求解、不儲存候選結果。"""

    def __init__(self, path: Path, project: Scheme, settings: SearchSettings,
                 identity: SearchIdentity, snapshot: _IdentitySnapshot) -> None:
        self._path = path
        self._project = project.model_copy(deep=True)
        self._settings = settings.model_copy(deep=True)
        self._identity = SearchIdentity(identity.physics_identity, identity.program_fingerprint,
                                        identity.purpose_settings.model_copy(deep=True))
        self._snapshot = snapshot.model_copy(deep=True)

    @classmethod
    def create(cls, root: Path, *, project: Scheme, settings: SearchSettings,
               identity: SearchIdentity, versions: Mapping[str, str]) -> SearchStore:
        """先驗輸入再建新資料夾；碰到既有代號就報錯，完全不寫入該資料夾。"""
        project = Scheme.model_validate(project.model_dump(mode="json"))
        settings = SearchSettings.model_validate(settings.canonical())
        purpose = PurposeSettings.model_validate(identity.purpose_settings.model_dump(mode="json"))
        _check_purpose(project, settings, purpose)
        check_search_axis(project)
        snapshot = _IdentitySnapshot(
            store_version=SEARCH_STORE_VERSION, search_id=uuid.uuid4().hex,
            physics_identity=identity.physics_identity, program_fingerprint=identity.program_fingerprint,
            versions=dict(versions),
        )
        path = root / snapshot.search_id
        root.mkdir(parents=True, exist_ok=True)
        path.mkdir(exist_ok=False)
        (path / CANDIDATES_DIR).mkdir(exist_ok=False)
        settings_snapshot = _SettingsSnapshot(settings=settings, fingerprint=settings.fingerprint)
        for name, document in (
            (PROJECT_FILE, project.model_dump(mode="json")),
            (SETTINGS_FILE, settings_snapshot.model_dump(mode="json")),
            (PURPOSE_FILE, purpose.model_dump(mode="json")),
            (IDENTITY_FILE, snapshot.model_dump(mode="json")),
        ):
            _write_snapshot(path / name, document)
        return cls(path, project, settings, SearchIdentity(identity.physics_identity, identity.program_fingerprint, purpose), snapshot)

    @classmethod
    def open(cls, path: Path) -> SearchStore:
        """四份快照皆驗型別、版號、搜尋摘要與用途，拒絕混用或被改過的設定。"""
        project = Scheme.model_validate_json((path / PROJECT_FILE).read_bytes())
        settings = _SettingsSnapshot.model_validate_json((path / SETTINGS_FILE).read_bytes()).settings
        purpose = PurposeSettings.model_validate_json((path / PURPOSE_FILE).read_bytes())
        snapshot = _IdentitySnapshot.model_validate_json((path / IDENTITY_FILE).read_bytes())
        _check_purpose(project, settings, purpose)
        if snapshot.search_id != path.name:
            raise ValueError("search id does not match the directory name")
        if not (path / CANDIDATES_DIR).is_dir():
            raise ValueError("missing candidates directory")
        identity = SearchIdentity(snapshot.physics_identity, snapshot.program_fingerprint, purpose)
        return cls(path, project, settings, identity, snapshot)

    @property
    def search_id(self) -> str:
        return self._snapshot.search_id

    @property
    def path(self) -> Path:
        return self._path

    @property
    def fem_path(self) -> Path:
        """共用分片根目錄；第一次計算才建立，讀舊搜尋不要求它已存在。"""
        return self.path / FEM_DIR

    @property
    def project(self) -> Scheme:
        return self._project.model_copy(deep=True)

    @property
    def settings(self) -> SearchSettings:
        return self._settings.model_copy(deep=True)

    @property
    def identity(self) -> SearchIdentity:
        return SearchIdentity(self._identity.physics_identity, self._identity.program_fingerprint,
                              self._identity.purpose_settings.model_copy(deep=True))

    @property
    def versions(self) -> Mapping[str, str]:
        return MappingProxyType(self._snapshot.versions)

    def candidate_path(self, trial_number: int) -> Path:
        """試算編號命名的完整結果路徑；這一步只給路徑。"""
        return self.path / candidate_name(trial_number)

    @property
    def refine_dir(self) -> Path:
        """細算結果的新子資料夾；只給路徑，不在開搜尋時建立。"""
        return self.path / REFINE_DIR

    def refine_result_path(self, trial_number: int | None) -> Path:
        """None 是原方案；編號格式沿用篩選，但所有結果另存細算資料夾。"""
        return self.path / refine_result_name(trial_number)

    @property
    def refine_ledger_path(self) -> Path:
        """細算帳放在搜尋根目錄，與篩選帳分開。"""
        return self.path / REFINE_LEDGER_FILE

    def ensure_refine_dir(self) -> None:
        """第一次需要保存細算結果才建；已有資料夾照樣沿用。"""
        self.refine_dir.mkdir(exist_ok=True)

    @staticmethod
    def scheme_path_for(result_path: Path) -> Path:
        """工作方案放在完整結果旁邊，命名由搜尋資料夾統一。"""
        return result_path.with_name(f"{result_path.stem}-scheme{JSON_SUFFIX}")

    @staticmethod
    def stderr_path_for(result_path: Path) -> Path:
        """子行程錯誤輸出放在同一資料夾，不跟結果或方案混用。"""
        return result_path.with_name(f"{result_path.stem}{STDERR_SUFFIX}")

    @property
    def ledger_path(self) -> Path:
        return self.path / LEDGER_FILE

    @property
    def baseline_path(self) -> Path:
        """主對話判斷：原方案單獨保存，不佔試算編號。"""
        return self.path / f"baseline{JSON_SUFFIX}"

    @property
    def status_path(self) -> Path:
        return self.path / f"status{JSON_SUFFIX}"

    @property
    def stop_path(self) -> Path:
        """每批開始前檢查的使用者停止記號。"""
        return self.path / "stop"

    @property
    def refine_stop_path(self) -> Path:
        """細算獨立停止記號；搜尋曾被停過不會停止後續細算。"""
        return self.path / "refine-stop"
