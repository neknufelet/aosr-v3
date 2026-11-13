"""替身裝進格子之前對准用清單的固定材料核過哪一組起跑配置（#1082 段 2-8）：建替身寫、網頁預覽與網頁開跑讀；不載入 JAX。

開跑前那一道固定材料核對（:func:`aosr.gradient.fixed_candidates.check_fixed_candidates`）用的是「這一次的起跑配置」：
起跑樣板、裝法、固定的區、深度上限、准用清單，加上方案場景（固定的區照方案存檔那一份材料）。建替身在裝進格子之前用同一組
配置核過、全部候選都過才裝，並在格子裡留一份紀錄（``fixed_checks.json``）：哪一組配置（摘要＋原文）、哪一份替身
（manifest 整份摘要）、每個候選的數字。網頁預覽與網頁開跑拿這一次的配置對紀錄：對得上、而且這一場每個候選都在紀錄裡
過了，才說「可以直接開搜尋」；對不上（舊替身沒有紀錄、起跑配置換過、核的不是這份替身）就照實講、要照這一次的設定重建。

開跑那一道（#1142，老闆 2026-11-13 拍 A）：紀錄的身分逐欄都跟這一次相同才沿用紀錄的數字、不再完整解一次
（:func:`reused_rows`）；任何一欄不同、紀錄是舊格式（v1，沒有沿用要比的欄位）或讀不懂、替身是接續（carry-over）沿用的，
一律照舊自己重核。身分：替身（manifest 整份檔案的 SHA-256）、起跑配置（正規寫法的摘要）、頻率軸（逐位 ``float.hex()``）、
兩把尺（登記簿現值）、每個候選的材料內容（兩種摘要都比）與准用它的片、以及每個候選核對配置的逐片逐頻阻抗（逐位摘要；
起跑配置的正規寫法只記代號，家族的構造與盒界、使用者材料庫的內容改了阻抗就會變，這一格擋得住）。

准用清單沒有固定材料的切法不寫也不讀這份紀錄（格子裡的檔與網頁行為逐位照舊）。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aosr.gradient.surrogate_coverage import FixedTarget
from aosr.materials.search_library import MaterialsRefused
from aosr.physics.fem_reduced_receipt import ReducedModelNotValidated

if TYPE_CHECKING:
    from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances

FIXED_CHECKS_FILE: Final = "fixed_checks.json"
FIXED_CHECKS_VERSION: Final = "aosr.gradient.surrogate_fixed_checks.v1"
# v2（#1142）多記開跑沿用要比的身分欄位；v1 照讀（網頁預覽照舊），只是開跑不沿用、照舊重核。
FIXED_CHECKS_VERSION_V2: Final = "aosr.gradient.surrogate_fixed_checks.v2"
NO_RECORD_TEXT: Final = ("這份替身裝進格子之前沒有用開跑那一組起跑配置核過准用清單的固定材料（舊版建替身沒有這一道），"
                         "開跑前的核對可能不收；要重建（重建時會先用這一組配置核，不夠準就補訓練）")
OTHER_CONFIGURATION_TEXT: Final = ("這份替身核固定材料時用的起跑配置跟這一次不同（起跑樣板、裝法、深度上限、固定的區、"
                                   "准用清單或方案場景有改），開跑前的核對沒驗過這一組；要照這一次的設定重建")


INACCURATE_STATE_TEXT: Final = ("替身對准用清單的固定材料在這一組起跑配置不夠準，要重建替身"
                                "（重建時會先用這一組配置核，不夠準就補訓練）")


class FixedCandidateInaccurate(MaterialsRefused, ReducedModelNotValidated):
    """准用清單的固定材料在開跑那一組配置替身不夠準：是替身核不過的一種（網頁照「要重建」那條走），
    也照舊是材料拒收（命令列、舊考卷認的那一種）。開跑那一道（:mod:`aosr.gradient.fixed_candidates`）丟它。"""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


_SHA256 = r"^[0-9a-f]{64}$"
_ROW_IDENTITY: Final = ("surfaces", "material_digest", "full_solves", "impedances_sha256")


class FixedCheckRow(_Frozen):
    """一個候選在那一組起跑配置的核對：材料內容摘要、兩個差、兩把尺、過了沒。

    v2 多四格（開跑沿用要的）：准用它的片、材料的身分摘要（``endpoint.material_digest``，開跑記進 ``project.json`` 的那一種）、
    完整解次數、核對配置的逐片逐頻阻抗摘要（:func:`impedances_sha256`）。v1 沒有這四格。"""

    target: str = Field(pattern=r"^fixed:[a-z][a-z0-9_]*$")
    material_sha256: str = Field(pattern=_SHA256)
    frequency_db: float = Field(ge=0.0)
    band_db: float = Field(ge=0.0)
    frequency_ruler: float = Field(gt=0.0)
    band_ruler: float = Field(gt=0.0)
    passed: bool
    surfaces: tuple[str, ...] | None = Field(default=None, min_length=1)
    material_digest: str | None = Field(default=None, pattern=_SHA256)
    full_solves: int | None = Field(default=None, ge=1, strict=True)
    impedances_sha256: str | None = Field(default=None, pattern=_SHA256)


class FixedChecksRecord(_Frozen):
    """v1：建替身那一側只給網頁預覽用；v2（#1142）每一格沿用要比的身分都齊，開跑才可能沿用。版本跟欄位要一致：
    v2 缺一格、v1 多一格都不收（讀成「沒有紀錄」，網頁照「要重建」講、開跑照舊重核，不當成核過）。"""

    schema_version: Literal["aosr.gradient.surrogate_fixed_checks.v1",
                            "aosr.gradient.surrogate_fixed_checks.v2"] = FIXED_CHECKS_VERSION
    configuration_sha256: str = Field(pattern=_SHA256)
    configuration: dict[str, object]
    manifest_sha256: str = Field(pattern=_SHA256)
    checks: tuple[FixedCheckRow, ...] = Field(min_length=1)
    frequencies_hz: tuple[str, ...] | None = Field(default=None, min_length=1,
                                                   description="核對用的頻率軸，逐點 float.hex()（v2）")

    @model_validator(mode="after")
    def _fields_match_version(self) -> Self:
        v2 = self.schema_version == FIXED_CHECKS_VERSION_V2
        missing = [f"{row.target}.{name}" for row in self.checks for name in _ROW_IDENTITY
                   if (getattr(row, name) is None) == v2]
        if (self.frequencies_hz is None) == v2:
            missing.append("frequencies_hz")
        if missing:
            raise ValueError(f"核對紀錄 {self.schema_version} 的欄位跟版本不一致：{missing}")
        return self


@dataclass(frozen=True)
class CandidateIdentity:
    """開跑這一次、一個固定材料候選要跟紀錄比的身分：兩種材料摘要、准用它的片、核對配置的阻抗摘要。"""

    target: str
    material_sha256: str
    material_digest: str
    surfaces: tuple[str, ...]
    impedances_sha256: str


def axis_hex(frequencies_hz: Sequence[float]) -> tuple[str, ...]:
    """頻率軸的逐位寫法（``float.hex()``）：建替身與開跑各自算，逐字相同才算同一條軸。"""
    return tuple(float(value).hex() for value in frequencies_hz)


def impedances_sha256(impedances: FrequencyBoundaryImpedances) -> str:
    """一組核對配置（牆或面片 → 每個頻點的阻抗、單一值、或 None＝剛性）的逐位摘要：鍵寫成面片代號、照字排，每個值的
    實部虛部各取 ``float.hex()``。建替身與開跑用同一支，阻抗差一個位元就不同。不載入 JAX（值先轉成 Python 的 complex）。"""
    def number(value: float | complex | None) -> list[str] | None:
        if value is None:
            return None
        found = complex(value)
        return [float(found.real).hex(), float(found.imag).hex()]

    rows: dict[str, object] = {}
    for key, row in impedances.items():
        wall, zone = key if isinstance(key, tuple) else (key, "")
        name = wall.wall_name() + ("@" + zone if zone else "")
        rows[name] = (None if row is None else ["scalar", number(row)] if isinstance(row, (int, float, complex))
                      else [number(value) for value in row])
    content = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def configuration_document(*, presets: Sequence[str | None], mount: str, fixed_surfaces: Mapping[str, str],
                           max_depths: Mapping[str, float], allowed: Mapping[str, Sequence[str]],
                           scene_json: str) -> dict[str, object]:
    """一組起跑配置的正規寫法（面片序照起跑樣板；字典照鍵排、准用目標照字排；深度用 ``float.hex()`` 逐位比）。
    建替身由命令列參數、網頁由搜尋計畫各自算，兩邊要逐位相同才算同一組。``scene_json`` 是方案場景的 JSON 原文。"""
    return {"presets": [None if preset is None else str(preset) for preset in presets], "mount": str(mount),
            "fixed_surfaces": {name: str(value) for name, value in sorted(fixed_surfaces.items())},
            "max_depths": {name: float(value).hex() for name, value in sorted(max_depths.items())},
            "allowed": {name: sorted(str(target) for target in targets) for name, targets in sorted(allowed.items())},
            "scene_sha256": hashlib.sha256(scene_json.encode("utf-8")).hexdigest()}


def configuration_sha256(document: Mapping[str, object]) -> str:
    content = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def manifest_digest(manifest: Path) -> str:
    """替身 manifest 檔整份的 SHA-256（不信 manifest 自己記的那一格；網頁行程不載入替身）。"""
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def write_record(path: Path, record: FixedChecksRecord) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(record.model_dump_json(indent=1) + "\n")


def read_record(path: Path) -> FixedChecksRecord | None:
    """格子裡的紀錄；檔不在是 None（舊替身），檔壞了也當沒有（照「沒核過、要重建」講，不當成核過）。"""
    if not path.is_file():
        return None
    try:
        return FixedChecksRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ValidationError):
        return None


def unchecked_reason(slot: Path, manifest: Path, configuration: Mapping[str, object],
                     fixed: Sequence[FixedTarget]) -> str | None:
    """這一場的固定材料候選（``fixed``）在這一組起跑配置有沒有核過；核過回 None，沒有回一句白話原因。
    ``slot`` 是格子資料夾、``manifest`` 是替身 manifest；沒有候選一律 None（不讀紀錄）。"""
    if not fixed:
        return None
    record = read_record(slot / FIXED_CHECKS_FILE)
    if record is None or not manifest.is_file() or record.manifest_sha256 != manifest_digest(manifest):
        return NO_RECORD_TEXT
    if record.configuration_sha256 != configuration_sha256(configuration):
        return OTHER_CONFIGURATION_TEXT
    rows = {row.target: row for row in record.checks}
    missing = [item.target for item in fixed
               if (row := rows.get(item.target)) is None or not row.passed
               or row.material_sha256 != item.coverage().material_sha256]
    if missing:
        return f"固定材料 {'、'.join(missing)} 沒有在這一組起跑配置核過；要重建（重建時會先核，不夠準就補訓練）"
    return None


def reused_rows(record: FixedChecksRecord | None, *, manifest_sha256: str, configuration: Mapping[str, object],
                frequencies_hz: Sequence[float], rulers: tuple[float, float],
                candidates: Sequence[CandidateIdentity]) -> tuple[FixedCheckRow, ...] | str:
    """開跑那一道能不能沿用建替身時核過的數字（#1142）：身分逐欄都相同、每個候選都在紀錄裡而且過了，回照 ``candidates``
    順序排好的紀錄列；有任何一欄不同，回一句白話原因（開跑照舊重核）。``rulers`` 是登記簿現值（逐頻、帶能量）。
    全部候選一起沿用或一起重核，不拆。"""
    if record is None:
        return "替身旁邊沒有建替身時的核對紀錄（或讀不懂）"
    if record.schema_version != FIXED_CHECKS_VERSION_V2:
        return "建替身時的核對紀錄是舊格式，沒有沿用要比的欄位"
    if record.manifest_sha256 != manifest_sha256:
        return "核對紀錄核的不是這一份替身"
    if record.configuration_sha256 != configuration_sha256(configuration):
        return "起跑配置跟建替身時核的那一組不同"
    if record.frequencies_hz != axis_hex(frequencies_hz):
        return "頻率軸跟建替身時核的不同"
    rows = {row.target: row for row in record.checks}
    if set(rows) != {item.target for item in candidates}:
        return "固定材料候選跟建替身時核的那幾個不同"
    found = []
    for item in candidates:
        row = rows[item.target]
        if (row.frequency_ruler, row.band_ruler) != rulers:
            return f"固定材料 {item.target} 核的時候用的尺跟登記簿現在的不同"
        if row.material_sha256 != item.material_sha256 or row.material_digest != item.material_digest:
            return f"固定材料 {item.target} 的材料內容跟建替身時核的不同"
        if row.surfaces != item.surfaces:
            return f"固定材料 {item.target} 准用的片跟建替身時核的不同"
        if row.impedances_sha256 != item.impedances_sha256:
            return f"固定材料 {item.target} 的核對配置阻抗跟建替身時核的不同"
        if not row.passed or row.frequency_db > rulers[0] or row.band_db > rulers[1]:
            return f"固定材料 {item.target} 建替身時核的沒過"
        found.append(row)
    return tuple(found)
