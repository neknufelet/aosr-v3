"""替身裝進格子之前對准用清單的固定材料核過哪一組起跑配置（#1082 段 2-8）：建替身寫、網頁預覽與網頁開跑讀；不載入 JAX。

開跑前那一道固定材料核對（:func:`aosr.gradient.fixed_candidates.check_fixed_candidates`）用的是「這一次的起跑配置」：
起跑樣板、裝法、固定的區、深度上限、准用清單，加上方案場景（固定的區照方案存檔那一份材料）。建替身在裝進格子之前用同一組
配置核過、全部候選都過才裝，並在格子裡留一份紀錄（``fixed_checks.json``）：哪一組配置（摘要＋原文）、哪一份替身
（manifest 整份摘要）、每個候選的數字。網頁預覽與網頁開跑拿這一次的配置對紀錄：對得上、而且這一場每個候選都在紀錄裡
過了，才說「可以直接開搜尋」；對不上（舊替身沒有紀錄、起跑配置換過、核的不是這份替身）就照實講、要照這一次的設定重建。
開跑那一道照樣自己再核一次（紀錄只讓網頁先講，不取代開跑的核對）。

准用清單沒有固定材料的切法不寫也不讀這份紀錄（格子裡的檔與網頁行為逐位照舊）。
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aosr.gradient.surrogate_coverage import FixedTarget
from aosr.materials.search_library import MaterialsRefused
from aosr.physics.fem_reduced_receipt import ReducedModelNotValidated

FIXED_CHECKS_FILE: Final = "fixed_checks.json"
FIXED_CHECKS_VERSION: Final = "aosr.gradient.surrogate_fixed_checks.v1"
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


class FixedCheckRow(_Frozen):
    """一個候選在那一組起跑配置的核對：材料內容摘要、兩個差、兩把尺、過了沒。"""

    target: str = Field(pattern=r"^fixed:[a-z][a-z0-9_]*$")
    material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frequency_db: float = Field(ge=0.0)
    band_db: float = Field(ge=0.0)
    frequency_ruler: float = Field(gt=0.0)
    band_ruler: float = Field(gt=0.0)
    passed: bool


class FixedChecksRecord(_Frozen):
    schema_version: Literal["aosr.gradient.surrogate_fixed_checks.v1"] = FIXED_CHECKS_VERSION
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: dict[str, object]
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checks: tuple[FixedCheckRow, ...] = Field(min_length=1)


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
