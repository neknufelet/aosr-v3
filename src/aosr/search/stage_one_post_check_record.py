"""入選後檢查自己的摘要、原子保存與診斷讀回；不擴充搜尋角色。"""
from __future__ import annotations

import os
import re
from pathlib import Path
from tempfile import mkstemp
from typing import Literal

from pydantic import BaseModel

from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.modal_lookup import placement_matches
from aosr.reporting.scheme import Scheme
from aosr.search.modal_record import AttachmentState
from aosr.search.stage_one_selection_contract import Role as SelectionRole, StageOneSelection
from aosr.search.store import FROZEN

OWNED_FILE = re.compile(r"trial-\d{6}-[0-9a-f]{32}\.(?:json|stderr)\Z")


class PostCheckSeat(BaseModel):
    model_config = FROZEN
    trial_number: int
    roles: tuple[SelectionRole, ...]
    scheme_file: str
    placement_digest: str | None = None
    state: AttachmentState = "not_started"
    reason_text: str = ""
    exit_code: int | None = None
    diagnosis_file: str | None = None
    seconds: float | None = None


class StageOnePostCheckSummary(BaseModel):
    model_config = FROZEN
    schema_version: Literal["aosr.stage_one_post_check.v1"] = "aosr.stage_one_post_check.v1"
    selection: StageOneSelection
    modal_identity: str
    state: Literal["running", "done", "stopped", "failed"] = "running"
    message: str = ""
    seats: tuple[PostCheckSeat, ...] = ()


def summary_path(folder: Path) -> Path:
    return folder / "stage-one-check" / "summary.json"


def read_summary(folder: Path) -> StageOnePostCheckSummary | None:
    path = summary_path(folder)
    return StageOnePostCheckSummary.model_validate_json(path.read_bytes()) if path.exists() else None


def write_summary(folder: Path, summary: StageOnePostCheckSummary) -> None:
    """同目錄暫存、flush（清緩衝）、fsync（同步磁碟）後原子替換；只清自己的暫存。"""
    path = summary_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = mkstemp(dir=path.parent, prefix="summary-write-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(summary.model_dump_json() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def read_diagnosis(folder: Path, seat: PostCheckSeat, scheme: Scheme, *, identity: str) -> ModalDiagnosis:
    """限定自己的附件資料夾；成功診斷核完整房間、身分與擺位，與既有附件同一標準。"""
    if seat.diagnosis_file is None:
        raise ValueError("沒有診斷文件")
    root = summary_path(folder).parent.resolve()
    path = (root / seat.diagnosis_file).resolve()
    if path.parent != root:
        raise ValueError("診斷文件不能指向附件資料夾外")
    diagnosis = ModalDiagnosis.model_validate_json(path.read_bytes())
    if diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED and not placement_matches(diagnosis, scheme, identity=identity):
        raise ValueError("診斷的房間、身分或擺位與方案不符")
    return diagnosis


def prune(folder: Path, *summaries: StageOnePostCheckSummary | None) -> None:
    """收尾才清：只留這一代與上一代摘要指到的文件（報告不拿鎖，可能剛讀完上一份摘要、還沒讀它指到的文件），
    只刪自己命名樣式的檔，資料夾不會一直長。"""
    referenced = {Path(seat.diagnosis_file).stem for summary in summaries if summary is not None
                  for seat in summary.seats if seat.diagnosis_file is not None}
    for path in summary_path(folder).parent.iterdir():
        if OWNED_FILE.fullmatch(path.name) and path.stem not in referenced and path.is_file():
            path.unlink(missing_ok=True)
