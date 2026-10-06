"""搜尋附件的獨立紀錄與方案取法；不寫搜尋狀態、帳本或排名。"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkstemp
from typing import Literal, TypeAlias

from pydantic import BaseModel

from aosr.reporting.display import MODAL_STATE_TEXT, modal_reason
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.modal_lookup import placement_digest, placement_matches
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.search.ledger import Ledger
from aosr.search.outer_status import OuterConclusion, OuterSnapshot, snapshot_of
from aosr.search.refine import refine_order
from aosr.search.report_comparison import read_refinement_rows, scored_refinements
from aosr.search.run import SearchStatus
from aosr.search.store import FROZEN, SearchStore

Role: TypeAlias = Literal["baseline", "search_best", "refine_best"]
AttachmentState: TypeAlias = Literal["not_started", "running", "diagnosed_not_scored", "not_computed",
                                    "failed", "out_of_scope", "stopped", "skipped"]
TITLE = "低頻模態診斷（不計分）"
NOT_SCORED = "不計分、不改名次；搜尋結果與外圈結論不受診斷影響"
INCOMPLETE = "上次沒做完（可能進行中或被中斷）"
RESULT_PAGE = "其他座位與完整排序放進結果清單後到網頁結果頁看"
ROLE_LABELS: dict[Role, str] = {"baseline": "原方案", "search_best": "搜尋第一名", "refine_best": "細算第一名"}


class ModalRole(BaseModel):
    model_config = FROZEN
    role: Role
    trial_number: int | None = None
    temporary: bool = False
    placement_digest: str | None = None
    state: AttachmentState = "not_started"
    reason_text: str = ""
    diagnosis_file: str | None = None
    duplicate_of: Role | None = None
    exit_code: int | None = None


class ModalSummary(BaseModel):
    model_config = FROZEN
    schema_version: Literal["aosr.search_modal.v1"] = "aosr.search_modal.v1"
    cache_dir: str
    conclusion: OuterConclusion | None = None
    snapshot: OuterSnapshot = OuterSnapshot()
    completed: bool = False
    reason_text: str = ""
    roles: tuple[ModalRole, ...] = ()


@dataclass(frozen=True)
class RoleInput:
    record: ModalRole
    scheme: Scheme | None
    scheme_path: Path
    result_path: Path
    scope: str | None = None


def summary_path(folder: Path) -> Path:
    return folder / "modal-diagnosis" / "summary.json"


def read_summary(folder: Path) -> ModalSummary | None:
    path = summary_path(folder)
    return ModalSummary.model_validate_json(path.read_bytes()) if path.exists() else None


def write_summary(folder: Path, summary: ModalSummary) -> None:
    """每個角色收尾後原子替換；未完成的最後一份也可讀，不偽造完成記號。"""
    path = summary_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = mkstemp(dir=path.parent, prefix="summary-write-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(summary.model_dump_json() + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def role_inputs(store: SearchStore, status: SearchStatus) -> tuple[RoleInput, ...]:
    """名次只取兩本帳；細算尚未正常完成時，沿用擺位表的首名並明示暫時。"""
    order = refine_order(Ledger.read(store.ledger_path)[1]) if store.ledger_path.exists() else ()
    refined = scored_refinements(read_refinement_rows(store))
    choices: list[tuple[Role, int | None, Path, Path]] = [
        ("baseline", None, store.path / "project.json", store.baseline_path)]
    if order:
        result = store.candidate_path(order[0])
        choices.append(("search_best", order[0], store.scheme_path_for(result), result))
    if refined:
        result = store.refine_result_path(refined[0].trial_number)
        choices.append(("refine_best", refined[0].trial_number, store.scheme_path_for(result), result))
    inputs: list[RoleInput] = []
    for role, number, scheme_path, result in choices:
        temporary = role == "refine_best" and status.outer.conclusion != "complete"
        record = ModalRole(role=role, trial_number=number, temporary=temporary)
        scope = None
        try:
            scheme = store.project if role == "baseline" else load_scheme(scheme_path)
            record = record.model_copy(update={"placement_digest": placement_digest(scheme)})
        except (OSError, ValueError) as error:
            scheme = None
            record = record.model_copy(update={"state": "failed", "reason_text": f"方案檔讀不回：{error}"})
        if scheme is not None:
            try:
                document = json.loads(result.read_bytes())
                if not isinstance(document, dict):
                    raise ValueError("結果必須是資料物件")
                scope = str(document.get("scope", ""))
            except (OSError, ValueError) as error:
                record = record.model_copy(update={"state": "failed", "reason_text": f"結果範圍讀不回：{error}"})
        inputs.append(RoleInput(record, scheme, scheme_path, result, scope))
    return tuple(inputs)


def read_diagnosis(folder: Path, record: ModalRole, scheme: Scheme, *, identity: str | None = None) -> ModalDiagnosis:
    """純讀附件；報告可用文件自身身分，人手接續則要求目前程式身分。"""
    if record.diagnosis_file is None:
        raise ValueError("沒有診斷文件")
    root = summary_path(folder).parent.resolve()
    path = (root / record.diagnosis_file).resolve()
    if path.parent != root:
        raise ValueError("診斷文件不能指向附件資料夾外")
    diagnosis = ModalDiagnosis.model_validate_json(path.read_bytes())
    if diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED and not placement_matches(
            diagnosis, scheme, identity=identity or diagnosis.modal_identity or ""):
        raise ValueError("診斷的房間、身分或擺位與方案不符")
    return diagnosis


def role_label(record: ModalRole) -> str:
    name = ROLE_LABELS[record.role]
    if record.role != "baseline":
        name += "（原方案）" if record.trial_number is None else f"（試算 {record.trial_number}）"
    return name + ("（暫時）" if record.temporary else "")


def role_line(record: ModalRole) -> str:
    state = MODAL_STATE_TEXT.get(record.state, {"not_started": "未開始", "running": "進行中",
                                "stopped": "未計算：已停止", "skipped": "跳過"}.get(record.state, record.state))
    reason = f"；{record.reason_text}" if record.reason_text else ""
    same = f"；與 {ROLE_LABELS[record.duplicate_of]} 同擺位" if record.duplicate_of else ""
    # 原始錯誤仍完整存於文件；純文字報告一個角色只佔一行，段內不留空行。
    return f"{role_label(record)}：{state}{reason}{same}".replace("\r", " ").replace("\n", "；")


def summary_lines(summary: ModalSummary | None, *, running: bool = False) -> tuple[str, ...]:
    if summary is None:
        return (NOT_SCORED, *(f"{label}：未開始（搜尋正常收尾後才補）" for label in ROLE_LABELS.values()))
    lines = [NOT_SCORED]
    if not summary.completed:
        lines.append("診斷進行中；" + INCOMPLETE if running else INCOMPLETE)
    if summary.reason_text:
        lines.append(summary.reason_text.replace("\n", "；"))
    lines.extend(role_line(role) for role in summary.roles)
    found = {role.role for role in summary.roles}
    lines.extend(f"{label}：未計算（沒有可排名的方案）" for role, label in ROLE_LABELS.items() if role not in found)
    return tuple(lines)


def is_stale(summary: ModalSummary, inputs: tuple[RoleInput, ...], status: SearchStatus) -> bool:
    return (summary.snapshot != snapshot_of(status) or summary.conclusion != status.outer.conclusion
            or _role_ids(summary.roles) != _role_ids(tuple(item.record for item in inputs)))


def _role_ids(roles: tuple[ModalRole, ...]) -> tuple[tuple[Role, int | None, str | None], ...]:
    return tuple((r.role, r.trial_number, r.placement_digest) for r in roles)


def document_record(record: ModalRole, diagnosis: ModalDiagnosis) -> ModalRole:
    return record.model_copy(update={"state": diagnosis.state.value, "reason_text": modal_reason(diagnosis)})
