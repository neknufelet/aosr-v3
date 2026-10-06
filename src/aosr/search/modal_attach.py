"""外圈保存完成後補附件：獨立子行程、獨立停止、搜尋離開碼不受影響。"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from aosr.reporting.modal_diagnosis import modal_identity
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.modal_lookup import key_from_scheme, save_diagnosis, scheme_scope_reason
from aosr.runtime import child_process_env
from aosr.search.modal_record import (
    ModalRole, ModalSummary, RoleInput, document_record, read_diagnosis, read_summary, role_inputs,
    summary_path, write_summary,
)
from aosr.search.outer_status import OUTER_MESSAGES, snapshot_of
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore

DEFAULT_RUNNER = (sys.executable, "-m", "aosr.reporting.scheme_cli", "modal")
STOPPED_NOTE = "低頻診斷被停止，搜尋結果不受影響"
FIXED_ROOM_NOTE = "房間或材料不固定，這次不補（成本另評估）"


def _eligibility(status: SearchStatus, inputs: tuple[RoleInput, ...]) -> str:
    conclusion = status.outer.conclusion
    if conclusion is None or conclusion == "user_stopped" or conclusion.endswith(("_failed", "_interrupted")):
        text = OUTER_MESSAGES[conclusion] if conclusion is not None else "未判定"
        return f"搜尋沒有正常收尾（{text}），這次不補；接續跑完後會補"
    original = key_from_scheme(inputs[0].scheme) if inputs[0].scheme is not None else None
    for item in inputs:
        if item.scheme is None:
            continue
        key = key_from_scheme(item.scheme)
        if key != original:
            return FIXED_ROOM_NOTE
        if item.record.state != "failed" and item.scope != "stage_two_subset":
            return FIXED_ROOM_NOTE
    return ""


def _stop(process: subprocess.Popen[bytes]) -> None:
    """先讓命令列收尾清掉 modal-write-*，兩秒後仍在才強制停止整群。"""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _compute(item: RoleInput, output: Path, *, cache_dir: Path, lock_fd: int | None,
             runner: tuple[str, ...]) -> int:
    command = (*runner, str(item.scheme_path.resolve()), "--out", str(output), "--cache-dir", str(cache_dir))
    with output.with_suffix(".stderr").open("wb") as stderr:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=stderr,
            cwd=Path(__file__).resolve().parents[3], env=child_process_env(threads=1),
            pass_fds=() if lock_fd is None else (lock_fd,), start_new_session=True)
    try:
        return process.wait()
    except BaseException:
        _stop(process)
        raise


def _reuse(folder: Path, previous: ModalSummary | None, item: RoleInput, identity: str) -> ModalRole | None:
    if previous is None or item.scheme is None:
        return None
    for saved in sorted(previous.roles, key=lambda r: r.role != item.record.role):
        if saved.state != "diagnosed_not_scored" or saved.placement_digest != item.record.placement_digest:
            continue
        try:
            diagnosis = read_diagnosis(folder, saved, item.scheme, identity=identity)
        except (OSError, ValueError):
            continue
        if diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
            output = saved.diagnosis_file
            if saved.role != item.record.role:
                path = summary_path(folder).parent / f"{item.record.role}-{uuid4().hex}.json"
                save_diagnosis(diagnosis, path)
                output = path.name
            return item.record.model_copy(update={"state": diagnosis.state.value, "diagnosis_file": output})
    return None


def _persist_record(folder: Path, record: ModalRole) -> ModalRole:
    """入口失敗與停止也留角色文件；摘要保留 skipped／stopped 的執行狀態。"""
    state = ModalDiagnosisState.FAILED if record.state == "failed" else ModalDiagnosisState.NOT_COMPUTED
    diagnosis = ModalDiagnosis(state=state, reason_text=record.reason_text or "沒有算完")
    output = summary_path(folder).parent / f"{record.role}-{uuid4().hex}.json"
    save_diagnosis(diagnosis, output)
    return record.model_copy(update={"diagnosis_file": output.name})


def _duplicate(folder: Path, item: RoleInput, source: ModalRole) -> ModalRole:
    record = item.record.model_copy(update={"state": source.state, "reason_text": source.reason_text,
        "duplicate_of": source.role, "exit_code": source.exit_code})
    if source.diagnosis_file is None or item.scheme is None:
        return _persist_record(folder, record)
    diagnosis = read_diagnosis(folder, source, item.scheme)
    output = summary_path(folder).parent / f"{record.role}-{uuid4().hex}.json"
    save_diagnosis(diagnosis, output)
    return record.model_copy(update={"diagnosis_file": output.name})


def _diagnose(folder: Path, item: RoleInput, *, cache_dir: Path, identity: str,
              lock_fd: int | None, runner: tuple[str, ...]) -> ModalRole:
    assert item.scheme is not None
    output = summary_path(folder).parent / f"{item.record.role}-{uuid4().hex}.json"
    record = item.record.model_copy(update={"diagnosis_file": output.name})
    if key_from_scheme(item.scheme) is None:
        diagnosis = ModalDiagnosis(state=ModalDiagnosisState.OUT_OF_SCOPE, reason_code=scheme_scope_reason(item.scheme))
    else:
        code = _compute(item, output, cache_dir=cache_dir, lock_fd=lock_fd, runner=runner)
        record = record.model_copy(update={"exit_code": code})
        if code in (143, -signal.SIGTERM, -signal.SIGKILL, 129, -signal.SIGHUP):
            diagnosis = ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED, reason_text=f"已停止（離開碼 {code}）")
            save_diagnosis(diagnosis, output)
            return record.model_copy(update={"state": "stopped", "reason_text": diagnosis.reason_text})
        if code != 0:
            original = output.with_suffix(".stderr").read_text(errors="replace")
            diagnosis = ModalDiagnosis(state=ModalDiagnosisState.FAILED,
                reason_text=f"模態工作非正常結束（離開碼 {code}）；錯誤輸出原文：\n{original}")
        else:
            try:
                diagnosis = read_diagnosis(folder, record, item.scheme, identity=identity)
            except (OSError, ValueError) as error:
                diagnosis = ModalDiagnosis(state=ModalDiagnosisState.FAILED,
                    reason_text=f"離開碼 0 但沒有有效診斷文件：{error}")
    save_diagnosis(diagnosis, output)
    return document_record(record, diagnosis)


def _run_roles(store: SearchStore, summary: ModalSummary, inputs: tuple[RoleInput, ...], *,
               cache_dir: Path, lock_fd: int | None, runner: tuple[str, ...], previous: ModalSummary | None) -> ModalSummary:
    identity = modal_identity()
    records = list(summary.roles)
    seen: dict[str, ModalRole] = {}
    for index, item in enumerate(inputs):
        if item.scheme is None or item.record.state == "failed":
            records[index] = _persist_record(store.path, item.record)
            summary = summary.model_copy(update={"roles": tuple(records)})
            write_summary(store.path, summary)
            continue
        digest = item.record.placement_digest
        assert digest is not None
        duplicate = seen.get(digest)
        if duplicate is not None:
            records[index] = _duplicate(store.path, item, duplicate)
        else:
            records[index] = item.record.model_copy(update={"state": "running"})
            write_summary(store.path, summary.model_copy(update={"roles": tuple(records)}))
            try:
                records[index] = _reuse(store.path, previous, item, identity) or _diagnose(store.path, item,
                    cache_dir=cache_dir, identity=identity, lock_fd=lock_fd, runner=runner)
            except KeyboardInterrupt:
                records[index:] = [_persist_record(store.path, r.model_copy(update={"state": "stopped", "reason_text": "已停止，沒有算完；人手接續後會再試"}))
                                   for r in records[index:]]
                sys.stderr.write(STOPPED_NOTE + "\n")
                summary = summary.model_copy(update={"roles": tuple(records), "completed": True})
                write_summary(store.path, summary)
                return summary
            except Exception as error:
                records[index] = _persist_record(store.path, item.record.model_copy(update={"state": "failed", "reason_text": str(error)}))
            seen[digest] = records[index]
        summary = summary.model_copy(update={"roles": tuple(records)})
        write_summary(store.path, summary)
    return summary.model_copy(update={"completed": True})


def attach_modal(store: SearchStore, *, status: SearchStatus, cache_dir: Path, lock_fd: int | None = None,
                 runner: tuple[str, ...] = DEFAULT_RUNNER) -> ModalSummary:
    """在 auto 持有的鎖內執行；只讀兩本帳及結果，另存附件與每次新檔名。"""
    cache_dir = cache_dir.resolve()
    try:
        previous = read_summary(store.path)
    except (OSError, ValueError):
        previous = None
    inputs = role_inputs(store, status)
    summary = ModalSummary(cache_dir=str(cache_dir), conclusion=status.outer.conclusion,
                           snapshot=snapshot_of(status), roles=tuple(item.record for item in inputs))
    write_summary(store.path, summary)
    reason = _eligibility(status, inputs)
    if reason:
        summary = summary.model_copy(update={"reason_text": reason, "completed": True,
            "roles": tuple(_persist_record(store.path, r.model_copy(update={"state": "skipped", "reason_text": reason})) for r in summary.roles)})
    else:
        write_summary(store.path, summary)
        summary = _run_roles(store, summary, inputs, cache_dir=cache_dir, lock_fd=lock_fd, runner=runner, previous=previous)
    write_summary(store.path, summary)
    return summary


def record_attachment_error(store: SearchStore, status: SearchStatus, cache_dir: Path,
                            error: BaseException, *, stopped: bool) -> None:
    """附件整段出错也不能越過 CLI 的獨立出口；讀寫本身失敗則標準錯誤保留原文。"""
    try:
        summary = read_summary(store.path) or ModalSummary(cache_dir=str(cache_dir.resolve()),
            conclusion=status.outer.conclusion, snapshot=snapshot_of(status),
            roles=tuple(item.record for item in role_inputs(store, status)))
        state = "stopped" if stopped else "failed"
        reason = "已停止，沒有算完；人手接續後會再試" if stopped else str(error)
        roles = tuple(r if r.state == "diagnosed_not_scored" else _persist_record(store.path, r.model_copy(update={"state": state, "reason_text": reason}))
                      for r in summary.roles)
        write_summary(store.path, summary.model_copy(update={"roles": roles, "reason_text": reason, "completed": True}))
    except (Exception, KeyboardInterrupt) as recording_error:
        sys.stderr.write(f"低頻診斷摘要未能保存：{recording_error}\n")
