"""外圈保存完成後補附件：獨立子行程、獨立停止、搜尋離開碼不受影響。"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from aosr.reporting.modal_diagnosis import modal_identity
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.modal_lookup import key_from_scheme, save_diagnosis, scheme_scope_reason
from aosr.runtime import child_process_env
from aosr.reporting.scheme import Scheme
from aosr.search.modal_record import (
    AttachmentState, ModalRole, ModalSummary, RoleInput, document_record, read_diagnosis, read_summary, role_inputs,
    scheme_for_role, summary_path, write_summary,
)
from aosr.search.outer_status import OUTER_MESSAGES, snapshot_of
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore

DEFAULT_RUNNER = (sys.executable, "-m", "aosr.reporting.scheme_cli", "modal")
STOPPED_NOTE = "低頻診斷被停止，搜尋結果不受影響"
FIXED_ROOM_NOTE = "房間或材料不固定，這次不補（成本另評估）"
OWNED_FILE = re.compile(r"(?:baseline|search_best|refine_best)-[0-9a-f]{32}\.(?:json|stderr)\Z")


def write_stderr(text: str) -> None:
    """標準錯誤寫不出去（例如接在被砍掉的 tee 後面）就不印；離開碼仍照外圈結論。

    只接住寫入當下不夠：寫失敗的字留在緩衝區，Python 結束前清緩衝又失敗，會把離開碼改成 120。
    所以寫失敗時把描述子 2 改接 /dev/null，讓結束前那一次清得掉。
    """
    try:
        sys.stderr.write(text)
        sys.stderr.flush()
    except OSError:
        try:
            if sys.stderr.fileno() == 2:
                devnull = os.open(os.devnull, os.O_WRONLY)
                os.dup2(devnull, 2)
                os.close(devnull)
        except (OSError, ValueError, AttributeError):
            pass


def _notice(text: str) -> None:
    write_stderr(text + "\n")


class AttachmentRecorded(Exception):
    """附件出錯或被停，已經照上一份摘要寫好收尾；命令列只要印一行。"""

    def __init__(self, error: BaseException, *, stopped: bool) -> None:
        super().__init__(str(error))
        self.stopped = stopped


def _finish(folder: Path, summary: ModalSummary, *keep: ModalSummary | None) -> None:
    """先保存有完成記號的摘要，再清掉本附件命名、新摘要與上一份摘要都沒指到的文件及錯誤輸出。

    留上一份那一代：報告不拿資料夾鎖，可能剛讀完上一份摘要、還沒讀它指到的文件；最多留兩代，資料夾不會一直長。
    """
    write_summary(folder, summary)
    if not summary.completed:
        return
    roles = (*summary.roles, *(role for kept in keep if kept is not None for role in kept.roles))
    referenced = {Path(r.diagnosis_file).stem for r in roles if r.diagnosis_file is not None}
    for path in summary_path(folder).parent.iterdir():
        if OWNED_FILE.fullmatch(path.name) and path.stem not in referenced and path.is_file():
            path.unlink(missing_ok=True)


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
        pass
    try:
        process.wait(timeout=2)
    except BaseException as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        if not isinstance(error, subprocess.TimeoutExpired):
            raise


def _compute(item: RoleInput, output: Path, *, cache_dir: Path, lock_fd: int | None,
             runner: tuple[str, ...]) -> int:
    command = (*runner, str(item.scheme_path.resolve()), "--out", str(output.resolve()), "--cache-dir", str(cache_dir))
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
            return item.record.model_copy(update={"state": diagnosis.state.value, "diagnosis_file": saved.diagnosis_file})
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
    read_diagnosis(folder, source, item.scheme)
    return record.model_copy(update={"diagnosis_file": source.diagnosis_file})


def _nonzero_diagnosis(output: Path, code: int) -> ModalDiagnosis:
    original = output.with_suffix(".stderr").read_text(errors="replace")
    external_signal = {-15: 15, 143: 15, -1: 1, 129: 1}.get(code)
    if external_signal is not None:
        reason = f"已停止（子行程被外部訊號 {external_signal} 停止）"
        state = ModalDiagnosisState.NOT_COMPUTED
    else:
        reason = f"模態工作非正常結束（離開碼 {code}）"
        if code in (-9, 137):
            reason += "；可能是記憶體不足被系統強制結束"
        state = ModalDiagnosisState.FAILED
    return ModalDiagnosis(state=state, reason_text=f"{reason}；錯誤輸出原文：\n{original}")


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
        if code != 0:
            diagnosis = _nonzero_diagnosis(output, code)
            if code in (143, -signal.SIGTERM, 129, -signal.SIGHUP):
                save_diagnosis(diagnosis, output)
                return record.model_copy(update={"state": "stopped", "reason_text": diagnosis.reason_text})
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
                summary = summary.model_copy(update={"roles": tuple(records), "completed": True})
                _finish(store.path, summary, previous)
                _notice(STOPPED_NOTE)
                return summary
            except Exception as error:
                records[index] = _persist_record(store.path, item.record.model_copy(update={"state": "failed", "reason_text": str(error)}))
            seen[digest] = records[index]
        summary = summary.model_copy(update={"roles": tuple(records)})
        write_summary(store.path, summary)
    return summary.model_copy(update={"completed": True})


def _kept_diagnosis(folder: Path, record: ModalRole, scheme: Scheme | None, previous: ModalSummary | None) -> ModalRole | None:
    """上一份摘要裡角色與擺位都相同、文件讀得回且擺位對得上的已診斷結果；沒有就回 None。"""
    if previous is None or scheme is None:
        return None
    for saved in previous.roles:
        if (saved.role, saved.placement_digest) != (record.role, record.placement_digest):
            continue
        if saved.state != "diagnosed_not_scored":
            continue
        try:
            diagnosis = read_diagnosis(folder, saved, scheme)
            if diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
                return record.model_copy(update={"state": saved.state, "reason_text": saved.reason_text,
                    "diagnosis_file": saved.diagnosis_file, "duplicate_of": saved.duplicate_of, "exit_code": saved.exit_code})
        except (OSError, ValueError):
            pass
    return None


def _settled_duplicates(roles: tuple[ModalRole, ...]) -> tuple[ModalRole, ...]:
    """沿用上一份的已診斷時，「與 X 同擺位」只在 X 這一份也是已診斷、指同一份文件時才留著。

    上一次搜尋第一名與細算第一名同擺位、這一次搜尋第一名換了擺位：照抄記號會寫出錯的「同擺位」，
    報告也會因為以為是複本而不印細算第一名的診斷內容。
    """
    by_role = {role.role: role for role in roles}

    def settled(role: ModalRole) -> ModalRole:
        target = by_role.get(role.duplicate_of) if role.duplicate_of is not None else None
        if role.duplicate_of is None or (target is not None and target.state == "diagnosed_not_scored"
                                         and target.diagnosis_file == role.diagnosis_file):
            return role
        return role.model_copy(update={"duplicate_of": None})

    return tuple(settled(role) for role in roles)


def _skipped_role(folder: Path, item: RoleInput, previous: ModalSummary | None, reason: str) -> ModalRole:
    return (_kept_diagnosis(folder, item.record, item.scheme, previous)
            or _persist_record(folder, item.record.model_copy(update={"state": "skipped", "reason_text": reason})))


def attach_modal(store: SearchStore, *, status: SearchStatus, cache_dir: Path, lock_fd: int | None = None,
                 runner: tuple[str, ...] = DEFAULT_RUNNER) -> ModalSummary:
    """在 auto 持有的鎖內執行；只讀兩本帳及結果，另存附件與每次新檔名。

    出錯或被停時照上一份摘要收尾（保留還對得上的已診斷、上一份指到的文件不清），再丟 AttachmentRecorded。
    """
    cache_dir = cache_dir.resolve()
    try:
        previous = read_summary(store.path)
    except (OSError, ValueError):
        previous = None
    try:
        return _attach(store, status=status, cache_dir=cache_dir, lock_fd=lock_fd, runner=runner, previous=previous)
    except (Exception, KeyboardInterrupt) as error:
        stopped = isinstance(error, KeyboardInterrupt)
        record_attachment_error(store, status, cache_dir, error, stopped=stopped, previous=previous)
        raise AttachmentRecorded(error, stopped=stopped) from error


def _attach(store: SearchStore, *, status: SearchStatus, cache_dir: Path, lock_fd: int | None,
            runner: tuple[str, ...], previous: ModalSummary | None) -> ModalSummary:
    inputs = role_inputs(store, status)
    summary = ModalSummary(cache_dir=str(cache_dir), conclusion=status.outer.conclusion,
                           snapshot=snapshot_of(status), roles=tuple(item.record for item in inputs))
    write_summary(store.path, summary)
    reason = _eligibility(status, inputs)
    if reason:
        summary = summary.model_copy(update={"reason_text": reason, "completed": True,
            "roles": _settled_duplicates(tuple(_skipped_role(store.path, item, previous, reason) for item in inputs))})
    else:
        write_summary(store.path, summary)
        summary = _run_roles(store, summary, inputs, cache_dir=cache_dir, lock_fd=lock_fd, runner=runner, previous=previous)
    _finish(store.path, summary, previous)
    return summary


def _closed_role(store: SearchStore, record: ModalRole, previous: ModalSummary | None, state: AttachmentState,
                 reason: str) -> ModalRole:
    if record.state == "diagnosed_not_scored":
        return record
    try:
        scheme: Scheme | None = scheme_for_role(store, record)
    except (OSError, ValueError):
        scheme = None
    return (_kept_diagnosis(store.path, record, scheme, previous)
            or _persist_record(store.path, record.model_copy(update={"state": state, "reason_text": reason})))


def record_attachment_error(store: SearchStore, status: SearchStatus, cache_dir: Path,
                            error: BaseException, *, stopped: bool, previous: ModalSummary | None = None) -> None:
    """附件整段出错也不能越過 CLI 的獨立出口；讀寫本身失敗則標準錯誤保留原文。

    磁碟上的摘要可能已被這一次附件蓋成進行中那一份：還對得上的已診斷照上一份（`previous`）保留，
    清理時上一份與磁碟上那一份指到的文件都不刪。
    """
    try:
        summary = read_summary(store.path) or ModalSummary(cache_dir=str(cache_dir.resolve()),
            conclusion=status.outer.conclusion, snapshot=snapshot_of(status),
            roles=tuple(item.record for item in role_inputs(store, status)))
        state: AttachmentState = "stopped" if stopped else "failed"
        reason = "已停止，沒有算完；人手接續後會再試" if stopped else str(error)
        roles = _settled_duplicates(tuple(_closed_role(store, r, previous, state, reason) for r in summary.roles))
        _finish(store.path, summary.model_copy(update={"roles": roles, "reason_text": reason, "completed": True}),
                summary, previous)
    except (Exception, KeyboardInterrupt) as recording_error:
        _notice(f"低頻診斷摘要未能保存：{recording_error}")
