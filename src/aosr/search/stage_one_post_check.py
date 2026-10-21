"""第一階段每一入選席的串行低頻診斷；只寫自己的附件，不計分。"""
from __future__ import annotations

from pathlib import Path
from time import monotonic
from uuid import uuid4

from aosr.reporting.display import modal_reason
from aosr.reporting.modal_diagnosis import modal_identity
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.modal_lookup import placement_digest, save_diagnosis
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.search.modal_attach import DEFAULT_RUNNER, _compute, _nonzero_diagnosis
from aosr.search.modal_record import ModalRole, RoleInput
from aosr.search.run import ComputeFailed, IdentityChanged
from aosr.search.stage_one_post_check_record import (
    PostCheckSeat, StageOnePostCheckSummary, read_diagnosis, read_summary, summary_path, write_summary,
)
from aosr.search.stage_one_refine_record import StageOneRefineStatus, read_stage_one_refined_result
from aosr.search.store import SearchStore

STOPPED = "入選後檢查已停止，細算結果不受影響"


def _seat_inputs(store: SearchStore, refined: StageOneRefineStatus) -> tuple[list[PostCheckSeat], dict[int, Scheme]]:
    results = {row.trial_number: row for row in refined.results}
    seats = []
    schemes = {}
    for selected in refined.selection.candidates:
        number = selected.trial_number
        path = store.scheme_path_for(store.refine_result_path(number))
        seat = PostCheckSeat(trial_number=number, roles=selected.roles, scheme_file=path.relative_to(store.path).as_posix())
        row = results.get(number)
        try:
            if row is None or row.state not in ("scored", "unassessed", "incomparable"):
                raise ValueError("細算沒有可讀結果" if row is None else f"細算狀態 {row.state}：{row.reason}")
            read_stage_one_refined_result(store, number)
            scheme = load_scheme(path)
            seat = seat.model_copy(update={"placement_digest": placement_digest(scheme)})
            schemes[number] = scheme
        except (OSError, ValueError, ComputeFailed, IdentityChanged) as error:
            seat = seat.model_copy(update={"state": "skipped", "reason_text": f"細算結果讀不回或尚未完成，跳過：{error}"})
        seats.append(seat)
    return seats, schemes


def _reuse(store: SearchStore, previous: StageOnePostCheckSummary | None, seat: PostCheckSeat,
           scheme: Scheme, identity: str) -> PostCheckSeat | None:
    if previous is None:
        return None
    for saved in previous.seats:
        if (saved.trial_number, saved.placement_digest, saved.state) != (seat.trial_number, seat.placement_digest, "diagnosed_not_scored"):
            continue
        try:
            diagnosis = read_diagnosis(store.path, saved, scheme, identity=identity)
        except (OSError, ValueError):
            continue
        if diagnosis.state is ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
            return seat.model_copy(update={"state": saved.state, "reason_text": saved.reason_text,
                "exit_code": saved.exit_code, "diagnosis_file": saved.diagnosis_file, "seconds": saved.seconds})
    return None


def _diagnose(store: SearchStore, seat: PostCheckSeat, scheme: Scheme, *, cache_dir: Path,
              lock_fd: int | None, runner: tuple[str, ...], identity: str) -> PostCheckSeat:
    output = _output(store, seat)
    record = seat.model_copy(update={"diagnosis_file": output.name})
    # 只借子行程啟停機制：角色欄是轉接資料，從不送入第二階段附件的保存或資格判定。
    item = RoleInput(ModalRole(role="refine_best", trial_number=seat.trial_number), scheme,
                     store.scheme_path_for(store.refine_result_path(seat.trial_number)), store.refine_result_path(seat.trial_number))
    start = monotonic()
    try:
        code = _compute(item, output, cache_dir=cache_dir, lock_fd=lock_fd, runner=runner)
        record = record.model_copy(update={"exit_code": code})
        if code:
            diagnosis = _nonzero_diagnosis(output, code)
        else:
            diagnosis = read_diagnosis(store.path, record, scheme, identity=identity)
        state = "stopped" if code in (-15, 143, -1, 129) else diagnosis.state.value
        record = record.model_copy(update={"state": state, "reason_text": modal_reason(diagnosis)})
    except KeyboardInterrupt:
        diagnosis = ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED, reason_text=STOPPED)
        record = record.model_copy(update={"state": "stopped", "reason_text": STOPPED})
    except Exception as error:
        diagnosis = ModalDiagnosis(state=ModalDiagnosisState.FAILED, reason_text=f"模態診斷失敗：{error}")
        record = record.model_copy(update={"state": "failed", "reason_text": diagnosis.reason_text})
    record = record.model_copy(update={"seconds": monotonic() - start})
    save_diagnosis(diagnosis, output)
    return record


def run_stage_one_post_check(store: SearchStore, *, cache_dir: Path, lock_fd: int | None,
                             runner: tuple[str, ...] = DEFAULT_RUNNER) -> StageOnePostCheckSummary:
    """由持鎖的細算入口呼叫；細算完成才建立附件，席次順序照挑席、每席只跑一次。"""
    refined = StageOneRefineStatus.model_validate_json((store.path / "stage_one_refine.json").read_bytes())
    if store.settings.stage is None or refined.state != "done":
        raise ValueError("入選後檢查只收已完成細算的第一階段搜尋資料夾")
    identity = modal_identity()
    try:
        previous = read_summary(store.path)
    except (OSError, ValueError):
        previous = None
    if previous is not None and (previous.selection != refined.selection or previous.modal_identity != identity):
        previous = None
    seats, schemes = _seat_inputs(store, refined)
    summary = StageOnePostCheckSummary(selection=refined.selection, modal_identity=identity, seats=tuple(seats))
    write_summary(store.path, summary)
    try:
        for index, seat in enumerate(seats):
            if store.refine_stop_path.exists():
                return _stopped(store, summary, seats, index)
            if seat.state == "skipped":
                seats[index] = _uncomputed(store, seat)
                write_summary(store.path, summary.model_copy(update={"seats": tuple(seats)}))
                continue
            scheme = schemes[seat.trial_number]
            kept = _reuse(store, previous, seat, scheme, identity)
            if kept is not None:
                seats[index] = kept
            else:
                seats[index] = seat.model_copy(update={"state": "running"})
                write_summary(store.path, summary.model_copy(update={"seats": tuple(seats)}))
                seats[index] = _diagnose(store, seat, scheme, cache_dir=cache_dir.resolve(), lock_fd=lock_fd,
                                         runner=runner, identity=identity)
            if seats[index].state == "stopped":
                return _stopped(store, summary, seats, index + 1)
            write_summary(store.path, summary.model_copy(update={"seats": tuple(seats)}))
    except KeyboardInterrupt:
        return _stopped(store, summary, seats, index)
    failed = any(seat.state == "failed" for seat in seats)
    summary = summary.model_copy(update={"state": "failed" if failed else "done", "seats": tuple(seats),
        "message": "入選後檢查有診斷失敗，原因見各席" if failed else "入選後檢查已收尾；不計分、不改名次"})
    write_summary(store.path, summary)
    return summary


def _stopped(store: SearchStore, summary: StageOnePostCheckSummary, seats: list[PostCheckSeat], index: int) -> StageOnePostCheckSummary:
    seats[index:] = [_uncomputed(store, seat.model_copy(update={"state": "stopped", "reason_text": STOPPED}))
                     for seat in seats[index:]]
    summary = summary.model_copy(update={"state": "stopped", "message": STOPPED, "seats": tuple(seats)})
    write_summary(store.path, summary)
    return summary


def _output(store: SearchStore, seat: PostCheckSeat) -> Path:
    output = summary_path(store.path).parent / f"trial-{seat.trial_number:06d}-{uuid4().hex}.json"
    output.with_suffix(".stderr").touch()
    return output


def _uncomputed(store: SearchStore, seat: PostCheckSeat) -> PostCheckSeat:
    """跳過或還沒開工就停止也留一份未計算文件；沒有子行程的錯誤輸出為空檔。"""
    output = _output(store, seat)
    save_diagnosis(ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED, reason_text=seat.reason_text), output)
    return seat.model_copy(update={"diagnosis_file": output.name})
