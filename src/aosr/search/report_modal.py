"""低頻附件的唯讀文字投影：房間一次、每份擺位主位前三名，不計分。"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from aosr.config.quality_targets import QualityTargets
from aosr.reporting.display import (
    MODAL_GROUP_NOTE, MODAL_PLACEMENT_NOTE, modal_check_text, modal_guarantee_text,
    modal_reference_text, speaker_label,
)
from aosr.reporting.modal_diagnosis import modal_identity
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis
from aosr.reporting.modal_diagnosis_readout import DiagnosisReadout, read_modal_diagnosis
from aosr.scoring.low_frequency_decay_reference import load_reference
from aosr.search.modal_record import (
    NOT_SCORED, RESULT_PAGE, STALE_NOTE, TITLE, ModalRole, document_record, is_stale, read_diagnosis,
    read_summary, role_inputs, role_label, scheme_for_role, summary_lines,
)
from aosr.search.run import SearchStatus
from aosr.search.store import FROZEN, SearchStore
from aosr.reporting.scheme import Scheme


class ModalReport(BaseModel):
    model_config = FROZEN
    lines: tuple[str, ...] = (NOT_SCORED, "未開始（搜尋正常收尾後才補）", RESULT_PAGE)


def _room_lines(diagnosis: ModalDiagnosis, readout: DiagnosisReadout) -> tuple[str, ...]:
    room = diagnosis.room_layer
    assert room is not None and readout.resonances is not None and readout.groups is not None
    exceeding = sum(row.excess_t60_s is not None and row.excess_t60_s > 0 for row in readout.resonances)
    outside = sum(row.target_t60_s is None for row in readout.resonances)
    overlap = sum(row.member_count > 1 for row in readout.groups)
    lines = [f"房間層：共振 {len(readout.resonances)} 個；超過目標線 {exceeding} 個；參考線範圍外 {outside} 個",
             f"重疊分組共 {len(readout.groups)} 組；成員數超過 1 的重疊組 {overlap} 組", MODAL_GROUP_NOTE]
    for row in sorted(readout.groups, key=lambda g: (-g.member_count, g.group_id))[:3]:
        group = room.groups[row.group_id]
        label = "重疊組" if row.member_count > 1 else "單獨共振組"
        lines.append(f"{label} {row.group_id}：成員 {row.member_count}；{group.lower_hz:.2f}–{group.upper_hz:.2f} Hz；"
                     f"T60 範圍 {row.t60_range_s[0]:.4g}–{row.t60_range_s[1]:.4g} 秒；"
                     f"超過目標線成員 {row.exceeding_member_count}；參考線範圍外成員 {row.not_assessed_member_count}")
    lines.append(f"求解自檢：{modal_guarantee_text(room.check_summary)}；{modal_check_text(room.check_summary)}")
    return tuple(lines)


def _placement_lines(record: ModalRole, readout: DiagnosisReadout, primary: str,
                     roles: dict[str, str]) -> tuple[str, ...]:
    assert readout.placements is not None and readout.resonances is not None
    by_index = {row.mode_index: row for row in readout.resonances}
    lines = []
    for pair in readout.placements:
        if pair.receiver_id != primary:
            continue
        resonances = []
        for size in pair.sorted_resonances[:3]:
            row = by_index[size.mode_index]
            relative = "無有限相對 dB" if size.db_relative_to_pair_strongest_resonance is None else f"{size.db_relative_to_pair_strongest_resonance:.2f} dB"
            resonances.append(f"{row.frequency_hz:.2f} Hz（T60 {row.t60_s:.4g} 秒，"
                              f"{modal_reference_text(row.excess_t60_s)}，{relative}）")
        label = speaker_label(roles.get(pair.speaker_id, pair.speaker_id))
        lines.append(f"{role_label(record)}／{label} → 主位：" + ("；".join(resonances) or "沒有共振"))
    return tuple(lines)


def modal_report(store: SearchStore, status: SearchStatus, registry: QualityTargets) -> ModalReport:
    """壞附件只壞這一段；從不呼叫快取鎖、求解或診斷入口。"""
    try:
        summary = read_summary(store.path)
        if summary is None:
            return ModalReport(lines=(*summary_lines(None, status=status), RESULT_PAGE))
        inputs = role_inputs(store, status)
        by_role = {item.record.role: item for item in inputs}
        diagnoses: list[tuple[ModalRole, ModalDiagnosis, Scheme]] = []
        records = []
        old_roles = []
        for record in summary.roles:
            item = by_role.get(record.role)
            if item is not None and item.record.trial_number != record.trial_number:
                then = "原方案" if record.trial_number is None else f"試算 {record.trial_number}"
                now = "原方案" if item.record.trial_number is None else f"試算 {item.record.trial_number}"
                old_roles.append(f"{role_label(record)}：舊的：當時是{then}，現在是{now}")
            if record.diagnosis_file is not None and record.state not in ("running", "stopped", "skipped"):
                try:
                    try:
                        scheme = scheme_for_role(store, record)
                    except (OSError, ValueError) as error:
                        raise ValueError(f"方案檔讀不回：{error}") from error
                    diagnosis = read_diagnosis(store.path, record, scheme)
                    if item is None or item.record.trial_number == record.trial_number:
                        record = document_record(record, diagnosis)
                    if record.state == "diagnosed_not_scored" and record.duplicate_of is None:
                        diagnoses.append((record, diagnosis, scheme))
                except (OSError, ValueError) as error:
                    record = record.model_copy(update={"state": "failed", "reason_text": f"診斷文件讀不回：{error}"})
            records.append(record)
        lines = list(summary_lines(summary.model_copy(update={"roles": tuple(records)})))
        lines.extend(old_roles)
        if is_stale(summary, inputs, status):
            lines.append(STALE_NOTE)
        reference = load_reference(registry.purpose(store.settings.purpose))
        for index, (record, diagnosis, scheme) in enumerate(diagnoses):
            readout = read_modal_diagnosis(diagnosis, reference)
            if index == 0:
                lines.extend(_room_lines(diagnosis, readout))
                lines.append(MODAL_PLACEMENT_NOTE)
            if diagnosis.modal_identity != modal_identity():
                lines.append(f"{role_label(record)}：模態身分跟現在的程式不同，網頁不會沿用")
            roles = {channel.speaker_id: channel.role for channel in scheme.channel_group.channels}
            lines.extend(_placement_lines(record, readout, scheme.receiver_set.primary.receiver_id, roles))
        return ModalReport(lines=(*lines, RESULT_PAGE))
    except (OSError, ValueError, KeyError) as error:
        return ModalReport(lines=(NOT_SCORED, f"低頻診斷附件讀不回：{error}", RESULT_PAGE))


def modal_text(report: ModalReport) -> str:
    return "\n".join((TITLE, *(line.replace("\r", " ").replace("\n", "；") for line in report.lines)))
