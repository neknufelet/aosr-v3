"""診斷自己的畫面資料；只讀品質參考線，不接候選、評分或排名。"""
from __future__ import annotations

from typing import TypedDict

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.gui.labels import listening_point_label, speaker_label
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
from aosr.reporting.modal_diagnosis_readout import DiagnosisReadout, read_modal_diagnosis
from aosr.reporting.scheme import Scheme
from aosr.scoring.low_frequency_decay_reference import ReferenceOrigin, load_reference

REFERENCE_SECONDS = 564
REFERENCE_ROOM = "6×4×3 m、六面 4ρc 的房間，2026-10-06 實測"
COMPUTE_NOTE = (f"首次計算參考約 {REFERENCE_SECONDS} 秒（{REFERENCE_ROOM}；這份方案若不同，時間尚未量測）；"
                "記憶體約 5 GB。同房同材料算過之後，換擺位約十幾秒；同擺位直接沿用報告。低頻拖尾不計分，不改排名。")
GROUP_NOTE = "重疊分組：相鄰共振的峰寬互相重疊就連成一組；不是一個共振，組內每個共振仍在模態表逐列列出"
PLACEMENT_NOTE = "相對 dB：相對同一喇叭到同一座位最強的共振。前幾名只是版面長度，不是品質門檻；完整排序與整群大小剖面可展開。"
STATE_TEXT = {ModalDiagnosisState.DIAGNOSED_NOT_SCORED: "已診斷不計分",
              ModalDiagnosisState.NOT_COMPUTED: "未計算", ModalDiagnosisState.FAILED: "失敗",
              ModalDiagnosisState.OUT_OF_SCOPE: "範圍外"}
ORIGIN_TEXT = {ReferenceOrigin.LITERATURE_ANCHOR: "文獻錨點",
               ReferenceOrigin.ENGINEERING_INTERPOLATION: "工程內插",
               ReferenceOrigin.ENGINEERING_EXTENSION: "工程延伸",
               ReferenceOrigin.NOT_ASSESSED: "參考線範圍外，未評估"}


class ModeRow(TypedDict):
    index: int
    frequency_text: str
    kind_text: str
    t60_text: str
    q_text: str
    target_text: str
    excess_text: str
    origin_text: str
    group_text: str


class GroupRow(TypedDict):
    member_text: str
    lower_text: str
    upper_text: str
    group_text: str
    t60_range_text: str
    exceeding_text: str
    outside_text: str


class PlacementRow(TypedDict):
    index: int
    frequency_text: str
    magnitude_text: str
    group_magnitude_text: str
    relative_text: str


class PlacementView(TypedDict):
    heading_text: str
    rows: list[PlacementRow]
    profiles: list[list[str]]


class ModalView(TypedDict, total=False):
    state: str
    state_text: str
    reason_text: str
    can_calculate: bool
    compute_note: str
    group_note: str
    placement_note: str
    modes: list[ModeRow]
    groups: list[GroupRow]
    placements: list[PlacementView]
    guarantee_text: str
    counts: list[list[str]]
    check_note: str


def _number(value: float | None, unit: str) -> str:
    return "未定義" if value is None else f"{value:.4g} {unit}".strip()


def _modes(diagnosis: ModalDiagnosis, readout: DiagnosisReadout) -> list[ModeRow]:
    assert diagnosis.room_layer is not None and readout.resonances is not None
    references = {row.mode_index: row for row in readout.resonances}
    rows: list[ModeRow] = []
    kinds = {"oscillating_resonance": "共振", "static": "靜態", "nonoscillating_decay": "純衰減"}
    for mode in diagnosis.room_layer.modes:
        target = references.get(mode.mode_index)
        outside = target is not None and target.target_t60_s is None
        rows.append(ModeRow(index=mode.mode_index, frequency_text=f"{mode.frequency_hz:.2f} Hz",
            kind_text=kinds.get(mode.kind.value, mode.kind.value), t60_text=_number(mode.t60_s, "秒"),
            q_text=_number(mode.q, ""), group_text="不進群" if mode.group_id is None else str(mode.group_id),
            target_text="參考線範圍外，未評估" if outside else _number(target.target_t60_s, "秒") if target else "未評估",
            excess_text="未評估" if target is None or outside else _number(target.excess_t60_s, "秒"),
            origin_text=ORIGIN_TEXT[target.target_origin] if target else "未評估"))
    return rows


def _groups(diagnosis: ModalDiagnosis, readout: DiagnosisReadout) -> list[GroupRow]:
    assert diagnosis.room_layer is not None and readout.groups is not None
    bounds = {group.group_id: group for group in diagnosis.room_layer.groups}
    return [GroupRow(group_text=str(row.group_id), member_text=str(row.member_count),
        lower_text=f"{bounds[row.group_id].lower_hz:.2f} Hz", upper_text=f"{bounds[row.group_id].upper_hz:.2f} Hz",
        t60_range_text=f"{row.t60_range_s[0]:.4g}–{row.t60_range_s[1]:.4g} 秒",
        exceeding_text=str(row.exceeding_member_count), outside_text=str(row.not_assessed_member_count))
        for row in readout.groups]


def _placements(diagnosis: ModalDiagnosis, readout: DiagnosisReadout, scheme: Scheme) -> list[PlacementView]:
    assert diagnosis.room_layer is not None and readout.placements is not None
    modes = diagnosis.room_layer.modes
    roles = {c.speaker_id: c.role for c in scheme.channel_group.channels}
    views: list[PlacementView] = []
    for pair in readout.placements:
        rows = [PlacementRow(index=row.mode_index, frequency_text=f"{modes[row.mode_index].frequency_hz:.2f} Hz",
            magnitude_text=f"{row.resonance_magnitude:.6g}", group_magnitude_text=f"{row.group_magnitude:.6g}",
            relative_text="無有限相對 dB" if row.db_relative_to_pair_strongest_resonance is None
            else f"{row.db_relative_to_pair_strongest_resonance:.2f} dB") for row in pair.sorted_resonances]
        profiles = [[str(group.group_id), str(index), f"{frequency:.2f} Hz", f"{magnitude:.6g}"]
                    for group in pair.group_profiles for index, frequency, magnitude in zip(
                        group.mode_indices, group.frequencies_hz, group.group_magnitude, strict=True)]
        views.append(PlacementView(heading_text=f"{speaker_label(roles.get(pair.speaker_id, pair.speaker_id))} → "
            f"{listening_point_label(pair.receiver_id)}", rows=rows, profiles=profiles))
    return views


def build_modal_view(diagnosis: ModalDiagnosis, scheme: Scheme) -> ModalView:
    """四態分開；未評估不印零，群沒有代表成員，保證高度零不印無限大。"""
    reason = diagnosis.reason_text or {"unsupported_room": "房型不是長方形",
        "unsupported_impedance": "六面阻抗必須齊全、為有限正實數"}.get(diagnosis.reason_code or "", diagnosis.reason_code or "")
    view = ModalView(state=diagnosis.state.value, state_text=STATE_TEXT[diagnosis.state], reason_text=reason,
        can_calculate=diagnosis.state is ModalDiagnosisState.NOT_COMPUTED, compute_note=COMPUTE_NOTE)
    if diagnosis.state is not ModalDiagnosisState.DIAGNOSED_NOT_SCORED:
        return view
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose(scheme.purpose)
    readout = read_modal_diagnosis(diagnosis, load_reference(purpose))
    assert diagnosis.room_layer is not None
    summary = diagnosis.room_layer.check_summary
    view["group_note"], view["placement_note"] = GROUP_NOTE, PLACEMENT_NOTE
    view["modes"], view["groups"] = _modes(diagnosis, readout), _groups(diagnosis, readout)
    view["placements"] = _placements(diagnosis, readout, scheme)
    view["guarantee_text"] = ("沒有保證（保證找齊的衰減高度為零，最低 T60 未定義）" if summary.guaranteed_decay_rate_rad_s == 0
        else f"保證找齊的衰減高度：{summary.guaranteed_decay_rate_rad_s:.6g} rad/s（弧度／秒）；最低 T60：{summary.guaranteed_min_t60_s:.6g} 秒")
    view["counts"] = [[f"{b.lower_hz:.2f}–{b.upper_hz:.2f} Hz", str(b.found_resonances), f"{b.weyl_estimate:.4g}",
                 f"{b.found_minus_weyl:.4g}", "未提供" if b.rigid_reference_count is None else str(b.rigid_reference_count),
                 "未提供" if b.found_minus_rigid is None else str(b.found_minus_rigid)] for b in summary.count_bands]
    view["check_note"] = (f"{summary.weyl_terms}；靜態 {summary.static_count}、零根延續 {summary.zero_mode_continuation_count}、"
        f"過阻尼 {summary.overdamped_count}、未確認衰減 {summary.unconfirmed_decay_count}、上限外返回 {summary.returned_above_limit_count}。"
        "保證以求解收斂且最近根選取完整為前提，只在開圓內成立；強阻尼完備性仍有未驗限制。")
    return view
