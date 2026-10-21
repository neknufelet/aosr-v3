"""入選後檢查的唯讀報告；房間層一次、每席主位超線共振全部列出。"""
from __future__ import annotations

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.display import MODAL_STATE_TEXT, speaker_label
from aosr.reporting.modal_diagnosis import modal_identity
from aosr.reporting.modal_diagnosis_readout import DiagnosisReadout, ResonanceReadout, read_modal_diagnosis
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.scoring.low_frequency_decay_reference import LowFrequencyDecayReference, load_reference
from aosr.search.modal_record import NOT_SCORED
from aosr.search.stage_one_post_check_record import PostCheckSeat, read_diagnosis, read_summary
from aosr.search.stage_one_selection_contract import StageOneSelection
from aosr.search.store import SearchStore

TITLE = "入選後檢查（低頻模態診斷）"
PARTICIPATION = "主位參與門檻（『主位參與大』的量法與數值）還沒定，超線的先全列；每一席的主位參與附在下面，只排序、不設 dB 切線"
SUMMARY_STATE_TEXT = {"running": "開始過、沒有收尾紀錄", "done": "已收尾", "stopped": "已停止", "failed": "有診斷失敗"}
OLD_IDENTITY = "這一段的診斷是舊版模態程式算的（模態身分跟現在的程式不同）；再跑一次細算會整段重算"


def _label(seat: PostCheckSeat, selection: StageOneSelection) -> str:
    selected = next(row for row in selection.candidates if row.trial_number == seat.trial_number)
    roles = {"top_n": f"前 N（搜尋第 {selected.search_rank} 名）", "big_peak_control": "大峰對照", "far_control": "較遠對照"}
    return f"試算 {seat.trial_number}／" + "、".join(roles[role] for role in seat.roles)


def _exceeding(readout: DiagnosisReadout) -> tuple[ResonanceReadout, ...]:
    return tuple(sorted((row for row in readout.resonances or () if row.excess_t60_s is not None and row.excess_t60_s > 0),
                        key=lambda row: (-float(row.excess_t60_s or 0), row.frequency_hz, row.mode_index)))


def _reference_span(reference: LowFrequencyDecayReference) -> str:
    """參考線涵蓋的頻段照登記簿讀：延伸段下端到最高錨點；報告不另抄一份數字。"""
    line = reference.target
    low = line.extension[0] if line.extension is not None else line.frequencies_hz[0]
    return f"{low:g}～{line.frequencies_hz[-1]:g} Hz"


def _room_lines(readout: DiagnosisReadout, span: str) -> list[str]:
    resonances = readout.resonances or ()
    exceeding = _exceeding(readout)
    outside = sum(row.target_t60_s is None for row in resonances)
    lines = [f"房間層：共振 {len(resonances)} 個；超過參考線 {len(exceeding)} 個；參考線範圍外 {outside} 個",
             "超過聽感參考線的共振（照超出量排序）"]
    lines.extend(f"{row.frequency_hz:.2f} Hz｜T60 {row.t60_s:.6f} 秒｜參考線 T60 {row.target_t60_s:.6f} 秒｜"
                 f"超出量 {row.excess_t60_s:.6f} 秒；改善能力尚未確認" for row in exceeding)
    if not exceeding:
        lines.append("這份診斷沒有超線共振")
    lines.append(f"{outside} 根參考線範圍外（{span} 以外），未評估")
    return lines


def _placement_lines(label: str, scheme: Scheme, readout: DiagnosisReadout) -> list[str]:
    roles = {channel.speaker_id: channel.role for channel in scheme.channel_group.channels}
    lines = []
    for pair in readout.placements or ():
        if pair.receiver_id != scheme.receiver_set.primary.receiver_id:
            continue
        sizes = {row.mode_index: row.db_relative_to_pair_strongest_resonance for row in pair.sorted_resonances}
        values = []
        for row in _exceeding(readout):
            db = sizes[row.mode_index]
            relative = "無有限相對 dB" if db is None else f"{db:.2f} dB"
            values.append(f"{row.frequency_hz:.2f} Hz（{relative}）")
        speaker = speaker_label(roles.get(pair.speaker_id, pair.speaker_id))
        lines.append(f"{label}／{pair.speaker_id}（{speaker}） → 主位：" + ("；".join(values) or "沒有超線共振"))
    return lines


def stage_one_post_check_lines(store: SearchStore) -> tuple[str, ...]:
    """壞摘要只壞這段；壞診斷只壞那席，其他名次與診斷繼續顯示。"""
    lines = [TITLE, NOT_SCORED]
    try:
        summary = read_summary(store.path)
        if summary is None:
            return (*lines, "入選後檢查還沒做")
        lines.append(f"入選後檢查狀態：{SUMMARY_STATE_TEXT[summary.state]}；{summary.message}")
        if summary.modal_identity != modal_identity():
            lines.append(OLD_IDENTITY)
        reference = load_reference(load_quality_targets(config_path("quality_targets.toml")).purpose(store.settings.purpose))
        room_lines: list[str] = []
        placement_lines: list[str] = []
        for seat in summary.seats:
            label = f"試算 {seat.trial_number}"
            try:
                label = _label(seat, summary.selection)
                state, reason = seat.state, seat.reason_text
                if state not in ("not_started", "running", "stopped", "skipped"):
                    expected = store.scheme_path_for(store.refine_result_path(seat.trial_number))
                    if seat.scheme_file != expected.relative_to(store.path).as_posix():
                        raise ValueError("摘要的方案路徑不是這一席的細算方案")
                    scheme = load_scheme(expected)
                    diagnosis = read_diagnosis(store.path, seat, scheme, identity=summary.modal_identity)
                    if diagnosis.state.value != state:
                        raise ValueError("診斷狀態與摘要不同")
                    readout = read_modal_diagnosis(diagnosis, reference)
                    if state == "diagnosed_not_scored":
                        if not room_lines:
                            room_lines = _room_lines(readout, _reference_span(reference))
                        placement_lines.extend(_placement_lines(label, scheme, readout))
                state_text = MODAL_STATE_TEXT.get(state, {"not_started": "未開始", "running": "開始過、沒有收尾紀錄",
                    "stopped": "未計算：已停止", "skipped": "跳過"}.get(state, state))
                lines.append(f"{label}：{state_text}；{reason or '無附加原因'}")
            except (OSError, ValueError, KeyError, StopIteration) as error:
                lines.append(f"{label}：失敗；診斷文件讀不回或內容不完整：{error}")
        lines.extend(room_lines or ["房間層尚無可讀診斷，未評估"])
        lines.append(PARTICIPATION)
        lines.extend(placement_lines)
        seconds = [seat.seconds for seat in summary.seats if seat.seconds is not None]
        lines.append(f"總秒數：{sum(seconds):.6f} 秒（有計時的席加總）；沒有計時紀錄 {sum(seat.seconds is None for seat in summary.seats)} 份")
    except (OSError, ValueError, KeyError) as error:
        lines.append(f"入選後檢查摘要讀不回或內容不完整：{error}")
    return tuple(line.replace("\r", " ").replace("\n", "；") for line in lines)
