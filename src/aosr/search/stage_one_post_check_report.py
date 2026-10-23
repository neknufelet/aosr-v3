"""入選後檢查的唯讀報告；依房間鍵各列一份房間層、每席主位超線共振全部列出。"""
from __future__ import annotations

from dataclasses import dataclass

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.display import MODAL_STATE_TEXT, speaker_label
from aosr.reporting.modal_diagnosis import modal_identity
from aosr.reporting.modal_diagnosis_model import ModalKey
from aosr.reporting.modal_diagnosis_readout import DiagnosisReadout, ResonanceReadout, read_modal_diagnosis
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.scoring.low_frequency_decay_reference import LowFrequencyDecayReference, load_reference
from aosr.search.modal_record import NOT_SCORED
from aosr.search.stage_one_post_check_record import PostCheckSeat, read_diagnosis, read_summary
from aosr.search.stage_one_selection_contract import StageOneSelection, role_text
from aosr.search.store import SearchStore

TITLE = "入選後檢查（低頻模態診斷）"
PARTICIPATION = "主位參與門檻（『主位參與大』的量法與數值）還沒定，超線的先全列；每一席的主位參與附在下面，只排序、不設 dB 切線"
SUMMARY_STATE_TEXT = {"running": "開始過、沒有收尾紀錄", "done": "已收尾", "stopped": "已停止", "failed": "有診斷失敗"}
RELATIVE = "主位參與的 dB 是相對值：相對同一支喇叭到同一個座位最強的那根共振（每一席、每支喇叭各自為基準）"
OLD_IDENTITY = "這一段的診斷是舊版模態程式算的（模態身分跟現在的程式不同）；再跑一次細算會整段重算"
NOT_YET = "入選後檢查還沒做"
NO_DIAGNOSIS = ("not_started", "running", "stopped", "skipped")
RESONANCE_HEADINGS = ("席", "頻率（Hz）", "T60（秒）", "參考線 T60（秒）", "超出量（秒）", "主位參與（相對 dB）", "改善能力")
TABLE_NOTE = "下表每一席列出超過聽感參考線的共振（照超出量排序，席的角色見上面各行）；參考線範圍（{span}）以外的共振未評估，只在上面計個數"
_NO_DIAGNOSIS_TEXT = {"not_started": "未開始", "running": "開始過、沒有收尾紀錄", "stopped": "未計算：已停止", "skipped": "跳過"}


def _one_line(text: str) -> str:
    return text.replace("\r", " ").replace("\n", "；")


def _state_text(state: str) -> str:
    return MODAL_STATE_TEXT.get(state, _NO_DIAGNOSIS_TEXT.get(state, state))


def _label(seat: PostCheckSeat, selection: StageOneSelection) -> str:
    selected = next(row for row in selection.candidates if row.trial_number == seat.trial_number)
    return f"試算 {seat.trial_number}／" + "、".join(role_text(role, selected.search_rank) for role in seat.roles)


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


def _participation(scheme: Scheme, readout: DiagnosisReadout) -> dict[int, str]:
    """每根超線共振到主位的相對 dB，各支喇叭並列；跟命令列同一份讀回。"""
    roles = {channel.speaker_id: channel.role for channel in scheme.channel_group.channels}
    parts: dict[int, list[str]] = {row.mode_index: [] for row in _exceeding(readout)}
    for pair in readout.placements or ():
        if pair.receiver_id != scheme.receiver_set.primary.receiver_id:
            continue
        sizes = {row.mode_index: row.db_relative_to_pair_strongest_resonance for row in pair.sorted_resonances}
        speaker = speaker_label(roles.get(pair.speaker_id, pair.speaker_id))
        for index, values in parts.items():
            db = sizes[index]
            values.append(f"{speaker} " + ("無有限相對值" if db is None else f"{db:.1f}"))
    return {index: "／".join(values) or "沒有主位配對" for index, values in parts.items()}


def _resonance_rows(trial_number: int, scheme: Scheme, readout: DiagnosisReadout) -> list[tuple[str, ...]]:
    participation = _participation(scheme, readout)
    label = f"試算 {trial_number}"
    rows: list[tuple[str, ...]] = [(label, f"{row.frequency_hz:.2f}", f"{row.t60_s:.3f}", f"{row.target_t60_s:.3f}", f"{row.excess_t60_s:.3f}",
             participation[row.mode_index], "尚未確認") for row in _exceeding(readout)
            if row.target_t60_s is not None and row.excess_t60_s is not None]
    return rows or [(label, "這一席沒有超線共振", "", "", "", "", "")]


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


@dataclass
class _RoomGroup:
    key: ModalKey
    resonances: tuple[ResonanceReadout, ...] | None
    lines: list[str]


def _group_room(groups: list[_RoomGroup], key: ModalKey, readout: DiagnosisReadout, span: str, *, variable: bool) -> None:
    """完整房間鍵相等才共用一張表；組內共振不同仍是壞診斷。固定房間照舊只有一張表、每席跟第一份比共振。"""
    group = next((group for group in groups if group.key == key), None) if variable else next(iter(groups), None)
    if group is not None:
        if group.resonances != readout.resonances:
            raise ValueError("這一席的房間層跟上面那份不同，不能共用同一張超線表")
        return
    lines = _room_lines(readout, span)
    if variable:
        dimensions = "×".join(f"{float.fromhex(value):.6f}" for value in key.room_hex)
        lines = [f"房間 長×寬×高（公尺）：{dimensions}", f"房間鍵：{key.digest}", *lines]
    groups.append(_RoomGroup(key, readout.resonances, lines))


def stage_one_post_check_lines(store: SearchStore) -> tuple[str, ...]:
    """壞摘要只壞這段；壞診斷只壞那席，其他名次與診斷繼續顯示。"""
    lines = [TITLE, NOT_SCORED]
    try:
        summary = read_summary(store.path)
        if summary is None:
            return (*lines, NOT_YET)
        lines.append(f"入選後檢查狀態：{SUMMARY_STATE_TEXT[summary.state]}；{summary.message}")
        if summary.modal_identity != modal_identity():
            lines.append(OLD_IDENTITY)
        reference = load_reference(load_quality_targets(config_path("quality_targets.toml")).purpose(store.settings.purpose))
        rooms: list[_RoomGroup] = []
        placement_lines: list[str] = []
        for seat in summary.seats:
            label = f"試算 {seat.trial_number}"
            try:
                label = _label(seat, summary.selection)
                state, reason = seat.state, seat.reason_text
                if state not in NO_DIAGNOSIS:
                    expected = store.scheme_path_for(store.refine_result_path(seat.trial_number))
                    if seat.scheme_file != expected.relative_to(store.path).as_posix():
                        raise ValueError("摘要的方案路徑不是這一席的細算方案")
                    scheme = load_scheme(expected)
                    diagnosis = read_diagnosis(store.path, seat, scheme, identity=summary.modal_identity)
                    if diagnosis.state.value != state:
                        raise ValueError("診斷狀態與摘要不同")
                    readout = read_modal_diagnosis(diagnosis, reference)
                    if state == "diagnosed_not_scored":
                        assert diagnosis.key is not None
                        _group_room(rooms, diagnosis.key, readout, _reference_span(reference),
                                    variable=store.settings.layout.room_range is not None)
                        placement_lines.extend(_placement_lines(label, scheme, readout))
                lines.append(f"{label}：{_state_text(state)}；{reason or '無附加原因'}")
            except (OSError, ValueError, KeyError, StopIteration) as error:
                lines.append(f"{label}：失敗；診斷文件讀不回或內容不完整：{error}")
        lines.extend([line for room in rooms for line in room.lines] or ["房間層尚無可讀診斷，未評估"])
        lines.extend((PARTICIPATION, RELATIVE))
        lines.extend(placement_lines)
        seconds = [seat.seconds for seat in summary.seats if seat.seconds is not None]
        lines.append(f"總秒數：{sum(seconds):.6f} 秒（有計時的席加總）；沒有計時紀錄 {sum(seat.seconds is None for seat in summary.seats)} 份")
    except (OSError, ValueError, KeyError) as error:
        lines.append(f"入選後檢查摘要讀不回或內容不完整：{error}")
    return tuple(line.replace("\r", " ").replace("\n", "；") for line in lines)


@dataclass(frozen=True)
class PostCheckSummaryText:
    """警示照狀態判（讀不回、診斷失敗），不照字面：寫入端的原因原文本來就可能寫「讀不回」。"""

    lines: tuple[str, ...]
    warning: bool
    rows: tuple[tuple[str, ...], ...] = ()


def stage_one_post_check_summary(store: SearchStore) -> PostCheckSummaryText:
    """網頁摘要：狀態與逐席房間層，沿用命令列的角色、狀態、讀檔核對與共振計數；壞一席只壞那一行，表上也沒有那一席。"""
    lines: list[str] = []
    rows: list[tuple[str, ...]] = []
    warning = False
    try:
        summary = read_summary(store.path)
        if summary is None:
            return PostCheckSummaryText((NOT_YET,), False)
        warning = summary.state == "failed"
        lines.append(f"入選後檢查狀態：{SUMMARY_STATE_TEXT[summary.state]}；{summary.message}")
        if summary.modal_identity != modal_identity():
            lines.append(OLD_IDENTITY)
        reference = load_reference(load_quality_targets(config_path("quality_targets.toml")).purpose(store.settings.purpose))
        rooms: list[_RoomGroup] = []
        for seat in summary.seats:
            label = f"試算 {seat.trial_number}"
            try:
                label = _label(seat, summary.selection)
                state = seat.state
                room_text = "房間層尚無可讀診斷，未評估"
                # 沒有診斷的席（含細算讀不回而跳過的）不讀方案，跟命令列一樣；房間尺寸看入選清單。
                if state not in NO_DIAGNOSIS:
                    expected = store.scheme_path_for(store.refine_result_path(seat.trial_number))
                    if seat.scheme_file != expected.relative_to(store.path).as_posix():
                        raise ValueError("摘要的方案路徑不是這一席的細算方案")
                    scheme = load_scheme(expected)
                    diagnosis = read_diagnosis(store.path, seat, scheme, identity=summary.modal_identity)
                    if diagnosis.state.value != state:
                        raise ValueError("診斷狀態與摘要不同")
                    readout = read_modal_diagnosis(diagnosis, reference)
                    room = scheme.scene.room_m
                    room_text = f"房間 長×寬×高（公尺）：{room.Lx:.6f}×{room.Ly:.6f}×{room.Lz:.6f}"
                    seat_rows: list[tuple[str, ...]] = []
                    if state == "diagnosed_not_scored":
                        assert diagnosis.key is not None
                        # 同一間房的診斷要一致才算數（命令列共用超線表時同一道核對）。
                        _group_room(rooms, diagnosis.key, readout, _reference_span(reference),
                                    variable=store.settings.layout.room_range is not None)
                        room_text += "；" + _room_lines(readout, _reference_span(reference))[0]
                        seat_rows = _resonance_rows(seat.trial_number, scheme, readout)
                    rows.extend(seat_rows)
                lines.append(f"{label}：{_state_text(state)}；{seat.reason_text or '無附加原因'}；{room_text}")
                warning = warning or state == "failed"
            except (OSError, ValueError, KeyError, StopIteration) as error:
                lines.append(f"{label}：失敗；診斷文件讀不回或內容不完整：{error}")
                warning = True
        if rows:
            lines.extend((TABLE_NOTE.format(span=_reference_span(reference)), PARTICIPATION, RELATIVE))
    except (OSError, ValueError, KeyError) as error:
        lines.append(f"入選後檢查摘要讀不回或內容不完整：{error}")
        warning = True
        rows = []
    return PostCheckSummaryText(tuple(_one_line(line) for line in lines), warning,
                                tuple(tuple(_one_line(cell) for cell in row) for row in rows))
