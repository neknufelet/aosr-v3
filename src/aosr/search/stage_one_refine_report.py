"""第一階段細算的唯讀文字投影；不呼叫選席或計算。"""
from __future__ import annotations

from aosr.reporting.stage_one_result import StageOneResult
from aosr.search.run import ComputeFailed, IdentityChanged
from aosr.search.stage_one_refine_record import (RefinedSeat, StageOneRefineStatus, read_stage_one_refined_result)
from aosr.search.stage_one_selection_contract import SelectionSeat
from aosr.search.store import SearchStore


def _number(value: float | None) -> str:
    return "量不到" if value is None else f"{value:.6f}"


def _channels(result: StageOneResult) -> str:
    items = []
    for speaker in result.evaluation.speakers:
        p = speaker.payload
        items.append(f"{speaker.speaker_id} " + ("不可估" if p is None else
            f"{p.own_line_ripple_rms_db:.6f}／{p.seat_robustness_rms_db:.6f}"))
    return "；".join(items)


def _row(store: SearchStore, seat: SelectionSeat, refined: RefinedSeat | None) -> str:
    meters = seat.params_m
    roles = "、".join({"top_n": f"前 N（搜尋第 {seat.search_rank} 名）", "big_peak_control": "大峰對照",
                       "far_control": "較遠對照"}[role] for role in seat.roles)
    rank = str(refined.refine_rank) if refined is not None and refined.refine_rank is not None else "—"
    detail, score, peak = "尚未完成", "—", "量不到"
    if seat.error is not None:
        detail = f"讀不回：{seat.error}"
    elif refined is not None and refined.state != "pending":
        try:
            result = read_stage_one_refined_result(store, seat.trial_number)
            detail = _channels(result)
            score = _number(result.score.screening_total)
            peak = _number(refined.big_peak_db)
            if refined.reason:
                detail += f"；{refined.reason}"
        except (OSError, ValueError, ComputeFailed, IdentityChanged) as error:
            rank, detail = "—", f"細算結果讀不回：{error}"
    if seat.peak_reason is not None:
        detail += f"；搜尋軸大峰量不到：{seat.peak_reason}"
    return (f"{rank}｜試算 {seat.trial_number}｜{seat.search_rank}｜{roles}｜"
        f"{meters['front_distance']:.6f}／{meters['spacing']:.6f}／{meters['listening_distance']:.6f}｜"
        f"{seat.search_score:.6f}｜{score}｜{detail}｜{_number(seat.big_peak_db)}／{peak}")


def stage_one_refine_lines(store: SearchStore) -> tuple[str, ...]:
    """摘要缺席只印還沒細算；摘要壞掉只影響這一段。"""
    path = store.path / "stage_one_refine.json"
    if not path.exists():
        return ("還沒細算",)
    lines = ["細算（每 1 Hz 一點，只算有限元素）",
        "大峰＝最強峰高出這支喇叭自己那條直線多少，左右兩支取較差",
        "房間固定，面積對照不適用", "入選後檢查（低頻模態診斷）還沒做"]
    try:
        status = StageOneRefineStatus.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        return tuple([*lines, f"細算摘要讀不回：{error}"])
    selected = status.selection
    lines.extend([f"細算狀態：{status.state}；{status.message}",
        f"前 N 設定 {selected.top_n}；本次實際 {selected.actual_top_n} 個",
        "細算名次｜試算｜搜尋名次｜角色｜離前牆／喇叭間距／聆聽距離（公尺）｜搜尋軸合分｜1 Hz 合分｜"
        "每支喇叭主項／穩健性｜大峰（搜尋軸／1 Hz）"])
    results = {seat.trial_number: seat for seat in status.results}
    top = [seat for seat in selected.candidates if "top_n" in seat.roles]
    top.sort(key=lambda seat: ((results[seat.trial_number].refine_rank or float("inf"))
        if seat.trial_number in results else float("inf"), seat.search_rank))
    lines.extend(_row(store, seat, results.get(seat.trial_number)) for seat in top)
    lines.append("對照（不算入選名額、不給細算名次；角色重複時共用上表結果）")
    lines.extend(_row(store, seat, results.get(seat.trial_number)) for seat in selected.candidates if "top_n" not in seat.roles)
    lines.extend(_control_lines(status))
    lines.append(f"總秒數：{sum(seat.seconds or 0. for seat in status.results):.6f} 秒（各候選第二波牆鐘加總）")
    return tuple(lines)


def _control_lines(status: StageOneRefineStatus) -> list[str]:
    selected = status.selection
    threshold = selected.big_peak_threshold_db
    lines = [f"大峰對照門檻：{_number(threshold)} dB；前 N 裡最低的大峰減差距 "
        f"{selected.big_peak_control_margin_db:g} dB"]
    if "big_peak_control" in selected.missing:
        lines.append(f"本次未取得大峰對照：{selected.missing['big_peak_control']}")
    x = selected.far_control_min_front_distance_m
    lines.append(f"主搜尋已採到的較遠候選中最佳（離前牆至少 {x:g} m；{x:g} 是工程暫定的選席分界）")
    if "far_control" in selected.missing:
        lines.append("本次未取得較遠對照，不因此判定較遠區不好")
    return lines
