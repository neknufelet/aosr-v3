"""第一階段細算的唯讀文字投影；不呼叫選席或計算。"""
from __future__ import annotations

from aosr.reporting.stage_one_result import StageOneResult, stage_one_comparison_identity
from aosr.search.run import ComputeFailed, IdentityChanged
from aosr.search.stage_one_refine_record import (RefinedSeat, StageOneRefineStatus, read_stage_one_refined_result)
from aosr.search.stage_one_selection_contract import SelectionSeat, big_peak
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


def _mismatch(result: StageOneResult, refined: RefinedSeat, status: StageOneRefineStatus) -> str | None:
    """重新讀到的檔要跟摘要記的一致；細算後被換掉的檔不准拿新數字配舊名次與舊的同表判定。"""
    if big_peak(result) != refined.big_peak_db:
        return "大峰"
    if refined.state == "scored" and result.score.screening_total != refined.score:
        return "合分"
    if refined.state in ("scored", "unassessed") and (result.score.state == "scored") != (refined.state == "scored"):
        return "可估狀態"
    if status.comparison_identity is not None and (
            (stage_one_comparison_identity(result) == status.comparison_identity) != (refined.state != "incomparable")):
        return "比較身分"
    return None


def _row(store: SearchStore, status: StageOneRefineStatus, seat: SelectionSeat, refined: RefinedSeat | None) -> str:
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
        except (OSError, ValueError, ComputeFailed, IdentityChanged) as error:
            rank, detail = "—", f"細算結果讀不回：{error}"
        else:
            changed = _mismatch(result, refined, status)
            if changed is not None:
                rank, detail = "—", f"細算結果跟摘要記的{changed}不同（細算後結果檔被換過），要重跑細算"
            else:
                detail = _channels(result)
                score = _number(result.score.screening_total)
                peak = _number(refined.big_peak_db)
                if refined.reason:
                    detail += f"；{refined.reason}"
    if seat.peak_reason is not None:
        detail += f"；搜尋軸大峰量不到：{seat.peak_reason}"
    room = (f"{meters['room_Lx']:.6f}×{meters['room_Ly']:.6f}×{meters['room_Lz']:.6f}｜"
            if store.settings.layout.room_range is not None else "")
    return (f"{rank}｜試算 {seat.trial_number}｜{seat.search_rank}｜{roles}｜"
        f"{meters['front_distance']:.6f}／{meters['spacing']:.6f}／{meters['listening_distance']:.6f}｜"
        f"{room}{seat.search_score:.6f}｜{score}｜{detail}｜{_number(seat.big_peak_db)}／{peak}")


def stage_one_refine_lines(store: SearchStore) -> tuple[str, ...]:
    """摘要缺席只印還沒細算；摘要壞掉只影響這一段。"""
    path = store.path / "stage_one_refine.json"
    if not path.exists():
        return ("還沒細算",)
    lines = ["細算（每 1 Hz 一點，只算有限元素）",
        "大峰＝最強峰高出這支喇叭自己那條直線多少，左右兩支取較差",
        ("房間固定，面積對照不適用" if store.settings.layout.room_range is None else "房間隨候選而變，面積對照不適用"),
        "入選後檢查見下一段"]
    try:
        status = StageOneRefineStatus.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as error:
        return tuple([*lines, f"細算摘要讀不回：{error}"])
    try:
        return tuple([*lines, *_status_lines(store, status)])
    except (KeyError, ValueError) as error:  # 摘要通過模型、內容卻缺格：只讓這一段報錯，不拖垮整份報告。
        return tuple([*lines, f"細算摘要內容不完整：{error!r}"])


def _status_lines(store: SearchStore, status: StageOneRefineStatus) -> list[str]:
    selected = status.selection
    lines: list[str] = []
    lines.extend([f"細算狀態：{status.state}；{status.message}",
        f"前 N 設定 {selected.top_n}；本次實際 {selected.actual_top_n} 個",
        "細算名次｜試算｜搜尋名次｜角色｜離前牆／喇叭間距／聆聽距離（公尺）｜"
        + ("房間 長×寬×高（公尺）｜" if store.settings.layout.room_range is not None else "") + "搜尋軸合分｜1 Hz 合分｜"
        "每支喇叭主項／穩健性｜大峰（搜尋軸／1 Hz）"])
    results = {seat.trial_number: seat for seat in status.results}
    top = [seat for seat in selected.candidates if "top_n" in seat.roles]
    top.sort(key=lambda seat: ((results[seat.trial_number].refine_rank or float("inf"))
        if seat.trial_number in results else float("inf"), seat.search_rank))
    lines.extend(_row(store, status, seat, results.get(seat.trial_number)) for seat in top)
    lines.append("對照（不算入選名額、不給細算名次；角色重複時共用上表結果）")
    lines.extend(_row(store, status, seat, results.get(seat.trial_number))
                 for seat in selected.candidates if "top_n" not in seat.roles)
    lines.extend(_control_lines(status))
    timed = [seat.seconds for seat in status.results if seat.seconds is not None]
    # 結果檔存好、摘要還沒記到秒數就被砍：接續沿用那份結果，秒數查不回來，照實列出、不當成 0。
    untimed = sum(seat.seconds is None and seat.state in ("scored", "unassessed", "incomparable") for seat in status.results)
    lines.append(f"總秒數：{sum(timed):.6f} 秒（各候選第二波牆鐘加總）" + (
        f"；另有 {untimed} 份沒有計時紀錄（上次被砍時結果已存、秒數還沒記），不在總數裡" if untimed else ""))
    return lines


def _control_lines(status: StageOneRefineStatus) -> list[str]:
    selected = status.selection
    threshold = selected.big_peak_threshold_db
    shown = "無" if threshold is None else f"{threshold:.6f} dB"
    lines = [f"大峰對照門檻：{shown}；前 N 裡最低的大峰減差距 {selected.big_peak_control_margin_db:g} dB"]
    # 標籤只掛在真的取到的那一席；沒取到時只印原因，不留一行沒有候選的標籤。
    control = next((seat for seat in selected.candidates if "big_peak_control" in seat.roles), None)
    if control is None:
        lines.append(f"本次未取得大峰對照：{selected.missing['big_peak_control']}")
    else:
        lines.append(f"大峰對照＝試算 {control.trial_number}：前 N 名以外、大峰低過門檻的候選中搜尋軸合分最好")
    x = selected.far_control_min_front_distance_m
    far = next((seat for seat in selected.candidates if "far_control" in seat.roles), None)
    if far is None:
        lines.append(f"本次未取得較遠對照，不因此判定較遠區不好（選席分界：離前牆至少 {x:g} m，工程暫定）")
    else:
        lines.append(f"較遠對照＝試算 {far.trial_number}：主搜尋已採到的較遠候選中最佳"
                     f"（離前牆至少 {x:g} m；{x:g} 是工程暫定的選席分界）")
    return lines
