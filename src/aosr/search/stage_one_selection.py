"""第一階段選席：只讀搜尋報告與品質登記簿，不求解、不保存。"""
from __future__ import annotations

from pathlib import Path

from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.search.settings import StageOneRefineSettings
from aosr.search.stage_one_selection_contract import Role, SelectionSeat, StageOneSelection, big_peak, percent
from aosr.search.store import SearchStore

MARGIN_KEY = "stage_one_selection.big_peak_control_margin_db"


def _floor_area(seat: SelectionSeat) -> float:
    return seat.params_m["room_Lx"] * seat.params_m["room_Ly"]


def _area_control(seats: list[SelectionSeat], top: list[SelectionSeat], settings: StageOneRefineSettings,
                  room_varies: bool) -> tuple[SelectionSeat | None, str | None, float | None]:
    """前 N 面積跨度不超過門檻才開；只從前 N 以外、讀得回的候選裡挑搜尋軸合分最好的一個（老闆 2026-10-23）。

    面積取帳本記的房間長寬（換房時就是那一題實際的房），結果檔壞掉也量得到。
    """
    limit, difference = settings.area_control_max_spread, settings.area_control_min_difference
    if limit is None or difference is None:
        return None, None, None
    if not room_varies:
        return None, "房間固定，面積對照不適用", None
    if not top:
        return None, "沒有前 N 候選，面積對照不適用", None
    low, high = min(map(_floor_area, top)), max(map(_floor_area, top))
    spread = (high - low) / low
    if spread > limit:
        return None, f"前 N 面積跨度超過 {percent(limit)}，面積本來就不接近，不開這一席", spread
    chosen = next((seat for seat in seats[len(top):] if seat.error is None and (
        (low - _floor_area(seat)) / low >= difference or (_floor_area(seat) - high) / high >= difference)), None)
    if chosen is None:
        return None, (f"本次未找到面積對照：前 N 以外沒有面積比前 N 最小再小至少 {percent(difference)}、"
                      f"或比最大再大至少 {percent(difference)} 的候選"), spread
    return chosen, None, spread


def select_stage_one_candidates(store: SearchStore, *, registry_path: Path) -> StageOneSelection:
    """候選順序與讀回守門一律取自搜尋報告；前段壞檔不遞補。"""
    from aosr.search.stage_one_report import stage_one_ranked

    settings = store.settings.stage_one_refine
    if store.settings.stage is None or settings is None:
        raise ValueError("這個第一階段搜尋沒有設定細算（stage_one_refine），不能細算")
    entry = load_quality_targets(registry_path).stage_one(store.project.purpose).entry(MARGIN_KEY)
    if not isinstance(entry, SettingEntry) or entry.unit != "dB" or not isinstance(entry.value, (int, float)) or entry.value <= 0:
        raise ValueError(f"{MARGIN_KEY} 必須是正的 dB 差距設定")
    margin = float(entry.value)
    seats = []
    for rank, item in enumerate(stage_one_ranked(store), 1):
        error, peak = item.error, None
        if item.result is not None:
            try:
                peak = big_peak(item.result)
            except ValueError as exc:  # 壞的結果檔只讓這一列讀不回，不中斷整個挑席。
                error = f"結果讀回失敗：{exc}"
        assert item.row.score is not None
        seats.append(SelectionSeat(trial_number=item.trial_number, roles=(), search_rank=rank,
            search_score=item.row.score, big_peak_db=peak, params_m=item.row.params_m, error=error,
            peak_reason=(error or "任一支喇叭量不到最強峰") if peak is None else None))
    top = seats[:settings.top_n]
    peaks = [seat.big_peak_db for seat in top if seat.big_peak_db is not None]
    threshold = min(peaks) - margin if peaks else None
    control = next((seat for seat in seats[settings.top_n:] if seat.error is None and seat.big_peak_db is not None
                    and threshold is not None and seat.big_peak_db <= threshold), None)
    far = next((seat for seat in seats if seat.error is None
                and seat.params_m["front_distance"] >= settings.far_control_min_front_distance_m), None)
    area, area_missing, spread = _area_control(seats, top, settings, store.settings.layout.room_range is not None)
    missing = {}
    if control is None:
        missing["big_peak_control"] = "前 N 名都量不到大峰" if threshold is None else "前 N 名以外沒有大峰低過門檻的候選"
    if far is None:
        missing["far_control"] = "本次未取得較遠對照"
    if area_missing is not None:
        missing["area_control"] = area_missing
    selected = []
    for seat in seats:
        roles: list[Role] = []
        if seat in top:
            roles.append("top_n")
        if seat == control:
            roles.append("big_peak_control")
        if seat == far:
            roles.append("far_control")
        if seat == area:
            roles.append("area_control")
        if roles:
            selected.append(seat.model_copy(update={"roles": tuple(roles)}))
    return StageOneSelection(top_n=settings.top_n, actual_top_n=len(top),
        far_control_min_front_distance_m=settings.far_control_min_front_distance_m,
        big_peak_control_margin_db=margin, big_peak_threshold_db=threshold, candidates=tuple(selected), missing=missing,
        unreadable={seat.trial_number: seat.error for seat in seats if seat.error is not None},
        area_control_max_spread=settings.area_control_max_spread,
        area_control_min_difference=settings.area_control_min_difference, top_n_area_spread=spread)
