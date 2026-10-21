"""第一階段選席：只讀搜尋報告與品質登記簿，不求解、不保存。"""
from __future__ import annotations

from pathlib import Path

from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.search.stage_one_selection_contract import Role, SelectionSeat, StageOneSelection, big_peak
from aosr.search.store import SearchStore

MARGIN_KEY = "stage_one_selection.big_peak_control_margin_db"


def select_stage_one_candidates(store: SearchStore, *, registry_path: Path) -> StageOneSelection:
    """候選順序與讀回守門一律取自搜尋報告；前段壞檔不遞補。"""
    from aosr.search.stage_one_report import build_stage_one_report

    settings = store.settings.stage_one_refine
    if store.settings.stage is None or settings is None:
        raise ValueError("這個第一階段搜尋沒有設定細算（stage_one_refine），不能細算")
    entry = load_quality_targets(registry_path).stage_one(store.project.purpose).entry(MARGIN_KEY)
    if not isinstance(entry, SettingEntry) or entry.unit != "dB" or not isinstance(entry.value, (int, float)) or entry.value <= 0:
        raise ValueError(f"{MARGIN_KEY} 必須是正的 dB 差距設定")
    margin = float(entry.value)
    ranked = build_stage_one_report(store).ranked
    seats = []
    for rank, item in enumerate(ranked, 1):
        peak = big_peak(item.result) if item.result is not None else None
        assert item.row.score is not None
        seats.append(SelectionSeat(trial_number=item.trial_number, roles=(), search_rank=rank,
            search_score=item.row.score, big_peak_db=peak, params_m=item.row.params_m, error=item.error,
            peak_reason=(item.error or "任一支喇叭量不到最強峰") if peak is None else None))
    top = seats[:settings.top_n]
    peaks = [seat.big_peak_db for seat in top if seat.big_peak_db is not None]
    threshold = min(peaks) - margin if peaks else None
    control = next((seat for seat in seats[settings.top_n:] if seat.error is None and seat.big_peak_db is not None
                    and threshold is not None and seat.big_peak_db <= threshold), None)
    far = next((seat for seat in seats if seat.error is None
                and seat.params_m["front_distance"] >= settings.far_control_min_front_distance_m), None)
    missing = {}
    if control is None:
        missing["big_peak_control"] = "前 N 名都量不到大峰" if threshold is None else "前 N 名以外沒有大峰低過門檻的候選"
    if far is None:
        missing["far_control"] = "本次未取得較遠對照"
    selected = []
    for seat in seats:
        roles: list[Role] = []
        if seat in top:
            roles.append("top_n")
        if seat == control:
            roles.append("big_peak_control")
        if seat == far:
            roles.append("far_control")
        if roles:
            selected.append(seat.model_copy(update={"roles": tuple(roles)}))
    return StageOneSelection(top_n=settings.top_n, actual_top_n=len(top),
        far_control_min_front_distance_m=settings.far_control_min_front_distance_m,
        big_peak_control_margin_db=margin, big_peak_threshold_db=threshold, candidates=tuple(selected), missing=missing,
        unreadable={seat.trial_number: seat.error for seat in seats if seat.error is not None})
