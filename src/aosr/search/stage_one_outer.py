"""第一階段自動外圈：搜尋停下、固定選席細算，不為對照追加搜尋。"""
from datetime import date
from pathlib import Path

from aosr.search.outer import _read, _stopped, conclude
from aosr.search.outer_status import OuterConclusion
from aosr.search.run import IdentityProbe, SearchStatus
from aosr.search.stage_one_refine_record import StageOneRefineStatus
from aosr.search.stage_one_refine_stop import StageOneRefineStopped
from aosr.search.stage_one_run import resume_stage_one_search
from aosr.search.stage_one_worker import StageOneComputeFunction
from aosr.search.store import SearchStore


def _terminal(store: SearchStore, status: SearchStatus) -> OuterConclusion | None:
    if _stopped(store) or status.state == "user_stopped":
        return "user_stopped"
    if status.state == "failed":
        return "search_failed"
    if status.state == "interrupted":
        return "search_interrupted"
    if status.state in ("converged", "budget_exhausted"):
        path = store.path / "stage_one_refine.json"
        refined = StageOneRefineStatus.model_validate_json(path.read_bytes()) if path.exists() else None
        if refined is not None:
            if refined.state == "done":
                return "complete"
            if refined.state == "failed":
                return "refine_failed"
            if refined.state == "interrupted":
                return "refine_interrupted"
    return None


def auto_stage_one_search(store: SearchStore, *, compute: StageOneComputeFunction, probe: IdentityProbe,
                          registry_path: Path, run_date: date, engine_version: str) -> SearchStatus:
    """每步重讀既有紀錄，沒有回饋步驟，不為對照追加搜尋。

    第一階段 complete（細算完成）＝搜尋收斂或用完預算，且第一階段細算 done（完成）；
    不是第二階段「細算第一名跟回饋參考一致」的意思。結論沿用共用外圈欄位。
    """
    from aosr.search.stage_one_refine import refine_stage_one_search

    while True:
        status = _read(store)
        terminal = _terminal(store, status)
        if terminal is not None:
            return conclude(store, status, terminal)
        try:
            if status.state == "running":
                resume_stage_one_search(store, compute=compute, probe=probe, registry_path=registry_path,
                    run_date=run_date, engine_version=engine_version)
            else:
                refine_stage_one_search(store, compute=compute, probe=probe, registry_path=registry_path,
                    run_date=run_date, engine_version=engine_version, keep_stop_marker=True)
        except StageOneRefineStopped:
            continue
        except Exception:
            current = _read(store)
            terminal = _terminal(store, current)
            if terminal is not None:
                concluded = conclude(store, current, terminal)
                if terminal == "user_stopped":
                    return concluded
            raise
