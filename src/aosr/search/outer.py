"""自動外圈：每步重讀狀態，既有帳本就是接續進度。"""

from datetime import date
from functools import partial
from pathlib import Path
from typing import Literal

from aosr.search.feedback import FeedbackUnavailable, comparison_trial, feedback_search
from aosr.search.outer_status import OUTER_MESSAGES, OuterConclusion, OuterStatus, snapshot_of
from aosr.search.refine_run import refine_search
from aosr.search.run import Compute, IdentityProbe, SearchStatus, _write_status, resume_search
from aosr.search.store import SearchStore


def _terminal(status: SearchStatus, stopped: bool) -> OuterConclusion | None:
    # 細算的使用者停止只算本輪的；上一輪留下的（之後人手回饋過）不擋新一輪搜尋。
    current_refine = status.refine.round == status.round
    if stopped or status.state == "user_stopped" or (current_refine and status.refine.stop_reason == "user_stopped"):
        return "user_stopped"
    if status.state == "failed":
        return "search_failed"
    if status.state == "interrupted":
        return "search_interrupted"
    if status.refine.state == "failed":
        return "refine_failed"
    if status.refine.state == "interrupted":
        return "refine_interrupted"
    return None


def decide_outer(status: SearchStatus, *, reference: int | None, budget: int,
                 feedback: bool, stopped: bool = False) -> OuterConclusion | Literal["feedback"]:
    """依判準表先後判斷；feedback（要回饋）是動作，並非結論。"""
    terminal = _terminal(status, stopped)
    if terminal is not None:
        return terminal
    refinement = status.refine
    if refinement.state != "stopped" or refinement.round != status.round:
        raise ValueError("這一輪細算尚未停下，不能判外圈結論")
    if refinement.stop_reason == "refine_budget":
        return "refine_budget"
    if refinement.stop_reason == "candidates_exhausted":
        return "refine_candidates_exhausted"
    if refinement.best is None:
        return "no_rankable_first"
    if refinement.best == "baseline":
        return "baseline_first"
    if status.state not in ("converged", "budget_exhausted"):
        raise ValueError("搜尋尚未停下，不能判外圈結論")
    if refinement.best == reference:
        return "complete" if status.state == "converged" else "stable_but_search_budget"
    if status.state == "budget_exhausted" or status.asked >= budget:
        return "feedback_blocked_by_budget"
    if not feedback:
        return "feedback_not_configured"
    return "feedback"


def conclude(store: SearchStore, status: SearchStatus, conclusion: OuterConclusion) -> SearchStatus:
    outer = OuterStatus(conclusion=conclusion, message=OUTER_MESSAGES[conclusion], snapshot=snapshot_of(status))
    return _write_status(store, status.model_copy(update={"outer": outer}))


def _read(store: SearchStore) -> SearchStatus:
    return SearchStatus.model_validate_json(store.status_path.read_bytes())


def _stopped(store: SearchStore) -> bool:
    return store.stop_path.exists() or store.refine_stop_path.exists()


def auto_search(store: SearchStore, *, compute: Compute, probe: IdentityProbe,
                registry_path: Path, run_date: date, engine_version: str) -> SearchStatus:
    """每步各自可接續；訊號與被砍不捕捉、不另寫結論。"""
    while True:
        status = _read(store)
        terminal = _terminal(status, _stopped(store))
        if terminal is not None:
            return conclude(store, status, terminal)
        if status.state == "running":
            entry = resume_search
        elif status.refine.state in ("not_started", "running") or status.refine.round < status.round:
            entry = partial(refine_search, keep_stop_marker=True)
        else:
            decision = decide_outer(status, reference=comparison_trial(store, status), budget=store.settings.budget,
                                    feedback=store.settings.feedback is not None)
            if decision != "feedback":
                return conclude(store, status, decision)
            try:
                feedback_search(store)
            except FeedbackUnavailable:
                return conclude(store, _read(store), "feedback_unavailable")
            except ValueError:
                # 迴圈頂查完記號之後才放的搜尋停止記號，回饋會拒絕；照使用者停止收，不當完整性錯誤。
                if _stopped(store):
                    return conclude(store, _read(store), "user_stopped")
                raise
            continue
        try:
            entry(store, compute=compute, probe=probe, registry_path=registry_path,
                  run_date=run_date, engine_version=engine_version)
        except Exception:
            current = _read(store)
            terminal = _terminal(current, _stopped(store))
            if terminal is not None:
                concluded = conclude(store, current, terminal)
                if terminal == "user_stopped":
                    return concluded
            raise
