"""外圈結論與判定快照；搜尋、細算停止原因各自保留。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from aosr.search.run import SearchStatus

OuterConclusion: TypeAlias = Literal[
    "complete", "stable_but_search_budget", "refine_budget", "refine_candidates_exhausted",
    "feedback_blocked_by_budget", "feedback_not_configured", "feedback_unavailable",
    "baseline_first", "no_rankable_first", "user_stopped", "search_failed", "search_interrupted",
    "refine_failed", "refine_interrupted",
]

OUTER_MESSAGES: dict[OuterConclusion, str] = {
    "complete": "細算完成",
    "stable_but_search_budget": "細算第一名沒換，但搜尋因預算停止，未完成",
    "refine_budget": "細算用完上限，未完成",
    "refine_candidates_exhausted": "細算候選用完，未完成",
    "feedback_blocked_by_budget": "第一名換了，但回饋被搜尋預算擋住，未完成",
    "feedback_not_configured": "第一名換了，但搜尋未設定回饋，未完成",
    "feedback_unavailable": "第一名換了，但沒有可回饋的點或上一輪回饋點尚未問完，未完成",
    "baseline_first": "細算第一名是原方案，未完成",
    "no_rankable_first": "沒有可排名的第一名，未完成",
    "user_stopped": "使用者停止，未完成",
    "search_failed": "搜尋失敗，未完成",
    "search_interrupted": "搜尋中斷，未完成",
    "refine_failed": "細算失敗，未完成",
    "refine_interrupted": "細算中斷，未完成",
}


class OuterSnapshot(BaseModel):
    """六格各有預設；狀態推進後不能把舊結論當成本次判定。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    search_state: str | None = None
    search_round: int | None = None
    asked: int | None = None
    refine_state: str | None = None
    refine_round: int | None = None
    refined: int | None = None


class OuterStatus(BaseModel):
    """獨立的第三格，不覆寫搜尋或細算的訊息。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    conclusion: OuterConclusion | None = None
    message: str = "未判定"
    snapshot: OuterSnapshot = OuterSnapshot()


def snapshot_of(status: SearchStatus) -> OuterSnapshot:
    return OuterSnapshot(search_state=status.state, search_round=status.round, asked=status.asked,
                         refine_state=status.refine.state, refine_round=status.refine.round, refined=status.refine.refined)


def attachment_skip_reason(conclusion: OuterConclusion | None) -> str:
    """搜尋附件共用收尾資格；不改外圈結論或離開碼。"""
    if conclusion is None or conclusion == "user_stopped" or conclusion.endswith(("_failed", "_interrupted")):
        text = OUTER_MESSAGES[conclusion] if conclusion is not None else "未判定"
        return f"搜尋沒有正常收尾（{text}），這次不補；接續跑完後會補"
    return ""


def conclusion_message(status: SearchStatus) -> str:
    if status.outer.conclusion is None:
        return "未判定"
    if status.outer.snapshot != snapshot_of(status):
        return "未判定（判定之後狀態又變了）"
    return status.outer.message
