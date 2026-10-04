"""搜尋與細算各輪的牆鐘；只在狀態寫回時累加，不從候選帳本分攤。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from aosr.search.run import SearchStatus

def _now() -> float:
    """牆鐘入口：考卷換這一個就好，不必凍住全域的單調時鐘（子行程逾時靠它）。"""
    return time.monotonic()


RoundNumber = Annotated[int, Field(ge=1)]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SearchTimings(BaseModel):
    """報告用的檢視（不存檔）：空字典表示沒有紀錄；輪次作鍵，零秒也表示確實記過。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    search: dict[RoundNumber, Seconds] = Field(default_factory=dict)
    refine: dict[RoundNumber, Seconds] = Field(default_factory=dict)
    from_start: bool = False


@dataclass
class WallClock:
    """每次命令重建單調起點，接續只加當次經過時間，不計命令之間的空檔。"""

    previous: float = field(default_factory=lambda: _now())

    def record(self, status: SearchStatus, phase: Literal["search", "refine"]) -> SearchStatus:
        """搜尋的秒數記在搜尋自己那一格、細算的記在細算子物件裡：細算仍只改細算子物件。"""
        now = _now()
        elapsed = now - self.previous
        self.previous = now
        if phase == "search":
            rounds = dict(status.search_seconds)
            rounds[status.round] = rounds.get(status.round, 0.0) + elapsed
            return status.model_copy(update={"search_seconds": rounds})
        refined = dict(status.refine.seconds)
        refined[status.refine.round] = refined.get(status.refine.round, 0.0) + elapsed
        return status.model_copy(update={"refine": status.refine.model_copy(update={"seconds": refined})})


def timings_of(status: SearchStatus) -> SearchTimings:
    """報告用的檢視：兩段秒數各自住在搜尋與細算自己那一格，這裡只並起來看。"""
    return SearchTimings(search=status.search_seconds, refine=status.refine.seconds, from_start=status.timed_from_start)


NO_TIMINGS = "這個搜尋資料夾沒有時間紀錄（加上時間紀錄之前開的搜尋）"
NOT_YET = "還沒有時間紀錄（還沒算到第一次存檔）"
PARTIAL = "有一部分是加上時間紀錄之前的程式跑的，上面的時間與合計不含那一部分"


def round_text(rounds: dict[int, float], phase: Literal["搜尋", "細算"], *, from_start: bool) -> str:
    """各輪排序後列分鐘，合計先加秒數才四捨五入。"""
    if not rounds:
        return f"各輪{phase}花的時間：" + ("尚無紀錄" if from_start else "沒有紀錄（還沒跑，或是加上時間紀錄之前的程式跑的）")
    entries = "、".join(f"第 {number} 輪 {seconds / 60:.1f} 分" for number, seconds in sorted(rounds.items()))
    return f"各輪{phase}花的時間：{entries}；{phase}合計 {sum(rounds.values()) / 60:.1f} 分"


def total_text(timings: SearchTimings) -> str:
    if not timings.search and not timings.refine:
        return NO_TIMINGS
    seconds = sum(timings.search.values()) + sum(timings.refine.values())
    return (f"搜尋＋細算合計 {seconds / 60:.1f} 分"
            "（牆鐘，含共用有限元素；被砍後未寫回狀態的那一小段不在內）")
