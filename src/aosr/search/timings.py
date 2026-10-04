"""搜尋與細算各輪的牆鐘；只在狀態寫回時累加，不從候選帳本分攤。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from aosr.search.run import SearchStatus

RoundNumber = Annotated[int, Field(ge=1)]
Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class SearchTimings(BaseModel):
    """空字典表示沒有紀錄；輪次作鍵，零秒也表示確實記過。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    search: dict[RoundNumber, Seconds] = Field(default_factory=dict)
    refine: dict[RoundNumber, Seconds] = Field(default_factory=dict)


@dataclass
class WallClock:
    """每次命令重建單調起點，接續只加當次經過時間，不計命令之間的空檔。"""

    previous: float = field(default_factory=lambda: time.monotonic())

    def record(self, status: SearchStatus, phase: Literal["search", "refine"]) -> SearchStatus:
        now = time.monotonic()
        elapsed = now - self.previous
        round_number = status.round if phase == "search" else status.refine.round
        rounds = dict(getattr(status.timings, phase))
        rounds[round_number] = rounds.get(round_number, 0.0) + elapsed
        timings = status.timings.model_copy(update={phase: rounds})
        self.previous = now
        return status.model_copy(update={"timings": timings})


NO_TIMINGS = "這個搜尋資料夾沒有時間紀錄（加上時間紀錄之前開的搜尋）"


def round_text(rounds: dict[int, float], phase: Literal["搜尋", "細算"], *, recorded: bool) -> str:
    """各輪排序後列分鐘，合計先加秒數才四捨五入。"""
    if not rounds:
        return f"各輪{phase}花的時間：尚無紀錄" if recorded else NO_TIMINGS
    entries = "、".join(f"第 {number} 輪 {seconds / 60:.1f} 分" for number, seconds in sorted(rounds.items()))
    return f"各輪{phase}花的時間：{entries}；{phase}合計 {sum(rounds.values()) / 60:.1f} 分"


def total_text(timings: SearchTimings) -> str:
    if not timings.search and not timings.refine:
        return NO_TIMINGS
    seconds = sum(timings.search.values()) + sum(timings.refine.values())
    return (f"搜尋＋細算合計 {seconds / 60:.1f} 分"
            "（牆鐘，含共用有限元素；被砍後未寫回狀態的那一小段不在內）")
