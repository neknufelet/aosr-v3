"""狀態頁上那幾格資料的形狀。

全部凍結（`frozen=True`）：這一頁是「某一跑算出來的一張快照」，算完就不准再被改。
每一格都是已經算好的值，不放待解析的原始字典——`render.py` 只准排版，不准在排版的時候
再去解讀資料，那是 v2「文件抄了現況、原始資料改了文件沒改」的同一條路。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuleCard:
    """一張已經立在主線上的規矩卡。"""

    card_id: str
    human: str
    blood_debt: tuple[str, ...]
    merged_day: str  # 這張卡合進主線那天（ISO 日期），算不出來就是空字串


@dataclass(frozen=True)
class Blueprint:
    """藍圖 38 張卡的分組（`blueprint/cards-38.json`，只讀不寫）。"""

    total: int
    groups: tuple[tuple[str, int], ...]  # (feasibility_v2 的值, 幾張)
    waiting: tuple[tuple[str, tuple[str, ...]], ...]  # (在等什麼 blocked_on, 哪幾張卡)


@dataclass(frozen=True)
class Milestone:
    """GitHub 里程碑一條。"""

    title: str
    state: str
    open_issues: int
    closed_issues: int
    url: str


@dataclass(frozen=True)
class Ticket:
    """開著的一張票：issue（待辦票）或 PR（合併請求）。"""

    number: int
    title: str
    labels: tuple[str, ...]
    is_pr: bool
    updated: str
    url: str


@dataclass(frozen=True)
class StepResult:
    """雲端那一跑裡的一步（一步就是一支檢查）。"""

    name: str
    conclusion: str
    seconds: int


@dataclass(frozen=True)
class CloudRun:
    """主線最近一次 verify（雲端檢查）那一跑。"""

    run_id: int
    conclusion: str
    status: str
    started: str
    finished: str
    head_sha: str
    url: str
    job_seconds: int
    steps: tuple[StepResult, ...]


@dataclass(frozen=True)
class RepoState:
    """版控那一邊：主線最新一筆、以及本機跟遠端差多少。"""

    branch: str
    head_sha: str
    head_subject: str
    head_when: str
    ahead: int
    behind: int


@dataclass(frozen=True)
class PageData:
    """整頁的資料。`render.render_page` 只吃這一個東西。"""

    computed_at: str  # 算出這一頁的時間（台北時間，人看的字串）
    computed_by: str  # 是哪一跑算的：雲端就寫 run id，本機就說是本機
    reflects: str  # 這一頁反映的是哪一跑／哪一顆 commit
    repo: str  # owner/repo
    page_url: str
    cards: tuple[RuleCard, ...]
    lessons_total: int
    blueprint: Blueprint
    milestones: tuple[Milestone, ...]
    tickets: tuple[Ticket, ...]
    cloud: CloudRun | None
    repo_state: RepoState
