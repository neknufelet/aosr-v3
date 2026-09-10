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
class ClosedTicket:
    """一張最近關掉的 issue（待辦票），以及它落地的證據對不對得起來。

    三樣東西要串得起來：一個把它關掉的 PR（合併請求，GitHub 自己記的關票來源）、那個 PR
    併出來的那一顆 commit、以及那一顆 commit 的 verify（雲端檢查）在 status 分支上留下的一份
    綠收據。串不起來的那一句寫在 `problem` 裡——上一代出過的事（票關了、東西沒真的落地）
    就是這一格要照出來的。
    """

    number: int
    title: str
    closed: str  # 關掉的時間（台北時間，人看的字串）
    url: str
    pr_number: int  # 把它關掉、而且合進主線的那個 PR；沒有就 0（人手關的票就是 0）
    pr_url: str
    merge_sha: str  # 那個 PR 併出來的那一顆 commit（短的）；找不到就空字串
    receipt_run_id: int  # 那一顆 commit 的收據是哪一跑寫的；沒有收據就 0
    receipt_url: str
    receipt_green: bool
    problem: str  # 空字串＝三樣都對得上；有字＝要用紅字寫出來的那一句


@dataclass(frozen=True)
class ClosedReview:
    """「關掉的票對不對得到綠收據」那一格。這一格只給人看，不擋合併。"""

    source: str  # 收據是從哪裡讀的（哪一個 ref、幾份），這一頁自己要說得出來
    tickets: tuple[ClosedTicket, ...]


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
class MissingReceipt:
    """主線上一跑 verify（雲端檢查），而收據分支上找不到它的收據。"""

    run_id: int
    conclusion: str
    started: str
    head_sha: str
    url: str


@dataclass(frozen=True)
class ReceiptGaps:
    """「主線最近那幾跑裡，哪幾跑沒有收據」那一格。

    收據會無聲消失過一次（2026-09-10，主線 verify run 34452456923：兩個 job 共用同一個
    併發組，後排的那一輪被下一個進來的取消掉），而當時頁面上沒有任何一格看得見。
    這一格就是那件事的眼睛：`missing` 不是空的就用紅字列出來。
    """

    looked: int  # 這一頁看了主線最近幾跑（只算已經跑完的）
    source: str  # 收據是從哪裡讀的（哪一個 ref、幾份）
    missing: tuple[MissingReceipt, ...]


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
    closed_review: ClosedReview
    cloud: CloudRun | None
    receipt_gaps: ReceiptGaps
    repo_state: RepoState
