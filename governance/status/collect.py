"""去問版控與 GitHub，把狀態頁要的每一格算出來。

**這個檔是唯一會跑子程序、唯一會上網的一層。** `render.py` 完全不碰它們，所以那一層測得動
（測試餵假資料，不上網、不 spawn 任何東西）。

**外部工具不在就回 2，不准安靜產一頁沒資料的。** 版控工具或 `gh`（GitHub 的命令列工具）
叫不動、沒登入、非零退出、吐出來的東西不是解得開的 JSON、形狀跟預期不一樣——一律
`ToolBroken`，由入口翻成離開碼 2。理由跟每支檢查一樣：量不到就不出結論。一頁「什麼都沒有」
的狀態頁比沒有狀態頁更糟，因為它看起來像是「真的什麼都沒發生」。
"""
from __future__ import annotations

import json
import subprocess
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from governance.exit_codes import ToolBroken
from governance.status.mirror_receipts import DEFAULT_REF, list_receipts, resolve_ref
from governance.status.model import (
    Blueprint,
    ClosedReview,
    ClosedTicket,
    CloudRun,
    Milestone,
    PageData,
    RepoState,
    RuleCard,
    StepResult,
    Ticket,
)

# 外部工具。兩個都不在 PATH 上就沒有這一頁。
VCS = "git"
GH = "gh"

# 卡住在哪、藍圖與事故資料住在哪（相對 repo 根）。
RULES_DIR = "governance/rules"
BLUEPRINT_CARDS = "blueprint/cards-38.json"
LESSONS = "v2-audit/lessons.json"

# 主線的名字，以及雲端那一跑的 workflow 名（要挑出「最近一次 main 的 verify」）。
# 藍圖上「暫緩」的那一格值，以及沒寫「在等什麼」時歸的那一組。
DEFERRED = "deferred"
NO_BLOCKER = "藍圖上沒寫在等什麼"

MAIN = "main"
REMOTE_MAIN = "origin/main"
VERIFY_WORKFLOW = "verify"

# GitHub 記「這一跑／這個 job 是綠的」用的那個字。
SUCCESS = "success"

# 收據住在哪：機器分支 status 的鏡像 ref（本機已經有的那一份，這一支不上網去抓）。
# 名字與目錄跟 mirror_receipts 共用一份，不在這裡再抄一次。
RECEIPT_REF = DEFAULT_REF

# 台北時間。這不是門檻是事實（台灣沒有日光節約時間），刻意不靠系統的時區資料庫——
# 雲端那台機器上 zoneinfo 缺一份 tzdata 就會炸，而這一頁只是要把時間寫成老闆看的樣子。
TAIPEI = timezone(timedelta(hours=8))

# 一行裡分欄用的字元（版控的輸出裡不會有它）。
SEP = "\x1f"


@dataclass(frozen=True)
class Shell:
    """跑外部指令的那一層。`cwd` 是 repo 根，`timeout` 由入口從命令列傳進來。"""

    cwd: Path
    timeout: int

    def out(self, argv: Sequence[str], what: str) -> str:
        """跑一個指令，拿它的 stdout。叫不動、逾時、非零退出，一律 ToolBroken。"""
        try:
            proc = subprocess.run(
                [*argv],
                cwd=self.cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except FileNotFoundError as exc:
            raise ToolBroken(
                f"{argv[0]} 不在 PATH 上（{what}）——這一頁的資料全靠它，"
                "沒有它就不准產一頁沒資料的"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise ToolBroken(f"{argv[0]} 超過 {self.timeout} 秒沒回話（{what}）") from exc
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout).strip()[:300]
            raise ToolBroken(f"{argv[0]} 回 {proc.returncode}（{what}）：{tail}")
        return proc.stdout


@dataclass(frozen=True)
class Hub:
    """問 GitHub 的那一把手：跑指令的殼、這個 repo 的全名、問清單時一頁抓幾筆。

    `per_page`（一頁幾筆）跟 `timeout` 一樣由入口從命令列傳進來，不是程式裡的常數：
    它決定的只有「翻幾次」，翻到底這件事本身由 `api_pages` 保證，不靠這個數字夠大。
    """

    shell: Shell
    slug: str
    per_page: int


# ── 把讀回來的動態資料收窄成具體型別 ────────────────────────────────────────
# 形狀不對就 ToolBroken（回 2）。刻意不用 Any：規矩卡 type-guard 的第②層禁它，
# 而這裡真正該做的事本來就是「一個地方收窄、順手 fail closed」。


def parse_json(text: str, what: str) -> object:
    """解 JSON。解不開就 ToolBroken——我沒看懂就不出結論。"""
    try:
        parsed: object = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ToolBroken(f"{what} 吐出來的不是解得開的 JSON（{exc}）") from exc
    return parsed


def as_rows(value: object, what: str) -> list[object]:
    """一張清單。"""
    if not isinstance(value, list):
        raise ToolBroken(f"{what} 應該是一張清單，實際是 {type(value).__name__}")
    return list(value)


def as_table(value: object, what: str) -> dict[str, object]:
    """一張表。"""
    if not isinstance(value, dict):
        raise ToolBroken(f"{what} 應該是一張表，實際是 {type(value).__name__}")
    return {str(key): item for key, item in value.items()}


def field_text(row: Mapping[str, object], key: str) -> str:
    """表裡的一格字串。缺席或 null 就是空字串（GitHub 很多欄位真的會是 null）。"""
    value = row.get(key)
    return value if isinstance(value, str) else ""


def field_int(row: Mapping[str, object], key: str) -> int:
    """表裡的一格整數。缺席、null、型別不對，一律 0。"""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value


def field_labels(row: Mapping[str, object]) -> tuple[str, ...]:
    """一張票上的標籤名（GitHub 的 labels 是一張張表）。"""
    names: list[str] = []
    for item in as_rows(row.get("labels", []), "票上的 labels"):
        if isinstance(item, dict):
            names.append(field_text(as_table(item, "一個 label"), "name"))
        elif isinstance(item, str):
            names.append(item)
    return tuple(name for name in names if name)


# ── 時間 ──────────────────────────────────────────────────────────────────


def now_taipei() -> str:
    """現在幾點（台北時間，人看的樣子）。"""
    return datetime.now(TAIPEI).strftime("%Y-%m-%d %H:%M")


def to_taipei(iso: str) -> str:
    """把 GitHub／版控給的 ISO 時間換成台北時間。看不懂就照原樣還回去。"""
    if not iso:
        return ""
    try:
        stamp = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(TAIPEI).strftime("%Y-%m-%d %H:%M")


def elapsed_seconds(start: str, finish: str) -> int:
    """兩個 ISO 時間之間幾秒。任一頭看不懂就回 0（頁面會寫「算不出來」）。"""
    try:
        began = datetime.fromisoformat(start)
        ended = datetime.fromisoformat(finish)
    except ValueError:
        return 0
    return max(int((ended - began).total_seconds()), 0)


# ── 版控那一邊 ────────────────────────────────────────────────────────────


def read_repo_state(shell: Shell) -> RepoState:
    """主線最新一筆、以及這棵樹跟遠端差多少。"""
    branch = shell.out([VCS, "rev-parse", "--abbrev-ref", "HEAD"], "問現在在哪一條分支").strip()
    head = shell.out(
        [VCS, "log", "-1", f"--format=%H{SEP}%s{SEP}%cI"], "問最新一筆提交"
    ).strip()
    parts = head.split(SEP)
    if len(parts) != 3:  # sha、標題、時間
        raise ToolBroken(f"最新一筆提交的格式看不懂：{head!r}")
    counts = shell.out(
        [VCS, "rev-list", "--left-right", "--count", f"{REMOTE_MAIN}...HEAD"],
        f"問這棵樹跟 {REMOTE_MAIN} 差多少",
    ).split()
    if len(counts) != 2:  # 落後幾筆、領先幾筆
        raise ToolBroken(f"領先落後的格式看不懂：{counts!r}")
    return RepoState(
        branch=branch,
        head_sha=parts[0][:12],
        head_subject=parts[1],
        head_when=to_taipei(parts[2]),
        behind=int(counts[0]),
        ahead=int(counts[1]),
    )


def card_merge_days(shell: Shell) -> dict[str, str]:
    """每張卡合進主線那天：卡的 toml 第一次出現在歷史裡的那筆提交日期。"""
    log = shell.out(
        [
            VCS,
            "log",
            f"--format={SEP}%cI",
            "--name-status",
            "--diff-filter=A",
            "--",
            RULES_DIR,
        ],
        "問每張卡是哪天進主線的",
    )
    days: dict[str, str] = {}
    day = ""
    for line in log.splitlines():
        if line.startswith(SEP):
            day = to_taipei(line[1:].strip()).split(" ")[0]
            continue
        row = line.split("\t")
        if len(row) < 2 or not row[-1].endswith(".toml"):  # 一行是「狀態<tab>路徑」
            continue
        # 由新往舊掃，同一張卡後面看到的那筆更早，所以一律覆蓋。
        days[Path(row[-1]).stem] = day
    return days


# ── 卡與藍圖（只讀檔，不上網）────────────────────────────────────────────────


def read_rule_cards(root: Path, days: Mapping[str, str]) -> tuple[RuleCard, ...]:
    """主線上已經立好的規矩卡。讀不開一張就 ToolBroken——半份清單不算清單。"""
    cards: list[RuleCard] = []
    rules = root / RULES_DIR
    for path in sorted(rules.glob("*.toml")):
        rel = path.relative_to(root).as_posix()
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開規矩卡 {rel}：{exc}") from exc
        table = as_table(data, rel)
        debts = tuple(
            item for item in as_rows(table.get("blood_debt", []), f"{rel} 的血債") if isinstance(item, str)
        )
        card_id = field_text(table, "id") or path.stem
        cards.append(
            RuleCard(
                card_id=card_id,
                human=field_text(table, "human"),
                blood_debt=debts,
                merged_day=days.get(path.stem, ""),
            )
        )
    if not cards:
        raise ToolBroken(f"{rules} 底下一張卡都沒有——這一頁沒東西可算")
    return tuple(cards)


def read_blueprint(root: Path) -> Blueprint:
    """藍圖 38 張卡的分組。只讀不寫。"""
    path = root / BLUEPRINT_CARDS
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolBroken(f"讀不到 {BLUEPRINT_CARDS}：{exc}") from exc
    rows = as_rows(as_table(parse_json(raw, BLUEPRINT_CARDS), BLUEPRINT_CARDS).get("cards"), "藍圖的卡")
    tally: dict[str, int] = {}
    waiting: dict[str, list[str]] = {}
    for item in rows:
        card = as_table(item, "藍圖的一張卡")
        state = field_text(card, "feasibility_v2") or "未判"
        tally[state] = tally.get(state, 0) + 1
        if state != DEFERRED:
            continue
        # 在等什麼：藍圖上的 blocked_on。沒寫的另外一組，不准悄悄併進別人那一組。
        blocker = field_text(card, "blocked_on") or NO_BLOCKER
        waiting.setdefault(blocker, []).append(field_text(card, "id"))
    return Blueprint(
        total=len(rows),
        groups=tuple(sorted(tally.items())),
        waiting=tuple((key, tuple(sorted(ids))) for key, ids in sorted(waiting.items())),
    )


def read_lessons_total(root: Path) -> int:
    """v2 事故總共幾筆（血債的分母）。"""
    path = root / LESSONS
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ToolBroken(f"讀不到 {LESSONS}：{exc}") from exc
    table = as_table(parse_json(raw, LESSONS), LESSONS)
    return len(as_rows(table.get("incidents"), "v2 事故清單"))


# ── GitHub 那一邊 ─────────────────────────────────────────────────────────


def read_slug(shell: Shell, env: Mapping[str, str]) -> str:
    """這個 repo 在 GitHub 上叫什麼（owner/repo）。雲端那一跑環境變數裡就有。"""
    given = env.get("GITHUB_REPOSITORY", "").strip()
    if given:
        return given
    table = as_table(
        parse_json(
            shell.out([GH, "repo", "view", "--json", "nameWithOwner"], "問這個 repo 的全名"),
            "gh repo view",
        ),
        "gh repo view 的輸出",
    )
    slug = field_text(table, "nameWithOwner")
    if not slug:
        raise ToolBroken("問不出這個 repo 在 GitHub 上的全名（nameWithOwner 是空的）")
    return slug


def api(shell: Shell, path: str, what: str) -> object:
    """打一支 GitHub API。回不了話就 ToolBroken。"""
    return parse_json(shell.out([GH, "api", path], what), f"gh api {path}")


def page_rows(raw: object, what: str, key: str) -> list[object]:
    """一頁的列。清單型的直接是列；`workflow_runs`／`jobs` 那種包在一格裡的先開盒。"""
    if not key:
        return as_rows(raw, what)
    return as_rows(as_table(raw, what).get(key), f"{what} 的 {key}")


def api_pages(hub: Hub, path: str, what: str, key: str = "") -> Iterator[list[object]]:
    """把一支問清單的 GitHub API 翻到底，一次交一頁出去。

    **收到滿滿一頁就一定再翻下一頁**，直到某一頁短於 `per_page`（或者一筆都沒有）為止：
    只問第一頁等於「清單超過一頁的時候，多出來的那些在這一頁上不存在」，而那正是狀態頁
    最不能出的錯——看起來像「真的沒有」。回的是產生器（generator，要一頁才去拿一頁），
    所以找到就收工的那幾格（例如「主線最近一次 verify」）不會白翻後面的頁。
    """
    page = 1
    while True:
        joiner = "&" if "?" in path else "?"
        raw = api(
            hub.shell,
            f"{path}{joiner}per_page={hub.per_page}&page={page}",
            f"{what}（第 {page} 頁）",
        )
        rows = page_rows(raw, f"{what}（第 {page} 頁）", key)
        yield rows
        if len(rows) < hub.per_page:  # 短的一頁就是最後一頁
            return
        page += 1


def api_rows(hub: Hub, path: str, what: str, key: str = "") -> list[object]:
    """同上，但一次要全部：每一頁接起來成一張清單。"""
    everything: list[object] = []
    for rows in api_pages(hub, path, what, key):
        everything.extend(rows)
    return everything


def read_milestones(hub: Hub) -> tuple[Milestone, ...]:
    """里程碑（milestone，一段路線的名字）與各自開著／關掉的票數。"""
    rows = api_rows(hub, f"repos/{hub.slug}/milestones?state=all", "問里程碑")
    out: list[Milestone] = []
    for item in rows:
        row = as_table(item, "一條里程碑")
        out.append(
            Milestone(
                title=field_text(row, "title"),
                state=field_text(row, "state"),
                open_issues=field_int(row, "open_issues"),
                closed_issues=field_int(row, "closed_issues"),
                url=field_text(row, "html_url"),
            )
        )
    return tuple(out)


def read_tickets(hub: Hub) -> tuple[Ticket, ...]:
    """開著的票。GitHub 的 issues API 把 PR 也算 issue，所以順手分出來。"""
    rows = api_rows(hub, f"repos/{hub.slug}/issues?state=open", "問開著的票")
    out: list[Ticket] = []
    for item in rows:
        row = as_table(item, "一張票")
        out.append(
            Ticket(
                number=field_int(row, "number"),
                title=field_text(row, "title"),
                labels=field_labels(row),
                is_pr="pull_request" in row,
                updated=to_taipei(field_text(row, "updated_at")),
                url=field_text(row, "html_url"),
            )
        )
    return tuple(sorted(out, key=lambda t: t.number, reverse=True))


def read_run_steps(hub: Hub, run_id: int) -> tuple[int, tuple[StepResult, ...]]:
    """那一跑裡 verify 這個 job 的耗時，以及它每一步（每一步就是一支檢查）的結論。"""
    rows = api_rows(hub, f"repos/{hub.slug}/actions/runs/{run_id}/jobs", "問那一跑的 job", "jobs")
    seconds = 0
    steps: list[StepResult] = []
    for item in rows:
        job = as_table(item, "一個 job")
        if field_text(job, "name") != VERIFY_WORKFLOW:
            continue
        seconds = elapsed_seconds(field_text(job, "started_at"), field_text(job, "completed_at"))
        for entry in as_rows(job.get("steps", []), "job 的步驟"):
            step = as_table(entry, "一步")
            steps.append(
                StepResult(
                    name=field_text(step, "name"),
                    conclusion=field_text(step, "conclusion"),
                    seconds=elapsed_seconds(
                        field_text(step, "started_at"), field_text(step, "completed_at")
                    ),
                )
            )
    return seconds, tuple(steps)


def read_latest_verify(hub: Hub) -> CloudRun | None:
    """主線最近一次 verify（雲端檢查）那一跑。一次都沒跑過就回 None，頁面會寫成紅字。

    由新往舊翻，翻到第一個 verify 就收工（GitHub 那張清單本來就是新的在前）；翻完都沒有
    才是真的沒有。刻意不只看第一頁：主線上短時間內跑過一堆別的 workflow 的時候，
    「最近一次 verify」會被擠到第二頁去，只看第一頁就會謊報成「主線上還沒跑過」。
    """
    for rows in api_pages(hub, f"repos/{hub.slug}/actions/runs?branch={MAIN}", "問主線那幾跑", "workflow_runs"):
        for item in rows:
            row = as_table(item, "一跑")
            if field_text(row, "name") != VERIFY_WORKFLOW:
                continue
            run_id = field_int(row, "id")
            seconds, steps = read_run_steps(hub, run_id)
            return CloudRun(
                run_id=run_id,
                conclusion=field_text(row, "conclusion"),
                status=field_text(row, "status"),
                started=to_taipei(field_text(row, "run_started_at")),
                finished=to_taipei(field_text(row, "updated_at")),
                head_sha=field_text(row, "head_sha")[:12],
                url=field_text(row, "html_url"),
                job_seconds=seconds,
                steps=steps,
            )
    return None


# ── 關掉的票對不對得到綠收據（給人看的一格，不擋合併）──────────────────────────
# 一張關掉的 issue（待辦票）要串得起三樣東西：一個**把它關掉**的 PR（合併請求）、那個 PR 併出來
# 的那一顆 commit、以及那一顆 commit 的 verify（雲端檢查）在 status 分支上留下的一份綠收據。
# 串不起來的用紅字列出來——上一代出過的事就是「票關了、東西沒真的落地」。
#
# **「哪個 PR 做掉它」讀的是 GitHub 自己記的關票來源，不是猜的。** PR 內文寫 `Closes #n`，合併
# 的時候 GitHub 會自動關票並且把兩邊連起來，那條連結在 GraphQL（GitHub 的圖查詢介面）上叫
# `closedByPullRequestsReferences`。舊寫法拿票面上「互相提到」（cross-referenced）來猜，那是
# 一種習慣（記得在 PR 裡提票號）而不是機器守得住的事實：提到不等於做掉、做掉也可能沒提到。
# 所以人手 `gh issue close` 關掉的票在這一格是紅的，紅字明說「人手關的、沒有 PR 做掉它」——
# 跟「有 PR 但收據不綠」是兩種不一樣的紅，字面上分得開。這一格不擋合併，它只負責讓
# 「靠習慣」這件事變成機器看得見的東西。


# 問「哪個 PR 關掉這張票」的那一句圖查詢。`includeClosedPrs` 連關掉沒合的也要，
# 這樣「有 PR 但沒合進主線」跟「根本沒有 PR」在下面分得出來，不會併成同一種紅。
CLOSED_BY_QUERY = """
query($owner:String!,$name:String!,$number:Int!,$size:Int!,$after:String){
  repository(owner:$owner,name:$name){
    issue(number:$number){
      closedByPullRequestsReferences(first:$size, after:$after, includeClosedPrs:true){
        pageInfo{ hasNextPage endCursor }
        nodes{ number url merged mergedAt baseRefName mergeCommit{ oid } }
      }
    }
  }
}
"""


@dataclass(frozen=True)
class MergedPull:
    """一個已經合進主線、而且把某張票關掉的 PR（合併請求）。"""

    number: int
    url: str
    merged_at: str
    merge_sha: str


@dataclass(frozen=True)
class ClosedIssue:
    """一張關掉的 issue（待辦票）。`closed_at` 留 ISO 原樣，排序要用。"""

    number: int
    title: str
    url: str
    closed_at: str


@dataclass(frozen=True)
class ReceiptFacts:
    """status 分支上一份機器收據裡，這一格要用的那幾樣。"""

    run_id: int
    green: bool
    url: str


@dataclass(frozen=True)
class ClosingPulls:
    """GitHub 記的「這張票是被哪個 PR 關掉的」：合進主線的那一個，以及總共掛了幾個。

    `landed` 是合進主線、而且掛著關掉這張票的那個 PR（好幾個就是最後合進去的那一個）；
    `referenced` 是 GitHub 上掛著關掉它的 PR 總數（含還開著、關掉沒合的）。兩個都要，
    才分得出「人手關的、根本沒有 PR」跟「有 PR 掛著但沒合進主線」這兩種不一樣的斷法。
    """

    landed: MergedPull | None
    referenced: int


def graphql(
    hub: Hub, query: str, what: str, text: Mapping[str, str], numbers: Mapping[str, int]
) -> dict[str, object]:
    """打一句 GraphQL（GitHub 的圖查詢介面）。字串變數走 `-f`、數字變數走 `-F`。

    分兩種刻意不合併：`-F` 會把看起來像數字的字串真的變成數字，而翻頁用的游標
    （cursor，GitHub 給的一串位置代號）長得像數字的時候就會被送成整數、型別對不上。
    """
    argv = [GH, "api", "graphql", "-f", f"query={query}"]
    for key, value in text.items():
        argv.extend(["-f", f"{key}={value}"])
    for key, count in numbers.items():
        argv.extend(["-F", f"{key}={count}"])
    return as_table(parse_json(hub.shell.out(argv, what), what), what)


def nested(table: Mapping[str, object], path: Sequence[str], what: str) -> dict[str, object]:
    """一層層走進巢狀的表。中間任何一格不是表（含 GitHub 回 null）就 ToolBroken。"""
    node = dict(table)
    for key in path:
        node = as_table(node.get(key), f"{what} 的 {key}")
    return node


def owner_and_name(slug: str) -> tuple[str, str]:
    """把 `owner/repo` 拆成兩半（圖查詢要分開兩個變數）。"""
    owner, _, name = slug.partition("/")
    if not owner or not name:
        raise ToolBroken(f"這個 repo 的全名看不懂：{slug!r}（要長成 owner/repo）")
    return owner, name


def closing_pulls(hub: Hub, number: int) -> ClosingPulls:
    """問 GitHub：這張票是被哪個 PR 關掉的（`Closes #n` 合併時它自己記下來的那條連結）。

    刻意不看票面上的「互相提到」（cross-referenced）：那是人的習慣不是機器記的事實。
    翻到底（GraphQL 用游標翻頁），合進主線的有好幾個就取最後合進去的那一個
    ——這一格問的是「有沒有落地」，不是「哪一筆落地」。
    """
    owner, name = owner_and_name(hub.slug)
    what = f"問 #{number} 是被哪個 PR 關掉的"
    landed: list[MergedPull] = []
    seen = 0
    cursor = ""
    while True:
        text = {"owner": owner, "name": name}
        if cursor:
            text["after"] = cursor
        body = graphql(hub, CLOSED_BY_QUERY, what, text, {"number": number, "size": hub.per_page})
        block = nested(
            body, ("data", "repository", "issue", "closedByPullRequestsReferences"), what
        )
        for item in as_rows(block.get("nodes"), f"{what} 的那幾個 PR"):
            node = as_table(item, "一個關票的 PR")
            seen += 1
            if node.get("merged") is not True or field_text(node, "baseRefName") != MAIN:
                continue  # 掛著關掉它、但沒合進主線：不算落地
            landed.append(
                MergedPull(
                    number=field_int(node, "number"),
                    url=field_text(node, "url"),
                    merged_at=field_text(node, "mergedAt"),
                    merge_sha=field_text(as_table(node.get("mergeCommit"), "併出來那一顆"), "oid"),
                )
            )
        info = as_table(block.get("pageInfo"), f"{what} 的翻頁資訊")
        cursor = field_text(info, "endCursor")
        if info.get("hasNextPage") is not True or not cursor:
            break
    last = max(landed, key=lambda pull: pull.merged_at) if landed else None
    return ClosingPulls(landed=last, referenced=seen)


def read_closed_issues(hub: Hub, limit: int) -> tuple[ClosedIssue, ...]:
    """最近關掉的 issue（待辦票）。

    GitHub 的 issues API 把 PR 也算 issue，所以先分掉；它也排不出「按關掉時間」，
    所以整份翻回來自己按 `closed_at` 由新到舊排，取最前面那幾張（幾張由入口的命令列參數決定）。
    """
    rows = api_rows(hub, f"repos/{hub.slug}/issues?state=closed&sort=updated&direction=desc", "問關掉的票")
    picked: list[ClosedIssue] = []
    for item in rows:
        row = as_table(item, "一張關掉的票")
        if "pull_request" in row:
            continue
        picked.append(
            ClosedIssue(
                number=field_int(row, "number"),
                title=field_text(row, "title"),
                url=field_text(row, "html_url"),
                closed_at=field_text(row, "closed_at"),
            )
        )
    picked.sort(key=lambda issue: issue.closed_at, reverse=True)
    return tuple(picked[:limit])


def receipt_is_green(body: Mapping[str, object]) -> bool:
    """一份機器收據算不算綠：那一跑綠、verify 那個 job 綠，而且每一支檢查的離開碼都是 0。

    刻意不看收據自己的 `consistency` 欄——那一欄是收據自報的，而「中間有一層把離開碼吞掉」
    正是要抓的事（規矩卡 receipt-authority-is-the-cloud-run 同一個道理，這裡從欄位重算）。
    一支檢查都沒有的收據不算綠：沒掃過的乾淨不是乾淨。
    """
    run = as_table(body.get("run", {}), "收據的 run")
    job = as_table(body.get("job", {}), "收據的 job")
    if field_text(run, "conclusion") != SUCCESS or field_text(job, "conclusion") != SUCCESS:
        return False
    checks = as_rows(body.get("checks", []), "收據的 checks")
    if not checks:
        return False
    for item in checks:
        code = as_table(item, "一支檢查").get("exit_code")
        if isinstance(code, bool) or not isinstance(code, int) or code:
            return False
    return True


def read_receipts_by_sha(shell: Shell) -> tuple[str, dict[str, ReceiptFacts]]:
    """機器分支上的每一份收據，索引成「那一跑對著的 commit → 這一份」。

    只讀本機已經有的 ref（`origin/status`），不上網——跟 mirror_receipts 走同一條路，也共用
    它的列舉：ref 不在的時候那一層會說「本機沒有這個 ref」。同一顆 commit 重跑過就有好幾份，
    按檔名（run id）由小到大讀，留下最後那一份（比較新的那一跑）。
    """
    root = shell.cwd
    resolve_ref(root, RECEIPT_REF, shell.timeout)
    index: dict[str, ReceiptFacts] = {}
    for path in list_receipts(root, RECEIPT_REF, shell.timeout):
        where = f"收據 {path}"
        body = as_table(
            parse_json(shell.out([VCS, "show", f"{RECEIPT_REF}:{path}"], f"讀{where}"), where), where
        )
        run = as_table(body.get("run", {}), f"{where} 的 run")
        sha = field_text(run, "head_sha")
        if not sha:
            continue
        index[sha] = ReceiptFacts(
            run_id=field_int(run, "run_id"),
            green=receipt_is_green(body),
            url=field_text(run, "url"),
        )
    return f"{RECEIPT_REF}（{len(index)} 顆 commit 有收據）", index


def judge_closed(
    issue: ClosedIssue, closing: ClosingPulls, receipt: ReceiptFacts | None
) -> ClosedTicket:
    """一張關掉的票的那一列：串得起來就沒話說，串不起來就寫下是哪一段斷掉的。

    四種斷法字面上分得開：人手關的（GitHub 上沒有任何 PR 掛著關它）、有 PR 掛著但沒合進主線、
    合了但那一顆 commit 沒有收據、有收據但不是綠的。
    """
    pull = closing.landed
    if pull is None and not closing.referenced:
        problem = f"人手關的、沒有 PR 做掉它——GitHub 上沒有任何 PR 掛著關掉 #{issue.number}"
    elif pull is None:
        problem = f"掛著關掉它的 {closing.referenced} 個 PR 沒有一個合進 {MAIN}——東西沒進主線"
    elif receipt is None:
        problem = f"PR #{pull.number} 併出來的 {pull.merge_sha[:12]} 在收據分支上沒有收據"
    elif not receipt.green:
        problem = f"收據 run {receipt.run_id} 不是綠的（那一跑或某一支檢查沒過）"
    else:
        problem = ""
    return ClosedTicket(
        number=issue.number,
        title=issue.title,
        closed=to_taipei(issue.closed_at),
        url=issue.url,
        pr_number=pull.number if pull else 0,
        pr_url=pull.url if pull else "",
        merge_sha=pull.merge_sha[:12] if pull else "",
        receipt_run_id=receipt.run_id if receipt else 0,
        receipt_url=receipt.url if receipt else "",
        receipt_green=bool(receipt and receipt.green),
        problem=problem,
    )


def review_closed(hub: Hub, limit: int) -> ClosedReview:
    """整格算出來：最近關掉的那幾張票，各自對不對得到一份綠收據。"""
    source, receipts = read_receipts_by_sha(hub.shell)
    rows: list[ClosedTicket] = []
    for issue in read_closed_issues(hub, limit):
        closing = closing_pulls(hub, issue.number)
        pull = closing.landed
        rows.append(judge_closed(issue, closing, receipts.get(pull.merge_sha) if pull else None))
    return ClosedReview(source=source, tickets=tuple(rows))


# ── 這一頁是誰算的（「只認雲端」的那一格）────────────────────────────────────


def provenance(env: Mapping[str, str], slug: str, state: RepoState) -> tuple[str, str]:
    """回（是哪一跑算的、這一頁反映的是什麼）。雲端那一跑寫得出 run id，本機寫不出。"""
    run_id = env.get("GITHUB_RUN_ID", "").strip()
    server = env.get("GITHUB_SERVER_URL", "").strip() or "https://github.com"
    if run_id:
        attempt = env.get("GITHUB_RUN_ATTEMPT", "1").strip()
        by = f"雲端 run {run_id}（第 {attempt} 次嘗試）：{server}/{slug}/actions/runs/{run_id}"
    else:
        by = "本機跑的（不是雲端那一跑，所以這一頁不算權威，只是開工時先看一眼）"
    upstream = env.get("AOSR_VERIFY_RUN_ID", "").strip()
    reflects = f"{state.branch} 的 {state.head_sha}（{state.head_subject}）"
    if upstream:
        reflects = f"{reflects}；接在 verify run {upstream} 之後算的"
    return by, reflects


def collect(
    root: Path,
    shell: Shell,
    env: Mapping[str, str],
    page_url: str,
    recent_closed: int,
    per_page: int,
) -> PageData:
    """把整頁的資料算出來。任何一格算不出來就 ToolBroken，不補假值。"""
    state = read_repo_state(shell)
    slug = read_slug(shell, env)
    hub = Hub(shell=shell, slug=slug, per_page=per_page)
    by, reflects = provenance(env, slug, state)
    return PageData(
        computed_at=now_taipei(),
        computed_by=by,
        reflects=reflects,
        repo=slug,
        page_url=page_url,
        cards=read_rule_cards(root, card_merge_days(shell)),
        lessons_total=read_lessons_total(root),
        blueprint=read_blueprint(root),
        milestones=read_milestones(hub),
        tickets=read_tickets(hub),
        closed_review=review_closed(hub, recent_closed),
        cloud=read_latest_verify(hub),
        repo_state=state,
    )
