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
from collections.abc import Mapping, Sequence
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


def read_milestones(shell: Shell, slug: str) -> tuple[Milestone, ...]:
    """里程碑（milestone，一段路線的名字）與各自開著／關掉的票數。"""
    rows = as_rows(
        api(shell, f"repos/{slug}/milestones?state=all&per_page=100", "問里程碑"),
        "里程碑清單",
    )
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


def read_tickets(shell: Shell, slug: str) -> tuple[Ticket, ...]:
    """開著的票。GitHub 的 issues API 把 PR 也算 issue，所以順手分出來。"""
    rows = as_rows(
        api(shell, f"repos/{slug}/issues?state=open&per_page=100", "問開著的票"),
        "開著的票",
    )
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


def read_run_steps(shell: Shell, slug: str, run_id: int) -> tuple[int, tuple[StepResult, ...]]:
    """那一跑裡 verify 這個 job 的耗時，以及它每一步（每一步就是一支檢查）的結論。"""
    rows = as_rows(
        as_table(
            api(shell, f"repos/{slug}/actions/runs/{run_id}/jobs", "問那一跑的 job"),
            "那一跑的 job",
        ).get("jobs"),
        "job 清單",
    )
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


def read_latest_verify(shell: Shell, slug: str) -> CloudRun | None:
    """主線最近一次 verify（雲端檢查）那一跑。一次都沒跑過就回 None，頁面會寫成紅字。"""
    rows = as_rows(
        as_table(
            api(
                shell,
                f"repos/{slug}/actions/runs?branch={MAIN}&per_page=30",
                "問主線最近幾跑",
            ),
            "主線最近幾跑",
        ).get("workflow_runs"),
        "那幾跑",
    )
    for item in rows:
        row = as_table(item, "一跑")
        if field_text(row, "name") != VERIFY_WORKFLOW:
            continue
        run_id = field_int(row, "id")
        seconds, steps = read_run_steps(shell, slug, run_id)
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
# 一張關掉的 issue（待辦票）要串得起三樣東西：一個合進主線的 PR（合併請求）、那個 PR 併出來的
# 那一顆 commit、以及那一顆 commit 的 verify（雲端檢查）在 status 分支上留下的一份綠收據。
# 串不起來的用紅字列出來——上一代出過的事就是「票關了、東西沒真的落地」。


@dataclass(frozen=True)
class MergedPull:
    """一個已經合進主線的 PR（合併請求）：號碼、什麼時候合的、併出來的那一顆 commit。"""

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


def read_merged_pulls(shell: Shell, slug: str) -> dict[int, MergedPull]:
    """已經合進主線的 PR（合併請求），號碼 → 它併出來的那一顆 commit。

    關掉但沒合的 PR 不算落地，直接跳過：這一格問的是「東西進主線了沒」。
    一頁問完（這個 repo 的 PR 遠少於一頁；更舊的那些不在「最近關掉的票」的射程裡）。
    """
    rows = as_rows(
        api(
            shell,
            f"repos/{slug}/pulls?state=closed&base={MAIN}&sort=updated&direction=desc&per_page=100",
            "問合進主線的 PR",
        ),
        "合進主線的 PR",
    )
    found: dict[int, MergedPull] = {}
    for item in rows:
        row = as_table(item, "一個 PR")
        merged_at = field_text(row, "merged_at")
        if not merged_at:
            continue
        number = field_int(row, "number")
        found[number] = MergedPull(
            number=number,
            url=field_text(row, "html_url"),
            merged_at=merged_at,
            merge_sha=field_text(row, "merge_commit_sha"),
        )
    return found


def read_closed_issues(shell: Shell, slug: str, limit: int) -> tuple[ClosedIssue, ...]:
    """最近關掉的 issue（待辦票）。

    GitHub 的 issues API 把 PR 也算 issue，所以先分掉；它也排不出「按關掉時間」，
    所以拿一頁回來自己按 `closed_at` 由新到舊排，取最前面那幾張（幾張由入口的命令列參數決定）。
    """
    rows = as_rows(
        api(
            shell,
            f"repos/{slug}/issues?state=closed&sort=updated&direction=desc&per_page=100",
            "問關掉的票",
        ),
        "關掉的票",
    )
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


def linked_merged_pull(
    shell: Shell, slug: str, number: int, merged: Mapping[int, MergedPull]
) -> MergedPull | None:
    """這張票的 timeline（GitHub 記的事件流）上，有沒有一個已經合進主線的 PR 提到它。

    這個 repo 的票是人手關的（不是靠 PR 內文的關鍵字自動關），所以 GitHub 沒有「被誰關掉」
    那條連結可讀；看得到的證據就是票面上那幾條互相提到（cross-referenced）。有好幾個就取
    最後合進去的那一個——這一格問的是「有沒有落地」，不是「哪一筆落地」。
    """
    rows = as_rows(
        api(shell, f"repos/{slug}/issues/{number}/timeline?per_page=100", f"問 #{number} 的事件"),
        f"#{number} 的事件",
    )
    seen: list[MergedPull] = []
    for item in rows:
        row = as_table(item, "一件事")
        source = row.get("source")
        if not isinstance(source, dict):
            continue
        referenced = source.get("issue")
        if not isinstance(referenced, dict):
            continue
        pull = merged.get(field_int(as_table(referenced, "被提到的那一張"), "number"))
        if pull is not None:
            seen.append(pull)
    if not seen:
        return None
    return max(seen, key=lambda pull: pull.merged_at)


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
    issue: ClosedIssue, pull: MergedPull | None, receipt: ReceiptFacts | None
) -> ClosedTicket:
    """一張關掉的票的那一列：串得起來就沒話說，串不起來就寫下是哪一段斷掉的。"""
    if pull is None:
        problem = "沒有一個合進主線的 PR 提到它——票關了，東西不知道有沒有進主線"
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


def review_closed(shell: Shell, slug: str, limit: int) -> ClosedReview:
    """整格算出來：最近關掉的那幾張票，各自對不對得到一份綠收據。"""
    merged = read_merged_pulls(shell, slug)
    source, receipts = read_receipts_by_sha(shell)
    rows: list[ClosedTicket] = []
    for issue in read_closed_issues(shell, slug, limit):
        pull = linked_merged_pull(shell, slug, issue.number, merged)
        rows.append(judge_closed(issue, pull, receipts.get(pull.merge_sha) if pull else None))
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
    root: Path, shell: Shell, env: Mapping[str, str], page_url: str, recent_closed: int
) -> PageData:
    """把整頁的資料算出來。任何一格算不出來就 ToolBroken，不補假值。"""
    state = read_repo_state(shell)
    slug = read_slug(shell, env)
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
        milestones=read_milestones(shell, slug),
        tickets=read_tickets(shell, slug),
        closed_review=review_closed(shell, slug, recent_closed),
        cloud=read_latest_verify(shell, slug),
        repo_state=state,
    )
