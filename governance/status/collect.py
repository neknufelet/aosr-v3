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
from governance.status.model import (
    Blueprint,
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


def collect(root: Path, shell: Shell, env: Mapping[str, str], page_url: str) -> PageData:
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
        cloud=read_latest_verify(shell, slug),
        repo_state=state,
    )
