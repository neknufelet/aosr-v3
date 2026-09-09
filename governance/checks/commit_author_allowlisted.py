#!/usr/bin/env python3
"""進主線的每一筆提交，author 與 committer 的 email 都要在名單裡。

掃的不是檔案，是**這次 PR 整段提交範圍**（``base..head``）每一筆的身分兩欄。三關：

**第一關 名單**
每一筆的 author 與 committer email 都必須在 ``governance/authors.txt`` 裡（比對前兩邊轉小寫）。
名單檔是這張卡自己帶進來的依賴：讀不到、或裡面一個 email 都沒有，一律回 2——不准當成
「名單是空的所以每一筆都紅」，也不准當成「沒名單就放行」。

**第二關 黑名單（不看名單，錨定比對）**
名單是人維護的檔，人可以把假身分加進去，所以再加一道不靠名單的：email 的網域精確等於
``example.com``／``example.invalid``／``localhost``（或是它們的子網域），或 local part 完整
等於 ``test``，一律紅。比對錨定在 ``@`` 與網域標籤的邊界上，**不是樸素子字串**：
``x@example.com.tw``（網域是 ``example.com.tw``）與 ``attest@corp.com``（local part 是
``attest``）都不會被誤咬。

**第三關 名單檔自己的守衛**
這段範圍新增進 ``governance/authors.txt`` 的 email，必須有提交在用它：同一段範圍裡有提交
用它，**或這條歷史裡已經在用**。後面那條是刻意放寬的：GitHub 合併 PR 時 squash 出來那筆的
committer 是 ``noreply@github.com``，這種機器身分不可能出現在同一個 PR 的提交裡，嚴格版
（只准同範圍）會讓「把它加進名單」這件事永遠做不到。放寬後仍然擋得住「先把陌生人加進名單、
下次再用」——那個 email 在這條歷史裡從來沒出現過。

**範圍怎麼定**
* ``AOSR_RANGE_BASE`` 有值：``base..head``（``AOSR_RANGE_HEAD`` 沒給就用 ``HEAD``）。
  CI 在 ``pull_request`` 事件把這兩個環境變數設成 ``base.sha`` 與 ``head.sha``——checkout 出來的
  ``HEAD`` 是 GitHub 合出來的假想合併提交，那一筆不會進主線，不該算在範圍裡。
* 沒給 base：``HEAD~1..HEAD``（push 到 main 就是一次 PR 合併；若那筆是真的合併提交，
  這個範圍連第二個父節點那一側的提交都會一起看到）。``HEAD`` 沒有父節點（根提交）就只看它自己。
* ``GITHUB_EVENT_NAME=pull_request`` 卻拿不到 base、或 base 那顆物件不在（CI 預設的淺 clone）、
  或範圍算出來是空集合——**一律回 2，不准回 0**。

每次都印一行 ``range=<範圍> commits=<幾筆> emails=<幾個不同 email> allowlist=<名單幾個>``，
外殼再印一行 ``scan_root= files= hits=``。

**每一次 git log 都帶 ``--no-use-mailmap``**：現在的 git 預設開 ``log.mailmap``，一份 ``.mailmap``
就能把 ``test@example.invalid`` 映射成一個在名單裡的位址，連 ``%ae``／``%ce`` 都會被改寫——
那等於在受檢的那個 PR 裡自己改尺。

**必紅樣本怎麼有歷史**
樣本是一棵進版控的迷你掃描根，裡面塞不進一個真的 ``.git``（巢狀 repo 進不了外層版控），
所以每份樣本用 ``governance/fixture-commit-range.txt`` 宣告要有哪幾筆提交、每筆的兩個 email，
檢查在暫存目錄照它 ``git init`` 出一段**真的** git 歷史再照上面三關跑。真的 git 工作樹裡若
出現那個宣告檔一律回 2——不然把它擺進主線就能繞過真歷史。

**前提（卡面上也寫了）**：這是 PR 範圍的閘。任何繞過 PR 直推主線的路徑都在它視野外，那要靠
主線 ruleset 擋（見 merge-gate-read-back）。它擋的是 v2 事故 worktree-hook-writes-into-real-repo
的下半場（污染身分的提交進主線），擋不住本機 hook 當下已造成的 ``.git`` 寫入，那半歸
tests-isolated-from-real-env 的 git_sandbox。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from governance.exit_codes import ToolBroken, run

AUTHORS_FILE = "governance/authors.txt"
FIXTURE_RANGE_FILE = "governance/fixture-commit-range.txt"

BASE_ENV = "AOSR_RANGE_BASE"
HEAD_ENV = "AOSR_RANGE_HEAD"
EVENT_ENV = "GITHUB_EVENT_NAME"

# 黑名單：錨定比對。網域要精確等於這些字（或是它們的子網域）、local part 要完整等於這些字，
# 所以 x@example.com.tw、attest@corp.com 不算命中。
BLOCKED_DOMAINS = ("example.com", "example.invalid", "localhost")
BLOCKED_LOCAL_PARTS = ("test",)

# 建樣本歷史時要丟掉的 git 環境變數：留著會讓暫存 repo 指到別人的 .git。
DROPPED_GIT_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)
FIXTURE_DATE = "2026-01-01T00:00:00+00:00"


class Range(NamedTuple):
    """要看的提交範圍。``base`` 是空字串代表沒有 base（根提交，只看 head 那一筆）。"""

    base: str
    head: str
    log_args: list[str]
    label: str


class Step(NamedTuple):
    """樣本宣告裡的一筆提交。``adds`` 是這筆要往名單檔加的 email。"""

    kind: str
    author: str
    committer: str
    adds: tuple[str, ...]
    message: str


def _run_git(
    argv: list[str],
    cwd: Path,
    *,
    what: str,
    env: dict[str, str] | None = None,
    allow: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    """跑一次 git。叫不動、或退出碼不在 ``allow`` 裡，一律 ToolBroken（回 2，不是回 0）。"""
    try:
        proc = subprocess.run(
            ["git", *argv], cwd=cwd, env=env, capture_output=True, text=True
        )
    except FileNotFoundError as exc:
        raise ToolBroken(f"外部工具 git 不在 PATH（{what}）：{exc}") from exc
    except OSError as exc:
        raise ToolBroken(f"叫不動 git（{what}）：{exc}") from exc
    if proc.returncode not in allow:
        raise ToolBroken(
            f"git {' '.join(argv)} 回 {proc.returncode}（{what}）：{proc.stderr.strip()[:300]}"
        )
    return proc


def _rev_exists(work_tree: Path, rev: str) -> bool:
    """這顆提交在不在（淺 clone 拿不到 base 就是不在）。"""
    proc = _run_git(
        ["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
        work_tree,
        what=f"確認 {rev} 這顆提交在不在",
        allow=(0, 1),
    )
    return proc.returncode == 0


def _read_allowlist(scan_root: Path, files: list[Path]) -> set[str]:
    """讀名單。讀不到、或空的，一律 ToolBroken——這一跑不算數。"""
    path = scan_root / AUTHORS_FILE
    if path not in files:
        raise ToolBroken(
            f"名單檔 {AUTHORS_FILE} 不在這棵樹的版控裡（{path}）"
            "——讀不到名單就不准回 0（放行），也不准當成空名單把每一筆都判紅"
        )
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"名單檔 {AUTHORS_FILE} 讀不開：{exc}") from exc
    listed = {
        line.strip().lower()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not listed:
        raise ToolBroken(
            f"名單檔 {AUTHORS_FILE} 裡一個 email 都沒有——空名單會讓每一筆都紅，"
            "那是「尺自己壞了」不是「大家都違規」，一律回 2"
        )
    return listed


def _blocked(email: str) -> str:
    """命中黑名單就回一句人話理由；沒命中回空字串。比對錨定在 @ 與網域標籤邊界上。"""
    local, sep, domain = email.rpartition("@")
    if not sep or not local or not domain:
        return f"{email!r} 不是 local-part@domain 的形狀，這種身分擋掉"
    if local in BLOCKED_LOCAL_PARTS:
        return f"local part 完整等於 {local!r}（測試用的假身分；attest@… 這種不算命中）"
    for blocked in BLOCKED_DOMAINS:
        if domain == blocked or domain.endswith("." + blocked):
            return (
                f"網域 {domain!r} 精確等於保留網域 {blocked!r}（或是它的子網域）；"
                "x@example.com.tw 這種不算命中"
            )
    return ""


def _toplevel(scan_root: Path) -> Path | None:
    """這棵樹是哪個 git 工作樹的一部分？不是 git 工作樹就回 None。"""
    proc = _run_git(
        ["rev-parse", "--show-toplevel"],
        scan_root,
        what="問這棵樹是不是 git 工作樹的根",
        allow=(0, 128),
    )
    line = proc.stdout.strip()
    if proc.returncode != 0 or not line:
        return None
    return Path(line).resolve()


def _range_from_env(work_tree: Path) -> Range:
    """真的工作樹要看哪一段。拿不到範圍一律 ToolBroken，不准退回「看 HEAD 一筆就好」。"""
    base = os.environ.get(BASE_ENV, "").strip()
    head = os.environ.get(HEAD_ENV, "").strip() or "HEAD"
    event = os.environ.get(EVENT_ENV, "").strip()
    if not base and event == "pull_request":
        raise ToolBroken(
            f"這是 {event} 事件，卻沒有 {BASE_ENV}——PR 的範圍是 base..head，"
            "拿不到 base 就什麼都沒掃到，一律回 2 不准回 0"
        )
    if not _rev_exists(work_tree, head):
        raise ToolBroken(f"解不出 head（{head}）——沒有 head 就沒有範圍")
    if base:
        if not _rev_exists(work_tree, base):
            raise ToolBroken(
                f"解不出 base（{base}）——CI 預設的淺 clone 沒有這顆物件；"
                "把 actions/checkout 設 fetch-depth: 0。拿不到 base 一律回 2，不准回 0"
            )
        return Range(base=base, head=head, log_args=[f"{base}..{head}"], label=f"{base}..{head}")
    if _rev_exists(work_tree, f"{head}~1"):
        spec = f"{head}~1..{head}"
        return Range(base=f"{head}~1", head=head, log_args=[spec], label=spec)
    return Range(base="", head=head, log_args=["--max-count=1", head], label=f"{head}（根提交，沒有父節點）")


def _read_plan(decl: Path) -> list[Step]:
    """讀樣本宣告的歷史。格式看不懂就 ToolBroken——樣本建不出來就等於什麼都沒掃到。"""
    try:
        text = decl.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到樣本的範圍宣告 {decl}：{exc}") from exc
    steps: list[Step] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if tokens[0] not in ("base", "commit"):
            raise ToolBroken(f"{decl}:{lineno} 開頭只認 base 或 commit，實際是 {tokens[0]!r}")
        if len(tokens) < 3:
            raise ToolBroken(
                f"{decl}:{lineno} 一筆要寫滿 <base|commit> <author_email> <committer_email>：{line!r}"
            )
        strange = [t for t in tokens[3:] if not t.startswith("+")]
        if strange:
            raise ToolBroken(f"{decl}:{lineno} 認不出來的欄位 {strange}——第四欄之後只認 +<email>")
        steps.append(
            Step(
                kind=tokens[0],
                author=tokens[1],
                committer=tokens[2],
                adds=tuple(t[1:] for t in tokens[3:]),
                message=f"樣本歷史第 {len(steps) + 1} 筆（{tokens[0]}）",
            )
        )
    if not steps:
        raise ToolBroken(f"{decl} 裡一筆提交都沒有——這份樣本建不出歷史")
    if steps[0].kind != "base":
        raise ToolBroken(f"{decl} 的第一筆必須是 base（範圍起點，base..head 不含它）")
    if any(step.kind == "base" for step in steps[1:]):
        raise ToolBroken(f"{decl} 只准有一筆 base，而且必須是第一筆")
    return steps


def _fixture_env(tmp: Path) -> dict[str, str]:
    """建樣本歷史用的乾淨環境：不吃外面的 git 設定，也不吃外面的 GIT_DIR。"""
    env = {k: v for k, v in os.environ.items() if k not in DROPPED_GIT_ENV}
    nowhere = str(tmp / "no-such-gitconfig")
    env["GIT_CONFIG_GLOBAL"] = nowhere
    env["GIT_CONFIG_SYSTEM"] = nowhere
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _commit_env(env: dict[str, str], step: Step) -> dict[str, str]:
    out = dict(env)
    out["GIT_AUTHOR_NAME"] = "Fixture Author"
    out["GIT_AUTHOR_EMAIL"] = step.author
    out["GIT_AUTHOR_DATE"] = FIXTURE_DATE
    out["GIT_COMMITTER_NAME"] = "Fixture Committer"
    out["GIT_COMMITTER_EMAIL"] = step.committer
    out["GIT_COMMITTER_DATE"] = FIXTURE_DATE
    return out


def _materialize(scan_root: Path, decl: Path) -> tuple[Path, Range, Callable[[], None]]:
    """照樣本宣告，在暫存目錄裡建一段真的 git 歷史。回傳（工作樹, 範圍, 清乾淨的函式）。"""
    plan = _read_plan(decl)
    tmp = Path(tempfile.mkdtemp(prefix="aosr-range-", dir=scan_root))

    def cleanup() -> None:
        # 清不掉自己開的暫存目錄就是這一跑被汙染了，讓它回 2，不要吞。
        try:
            shutil.rmtree(tmp)
        except OSError as exc:
            raise ToolBroken(f"清不掉自己開的暫存目錄 {tmp}：{exc}") from exc

    work = tmp / "repo"
    try:
        work.mkdir()
        env = _fixture_env(tmp)
        _run_git(
            ["-c", "init.defaultBranch=main", "init", "--quiet"],
            work,
            what="開樣本用的暫存 repo",
            env=env,
        )
        listed: list[str] = []
        shas: list[str] = []
        for step in plan:
            for added in step.adds:
                listed.append(added)
                target = work / AUTHORS_FILE
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("\n".join(listed) + "\n", encoding="utf-8")
                _run_git(["add", "--", AUTHORS_FILE], work, what="把名單檔加進暫存 repo", env=env)
            _run_git(
                ["commit", "--allow-empty", "--no-verify", "--quiet", "-m", step.message],
                work,
                what=f"在暫存 repo 補一筆提交（author {step.author}）",
                env=_commit_env(env, step),
            )
            shas.append(
                _run_git(["rev-parse", "HEAD"], work, what="讀暫存 repo 的 HEAD", env=env).stdout.strip()
            )
    except Exception:
        cleanup()
        raise
    base, head = shas[0], shas[-1]
    label = f"{base[:9]}..{head[:9]}（樣本 {scan_root.name} 重建的歷史）"
    return work, Range(base=base, head=head, log_args=[f"{base}..{head}"], label=label), cleanup


def _nothing() -> None:
    """真的工作樹不用清什麼。"""


def _resolve_range(scan_root: Path) -> tuple[Path, Range, Callable[[], None]]:
    """決定要對哪棵樹的哪一段跑。兩條路：真的 git 工作樹，或樣本宣告的歷史。"""
    decl = scan_root / FIXTURE_RANGE_FILE
    top = _toplevel(scan_root)
    if top is not None and top == scan_root:
        if decl.exists():
            raise ToolBroken(
                f"這是真的 git 工作樹的根，卻放著樣本用的 {FIXTURE_RANGE_FILE}"
                "——那條路只給必紅樣本用；真的工作樹裡出現它，等於真歷史被一個檔案繞過"
            )
        return scan_root, _range_from_env(scan_root), _nothing
    if decl.is_file():
        return _materialize(scan_root, decl)
    raise ToolBroken(
        f"{scan_root} 既不是 git 工作樹的根，也沒有 {FIXTURE_RANGE_FILE}"
        "——我拿不到提交範圍，這一跑不算數"
    )


def _commits(work_tree: Path, rng: Range) -> list[tuple[str, str, str]]:
    """範圍裡每一筆的 (sha, author email, committer email)。空範圍一律 ToolBroken。"""
    raw = _run_git(
        ["rev-list", "--count", *rng.log_args], work_tree, what=f"數 {rng.label} 有幾筆提交"
    ).stdout.strip()
    try:
        count = int(raw)
    except ValueError as exc:
        raise ToolBroken(f"git rev-list --count 的輸出我沒看懂：{raw!r}（{exc}）") from exc
    if count == 0:
        raise ToolBroken(
            f"掃描範圍 {rng.label} 算出來是空的——一筆提交都沒掃到，"
            "「沒有違規」這句話不算數，一律回 2"
        )
    out = _run_git(
        ["log", "--no-use-mailmap", "--format=%H%x09%ae%x09%ce", *rng.log_args],
        work_tree,
        what=f"讀 {rng.label} 每一筆的身分兩欄",
    ).stdout
    rows: list[tuple[str, str, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            raise ToolBroken(f"git log 這一行我沒看懂：{line!r}")
        rows.append((parts[0], parts[1].strip().lower(), parts[2].strip().lower()))
    if len(rows) != count:
        raise ToolBroken(
            f"git log 讀到 {len(rows)} 筆、rev-list 數出 {count} 筆，兩邊對不上——這一跑不算數"
        )
    return rows


def _identity_problems(commits: list[tuple[str, str, str]], listed: set[str]) -> list[str]:
    bad: list[str] = []
    for sha, author, committer in commits:
        for role, email in (("author", author), ("committer", committer)):
            why = _blocked(email)
            if why:
                bad.append(f"{sha[:9]} 的 {role} email {email} 命中黑名單：{why}")
            elif email not in listed:
                bad.append(
                    f"{sha[:9]} 的 {role} email {email} 不在名單 {AUTHORS_FILE} 裡"
                    "——沒登記的身分不准進主線"
                )
    return bad


def _added_allowlist_emails(work_tree: Path, rng: Range) -> set[str]:
    """這段範圍往名單檔加了哪些 email。"""
    if rng.base:
        argv = ["diff", "--no-color", "--unified=0", rng.base, rng.head, "--", AUTHORS_FILE]
    else:
        argv = ["show", "--no-color", "--unified=0", "--format=", rng.head, "--", AUTHORS_FILE]
    out = _run_git(argv, work_tree, what=f"看 {rng.label} 有沒有把 email 加進名單").stdout
    added: set[str] = set()
    for line in out.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        body = line[1:].strip()
        if not body or body.startswith("#"):
            continue
        added.add(body.lower())
    return added


def _history_emails(work_tree: Path, head: str) -> set[str]:
    """這條歷史（從 head 往回）已經有哪些身分在用。"""
    out = _run_git(
        ["log", "--no-use-mailmap", "--format=%ae%n%ce", head],
        work_tree,
        what="看這條歷史已經有哪些身分在用",
    ).stdout
    return {line.strip().lower() for line in out.splitlines() if line.strip()}


def _growth_problems(work_tree: Path, rng: Range, used: set[str]) -> list[str]:
    """名單檔自己的守衛：新增的 email 必須有提交在用它。"""
    added = _added_allowlist_emails(work_tree, rng)
    if not added:
        return []
    history = _history_emails(work_tree, rng.head)
    bad: list[str] = []
    for email in sorted(added - used - history):
        bad.append(
            f"這段範圍把 {email} 加進 {AUTHORS_FILE}，但範圍裡沒有任何提交在用它，"
            "這條歷史裡也沒人用過它——名單被預先放寬，等於先開門再走進來；"
            "新增名單要跟使用它的提交一起進來"
        )
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    listed = _read_allowlist(scan_root, files)
    work_tree, rng, cleanup = _resolve_range(scan_root)
    try:
        commits = _commits(work_tree, rng)
        used = {email for _, author, committer in commits for email in (author, committer)}
        bad = _identity_problems(commits, listed)
        bad += _growth_problems(work_tree, rng, used)
    finally:
        cleanup()
    print(f"range={rng.label} commits={len(commits)} emails={len(used)} allowlist={len(listed)}")
    return bad


if __name__ == "__main__":
    sys.exit(run(check, description="進主線的每一筆提交，author 與 committer 都要在名單裡"))
