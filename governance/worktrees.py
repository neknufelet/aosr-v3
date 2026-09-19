"""worktree（同一個 repo 的第二個工作目錄）的家、名字、開與拆——只有這一支在決定。

決定在 ``docs/decisions/worktrees-live-with-the-work-folder.md``。一張票一個資料夾
``~/aosr-v3-work/<票號>-<短名>/``：工單與證據放那一層，樹住它底下的 ``tree/``，分支叫
``<種類>/<票號>-<短名>``——資料夾、樹、分支三樣東西一個名字。

三個動作：``new`` 開樹、``list`` 列樹並把不合規矩的標出來、``remove`` 拆樹。拆樹只拆
「PR（合併請求）已經合進主線、本機分支的頭就是 PR 的頭、樹是乾淨的」那一種：主線用
squash（把整條分支壓成一顆提交）合併，``git branch --merged`` 看不出合了沒，所以去問
GitHub；樹髒不髒交給 ``git worktree remove`` 自己擋，這裡不加 ``--force``、也不吞它的錯。
派工資料夾裡的工單與證據不動，只拿掉 ``tree/`` 與本機分支。

**這一支不是規矩卡。** 雲端看不到本機的樹，合併門口守不到這件事；守得到的只有「``list``
把放錯地方的、名字對不上的、該拆沒拆的標出來並回非零」。

用法：``uv run python -m governance.worktrees new 364 worktree-home``、
``... list``、``... remove 364-worktree-home``。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, note, repo_root

WORK_FOLDER = "aosr-v3-work"
TREE_DIR = "tree"
DEFAULT_BASE = "origin/main"
KINDS = ("feat", "fix", "docs", "card", "chore")
SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
MERGED = "MERGED"

Git = Callable[..., "subprocess.CompletedProcess[str]"]


class WorktreeError(Exception):
    """這一步做不下去。原文照印，不換一條路繞過去。"""


@dataclass(frozen=True)
class TreeName:
    """一張票的那一個名字：資料夾、樹、分支都從這裡出。"""

    kind: str
    issue: int
    slug: str

    @property
    def folder(self) -> str:
        return f"{self.issue}-{self.slug}"

    @property
    def branch(self) -> str:
        return f"{self.kind}/{self.folder}"

    def path(self, root: Path) -> Path:
        return root / self.folder / TREE_DIR


@dataclass(frozen=True)
class Linked:
    """``git worktree list`` 列出來、主樹以外的一棵。``branch`` 是 None 代表沒掛在分支上。"""

    path: Path
    branch: str | None


@dataclass(frozen=True)
class PullRequest:
    number: int
    state: str
    head_oid: str


PrLookup = Callable[[str], PullRequest | None]


def work_root() -> Path:
    return Path.home() / WORK_FOLDER


def tree_name(kind: str, issue: int, slug: str) -> TreeName:
    if kind not in KINDS:
        raise WorktreeError(f"種類只認 {'、'.join(KINDS)}，不認 {kind!r}")
    if issue <= 0:
        raise WorktreeError(f"票號要是正整數，不是 {issue}")
    if SLUG.fullmatch(slug) is None:
        raise WorktreeError(f"短名只准小寫英數與連字號（例：worktree-home），不是 {slug!r}")
    return TreeName(kind=kind, issue=issue, slug=slug)


def real_git(repo: Path) -> Git:
    """在 ``repo`` 底下跑 git 的那一支。非零退出就 raise，stderr 原文帶出來。"""
    tool = shutil.which("git")
    if tool is None:
        raise WorktreeError("找不到 git")

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run([tool, *args], cwd=repo, capture_output=True, text=True)
        if proc.returncode != 0:
            raise WorktreeError(f"git {' '.join(args)} 回 {proc.returncode}：{proc.stderr.strip()}")
        return proc

    return git


def gh_pull_request(repo: Path) -> PrLookup:
    """問 GitHub「這條分支的 PR 現在什麼狀態」。問不到就 raise，不當成沒有 PR。"""
    tool = shutil.which("gh")
    if tool is None:
        raise WorktreeError("找不到 gh（GitHub 的命令列工具），問不到 PR 合了沒")
    argv = [tool, "pr", "list", "--state", "all", "--limit", "1", "--json", "number,state,headRefOid"]

    def lookup(branch: str) -> PullRequest | None:
        proc = subprocess.run([*argv, "--head", branch], cwd=repo, capture_output=True, text=True)
        if proc.returncode != 0:
            raise WorktreeError(f"gh pr list --head {branch} 回 {proc.returncode}：{proc.stderr.strip()}")
        return parse_pull_request(proc.stdout)

    return lookup


def parse_pull_request(raw: str) -> PullRequest | None:
    """``gh pr list --json number,state,headRefOid`` 的輸出。空陣列＝這條分支沒有 PR。"""
    try:
        rows: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorktreeError(f"gh 的輸出不是 JSON（{exc}）") from exc
    if not isinstance(rows, list):
        raise WorktreeError("gh 的輸出不是陣列")
    if not rows:
        return None
    row: object = rows[0]
    if not isinstance(row, dict):
        raise WorktreeError("gh 的輸出裡那一筆不是物件")
    number, state, head = row.get("number"), row.get("state"), row.get("headRefOid")
    if not isinstance(number, int) or not isinstance(state, str) or not isinstance(head, str):
        raise WorktreeError(f"gh 的輸出欄位不齊：{row}")
    return PullRequest(number=number, state=state, head_oid=head)


def linked(git: Git) -> list[Linked]:
    """主樹以外的每一棵。``git worktree list --porcelain`` 第一段一定是主樹，丟掉。"""
    blocks = [b for b in git("worktree", "list", "--porcelain").stdout.split("\n\n") if b.strip()]
    out: list[Linked] = []
    for block in blocks[1:]:
        path: Path | None = None
        branch: str | None = None
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            if key == "worktree":
                path = Path(value)
            elif key == "branch":
                branch = value.removeprefix("refs/heads/")
        if path is None:
            raise WorktreeError(f"git worktree list 有一段沒有路徑：{block!r}")
        out.append(Linked(path=path, branch=branch))
    return out


def problems(item: Linked, root: Path) -> list[str]:
    """這一棵哪裡不合規矩。空 list＝位置與名字都對。"""
    out: list[str] = []
    if item.path.name != TREE_DIR or item.path.parent.parent != root:
        out.append(f"放錯地方：應該住 {root}/<票號>-<短名>/{TREE_DIR}")
    if item.branch is None:
        out.append("沒掛在分支上")
    elif item.path.name == TREE_DIR and item.branch.partition("/")[2] != item.path.parent.name:
        out.append(f"名字對不上：資料夾 {item.path.parent.name}、分支 {item.branch}")
    return out


def new(name: TreeName, git: Git, root: Path, base: str = DEFAULT_BASE) -> Path:
    path = name.path(root)
    if path.exists():
        raise WorktreeError(f"{path} 已經在了")
    git("worktree", "add", "-b", name.branch, str(path), base)
    return path


def find(key: str, git: Git) -> Linked:
    """用資料夾名（``364-worktree-home``）或完整分支名找那一棵；舊位置的樹只能用分支名找。"""
    hits = [
        item
        for item in linked(git)
        if item.branch == key or (item.path.name == TREE_DIR and item.path.parent.name == key)
    ]
    if not hits:
        raise WorktreeError(f"沒有哪一棵樹叫 {key}（用資料夾名或完整分支名）")
    first, *rest = hits
    if rest:
        raise WorktreeError(f"{key} 對到不只一棵：{[str(h.path) for h in hits]}")
    return first


def remove(key: str, git: Git, pull_request: PrLookup) -> Linked:
    """拆一棵。PR 沒合、合了之後本機又多了提交、樹是髒的，三種都不拆。"""
    item = find(key, git)
    if item.branch is None:
        raise WorktreeError(f"{item.path} 沒掛在分支上，這支工具不替你判斷，自己看過再用 git 拆")
    found = pull_request(item.branch)
    if found is None:
        raise WorktreeError(f"分支 {item.branch} 沒有 PR，不拆")
    if found.state != MERGED:
        raise WorktreeError(f"PR #{found.number} 現在是 {found.state}，還沒合，不拆")
    local = git("rev-parse", f"refs/heads/{item.branch}").stdout.strip()
    if local != found.head_oid:
        raise WorktreeError(
            f"本機 {item.branch} 的頭 {local} 不等於 PR #{found.number} 合進去的那一顆 {found.head_oid}"
            "——合了之後本機還有沒送出去的提交，不拆"
        )
    git("worktree", "remove", str(item.path))
    git("branch", "-D", item.branch)
    return item


def report(items: Sequence[Linked], root: Path, pull_request: PrLookup) -> int:
    """把每一棵印出來。有任何一棵放錯、名字不對、或 PR 合了還沒拆，就回非零。"""
    bad = 0
    for item in items:
        notes = problems(item, root)
        found = pull_request(item.branch) if item.branch is not None else None
        if found is not None and found.state == MERGED:
            notes.append(f"PR #{found.number} 已經合了，該拆")
        state = "沒有 PR" if found is None else f"PR #{found.number} {found.state}"
        note(f"{item.path}  [{item.branch}]  {state}")
        for line in notes:
            note(f"    紅：{line}")
        bad += bool(notes)
    note(f"共 {len(items)} 棵，{bad} 棵有事")
    return VIOLATION if bad else CLEAN


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="governance.worktrees", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    add = sub.add_parser("new", help="開一棵樹")
    add.add_argument("issue", type=int)
    add.add_argument("slug")
    add.add_argument("--kind", default=KINDS[0], choices=KINDS)
    add.add_argument("--base", default=DEFAULT_BASE)
    sub.add_parser("list", help="列出每一棵，不合規矩的標紅")
    drop = sub.add_parser("remove", help="拆一棵（PR 已合、樹乾淨才拆）")
    drop.add_argument("key", help="資料夾名或完整分支名")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo = repo_root()
        git = real_git(repo)
        if args.action == "new":
            name = tree_name(args.kind, args.issue, args.slug)
            if args.base == DEFAULT_BASE:
                git("fetch", "--quiet", "origin", "main")
            note(f"樹開在 {new(name, git, work_root(), args.base)}，分支 {name.branch}")
            return CLEAN
        if args.action == "list":
            return report(linked(git), work_root(), gh_pull_request(repo))
        gone = remove(args.key, git, gh_pull_request(repo))
        note(f"拆掉了 {gone.path}，本機分支 {gone.branch} 也刪了；派工資料夾留著")
        return CLEAN
    except WorktreeError as exc:
        note(f"做不下去：{exc}")
        return TOOL_BROKEN


if __name__ == "__main__":
    raise SystemExit(main())
