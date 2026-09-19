"""worktree（同一個 repo 的第二個工作目錄）的家、名字、開與拆——只有這一支在決定。

決定在 ``docs/decisions/worktrees-live-with-the-work-folder.md``。一張票一個資料夾
``~/aosr-v3-work/<票號>-<短名>/``：工單與證據放那一層，樹住它底下的 ``tree/``，分支叫
``<種類>/<票號>-<短名>``——資料夾、樹、分支三樣東西一個名字。

三個動作：``new`` 開樹、``list`` 列樹並把不合規矩的標出來、``remove`` 拆樹。拆樹四個條件
都成立才拆：PR（合併請求）已經合進主線（base 是主線，不是合進別條分支）、本機分支的頭就是
PR 的頭、樹裡沒有改過或沒進版控的檔、被 ``.gitignore`` 蓋住的檔只有重建得回來的那幾種。
主線用 squash（把整條分支壓成一顆提交）合併，``git branch --merged`` 看不出合了沒，所以去問
GitHub。沒進版控的檔由 ``git worktree remove`` 自己擋，這裡不加 ``--force``、也不吞它的錯；
但 git 的「乾淨」不看被忽略的檔、拆的時候會一起刪掉，所以那一格這裡自己先量。
派工資料夾裡的工單與證據不動，只拿掉 ``tree/`` 與本機分支。

**這一支不是規矩卡。** 雲端看不到本機的樹，合併門口守不到這件事；守得到的只有「``list``
把放錯地方的、名字對不上的、該拆沒拆的標出來並回非零」。

用法：``uv run python -m governance.worktrees new 364 worktree-home``、
``... list``、``... remove 364-worktree-home``。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION, ToolBroken, note, repo_root

WORK_FOLDER = "aosr-v3-work"
TREE_DIR = "tree"
DEFAULT_BASE = "origin/main"
KINDS = ("feat", "fix", "docs", "card", "chore")
SLUG = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
MERGED = "MERGED"
OPEN = "OPEN"
MAINLINE = "main"
IGNORED_MARK = "!! "
# 被 .gitignore 蓋住、拆樹時跟著消失也無所謂的東西：環境、快取、本機那份 junit 收據
# （本機跑出來的只是宣稱，雲端那一跑才算數）。不在這裡的被忽略檔一律擋下來給人看。
# 前一張只認樹的根層那一個（``notes/.venv/`` 不算）；位元組碼快取每一層都會長，認任何一層。
REBUILDABLE_AT_ROOT = frozenset({".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
REBUILDABLE_ANYWHERE = frozenset({"__pycache__"})
REBUILDABLE_PREFIXES = ("governance/receipts/",)
# 會把 git 指到別棵樹、別份 index 的環境變數。命令列的 --git-dir／--work-tree 蓋得過前兩個，
# 蓋不過第三個，所以真的那一支 runner 一律先把它們拿掉。
AMBIENT_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")

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
    base: str


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

    env = {key: value for key, value in os.environ.items() if key not in AMBIENT_GIT_ENV}

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run([tool, *args], cwd=repo, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise WorktreeError(f"git {' '.join(args)} 回 {proc.returncode}：{proc.stderr.strip()}")
        return proc

    return git


def gh_pull_request(repo: Path) -> PrLookup:
    """問 GitHub「這條分支的 PR 現在什麼狀態」。問不到就 raise，不當成沒有 PR。"""
    tool = shutil.which("gh")
    if tool is None:
        raise WorktreeError("找不到 gh（GitHub 的命令列工具），問不到 PR 合了沒")
    argv = [tool, "pr", "list", "--state", "all", "--json", "number,state,headRefOid,baseRefName"]

    def lookup(branch: str) -> PullRequest | None:
        proc = subprocess.run([*argv, "--head", branch], cwd=repo, capture_output=True, text=True)
        if proc.returncode != 0:
            raise WorktreeError(f"gh pr list --head {branch} 回 {proc.returncode}：{proc.stderr.strip()}")
        return parse_pull_request(proc.stdout)

    return lookup


def _pull_request(row: object) -> PullRequest:
    if not isinstance(row, dict):
        raise WorktreeError("gh 的輸出裡有一筆不是物件")
    number, state = row.get("number"), row.get("state")
    head, base = row.get("headRefOid"), row.get("baseRefName")
    if not (isinstance(number, int) and isinstance(state, str) and isinstance(head, str) and isinstance(base, str)):
        raise WorktreeError(f"gh 的輸出欄位不齊：{row}")
    return PullRequest(number=number, state=state, head_oid=head, base=base)


def parse_pull_request(raw: str) -> PullRequest | None:
    """``gh pr list --json ...`` 的輸出。空陣列＝這條分支沒有 PR。

    同一個分支名開過不只一個 PR 的時候：有還開著的就回開著的那一個（舊的合了不代表
    現在這一輪合了），都沒開著才回號碼最大的那一個。
    """
    try:
        rows: object = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise WorktreeError(f"gh 的輸出不是 JSON（{exc}）") from exc
    if not isinstance(rows, list):
        raise WorktreeError("gh 的輸出不是陣列")
    found = sorted((_pull_request(row) for row in rows), key=lambda pr: pr.number, reverse=True)
    still_open = [pr for pr in found if pr.state == OPEN]
    return next(iter(still_open or found), None)


def _blocks(git: Git) -> list[str]:
    return [b for b in git("worktree", "list", "--porcelain").stdout.split("\n\n") if b.strip()]


def main_tree(git: Git) -> Path:
    """主樹的路徑。從哪一棵樹裡面問，``git worktree list`` 的第一段都是主樹。"""
    blocks = _blocks(git)
    if not blocks:
        raise WorktreeError("git worktree list 什麼都沒印")
    first = blocks[0].splitlines()[0]
    key, _, value = first.partition(" ")
    if key != "worktree" or not value:
        raise WorktreeError(f"git worktree list 的第一段讀不出主樹：{first!r}")
    return Path(value)


def linked(git: Git) -> list[Linked]:
    """主樹以外的每一棵。第一段一定是主樹，丟掉。"""
    out: list[Linked] = []
    for block in _blocks(git)[1:]:
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
    # git 印的是解過符號連結的真實路徑，root 也要解過再比，不然工作區用符號連結接的時候每一棵都假紅。
    if item.path.name != TREE_DIR or item.path.parent.parent.resolve() != root.resolve():
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


def keep_reason(item: Linked, found: PullRequest | None, git: Git) -> str | None:
    """這一棵為什麼還不能拆。None＝PR 已經合進主線、而且本機的頭就是合進去的那一顆。

    這一支只管 PR 與頭那兩格；被忽略的檔那一格在 :func:`refusal`，``list`` 與 ``remove`` 問的都是它。
    """
    if item.branch is None:
        return "沒掛在分支上，這支工具不替你判斷，自己看過再用 git 拆"
    if found is None:
        return f"分支 {item.branch} 沒有 PR"
    if found.state != MERGED:
        return f"PR #{found.number} 現在是 {found.state}，還沒合"
    if found.base != MAINLINE:
        return f"PR #{found.number} 合進的是 {found.base}，不是主線 {MAINLINE}"
    local = git("rev-parse", f"refs/heads/{item.branch}").stdout.strip()
    if local != found.head_oid:
        return (
            f"本機 {item.branch} 的頭 {local} 不等於 PR #{found.number} 合進去的那一顆 {found.head_oid}"
            "——合了之後本機還有沒送出去的提交"
        )
    return None


def _rebuildable(rel: str) -> bool:
    parts = Path(rel).parts
    if rel.startswith(REBUILDABLE_PREFIXES) or not REBUILDABLE_ANYWHERE.isdisjoint(parts):
        return True
    return bool(parts) and parts[0] in REBUILDABLE_AT_ROOT


def ignored_keepsakes(item: Linked, git: Git) -> list[str]:
    """樹裡被 ``.gitignore`` 蓋住、又不是重建得回來的那幾種的檔。拆樹會把它們一起刪掉。

    明指 ``--git-dir``／``--work-tree`` 到那一棵：runner 的 cwd 是別棵樹，環境裡就算有
    ``GIT_DIR`` 也蓋不過命令列這兩格（``GIT_INDEX_FILE`` 蓋得過，所以 :func:`real_git` 先把它拿掉）。
    ``--untracked-files=normal``：全域設定關掉未追蹤檔顯示的機器上，``--ignored`` 會直接 fatal。
    ``--ignored=matching``：只印真的對上忽略樣式的那一層；預設模式會把「整個目錄底下只有
    被忽略的檔」收成上一層目錄，那樣就分不出裡面是快取還是別的東西。
    """
    proc = git(
        f"--git-dir={item.path / '.git'}", f"--work-tree={item.path}",
        "-c", "core.quotePath=false", "status", "--porcelain", "--untracked-files=normal", "--ignored=matching",
    )  # fmt: skip
    rels = [ln[len(IGNORED_MARK) :] for ln in proc.stdout.splitlines() if ln.startswith(IGNORED_MARK)]
    return [rel for rel in rels if not _rebuildable(rel)]


def refusal(item: Linked, found: PullRequest | None, git: Git) -> str | None:
    """這一棵為什麼不拆，None＝可以拆。``list`` 說「該拆」與 ``remove`` 真的去拆問的是同一支：

    兩邊條件不一樣的話，``list`` 會叫人去拆一棵 ``remove`` 自己不肯拆的樹，人就會改用硬拆。
    沒進版控的檔那一格不在這裡：那由 ``git worktree remove`` 當場擋。
    """
    reason = keep_reason(item, found, git)
    if reason is not None or not item.path.is_dir():
        return reason
    kept = ignored_keepsakes(item, git)
    if kept:
        return f"樹裡有被 .gitignore 蓋住的檔，拆了就跟著沒了，先搬到 {item.path.parent} 或刪掉：{kept}"
    return None


def remove(key: str, git: Git, pull_request: PrLookup) -> Linked:
    """拆一棵。任何一格不成立都不拆，原因照印。``git`` 的 cwd 不准是要拆的那一棵。"""
    item = find(key, git)
    reason = refusal(item, pull_request(item.branch) if item.branch is not None else None, git)
    if reason is not None or item.branch is None:
        raise WorktreeError(f"不拆：{reason}")
    git("worktree", "remove", str(item.path))
    git("branch", "-D", item.branch)
    return item


def report(items: Sequence[Linked], root: Path, pull_request: PrLookup, git: Git) -> int:
    """把每一棵印出來。有任何一棵放錯、名字不對、或可以拆了還沒拆，就回非零。

    某一棵量不下去（例：repo 搬過家，那棵樹的 ``.git`` 指到不存在的地方）不准把整份清單拖掉：
    那一棵照印、標「量不到」，其餘照量，最後回「這一跑不算數」——少量一棵的清單不能當成乾淨。
    """
    bad = 0
    unmeasured = 0
    for item in items:
        notes = problems(item, root)
        found = pull_request(item.branch) if item.branch is not None else None
        try:
            reason = refusal(item, found, git)
        except (WorktreeError, ValueError) as exc:
            # ValueError：樹裡有檔名不是合法 UTF-8 的被忽略檔，git 的輸出解不開。
            # 位置與名字那兩格不用 git 就算得出來，量不到也照印。
            unmeasured += 1
            note(f"{item.path}  [{item.branch}]  量不到：{exc}")
            for line in notes:
                note(f"    紅：{line}")
            continue
        if found is not None and reason is None:
            notes.append(f"PR #{found.number} 已經合進主線、頭也對得上，該拆（走 remove）")
        elif found is not None and found.state == MERGED:
            notes.append(f"PR 合了但別硬拆，先處理：{reason}")
        state = "沒有 PR" if found is None else f"PR #{found.number} {found.state}"
        note(f"{item.path}  [{item.branch}]  {state}")
        for line in notes:
            note(f"    紅：{line}")
        bad += bool(notes)
    note(f"共 {len(items)} 棵，{bad} 棵有事，{unmeasured} 棵量不到")
    if unmeasured:
        return TOOL_BROKEN
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
        # git 一律從主樹底下跑：站在要拆的那一棵裡面拆自己，樹拆掉之後 cwd 就不在了，
        # 下一步刪分支會死在半路，留下「樹沒了、分支還在」的殘骸。
        repo = main_tree(real_git(repo_root()))
        git = real_git(repo)
        if args.action == "new":
            name = tree_name(args.kind, args.issue, args.slug)
            if args.base == DEFAULT_BASE:
                git("fetch", "--quiet", "origin", MAINLINE)
            note(f"樹開在 {new(name, git, work_root(), args.base)}，分支 {name.branch}")
            return CLEAN
        if args.action == "list":
            return report(linked(git), work_root(), gh_pull_request(repo), git)
        gone = remove(args.key, git, gh_pull_request(repo))
        note(f"拆掉了 {gone.path}，本機分支 {gone.branch} 也刪了；派工資料夾留著")
        return CLEAN
    except (WorktreeError, ToolBroken, OSError, ValueError, RuntimeError) as exc:
        # 工具自己做不下去是「這一跑不算數」，不是「抓到違規」；不讓它變成 traceback 的離開碼 1。
        # ValueError：git 印出來的路徑不是合法 UTF-8；RuntimeError：家目錄解析不到。
        note(f"做不下去：{exc}")
        return TOOL_BROKEN


if __name__ == "__main__":
    raise SystemExit(main())
