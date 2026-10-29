"""量一段提交範圍的差異：範圍怎麼定、``git diff --name-status -z`` 怎麼讀。

這一支原本住在規矩卡 tests-land-with-code 的檢查程式裡（#170）；那張卡 2026-10-29 退掉（#977），
借用這套零件的 precision-contracts-live-in-one-registry（牙②判一段提交範圍）還在，所以零件搬到
這裡自己一個家。只有零件，沒有判準：這裡不判任何一條規矩。

**範圍怎麼定（三點，不是兩點）**
* ``AOSR_RANGE_BASE`` 有值：``base...head``（三點；``AOSR_RANGE_HEAD`` 沒給就用 ``HEAD``）。
  三點是從**共同分支點**算起。CI 餵的 ``base.sha`` 是主線當下的頂端，主線在你開分支之後合進來的
  任何變更，在 base 與 head 兩棵樹之間都不一樣——兩點差異會把那些算成「這個 PR 動了」。
* 沒給 base：``main`` 解得出來、而且它跟 ``HEAD`` 的共同分支點不等於 ``HEAD``，就從那個分支點
  算到 ``HEAD``——本機分支上量到的是「這個分支到目前為止帶進來的全部」，也就是 PR 的語意。
* 共同分支點就是 ``HEAD``（在主線上）：改用 ``HEAD~1..HEAD``——push 到 main 的那一筆就是一次
  PR 合併，等於整個 PR。
* ``GITHUB_EVENT_NAME=pull_request`` 卻拿不到 base、head 或 base 那顆物件不在（淺 clone）、
  兩邊算不出共同祖先、根提交連父節點都沒有——**一律 ToolBroken（回 2），不准回 0**。

**差異怎麼讀**
``git diff --name-status -z -M``。``-z`` 所以檔名裡的空白與換行讀得對（逐行剖析會漏）；
``-M`` 所以改名被認出來是改名，不是「刪一個、加一個」。改名與複製兩側的路徑都收下來。
認不出的狀態字母（含 ``U``，未合併）一律 ToolBroken——「我不知道這個檔怎麼了」不准當成沒改。
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import NamedTuple

from governance.exit_codes import ToolBroken, note

BASE_ENV = "AOSR_RANGE_BASE"
HEAD_ENV = "AOSR_RANGE_HEAD"
EVENT_ENV = "GITHUB_EVENT_NAME"
PULL_REQUEST_EVENT = "pull_request"

# 主線叫什麼（本機沒給 base 的時候要跟它算共同分支點）。按順序試第一個解得出來的：
# 本機的分支，其次是遠端追蹤分支（CI 上 checkout 出來的樹不一定有本機同名分支）。
MAINLINE_REFS = ("refs/heads/main", "refs/remotes/origin/main")

# git 的狀態字母。一側的：新增、修改、型別換了（檔變成連結那種）、刪除；
# 兩側的：改名、複製。其餘（U 未合併、X 未知、B 兩邊都壞）一律「我沒看懂」。
STATUS_ADDED = "A"
STATUS_DELETED = "D"
ONE_SIDED_STATUSES = (STATUS_ADDED, "M", "T", STATUS_DELETED)
TWO_SIDED_STATUSES = ("R", "C")

NUL = "\0"

# 建暫存 repo 時要丟掉的 git 環境變數：留著會讓暫存 repo 指到別人的 .git。
DROPPED_GIT_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)
FIXTURE_DATE = "2026-01-01T00:00:00+00:00"
FIXTURE_IDENTITY = "fixture@aosr.invalid"


class DiffEntry(NamedTuple):
    """差異裡的一筆。``old`` 空字串＝新增，``new`` 空字串＝刪除，兩邊都有＝改名／複製。"""

    status: str
    old: str
    new: str

    @property
    def sides(self) -> tuple[str, ...]:
        """這一筆碰到的路徑（改名兩側都算）。"""
        return tuple(side for side in (self.old, self.new) if side)


class Range(NamedTuple):
    """要量的那一段。``diff_args`` 是餵給 ``git diff`` 的範圍寫法。"""

    diff_args: list[str]
    label: str

    @classmethod
    def two_dot(cls, base: str, head: str) -> Range:
        """``base..head``：兩棵樹直接比。只用在「範圍起點就是分支點」已經算好的時候。"""
        return cls(diff_args=[f"{base}..{head}"], label=f"{_short(base)}..{_short(head)}")

    @classmethod
    def three_dot(cls, base: str, head: str) -> Range:
        """``base...head``：從共同分支點算起。PR 的範圍是這一種。"""
        return cls(diff_args=[f"{base}...{head}"], label=f"{_short(base)}...{_short(head)}")


def _short(rev: str) -> str:
    """範圍標籤裡的短寫法。看得懂就好，不是用來回查的（完整的 sha 才回查得到）。"""
    return rev[:9] if len(rev) > 12 else rev


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


def _rev_parse(work_tree: Path, rev: str) -> str:
    """這顆提交的完整 id。解不出來回空字串（淺 clone 拿不到 base 就是這種）。"""
    proc = _run_git(
        ["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"],
        work_tree,
        what=f"解 {rev} 這顆提交",
        allow=(0, 1),
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


def _merge_base(work_tree: Path, left: str, right: str) -> str:
    """兩顆提交的共同分支點。沒有共同祖先回空字串。"""
    proc = _run_git(
        ["merge-base", left, right],
        work_tree,
        what=f"算 {left} 與 {right} 的共同分支點",
        allow=(0, 1),
    )
    return proc.stdout.strip() if proc.returncode == 0 else ""


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


def _fixture_env(tmp: Path) -> dict[str, str]:
    """建暫存 repo 用的乾淨環境：不吃外面的 git 設定，也不吃外面的 GIT_DIR。"""
    env = {k: v for k, v in os.environ.items() if k not in DROPPED_GIT_ENV}
    nowhere = str(tmp / "no-such-gitconfig")
    env["GIT_CONFIG_GLOBAL"] = nowhere
    env["GIT_CONFIG_SYSTEM"] = nowhere
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["GIT_AUTHOR_NAME"] = "Fixture Author"
    env["GIT_AUTHOR_EMAIL"] = FIXTURE_IDENTITY
    env["GIT_AUTHOR_DATE"] = FIXTURE_DATE
    env["GIT_COMMITTER_NAME"] = "Fixture Committer"
    env["GIT_COMMITTER_EMAIL"] = FIXTURE_IDENTITY
    env["GIT_COMMITTER_DATE"] = FIXTURE_DATE
    return env


def parse_diff(raw: str) -> list[DiffEntry]:
    """讀 ``git diff --name-status -z`` 的輸出。

    ``-z`` 的一筆是「狀態 NUL 路徑 NUL」，改名／複製是「狀態 NUL 舊路徑 NUL 新路徑 NUL」。
    逐 NUL 走，所以檔名裡的空白與換行都在同一格裡，不會被切開。截斷、空路徑、認不出的狀態
    字母一律 ToolBroken——讀不懂差異就不出結論。

    **每一格都以 NUL 收尾，所以整段輸出也一定以 NUL 收尾。** 沒收尾就是被截斷了，最後那一格
    不准收下——獨立重跑實測過：一份「狀態、一個路徑、最後少一個收尾 NUL」的輸出原本照樣被收下，
    等於讀了半份差異還下結論。相似度數字的形狀也嚴格比：兩側的狀態（改名、複製）一定帶數字，一側的不准帶
    （``A123`` 不是「新增」，是我沒看懂的形狀）。
    """
    if not raw:
        return []
    if not raw.endswith(NUL):
        raise ToolBroken(
            "差異輸出沒有以 NUL 收尾——-z 的每一格都以 NUL 結束，沒收尾就是這一段被截斷了，"
            "最後那一格不准當成讀完整的一筆"
        )
    tokens = raw.split(NUL)
    tokens.pop()
    out: list[DiffEntry] = []
    index = 0
    while index < len(tokens):
        field = tokens[index]
        index += 1
        letter = field[:1].upper()
        score = field[1:]
        if letter in TWO_SIDED_STATUSES:
            wanted = 2
            if not score or not score.isdigit():
                raise ToolBroken(
                    f"狀態欄 {field!r} 我沒看懂：改名／複製一定帶相似度數字（R100 這種），"
                    "裸的字母不是這個協定裡的東西"
                )
        elif letter in ONE_SIDED_STATUSES:
            wanted = 1
            if score:
                raise ToolBroken(
                    f"狀態欄 {field!r} 我沒看懂：一側的狀態（新增、修改、刪除）不帶相似度數字"
                )
        else:
            raise ToolBroken(
                f"認不出的狀態字母 {field!r}——U（未合併）、X、B 這幾種是「我不知道這個檔怎麼了」，"
                "不准當成沒改，一律回 2"
            )
        if len(tokens) - index < wanted:
            raise ToolBroken(
                f"狀態 {field!r} 後面少了路徑（還要 {wanted} 格，只剩 {len(tokens) - index} 格）"
                "——差異讀不完整就不出結論"
            )
        paths = tokens[index : index + wanted]
        index += wanted
        if any(not path for path in paths):
            raise ToolBroken(f"狀態 {field!r} 的路徑是空的——差異我沒看懂就不出結論")
        if wanted == 2:
            out.append(DiffEntry(status=letter, old=paths[0], new=paths[1]))
        elif letter == STATUS_ADDED:
            out.append(DiffEntry(status=letter, old="", new=paths[0]))
        elif letter == STATUS_DELETED:
            out.append(DiffEntry(status=letter, old=paths[0], new=""))
        else:
            out.append(DiffEntry(status=letter, old=paths[0], new=paths[0]))
    return out


def resolve_range(work_tree: Path) -> Range:
    """真的工作樹要量哪一段。拿不到可信範圍一律 ToolBroken，不准退回「看一筆就好」。"""
    base = os.environ.get(BASE_ENV, "").strip()
    head_ref = os.environ.get(HEAD_ENV, "").strip() or "HEAD"
    event = os.environ.get(EVENT_ENV, "").strip()
    if not base and event == PULL_REQUEST_EVENT:
        raise ToolBroken(
            f"這是 {event} 事件，卻沒有 {BASE_ENV}——PR 的範圍是從共同分支點算起，"
            "拿不到 base 就什麼都沒量到，一律回 2 不准回 0"
        )
    head = _rev_parse(work_tree, head_ref)
    if not head:
        raise ToolBroken(f"解不出 head（{head_ref}）——沒有 head 就沒有範圍")
    if base:
        if not _rev_parse(work_tree, base):
            raise ToolBroken(
                f"解不出 base（{base}）——CI 預設的淺 clone 沒有這顆物件；"
                "把 actions/checkout 設 fetch-depth: 0。拿不到 base 一律回 2，不准回 0"
            )
        if not _merge_base(work_tree, base, head):
            raise ToolBroken(
                f"base（{base}）跟 head（{head}）算不出共同分支點——沒有分支點就量不出"
                "「這個 PR 自己帶進來的是什麼」，一律回 2"
            )
        return Range.three_dot(base, head)
    mainline = next((ref for ref in MAINLINE_REFS if _rev_parse(work_tree, ref)), "")
    if not mainline:
        raise ToolBroken(
            f"沒給 {BASE_ENV}，這棵樹裡又一個主線參照都找不到（試過 {list(MAINLINE_REFS)}）"
            "——認不出主線就分不出「這是主線上的一筆合併」還是「某個分支的最後一筆」，"
            "拿不到可信範圍一律回 2。不准默認任何分支都可以退成量最後一筆：那樣答案會取決於"
            "跑的時候 HEAD 剛好停在哪一筆"
        )
    fork = _merge_base(work_tree, mainline, head)
    if not fork:
        raise ToolBroken(
            f"找得到主線參照 {mainline}，卻跟 head（{head_ref}）算不出共同分支點"
            "——有主線卻沒有可信的分支點，一律回 2，不准退回「量最後一筆」"
        )
    if fork != head:
        note(f"沒給 {BASE_ENV}，改用跟 {mainline} 的共同分支點當範圍起點")
        return Range.two_dot(fork, head)
    # 共同分支點就是 head 自己：head 在主線上（或就是主線頂端），這是「push 到主線」的形狀。
    previous = _rev_parse(work_tree, f"{head_ref}~1")
    if not previous:
        raise ToolBroken(
            f"head（{head_ref}）就在主線上（{mainline}），卻沒有父節點——根提交前面沒有範圍，一律回 2"
        )
    note(f"沒給 {BASE_ENV}，head 就在 {mainline} 上——按主線 push 量上一個主線提交到 head")
    return Range.two_dot(previous, head)


def diff_entries(work_tree: Path, rng: Range) -> list[DiffEntry]:
    """量那一段的差異。空差異是合法的（空提交、只改別處），不是工具壞。"""
    proc = _run_git(
        ["diff", "--name-status", "-z", "-M", *rng.diff_args],
        work_tree,
        what=f"量 {rng.label} 的差異",
    )
    return parse_diff(proc.stdout)
