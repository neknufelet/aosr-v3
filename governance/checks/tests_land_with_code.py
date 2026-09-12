#!/usr/bin/env python3
"""產品程式與測試必須同一支合併請求落地。

掃的不是最終那棵樹，是**這支 PR 從共同分支點帶進來的那一段差異**。一句話：
``src/`` 底下的 Python 產品程式動了，同一段差異裡就必須有 ``tests/`` 底下的 Python 測試也動了。

**判的是同行，不是覆蓋**
它不問「這一行有沒有被測到」——那是語意，機器判不出來。它只問「產品變更有沒有帶著測試變更
一起進主線」。語意有效性仍然靠行為對照與獨立審查。

**範圍怎麼定（三點，不是兩點）**
* ``AOSR_RANGE_BASE`` 有值：``base...head``（三點；``AOSR_RANGE_HEAD`` 沒給就用 ``HEAD``）。
  三點是從**共同分支點**算起。這裡不能照抄 commit-author-allowlisted 的兩點寫法：CI 餵的
  ``base.sha`` 是主線當下的頂端，主線在你開分支之後合進來的任何測試變更，在 base 與 head
  兩棵樹之間都不一樣——兩點差異會把那些考卷算成「這個 PR 動了測試」，於是只改 ``src/`` 的
  PR 照樣綠。那道門縫不用惡意就走得過去（指揮 2026-09-12 在獨立暫存 repo 重跑確認）。
* 沒給 base：``main`` 解得出來、而且它跟 ``HEAD`` 的共同分支點不等於 ``HEAD``，就從那個分支點
  算到 ``HEAD``——本機分支上量到的因此是「這個分支到目前為止帶進來的全部」，也就是 PR 的語意；
  一筆一筆提交不會各自變紅（先提交產品、後補考卷是正常做法）。
* 共同分支點就是 ``HEAD``（在主線上）、或 ``main`` 不在這棵樹裡：改用 ``HEAD~1..HEAD``
  ——push 到 main 的那一筆就是一次 PR 合併，等於整個 PR。
* ``GITHUB_EVENT_NAME=pull_request`` 卻拿不到 base、head 或 base 那顆物件不在（淺 clone）、
  兩邊算不出共同祖先、根提交連父節點都沒有——**一律回 2，不准回 0**。

**差異怎麼讀**
``git diff --name-status -z -M``。``-z`` 所以檔名裡的空白與換行讀得對（逐行剖析會漏）；
``-M`` 所以改名被認出來是改名，不是「刪一個、加一個」。改名與複製兩側的路徑都收下來。
認不出的狀態字母（含 ``U``，未合併）一律回 2——「我不知道這個檔怎麼了」不准當成沒改。

**怎麼判**
* 產品側＝任一側路徑命中卡上登記的產品側底下、登記的副檔名。設定檔與標準答案不算。
* 測試側＝路徑命中卡上登記的測試側底下、登記的副檔名。考卷樹裡的說明文件與資料檔不算。
* 產品側有任何新增／修改／改名／複製 → 測試側必須有新增、修改，或**改名進來**
  （新路徑在 ``tests/`` 底下）。只刪測試不算，把考卷改名搬出去也不算。
* 產品側**全部**是刪除 → 測試側的刪除也算（整塊撤除的正常情況，考卷跟著走）。
* 產品側一筆都沒有 → 乾淨。空差異（空提交、只改治理檔）是有效範圍，回 0 不回 2。

**第一版不開豁免**
代價照實寫在卡面：改 ``src/`` 底下一行註解、一個 docstring、純結構調整，都要動一支考卷。
現有的放行機制（``[[settings.allow]]``）放行的單位是「這支檔的這個函式」，套不到「這一個 PR」；
要做 PR 層的豁免就得發明一個自報欄位，同一支 PR 的作者自己就寫得出來。哪天真的被擋到痛了，
另走一張決策紙。

**必紅樣本怎麼有歷史**
樣本是一棵進版控的迷你掃描根，裡面塞不進一個真的 ``.git``（巢狀 repo 進不了外層版控），
所以每份樣本用 ``governance/fixture-diff-range.txt`` 宣告要有哪幾筆提交、每筆碰哪些路徑，
檢查在暫存目錄 ``git init`` 出一段**真的** git 歷史再照上面那幾條跑。宣告只准提到樣本樹裡
**真的存在的檔**，內容逐位元從那個檔讀進來——宣告一串不存在的路徑字串建不出樹，回 2。
真的 git 工作樹裡出現那個宣告檔一律回 2（不然把它擺進主線就能繞過真歷史），而且真樹裡
那份宣告檔只准住在這張卡宣告的樣本樹底下（路徑從卡讀，不寫死在這裡）。

**前提（卡面上也寫了）**：這是 PR 範圍的閘，繞過 PR 直推主線的路徑在它視野外，那要靠主線
ruleset 擋（見 merge-gate-read-back）。

**它也不證明測試有效**：換掉一支真考卷、補一支只會過的，題數沒變，這一支放行；
green-must-be-real-green 的收集數地板擋的是「題數掉下去」，一換一它不會動。同行與題數
兩件事都不是有效性的證據——那一格今天沒有機器在守，靠行為對照與獨立審查。不要誤稱有牙。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import NamedTuple

from governance.exit_codes import ToolBroken, note, run
from governance.loader import RULES_DIR

# 樣本宣告歷史用的那份檔（相對掃描根）。真的工作樹的根出現它一律回 2。
FIXTURE_PLAN_FILE = "governance/fixture-diff-range.txt"

# 這張卡的 id。真樹那條路要從卡上讀「宣告檔只准住在哪幾棵樣本樹底下」。
CARD_ID = "tests-land-with-code"
# 卡上宣告樣本樹的那幾格。多一格少一格都不行：讀錯格子等於把守衛關掉。
FIXTURE_DIR_FIELDS = ("negative_fixture", "tool_broken_fixture")

BASE_ENV = "AOSR_RANGE_BASE"
HEAD_ENV = "AOSR_RANGE_HEAD"
EVENT_ENV = "GITHUB_EVENT_NAME"
PULL_REQUEST_EVENT = "pull_request"

# 主線叫什麼（本機沒給 base 的時候要跟它算共同分支點）。按順序試第一個解得出來的：
# 本機的分支，其次是遠端追蹤分支（CI 上 checkout 出來的樹不一定有本機同名分支）。
MAINLINE_REFS = ("refs/heads/main", "refs/remotes/origin/main")

# 登記簿的三個鍵：產品側住哪、測試側住哪、算哪幾種副檔名。**值住在卡上，這裡只有鍵名。**
# 順序有意義（read_rules 按順序取），多一個鍵、少一個鍵都回 2。
REGISTRY_KEYS = ("product_paths", "test_paths", "code_suffixes")

# git 的狀態字母。一側的：新增、修改、型別換了（檔變成連結那種）、刪除；
# 兩側的：改名、複製。其餘（U 未合併、X 未知、B 兩邊都壞）一律「我沒看懂」。
STATUS_ADDED = "A"
STATUS_DELETED = "D"
ONE_SIDED_STATUSES = (STATUS_ADDED, "M", "T", STATUS_DELETED)
TWO_SIDED_STATUSES = ("R", "C")

# 記在測試側「算同行」的那幾種：新增、修改、型別換了，加上改名／複製**進來**。
TEST_LANDING_STATUSES = (STATUS_ADDED, "M", "T", *TWO_SIDED_STATUSES)

NUL = "\0"

# 建樣本歷史時要丟掉的 git 環境變數：留著會讓暫存 repo 指到別人的 .git。
DROPPED_GIT_ENV = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
)
FIXTURE_DATE = "2026-01-01T00:00:00+00:00"
FIXTURE_IDENTITY = "fixture@aosr.invalid"

# 樣本宣告的語法。一段一筆提交：先一行 `commit <where>`，接著幾行 `<op> <路徑> [<新路徑>]`。
PLAN_COMMIT_KEYWORD = "commit"
PLAN_BASE = "base"
PLAN_MAIN = "main"
PLAN_HEAD = "head"
PLAN_WHERES = (PLAN_BASE, PLAN_MAIN, PLAN_HEAD)
PLAN_ADD = "add"
PLAN_MODIFY = "modify"
PLAN_DELETE = "delete"
PLAN_RENAME = "rename"
PLAN_OPS = (PLAN_ADD, PLAN_MODIFY, PLAN_DELETE, PLAN_RENAME)
# 樣本分支名。base 那一筆在主線上，候選那幾筆在另一支上。
PLAN_MAIN_BRANCH = "main"
PLAN_HEAD_BRANCH = "candidate"
# 宣告的路徑裡一律不准出現的段：前兩種會爬出樣本樹，最後一種是版控自己的內臟。
FORBIDDEN_SEGMENTS = ("..", ".", ".git")


class Rules(NamedTuple):
    """產品側／測試側住在哪、算哪幾種副檔名。**全部從卡的 ``[settings]`` 讀，程式裡沒有預設值。**"""

    product_paths: tuple[str, ...]
    test_paths: tuple[str, ...]
    code_suffixes: tuple[str, ...]


def _card_data(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """掃描根底下 id 是這張卡的那一張。找不到、找到多張、讀不開，一律 ToolBroken。"""
    mine: list[dict[str, object]] = []
    for path in sorted(f for f in files if f.parent == scan_root / RULES_DIR and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("id") == CARD_ID:
            mine.append(data)
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下 id={CARD_ID} 的卡有 {len(mine)} 張，要剛好一張"
            "——產品側／測試側住在哪只寫在卡上，讀不到這一跑就不算數"
        )
    return mine[0]


def read_rules(scan_root: Path, files: list[Path]) -> Rules:
    """從卡的 ``[settings]`` 讀判準：產品側、測試側、算哪幾種副檔名。

    為什麼不寫在程式裡：``scope_kind = "commits"`` 的卡，scan-scope-has-no-holes 只驗
    「宣告的每一條有沒有對象」，**不比集合**——所以「程式裡寫死、卡上也寫一份，不一樣會被抓」
    那句話是假的守衛（初讀點名，這裡照實改掉）。判準只有一個家：卡。
    """
    settings = _card_data(scan_root, files).get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(
            f"id={CARD_ID} 的卡沒有 [settings] 表（產品側、測試側、副檔名都寫在那裡）"
        )
    values: dict[str, tuple[str, ...]] = {}
    for key in REGISTRY_KEYS:
        raw = settings.get(key)
        if not isinstance(raw, list) or not raw or not all(isinstance(x, str) and x.strip() for x in raw):
            raise ToolBroken(
                f"id={CARD_ID} 的卡 [settings] {key} 必須是非空的字串 list，實際是 {raw!r}"
                "——打錯字的名單等於沒名單，一律回 2"
            )
        values[key] = tuple(x.strip() for x in raw)
    extra = [k for k in settings if k not in REGISTRY_KEYS]
    if extra:
        raise ToolBroken(
            f"id={CARD_ID} 的卡 [settings] 多了不認識的鍵 {sorted(extra)}，只認 {list(REGISTRY_KEYS)}"
        )
    return Rules(
        product_paths=values[REGISTRY_KEYS[0]],
        test_paths=values[REGISTRY_KEYS[1]],
        code_suffixes=values[REGISTRY_KEYS[2]],
    )


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


class Source(NamedTuple):
    """要對哪棵樹的哪一段跑，加一個清乾淨的函式。``real_tree`` 分得出這是真樹還是樣本重建的。"""

    work_tree: Path
    rng: Range
    cleanup: Callable[[], None]
    real_tree: bool


class PlanStep(NamedTuple):
    """樣本宣告裡的一筆提交：落在哪一支、碰哪些路徑。"""

    where: str
    ops: tuple[tuple[str, str, str], ...]


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


def _under(path: str, prefix: str, suffix: str) -> bool:
    """這個路徑在那個家底下，而且是那種副檔名。"""
    return path.startswith(prefix) and path.endswith(suffix)


def is_product(path: str, rules: Rules) -> bool:
    """這是產品程式嗎（卡上登記的產品側目錄底下、登記的副檔名）。"""
    return any(_under(path, home, suffix) for home in rules.product_paths for suffix in rules.code_suffixes)


def is_test(path: str, rules: Rules) -> bool:
    """這是測試嗎（卡上登記的測試側目錄底下、登記的副檔名）。文件與資料檔不算。"""
    return any(_under(path, home, suffix) for home in rules.test_paths for suffix in rules.code_suffixes)


def problems(entries: list[DiffEntry], rules: Rules) -> list[str]:
    """判這一段差異：產品程式有沒有帶著測試一起落地。"""
    product = [e for e in entries if any(is_product(side, rules) for side in e.sides)]
    if not product:
        return []

    landed = [e for e in entries if e.status in TEST_LANDING_STATUSES and is_test(e.new, rules)]
    dropped = [e for e in entries if e.status == STATUS_DELETED and is_test(e.old, rules)]
    renamed_out = [
        e
        for e in entries
        if e.status in TWO_SIDED_STATUSES and is_test(e.old, rules) and not is_test(e.new, rules)
    ]
    removal_only = all(e.status == STATUS_DELETED for e in product)
    if landed or (removal_only and dropped):
        return []

    seen: list[str] = []
    if dropped:
        seen.append(f"只刪掉測試（{_paths(e.old for e in dropped)}）")
    if renamed_out:
        seen.append(f"把測試改名搬出測試側（{_paths(e.old for e in renamed_out)}）")
    tail = (
        "測試側這一段" + ("、".join(seen) if seen else "一支 .py 都沒動")
        + f"——算同行的是卡上登記的測試側（{list(rules.test_paths)}）底下 {list(rules.code_suffixes)} 的新增、修改，或改名進來；"
        f"文件與資料檔不算，只刪測試也不算（產品側全是刪除才算）"
    )
    return [
        f"這一段差異{_describe(entry)}，卻沒有同一段裡的測試變更同行。{tail}"
        for entry in sorted(product, key=lambda e: e.sides)
    ]


def _paths(names: Iterable[str]) -> str:
    return "、".join(sorted(names))


def _describe(entry: DiffEntry) -> str:
    """一筆產品側變更的人話。"""
    if entry.status == STATUS_ADDED:
        return f"新增了產品程式 {entry.new}"
    if entry.status == STATUS_DELETED:
        return f"刪掉了產品程式 {entry.old}"
    if entry.status in TWO_SIDED_STATUSES:
        return f"把 {entry.old} 改名／複製成 {entry.new}（有一側是產品程式）"
    return f"改了產品程式 {entry.new}"


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


def _read_plan(decl: Path) -> list[PlanStep]:
    """讀樣本宣告的歷史。格式看不懂就 ToolBroken——樣本建不出來就等於什麼都沒量到。"""
    try:
        text = decl.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到樣本的差異宣告 {decl}：{exc}") from exc
    steps: list[PlanStep] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        if tokens[0] == PLAN_COMMIT_KEYWORD:
            if len(tokens) != 2 or tokens[1] not in PLAN_WHERES:
                raise ToolBroken(
                    f"{decl}:{lineno} 一段的開頭要寫 "
                    f"`{PLAN_COMMIT_KEYWORD} <{'|'.join(PLAN_WHERES)}>`，實際是 {line!r}"
                )
            steps.append(PlanStep(where=tokens[1], ops=()))
            continue
        if not steps:
            raise ToolBroken(
                f"{decl}:{lineno} 在第一段 `{PLAN_COMMIT_KEYWORD} …` 之前就出現動作 {line!r}"
            )
        if tokens[0] not in PLAN_OPS:
            raise ToolBroken(f"{decl}:{lineno} 認不出的動作 {tokens[0]!r}，只認 {list(PLAN_OPS)}")
        wanted = 3 if tokens[0] == PLAN_RENAME else 2
        if len(tokens) != wanted:
            raise ToolBroken(
                f"{decl}:{lineno} 動作 {tokens[0]} 要寫滿 {wanted - 1} 個路徑，實際是 {line!r}"
                "（路徑裡不准有空白——那種檔名由真樹那條路的 -z 剖析負責）"
            )
        first = _safe_relative(tokens[1], decl, lineno)
        second = _safe_relative(tokens[2], decl, lineno) if wanted == 3 else ""
        steps[-1] = PlanStep(
            where=steps[-1].where, ops=(*steps[-1].ops, (tokens[0], first, second))
        )
    _assert_plan_shape(steps, decl)
    return steps


def _safe_relative(rel: str, decl: Path, lineno: int) -> str:
    """宣告的路徑必須是「樣本樹底下的相對路徑」，而且不准碰版控自己的內臟。

    絕對路徑、``..``、``.``、空的路徑段、``.git``（任何一段）一律 ToolBroken。獨立重跑證實過
    這不是理論風險：宣告絕對路徑原本會讓建歷史那一步去改寫**樣本樹外面**的檔。
    這一關在剖析宣告的時候就把形狀擋掉，還沒開始動任何檔。
    """
    where = f"{decl}:{lineno} 的路徑 {rel!r}"
    if Path(rel).is_absolute() or rel.startswith("/"):
        raise ToolBroken(f"{where} 是絕對路徑——宣告只准寫成相對樣本樹的路徑，不准指到樹外面")
    segments = rel.split("/")
    if any(not segment for segment in segments):
        raise ToolBroken(f"{where} 有空的路徑段")
    for segment in segments:
        if segment.lower() in FORBIDDEN_SEGMENTS:
            raise ToolBroken(
                f"{where} 夾了 {segment!r}——{list(FORBIDDEN_SEGMENTS)} 這幾種段一律不准："
                "前兩種會爬出樣本樹，最後一種是版控自己的內臟"
            )
    return rel


def _inside(base: Path, target: Path) -> bool:
    """解析完（連結跟到底）之後，這個路徑還在那棵樹裡面嗎。"""
    try:
        return target.resolve().is_relative_to(base.resolve())
    except OSError:
        return False


def _assert_plan_shape(steps: list[PlanStep], decl: Path) -> None:
    """宣告的形狀：第一段是 base、只有一段 base、而且候選那一支至少有一筆。"""
    if not steps:
        raise ToolBroken(f"{decl} 裡一段提交都沒有——這份樣本建不出歷史")
    if steps[0].where != PLAN_BASE:
        raise ToolBroken(f"{decl} 的第一段必須是 {PLAN_BASE}（共同分支點，差異不含它自己）")
    if any(step.where == PLAN_BASE for step in steps[1:]):
        raise ToolBroken(f"{decl} 只准有一段 {PLAN_BASE}，而且必須是第一段")
    if not any(step.where == PLAN_HEAD for step in steps):
        raise ToolBroken(
            f"{decl} 一段 {PLAN_HEAD} 都沒有——候選分支上沒有提交，量出來永遠是空差異，"
            "這份樣本證明不了任何事"
        )


def _fixture_env(tmp: Path) -> dict[str, str]:
    """建樣本歷史用的乾淨環境：不吃外面的 git 設定，也不吃外面的 GIT_DIR。"""
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


def _seed_bytes(scan_root: Path, rel: str, decl: Path) -> bytes:
    """一個宣告到的路徑，內容從樣本樹裡**真的那個檔**逐位元讀進來。

    讀不到就 ToolBroken：宣告一串不存在的路徑字串建不出樹，那種樣本什麼都證明不了
    （准入第四條要的是真的產品／測試檔，不是路徑字串）。
    """
    source = scan_root / rel
    # 第二道（第一道是剖析宣告時的形狀）：解析完連結之後還要在樣本樹裡面。
    # 一條指到樹外面的 symlink 形狀完全合法，只有解析得出來才看得到它跑出去了。
    if source.is_symlink() or not _inside(scan_root, source):
        raise ToolBroken(
            f"{decl} 宣告的 {rel} 解析之後跑出樣本樹（或它自己是一條連結）"
            "——建歷史那一步只准讀樣本樹裡面的真檔"
        )
    if not source.is_file():
        raise ToolBroken(
            f"{decl} 宣告要碰 {rel}，但樣本樹裡沒有這個檔——宣告只准提到樣本樹裡真的存在的檔"
        )
    try:
        return source.read_bytes()
    except OSError as exc:
        raise ToolBroken(f"{decl} 宣告的 {rel} 讀不開：{exc}") from exc


def _write_target(work: Path, rel: str, decl: Path) -> Path:
    """要寫的那個路徑，確認它解析完還在暫存的那棵 repo 裡面。跑出去就 ToolBroken。"""
    target = work / rel
    parent = target.parent
    if parent.exists() and not _inside(work, parent):
        raise ToolBroken(f"{decl} 宣告的 {rel} 解析之後寫到暫存 repo 外面去了——一律回 2")
    if target.exists() and (target.is_symlink() or not _inside(work, target)):
        raise ToolBroken(f"{decl} 宣告的 {rel} 在暫存 repo 裡是一條連結或指到外面——一律回 2")
    return target


def _apply(
    work: Path, scan_root: Path, decl: Path, step: PlanStep, live: set[str], env: dict[str, str]
) -> None:
    """把一筆提交宣告的動作套到暫存樹上，再提交一筆。"""
    for op, first, second in step.ops:
        target = _write_target(work, first, decl)
        if op == PLAN_ADD:
            if first in live:
                raise ToolBroken(f"{decl} 要 {PLAN_ADD} {first}，但那個路徑這一段之前就已經在了")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(_seed_bytes(scan_root, first, decl))
            live.add(first)
        elif op == PLAN_MODIFY:
            if first not in live:
                raise ToolBroken(f"{decl} 要 {PLAN_MODIFY} {first}，但那個路徑這一段之前不在樹上")
            with target.open("ab") as handle:
                handle.write(f"# {decl.parent.parent.name} 改了一行\n".encode())
        elif op == PLAN_DELETE:
            if first not in live:
                raise ToolBroken(f"{decl} 要 {PLAN_DELETE} {first}，但那個路徑這一段之前不在樹上")
            target.unlink()
            live.discard(first)
        else:
            if first not in live:
                raise ToolBroken(f"{decl} 要 {PLAN_RENAME} {first}，但那個路徑這一段之前不在樹上")
            if second in live:
                raise ToolBroken(f"{decl} 要 {PLAN_RENAME} 成 {second}，但那個路徑已經在樹上了")
            moved = _write_target(work, second, decl)
            moved.parent.mkdir(parents=True, exist_ok=True)
            # 逐位元搬過去（不是重寫一份）：內容一模一樣，`-M` 才認得出這是改名不是刪一個加一個。
            moved.write_bytes(target.read_bytes())
            target.unlink()
            live.discard(first)
            live.add(second)
    _run_git(["add", "-A"], work, what="把這一筆的改動加進暫存 repo", env=env)
    _run_git(
        ["commit", "--allow-empty", "--no-verify", "--quiet", "-m", f"樣本歷史（{step.where}）"],
        work,
        what=f"在暫存 repo 補一筆提交（{step.where}）",
        env=env,
    )


def _materialize(scan_root: Path, decl: Path) -> Source:
    """照樣本宣告，在暫存目錄裡建一段真的 git 歷史，回傳要量的那一段。"""
    plan = _read_plan(decl)
    # 暫存區開在系統的暫存目錄，**不是**開在受檢的樣本樹裡面：樣本是證據，這一步只准讀它。
    # （上一版用 dir=scan_root，初讀點名那等於往受檢的樹寫檔；照抄 commit-author-allowlisted
    # 那條路是抄到了它的缺點。）
    tmp = Path(tempfile.mkdtemp(prefix="aosr-diff-"))

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
            ["-c", f"init.defaultBranch={PLAN_MAIN_BRANCH}", "init", "--quiet"],
            work,
            what="開樣本用的暫存 repo",
            env=env,
        )
        live: set[str] = set()
        base_step, rest = plan[0], plan[1:]
        _apply(work, scan_root, decl, base_step, live, env)
        fork = _rev_parse(work, "HEAD")
        main_live = set(live)
        for step in [s for s in rest if s.where == PLAN_MAIN]:
            _apply(work, scan_root, decl, step, main_live, env)
        main_tip = _rev_parse(work, "HEAD")
        _run_git(
            ["switch", "--quiet", "-c", PLAN_HEAD_BRANCH, fork],
            work,
            what="切到候選分支（從分支點長出去）",
            env=env,
        )
        for step in [s for s in rest if s.where == PLAN_HEAD]:
            _apply(work, scan_root, decl, step, live, env)
        head_tip = _rev_parse(work, "HEAD")
    except Exception:
        cleanup()
        raise
    label = f"{_short(main_tip)}...{_short(head_tip)}（樣本 {scan_root.name} 重建的歷史）"
    return Source(
        work_tree=work,
        rng=Range(diff_args=[f"{main_tip}...{head_tip}"], label=label),
        cleanup=cleanup,
        real_tree=False,
    )


def _nothing() -> None:
    """真的工作樹不用清什麼。"""


def _declared_fixture_dirs(scan_root: Path, files: list[Path]) -> list[str]:
    """這張卡宣告了哪幾棵樣本樹（路徑從卡讀，不寫死在這裡）。讀不到就 ToolBroken。"""
    data = _card_data(scan_root, files)
    mine: list[str] = []
    for field in FIXTURE_DIR_FIELDS:
        value = data.get(field)
        if isinstance(value, str) and value.strip():
            mine.append(value.strip().rstrip("/"))
    if not mine:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下找不到 id={CARD_ID} 的卡（或它沒宣告樣本樹）"
            f"——「{FIXTURE_PLAN_FILE} 只准住在樣本樹底下」這一條要從卡上讀，讀不到就不出結論"
        )
    return mine


def stray_declarations(scan_root: Path, files: list[Path]) -> list[str]:
    """真樹裡的宣告檔跑出樣本樹外面就是一筆違規：擺一個檔就能繞過真歷史。"""
    homes = _declared_fixture_dirs(scan_root, files)
    bad: list[str] = []
    for path in files:
        rel = path.relative_to(scan_root).as_posix()
        if not rel.endswith(FIXTURE_PLAN_FILE):
            continue
        if not any(rel.startswith(home + "/") for home in homes):
            bad.append(
                f"{rel} 是樣本用的差異宣告，卻沒住在卡宣告的樣本樹底下（{homes}）"
                "——那條路只給必紅樣本用；別處出現它，等於真歷史被一個檔案繞過"
            )
    return bad


def resolve_source(scan_root: Path) -> Source:
    """決定要對哪棵樹的哪一段跑。兩條路：真的 git 工作樹，或樣本宣告的歷史。"""
    decl = scan_root / FIXTURE_PLAN_FILE
    top = _toplevel(scan_root)
    if top is not None and top == scan_root:
        if decl.exists():
            raise ToolBroken(
                f"這是真的 git 工作樹的根，卻放著樣本用的 {FIXTURE_PLAN_FILE}"
                "——那條路只給必紅樣本用；真的工作樹裡出現它，等於真歷史被一個檔案繞過"
            )
        return Source(
            work_tree=scan_root, rng=resolve_range(scan_root), cleanup=_nothing, real_tree=True
        )
    if decl.is_file():
        return _materialize(scan_root, decl)
    raise ToolBroken(
        f"{scan_root} 既不是 git 工作樹的根，也沒有 {FIXTURE_PLAN_FILE}"
        "——我拿不到差異範圍，這一跑不算數"
    )


def check(scan_root: Path, files: list[Path]) -> list[str]:
    rules = read_rules(scan_root, files)
    source = resolve_source(scan_root)
    try:
        entries = diff_entries(source.work_tree, source.rng)
        bad = problems(entries, rules)
        if source.real_tree:
            bad += stray_declarations(scan_root, files)
    finally:
        source.cleanup()
    product = [e for e in entries if any(is_product(side, rules) for side in e.sides)]
    exams = [e for e in entries if any(is_test(side, rules) for side in e.sides)]
    note(f"range={source.rng.label} changed={len(entries)} product={len(product)} tests={len(exams)}")
    return bad


if __name__ == "__main__":
    sys.exit(
        run(check, description="產品程式與測試必須同一支合併請求落地（判同行，不判覆蓋）")
    )
