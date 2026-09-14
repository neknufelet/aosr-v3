"""開跑時丟掉舊 junit 收據，不預跑整套 pytest，並且不寫 .pyc。

**為什麼以前預跑、現在不用。** `green-must-be-real-green` 判的是 pytest 產的 junit 收據；
收據不進版控，剛 clone 的樹裡不存在。以前為了讓這張卡在後設測試第 1 回拿到收據，conftest
先跑一次真的全套，代價是每次把全套考兩遍。現在宣告了 `[junit]` 的卡在第 1 回固定證明
「收據不在就必須回 1」；真的收據本來就由 `.github/workflows/verify.yml` 的 pytest 步驟產生，
並在下一步「綠必須是真的綠」判，所以不需要為一張卡的第 1 回預跑整套。兩條禁令照舊：不是
「檔不存在就回 0」，也不是檢查自己造收據。主控開跑仍先丟掉卡宣告路徑上的舊收據，不准沿用
上一跑的檔。

**第二段：測試不准碰真環境（規矩卡 tests-isolated-from-real-env 的動態那半）。**

1. ``git_sandbox``——唯一准 spawn 版控工具的 fixture。它把 ``GIT_DIR``／``GIT_WORK_TREE``／
   ``GIT_INDEX_FILE`` **主動指到**暫存樹（不是只 unset）、``GIT_CONFIG_GLOBAL=/dev/null``
   ＋``GIT_CONFIG_NOSYSTEM=1``、身分走 ``GIT_AUTHOR_*``／``GIT_COMMITTER_*`` 環境變數而不是
   ``git config`` 寫檔，開樹之後先做負控制（``git rev-parse --show-toplevel`` 不等於暫存樹
   就 fail），跑完把樹刪掉。這一段逐條照 v2 事故 ``worktree-hook-writes-into-real-repo``
   自己寫的對策做。
2. session 起始的守衛——那三個變數只要已經被注入就直接停跑（hook 底下跑測試就是這個形狀），
   然後記下真 repo 的 ``git status --porcelain``。
3. session 收尾的守衛——再量一次。多一個未追蹤檔、少一個、或有檔被改，就是一筆 teardown
   error（收據上看得到，離開碼非零）。量測的呼叫收在 ``governance/repo_residue.py``，
   接線由 ``tests/test_repo_residue_guard.py`` 咬。

**第三段：這一跑為什麼跑得快（issue #44）。**

1. ``-n auto``（設定寫在 ``pyproject.toml`` 的 ``addopts``）——xdist（pytest 的平行外掛）把
   後設測試分給「跟核心數一樣多」的工人（worker，平行跑測試的子程序）。可以平行的理由是
   後設測試每一回合本來就是一個獨立子程序、掃描根是唯讀的樣本樹，回合之間不共用狀態。
   ``pytest_sessionstart`` 在主控（controller，分派測試的那個程序）與每個工人身上各跑一次，
   所以下面兩件「整跑只做一次」的事靠 ``PYTEST_XDIST_WORKER`` 認出自己是不是主控：鏡收據、
   丟掉舊 junit 收據。主控的 ``pytest_sessionstart`` 跑完才生工人，所以工人開始收集時鏡像
   已經在了，舊 junit 收據也已經不在。
2. ``AOSR_GH_REPLAY_DIR``——兩張准上網的卡（``merge-gate-read-back``、
   ``issues-closed-only-by-merged-pr``）的六回合裡有三回合真的會問 GitHub，同一句問題一次
   ``uv run pytest`` 仍會問很多次。這一格指到一個暫存的錄音目錄（不是版控樹裡），同一句只
   真的問一次，其餘重播；環境變數跟著傳給工人，所以整跑共用同一份錄音。
   **只有這裡設它**：CI 上跑檢查那幾步身上沒有這一格，雲端那一跑每一句都真的去問伺服器。
   重播不會讓判準變寬——它一樣要求那支工具真的在 ``PATH`` 上，所以第 4 回合（抽掉外部工具
   必須回 2）在快取是熱的時候照樣回 2。這兩句話由 ``tests/test_gh_replay.py`` 咬。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

# 不寫 .pyc：上一張卡撞過「改了程式，子程序卻拿 __pycache__ 裡的舊位元碼」的坑。
sys.dont_write_bytecode = True

import pytest  # noqa: E402  # expires=2026-12-08 reason=這幾個 import 必須排在 sys.dont_write_bytecode 與 REPO 那兩行之後，不是可以往上搬的；到期時重審

REPO = Path(__file__).resolve().parents[1]

from governance import gh_replay, repo_residue  # noqa: E402  # expires=2026-12-08 reason=這幾個 import 必須排在 sys.dont_write_bytecode 與 REPO 那兩行之後，不是可以往上搬的；到期時重審
from governance.exit_codes import ToolBroken  # noqa: E402  # expires=2026-12-08 reason=這幾個 import 必須排在 sys.dont_write_bytecode 與 REPO 那兩行之後，不是可以往上搬的；到期時重審
from governance.loader import load_all_cards  # noqa: E402  # expires=2026-12-08 reason=這幾個 import 必須排在 sys.dont_write_bytecode 與 REPO 那兩行之後，不是可以往上搬的；到期時重審
from governance.status import mirror_receipts  # noqa: E402  # expires=2026-12-08 reason=這幾個 import 必須排在 sys.dont_write_bytecode 與 REPO 那兩行之後，不是可以往上搬的；到期時重審

MIRROR_TIMEOUT = 60

# xdist（pytest 的平行外掛）給每個工人（worker，平行跑測試的子程序）設的環境變數；主控
# （controller，分派測試的那個程序）身上沒有這一格。`pytest_sessionstart` 在主控與每個工人
# 身上各跑一次，所以鏡收據與丟掉舊 junit 收據要靠這一格認出自己是不是主控——工人也刪一次，
# 會跟正在產新收據的主控搶同一個檔。
XDIST_WORKER_ENV = "PYTEST_XDIST_WORKER"

# 這一跑的 gh（GitHub 的命令列工具）錄音目錄。兩張准上網的卡的後設測試每張六回合裡有三回合
# 真的會問伺服器，同一句問題一次 `uv run pytest` 會問很多次；設了這一格之後同一句只真的問
# 一次，其餘重播。目錄由主控開在暫存區（不是版控樹裡），環境變數跟著傳給工人，所以整跑共用
# 同一份錄音。
# 只有這裡設它：CI 上跑檢查那幾步沒有這一格，雲端那一跑每一句都真的去問伺服器。
REPLAY_PREFIX = "aosr-gh-replay-"

# 這個程序是不是那個開錄音目錄的人（開的人才負責收尾刪掉）。
_OWNS_REPLAY_DIR = False

# 唯一准 spawn 版控工具、准對真樹寫檔的那支 fixture 的名字。這個字串同時寫在規矩卡
# governance/rules/tests-isolated-from-real-env.toml 的 [settings] sandbox_fixture，
# 靜態那半的判準讀的是卡上那一份，不是這裡。
SANDBOX_FIXTURE = "git_sandbox"
# 收尾守衛的名字。tests/test_repo_residue_guard.py 用 request.fixturenames 驗它真的 autouse。
GUARD_FIXTURE = "real_repo_left_untouched"

SANDBOX_IDENTITY = {
    "GIT_AUTHOR_NAME": "aosr sandbox",
    "GIT_AUTHOR_EMAIL": "sandbox@aosr.invalid",
    "GIT_COMMITTER_NAME": "aosr sandbox",
    "GIT_COMMITTER_EMAIL": "sandbox@aosr.invalid",
}

# 跑前那一份 git status --porcelain。session 起始記下來，收尾比對。
_BASELINE: str | None = None


def junit_paths() -> list[str]:
    """所有卡宣告的 junit 收據路徑（去重）。"""
    return sorted({card.junit_path for card in load_all_cards(REPO) if card.junit})


def _open_replay_dir() -> None:
    """開一個這一跑共用的 gh 錄音目錄。已經有人開過（工人）就沿用那一份。"""
    global _OWNS_REPLAY_DIR
    if os.environ.get(gh_replay.REPLAY_DIR_ENV):
        return
    os.environ[gh_replay.REPLAY_DIR_ENV] = tempfile.mkdtemp(prefix=REPLAY_PREFIX)
    _OWNS_REPLAY_DIR = True


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """收尾：把自己開的那個錄音目錄刪掉。錄音是這一跑的東西，不留給下一跑。"""
    if not _OWNS_REPLAY_DIR:
        return
    shutil.rmtree(os.environ[gh_replay.REPLAY_DIR_ENV], ignore_errors=True)


def pytest_sessionstart(session: pytest.Session) -> None:
    """開跑前：擋注入環境、記真 repo 狀態；主控另鏡收據並丟掉舊 junit。"""
    global _BASELINE
    injected = repo_residue.injected_env(os.environ)
    if injected:
        raise pytest.UsageError(
            f"環境裡已經被注入 {injected}——這幾個變數會把每一個 git 呼叫指到別棵樹去。"
            "v2 事故 worktree-hook-writes-into-real-repo 就是這個形狀：hook 底下跑測試，"
            "GIT_DIR 指回真 repo，暫存樹裡建的兩顆 fixture commit 全部落進真 repo。"
            "要在 hook 底下跑測試就先把它們 unset"
        )
    try:
        _BASELINE = repo_residue.porcelain(REPO)
    except repo_residue.ResidueError as exc:
        raise pytest.UsageError(
            f"量不到真 repo 跑前的狀態（{exc}）——量不到就不知道跑完有沒有留殘留，這一跑不算數"
        ) from exc

    _open_replay_dir()

    if os.environ.get(XDIST_WORKER_ENV):
        # 我是工人不是主控：跑前狀態已經記下（收尾守衛在每個工人身上都要能比對），
        # 但鏡收據與丟掉舊 junit 只由主控做一次。主控的 sessionstart 跑完才生工人，
        # 所以工人開始收集的時候鏡像已經在，舊 junit 也已經不在。
        return

    # 三張收據卡讀的是 status 分支上的機器收據；開跑前先鏡到被忽略的目錄（不上網，只讀本機的 ref）。
    # 鏡不成就停：那三張卡的第一回合會拿不到收據而回 2，第一回合要 0——與其在那裡紅，不如在這裡說清楚。
    try:
        mirror_receipts.mirror(REPO, mirror_receipts.DEFAULT_REF, REPO / mirror_receipts.DEFAULT_OUT, MIRROR_TIMEOUT)
    except ToolBroken as exc:
        raise pytest.UsageError(f"鏡不到 status 分支上的收據（{exc}）——先 git fetch origin status") from exc

    paths = junit_paths()
    if not paths:
        return
    _first, *extra = paths
    if extra:  # 不用 len 對數字：規矩卡 assertions-not-pinned-to-counts 連守門的 if 也咬，這裡要的是「不准第二條」不是「幾條」
        raise pytest.UsageError(
            f"有卡各自宣告了不同的 junit 收據路徑 {paths}——一跑 pytest 只產得出一份 junit。"
            "要嘛統一成同一個路徑，要嘛這裡改成一條路徑跑一次"
        )

    target = REPO / paths[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)  # 舊收據一律丟掉：不准拿上一跑的檔當這一跑的綠


@dataclass(frozen=True)
class GitSandbox:
    """暫存目錄裡的一棵樹，加一個只打在那棵樹上的版控指令。"""

    root: Path
    git: Callable[..., subprocess.CompletedProcess[str]]


@pytest.fixture
def git_sandbox(tmp_path: Path) -> Iterator[GitSandbox]:
    """唯一准 spawn 版控工具的 fixture。要碰版控就跟它要一棵暫存樹。

    照 v2 事故 worktree-hook-writes-into-real-repo 自己寫的對策做：三個 GIT_* 主動指到
    暫存路徑（不是只 unset）、全域與系統設定檔關掉、身分走環境變數而不是 config 寫檔、
    任何寫入前先做負控制、跑完把樹刪掉。
    """
    root = tmp_path / "sandbox"
    root.mkdir()
    home = tmp_path / "home"
    home.mkdir()

    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(SANDBOX_IDENTITY)
    env.update(
        {
            "HOME": str(home),
            "GIT_DIR": str(root / ".git"),
            "GIT_WORK_TREE": str(root),
            "GIT_INDEX_FILE": str(root / ".git" / "index"),
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_CONFIG_NOSYSTEM": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        """在暫存樹上跑一個版控指令。非零退出直接炸，不吞。"""
        proc = subprocess.run(["git", *args], cwd=root, env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise AssertionError(
                f"sandbox 裡 git {' '.join(args)} 回 {proc.returncode}：{proc.stderr.strip()[:300]}"
            )
        return proc

    git("init", "-q", "-b", "main")
    # 負控制：任何寫入之前先確認頂層就是這棵暫存樹。事故當下缺的就是這一步。
    top = Path(git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != root.resolve():
        raise AssertionError(f"sandbox 負控制不過：頂層是 {top}，不是 {root}——不准往下寫")
    if top.is_relative_to(REPO):
        raise AssertionError(f"sandbox 負控制不過：{top} 落在真 repo（{REPO}）裡面")

    yield GitSandbox(root=root, git=git)
    shutil.rmtree(root)


@pytest.fixture(scope="session", autouse=True)
def real_repo_left_untouched() -> Iterator[None]:
    """收尾守衛：全部跑完，真 repo 的 git status --porcelain 必須跟跑前一模一樣。

    寫成 session 級的 autouse fixture 而不是 sessionfinish 鉤子，是為了讓殘留在 junit
    收據上留下一筆 error——在鉤子裡改離開碼，收據上會看起來全綠，那就是這個 repo 最恨的
    「收據不誠實」。
    """
    yield
    if _BASELINE is None:
        raise AssertionError("跑前的狀態沒被記下來，這一跑的「沒有殘留」不算數")
    left = repo_residue.residue(_BASELINE, repo_residue.porcelain(REPO))
    if left:
        raise AssertionError(
            "測試跑完把真 repo 弄髒了：\n  "
            + "\n  ".join(left)
            + "\n測試不准在版控樹裡留東西。要寫檔就跟 git_sandbox 要暫存樹；真的非寫真樹不可，"
            "就在規矩卡 tests-isolated-from-real-env 的 [[settings.allow]] 開一條，"
            "而且那個路徑要被 .gitignore 蓋住"
        )
