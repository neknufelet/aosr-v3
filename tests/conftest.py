"""開跑前先產一份真的 junit 收據，並且不寫 .pyc。

**為什麼要有這一段。** `green-must-be-real-green` 那張卡判的是「pytest 產的 junit 收據」。
後設測試的第一回合要求每張卡在乾淨樹回 0，可是收據不進版控（見 `.gitignore`），
剛 clone 的樹裡它根本不存在。三條路：

1. 檢查程式「檔不存在就回 0」——這正是 v2 假綠的病根，不准。
2. 檢查程式缺檔就自己補一份——尺自己造證據，更不准。
3. 後設測試在第一回合之前，先跑一次**真的全套**，把收據產出來。

選 3。收據是那一跑的真實結果，不是這裡編的。產收據那一跑帶環境變數
`AOSR_GREEN_RECEIPT_SEED=1`，它自己不會再往下套一層；而且在那一跑裡，宣告了 `[junit]`
的卡的第一回合斷言的是**「收據不在的時候檢查必須回 1」**——比平常那一回合更凶，不是放水。
順序因此是：丟掉舊收據 → 產收據那一跑（此時收據不在，檢查必須紅）→ 外層這一跑
（收據在了，檢查必須綠）。舊收據一律丟掉，不准沿用上一跑的檔。

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
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

# 不寫 .pyc：上一張卡撞過「改了程式，子程序卻拿 __pycache__ 裡的舊位元碼」的坑。
sys.dont_write_bytecode = True

import pytest  # noqa: E402  # expires=2026-12-08 reason=這幾個 import 必須排在 sys.dont_write_bytecode 與 REPO 那兩行之後，不是可以往上搬的；到期時重審

REPO = Path(__file__).resolve().parents[1]

from governance import repo_residue  # noqa: E402  # expires=2026-12-08 reason=這幾個 import 必須排在 sys.dont_write_bytecode 與 REPO 那兩行之後，不是可以往上搬的；到期時重審
from governance.loader import load_all_cards  # noqa: E402  # expires=2026-12-08 reason=這幾個 import 必須排在 sys.dont_write_bytecode 與 REPO 那兩行之後，不是可以往上搬的；到期時重審

SEED_ENV = "AOSR_GREEN_RECEIPT_SEED"
SEED_TIMEOUT = 900

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


def _tail(proc: subprocess.CompletedProcess[str], rows: int = 15) -> str:
    return "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-rows:])


def pytest_sessionstart(session: pytest.Session) -> None:
    """開跑前：先擋掉被注入的 git 環境、記下真 repo 的狀態，再去產收據。"""
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

    if os.environ.get(SEED_ENV):
        return  # 我就是產收據那一跑，不再往下套一層

    paths = junit_paths()
    if not paths:
        return
    if len(paths) > 1:
        raise pytest.UsageError(
            f"有卡各自宣告了不同的 junit 收據路徑 {paths}——一跑 pytest 只產得出一份 junit。"
            "要嘛統一成同一個路徑，要嘛這裡改成一條路徑跑一次"
        )

    target = REPO / paths[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)  # 舊收據一律丟掉：不准拿上一跑的檔當這一跑的綠

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env[SEED_ENV] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", f"--junitxml={target}"],
            cwd=REPO,
            env=env,
            capture_output=True,
            text=True,
            timeout=SEED_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise pytest.UsageError(f"產收據那一跑超過 {SEED_TIMEOUT} 秒沒跑完") from exc
    if proc.returncode != 0 or not target.is_file():
        raise pytest.UsageError(
            f"產收據那一跑自己紅了（離開碼 {proc.returncode}），先修它——"
            f"收據是那一跑的真實結果，這裡不會替它補一份。最後幾行：\n{_tail(proc)}"
        )


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
