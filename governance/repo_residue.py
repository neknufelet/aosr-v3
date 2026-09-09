"""真 repo 有沒有被弄髒——量一次、跑完再量一次、比。

規矩卡 ``tests-isolated-from-real-env`` 的動態那半。靜態那半（``governance/checks/
tests_isolated_from_real_env.py``）只咬得到「直接 spawn」與「直接寫」；子程序間接寫進真樹、
或經過中間函式洗一手的寫入，它看不到。那些由這裡接住：**整套測試跑完，真 repo 的
``git status --porcelain`` 必須跟跑前一模一樣**——多一個未追蹤檔、少一個、或有檔被改，
都算殘留。

**為什麼放在 ``tests/`` 外面。** 這是本檔唯一要交代的事：靜態那半的判準是「``tests/`` 底下
會 spawn 版控工具的函式必須經過唯一那支 sandbox fixture」，而量真 repo 的狀態恰好是那條
規矩唯一的正當例外——它必須在真樹上跑，而且只讀。與其在卡上再開一條放行，這裡選擇把那個
唯讀呼叫收在 ``tests/`` 外面的這支模組：一個看得見、被 ``tests/test_repo_residue_guard.py``
咬著接線的地方。代價寫在卡面上：同一條路（把 git 呼叫搬出 ``tests/``）別人也走得過去，
靜態那半咬不到，只能靠 PR 審。

沒有預設值、不吞錯：量不到就 raise :class:`ResidueError`。「量不到就當沒殘留」是 v2 假綠的
病根（``worktree-hook-writes-into-real-repo``、``test-answer-depends-on-ambient-state``）。
"""
from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

# 量狀態的參數。``-c core.quotePath=false``：git 預設把非 ASCII 檔名印成八進位跳脫碼，
# 那串東西比較起來一樣，但人看不懂殘留是哪個檔。出處同 governance/exit_codes.py。
PORCELAIN_ARGV = ("git", "-c", "core.quotePath=false", "status", "--porcelain")
TOPLEVEL_ARGV = ("git", "rev-parse", "--show-toplevel")

# 會把 git 指到別棵樹的環境變數。v2 事故 worktree-hook-writes-into-real-repo 就是
# hook 底下 ``GIT_DIR`` 指回真 repo，測試在暫存樹建的 fixture commit 全部落進真 repo。
AMBIENT_GIT_ENV = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE")

ADDED = "多了"
GONE = "不見了"


class ResidueError(Exception):
    """量不到真 repo 的狀態。這一跑的「沒有殘留」不算數。"""


def injected_env(env: Mapping[str, str]) -> list[str]:
    """環境裡有沒有被注入那幾個會改寫 git 目標的變數。回被注入的名字。"""
    return [name for name in AMBIENT_GIT_ENV if env.get(name)]


def _run(argv: tuple[str, ...], root: Path) -> str:
    """在 ``root`` 底下跑一個唯讀的 git 指令，回它的 stdout。"""
    try:
        proc = subprocess.run(list(argv), cwd=root, capture_output=True)
    except (FileNotFoundError, NotADirectoryError) as exc:
        raise ResidueError(f"跑不起來 {' '.join(argv)}（{exc}）") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[:200]
        raise ResidueError(f"{' '.join(argv)} 非零退出（{proc.returncode}）：{detail}")
    try:
        return proc.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ResidueError(f"{' '.join(argv)} 的輸出不是合法 UTF-8（{exc}），我沒看懂就不出結論") from exc


def toplevel(root: Path) -> Path:
    """``root`` 那棵樹的頂層。負控制用：不等於 ``root`` 就代表量到別棵樹去了。"""
    return Path(_run(TOPLEVEL_ARGV, root).strip()).resolve()


def porcelain(root: Path) -> str:
    """``root`` 那棵樹現在的 ``git status --porcelain``。

    先做負控制：頂層必須就是 ``root``。不做這一步的話，環境被注入的時候量到的是別棵樹，
    而「沒有殘留」照樣印得出來——那正是事故當下的形狀。
    """
    top = toplevel(root)
    if top != root.resolve():
        raise ResidueError(
            f"負控制不過：在 {root} 底下問出來的頂層是 {top}——量到的不是這棵樹，這一跑不算數"
        )
    return _run(PORCELAIN_ARGV, root)


def residue(before: str, after: str) -> list[str]:
    """跑前跑後的差。空 list 就是「跟跑前一模一樣」。"""
    old = [ln for ln in before.splitlines() if ln.strip()]
    new = [ln for ln in after.splitlines() if ln.strip()]
    out = [f"{ADDED}：{ln}" for ln in sorted(set(new) - set(old))]
    out += [f"{GONE}：{ln}" for ln in sorted(set(old) - set(new))]
    return out
