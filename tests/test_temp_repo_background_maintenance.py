"""檢查自己開的暫存 git repo，每一次 commit 的那一刻背景自動維護都要已經關掉（#810）。

雲端的 git 在 commit 之後會在背景自動維護（重新打包、寫 maintenance.lock），跟檢查收尾刪暫存目錄搶同一個
目錄，偶發「清不掉自己開的暫存目錄」把主線打紅（#778 是考卷沙箱、#810 是檢查建樣本歷史的暫存 repo）。
本機的 git 不一定會在背景維護，等它真的搶起來才紅就太晚了；所以用一支假 git 包住真 git：每一次 commit
先用同一組全域選項問真 git 那兩格此刻的生效值、記下來，再原樣交給真 git 做。
每張卡的控制樣本都跑一次，不挑卡：以後新的檢查也開暫存 repo commit、忘了關，一樣紅。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from governance.checks.rule_card_required_fields import BITE_TIMEOUT, DEPTH_ENV
from governance.exit_codes import VIOLATION
from governance.loader import Card, load_all_cards
from tests.conftest import GitSandbox

REPO = Path(__file__).resolve().parents[1]
CARDS = load_all_cards(REPO)

# git 自己的兩個開關：maintenance.auto=false 是 commit 後不叫背景維護、gc.auto=0 是不自動打包。
# 刻意不從 governance.exit_codes 拿：那一份被清空時，寫的那端跟量的那端會一起變空，這一題照綠。
QUIET = {"maintenance.auto": "false", "gc.auto": "0"}

# #810 在雲端紅的就是這一張：它的控制樣本一筆 commit 都沒記到，就是假 git 沒接上，其他卡的綠不算數。
KNOWN_COMMITTER = "tests-land-with-code"

SHIM = '''#!{python}
"""假 git：commit 之前先記下那顆 repo 此刻兩格的生效值，再原樣交給真 git。"""
import os
import subprocess
import sys

REAL = {real!r}
LOG = {log!r}
KEYS = {keys!r}
args = sys.argv[1:]
lead = []
i = 0
# 子指令前面的全域選項（-C 目錄、-c 設定…）照樣帶去問：commit 那一刻真 git 看到的就是這一組。
while i < len(args) and args[i].startswith("-"):
    step = 2 if args[i] in ("-C", "-c") else 1
    lead += args[i:i + step]
    i += step
if i < len(args) and args[i] == "commit":
    values = [subprocess.run([REAL, *lead, "config", "--get", key], capture_output=True, text=True).stdout.strip()
              for key in KEYS]
    with open(LOG, "a", encoding="utf-8") as handle:
        handle.write("\\t".join([os.getcwd(), *values]) + "\\n")
os.execv(REAL, [REAL, *args])
'''


def _fake_git(git_sandbox: GitSandbox, tmp_path: Path) -> tuple[dict[str, str], Path]:
    """在 tmp_path 放一支假 git，回傳「PATH 先找到它」的環境與它的紀錄檔。

    真 git 在哪問沙箱裡那支自己（``--exec-path`` 底下就是同一支執行檔）：這幾題經過假 git 真的會跑 git，
    照規矩走唯一那支沙箱 fixture。
    """
    real = Path(git_sandbox.git("--exec-path").stdout.strip()) / "git"
    assert real.is_file(), f"沙箱的 git 說它住 {real.parent}，那裡卻沒有 git 執行檔：包不住真 git，這一題量不到"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commits.tsv"
    shim = bin_dir / "git"
    shim.write_text(
        SHIM.format(python=sys.executable, real=str(real), log=str(log), keys=tuple(QUIET)), encoding="utf-8"
    )
    shim.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(bin_dir), env.get("PATH", "")])
    # 跟後設測試餵樣本同一組：不寫 .pyc 進樣本樹、守卡者那張不往下再咬一層。
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env[DEPTH_ENV] = "0"
    return env, log


def _commits(log: Path) -> list[tuple[str, dict[str, str]]]:
    """假 git 記下的每一次 commit：在哪一顆 repo、那兩格當下是什麼。"""
    if not log.exists():
        return []
    rows: list[tuple[str, dict[str, str]]] = []
    for line in log.read_text(encoding="utf-8").splitlines():
        where, *values = line.split("\t")
        rows.append((where, dict(zip(QUIET, values, strict=True))))
    return rows


def _noisy(rows: list[tuple[str, dict[str, str]]]) -> list[str]:
    """commit 時背景維護還開著（或沒設、照 git 預設開著）的那幾筆。"""
    return [f"{where}：{values}" for where, values in rows if values != QUIET]


def _run_control(card: Card, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """拿假 git 跑這張卡的控制樣本。"""
    return subprocess.run(
        [sys.executable, "-m", card.check_module, "--scan-root", str(card.control_path(REPO))],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=BITE_TIMEOUT,
    )


@pytest.mark.parametrize("card", CARDS, ids=[card.id for card in CARDS])
def test_every_commit_in_a_checks_temp_repo_has_background_maintenance_off(
    card: Card, git_sandbox: GitSandbox, tmp_path: Path
) -> None:
    """跑控制樣本時檢查在哪顆 repo commit，那一刻兩格都要已經關掉；假 git 也不准改變結論。"""
    env, log = _fake_git(git_sandbox, tmp_path)
    proc = _run_control(card, env)
    assert proc.returncode == VIOLATION, (
        f"{card.id} 在假 git 底下跑控制樣本回 {proc.returncode}，不是 1：假 git 改變了結論，"
        f"這一題量到的不是它平常的路：{(proc.stdout + proc.stderr)[-600:]}"
    )
    noisy = _noisy(_commits(log))
    assert not noisy, f"{card.id} 在背景維護沒關的暫存 repo 裡 commit（git init 之後先寫兩格）：{noisy}"


def test_the_fake_git_sees_the_commits_of_a_known_committer(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """非空保證：已知會建樣本歷史的那張卡，假 git 一定要記到它的 commit，不然上一題的綠是沒量到。"""
    card = next(card for card in CARDS if card.id == KNOWN_COMMITTER)
    env, log = _fake_git(git_sandbox, tmp_path)
    proc = _run_control(card, env)
    assert proc.returncode == VIOLATION, (proc.stdout + proc.stderr)[-600:]
    assert _commits(log), f"{card.id} 的控制樣本跑完，假 git 一筆 commit 都沒記到：PATH 上的假 git 沒接上"


def test_a_commit_with_maintenance_on_is_reported(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """負控制：背景維護開著的那一次 commit，假 git 要記成開著、而且被判出來——不然第一題永遠綠。

    在沙箱裡、而且暫存區是空的：真 git 會回「沒東西可提交」不留提交，也就不會真的叫背景維護。
    """
    env, log = _fake_git(git_sandbox, tmp_path)
    env = {key: value for key, value in env.items() if not key.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    for lead in ([], ["-c", "maintenance.auto=true"]):
        subprocess.run(
            ["git", *lead, "commit", "--quiet", "--no-verify", "-m", "負控制"],
            cwd=git_sandbox.root,
            env=env,
            capture_output=True,
            text=True,
        )
    rows = _commits(log)
    assert [values for _where, values in rows] == [QUIET, QUIET | {"maintenance.auto": "true"}], rows
    assert _noisy(rows) == [f"{rows[-1][0]}：{rows[-1][1]}"]
    assert git_sandbox.git("rev-list", "--all").stdout.strip() == "", "負控制不准真的在沙箱留提交"
