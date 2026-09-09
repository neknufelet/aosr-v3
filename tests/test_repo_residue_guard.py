"""動態那半的活體證明：殘留量得出來、守衛真的在崗、沙盒真的不是真樹。

規矩卡 ``tests-isolated-from-real-env`` 的靜態那半由必紅樣本咬（後設測試六回合）；
動態那半（跑完真 repo 的 ``git status --porcelain`` 必須跟跑前一樣）沒有樣本可餵，
所以它的迴歸在這裡：

* ``governance/repo_residue.py`` 的差集算得對——多一個未追蹤檔就看得見。
* 那三個會把 git 指到別棵樹的環境變數認得出來。
* 收尾守衛真的是 autouse（用 ``request.fixturenames`` 驗，不是去 grep 原始碼）。
* ``git_sandbox`` 給的樹真的在真 repo 外面，而且它自己的負控制成立。

這一支自己也是被咬的對象：它要 spawn 版控工具，所以每一個這麼做的測試都跟
``git_sandbox`` 要樹——靜態那半的判準就是這個。
"""
from __future__ import annotations

from pathlib import Path

from governance import repo_residue
from tests.conftest import GUARD_FIXTURE, SANDBOX_FIXTURE

REPO = Path(__file__).resolve().parents[1]


def test_residue_spots_a_stray_file(git_sandbox) -> None:
    """多一個未追蹤檔，差集就看得見；沒動過就是空的。"""
    before = repo_residue.porcelain(git_sandbox.root)
    assert repo_residue.residue(before, before) == []
    (git_sandbox.root / "leftover.txt").write_text("殘留", encoding="utf-8")
    after = repo_residue.porcelain(git_sandbox.root)
    assert repo_residue.residue(before, after) == [f"{repo_residue.ADDED}：?? leftover.txt"]


def test_residue_spots_a_vanished_line(git_sandbox) -> None:
    """少一行也算殘留——「跑完跟跑前一樣」是雙向的。"""
    (git_sandbox.root / "leftover.txt").write_text("殘留", encoding="utf-8")
    before = repo_residue.porcelain(git_sandbox.root)
    (git_sandbox.root / "leftover.txt").unlink()
    after = repo_residue.porcelain(git_sandbox.root)
    assert repo_residue.residue(before, after) == [f"{repo_residue.GONE}：?? leftover.txt"]


def test_injected_git_env_is_spotted() -> None:
    """那三個會把 git 指到別棵樹的變數，認得出來；別的變數不誤咬。"""
    assert repo_residue.injected_env({"GIT_DIR": "/somewhere/else/.git"}) == ["GIT_DIR"]
    assert repo_residue.injected_env({"GIT_WORK_TREE": "/x", "GIT_INDEX_FILE": "/y"}) == [
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
    ]
    assert repo_residue.injected_env({"PATH": "/usr/bin", "GIT_DIR": ""}) == []


def test_the_session_guard_is_wired_and_autouse(request) -> None:
    """收尾守衛真的在這一跑的 fixture 清單裡（autouse 沒被拆掉）。"""
    assert GUARD_FIXTURE in request.fixturenames


def test_sandbox_is_outside_the_real_repo(git_sandbox) -> None:
    """負控制：沙盒的頂層就是沙盒自己，而且落在真 repo 外面。"""
    top = Path(git_sandbox.git("rev-parse", "--show-toplevel").stdout.strip()).resolve()
    assert top == git_sandbox.root.resolve()
    assert not top.is_relative_to(REPO)


def test_sandbox_identity_does_not_write_config(git_sandbox) -> None:
    """身分走環境變數，不是 git config 寫檔——事故當下就是 user.name=test 被寫進真 repo 的 config。"""
    git_sandbox.git("commit", "--allow-empty", "-m", "sandbox")
    who = git_sandbox.git("log", "-1", "--format=%an <%ae>").stdout.strip()
    assert who == "aosr sandbox <sandbox@aosr.invalid>"
    config = (git_sandbox.root / ".git" / "config").read_text(encoding="utf-8")
    assert "[user]" not in config


def test_the_sandbox_fixture_name_matches_the_card() -> None:
    """conftest 用的名字跟卡上宣告的那個是同一個（靜態那半讀的是卡）。"""
    card = (REPO / "governance" / "rules" / "tests-isolated-from-real-env.toml").read_text(
        encoding="utf-8"
    )
    assert f'sandbox_fixture = "{SANDBOX_FIXTURE}"' in card
