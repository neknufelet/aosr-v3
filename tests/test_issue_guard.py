"""票只准由 PR 關那一支的測試：全部餵假外殼，不上網、不真的重開任何一張票。

為什麼可以這樣測：`governance/status/issue_guard.py` 判斷的那一半（:func:`judge`）只吃
一個已經收窄過的 `ClosingPulls`（GitHub 記的關票來源），不碰網路也不碰子程序；上網的那一半
走的是 `collect.Shell`，測試打 `subprocess` 那一格把它換成假的，所以從頭到尾沒有真的叫起
任何一支外部程式，也沒有對 GitHub 寫過一個字。

假外殼（`Replies`）與那幾個假回應的組裝共用 `tests/test_build_status.py` 那一份，
不抄第二份——兩份假資料會各自漂。

斷言刻意不鎖死數量（規矩卡 assertions-not-pinned-to-counts）：比的是「有沒有下這一個
指令」、「重開了還是沒重開」，不是「必須剛好問了幾次」。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken
from governance.status import collect, issue_guard
from tests.test_build_status import FAKE_LATE_PULL, Replies, closed_by, fake_hub, pull_node

# 假的環境：`GITHUB_REPOSITORY` 在的時候這一層不必再問一次 repo 全名（雲端那一跑就是這樣）。
FAKE_ENV = {"GITHUB_REPOSITORY": "fake/fake"}
FAKE_ISSUE = 401


def test_a_ticket_closed_by_a_landed_pull_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """有一個合進主線的 PR 掛著關掉它：什麼都不做，判詞寫得出是哪一個 PR。"""
    hub, _ = fake_hub(monkeypatch, tmp_path, [closed_by([pull_node(FAKE_LATE_PULL)])], 1)
    verdict = issue_guard.verdict_for(hub, FAKE_ISSUE)
    assert not verdict.reopen
    assert str(FAKE_LATE_PULL.number) in verdict.why


def test_a_ticket_closed_by_hand_is_reopened(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """人手關的（GitHub 上一個掛著關它的 PR 都沒有）：重開。"""
    hub, _ = fake_hub(monkeypatch, tmp_path, [closed_by([])], 1)
    assert issue_guard.verdict_for(hub, FAKE_ISSUE).reopen


def test_a_closing_pull_that_never_landed_still_reopens() -> None:
    """掛著關它、但沒合進主線的 PR 也要重開——而且判詞跟「根本沒有 PR」不一樣。

    兩種都重開，可是原因不同：一種是有人開了 PR 還沒合，一種是根本沒人做。
    判詞混成同一句，看紀錄的人就分不出下一步該催誰。
    """
    hanging = issue_guard.judge(
        FAKE_ISSUE, collect.ClosingPulls(landed=None, referenced=len([FAKE_LATE_PULL.number]))
    )
    bare = issue_guard.judge(FAKE_ISSUE, collect.ClosingPulls(landed=None, referenced=0))
    assert hanging.reopen and bare.reopen
    assert hanging.why != bare.why


def test_reopening_leaves_the_human_sentence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """真的下了重開那一個指令，而且帶著那一句人話（`gh issue reopen --comment`）。"""
    replies = Replies([closed_by([]), ""])
    monkeypatch.setattr(subprocess, "run", replies)
    verdict = issue_guard.guard(
        tmp_path, FAKE_ENV, FAKE_ISSUE, 1, 1, issue_guard.REOPEN_COMMENT
    )
    assert verdict.reopen
    reopened = [call for call in replies.calls if "reopen" in call]
    assert reopened, f"沒有下重開那一個指令：{replies.calls}"
    assert str(FAKE_ISSUE) in reopened[-1]
    assert issue_guard.REOPEN_COMMENT in reopened[-1]
    # 那一句人話要指出正路（PR 內文寫 Closes #n），不是只說「不准」。
    assert "Closes" in issue_guard.REOPEN_COMMENT


def test_a_landed_pull_means_no_write_back_to_github(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """判成「不動它」的那一條路上，一個會改到 GitHub 的指令都不准下。

    假外殼備好的答案只有一份（問關票來源那一次），多問一次它就當場炸——
    「有沒有多下一個指令」因此斷言得到，不是靠它安靜地回一份空的。
    """
    replies = Replies([closed_by([pull_node(FAKE_LATE_PULL)])])
    monkeypatch.setattr(subprocess, "run", replies)
    verdict = issue_guard.guard(
        tmp_path, FAKE_ENV, FAKE_ISSUE, 1, 1, issue_guard.REOPEN_COMMENT
    )
    assert not verdict.reopen
    assert not [call for call in replies.calls if "reopen" in call]


def test_a_shape_we_do_not_understand_reopens_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """形狀看不懂就 ToolBroken：不重開也不留言（分不清就不動手）。"""
    replies = Replies(['{"data": {"repository": {"issue": null}}}'])
    monkeypatch.setattr(subprocess, "run", replies)
    with pytest.raises(ToolBroken):
        issue_guard.guard(tmp_path, FAKE_ENV, FAKE_ISSUE, 1, 1, issue_guard.REOPEN_COMMENT)
    assert not [call for call in replies.calls if "reopen" in call]


def test_entry_point_maps_tool_broken_to_exit_code_two(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """入口把「問不出來」翻成離開碼 2，不是 0 也不是 1（這一支不下判決）。"""

    def boom(*args: object, **kwargs: object) -> issue_guard.Verdict:
        raise ToolBroken("假的：gh 不在")

    monkeypatch.setattr(issue_guard, "guard", boom)
    assert issue_guard.main(["--issue", str(FAKE_ISSUE)]) == TOOL_BROKEN


def test_entry_point_returns_clean_when_it_could_tell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """問得出答案的兩條路（不動它／重開）都回 0：這一支沒有「抓到違規」這個結局。"""
    seen: list[int] = []

    def fine(*args: object, **kwargs: object) -> issue_guard.Verdict:
        seen.append(len(seen))
        return issue_guard.Verdict(reopen=True, why="假的：重開了")

    monkeypatch.setattr(issue_guard, "guard", fine)
    assert issue_guard.main(["--issue", str(FAKE_ISSUE)]) == CLEAN
    assert seen
