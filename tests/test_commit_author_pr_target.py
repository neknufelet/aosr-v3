"""commit-author-allowlisted 的「合併目標必須是預設分支」那一條（#520 第四支）。

必紅樣本 case-pr-targets-other-branch 守樣本那條路；這裡守真的工作樹那條路：環境變數怎麼讀、
缺了回 2、推送看推到的分支、其他事件一律紅、沒有事件（本機）不判。暫存 git 樹跟 ``git_sandbox`` 要，不碰真的 repo。
"""
from __future__ import annotations

import pytest

from governance.checks import commit_author_allowlisted as gate
from governance.exit_codes import ToolBroken
from tests.conftest import GitSandbox


def test_target_equal_to_default_branch_passes_and_other_branch_is_red() -> None:
    assert gate._target_problems(None) == []
    assert gate._target_problems(("main", "main")) == []
    problems = gate._target_problems(("feature-x", "main"))
    assert problems
    assert "feature-x" in problems[0]
    assert "main" in problems[0]


def test_pull_request_without_target_env_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = git_sandbox.root.resolve()
    monkeypatch.setenv(gate.EVENT_ENV, "pull_request")
    monkeypatch.setenv(gate.DEFAULT_BRANCH_ENV, "main")
    monkeypatch.delenv(gate.BASE_REF_ENV, raising=False)
    with pytest.raises(ToolBroken):
        gate._resolve_target(root)


def test_pull_request_target_is_read_from_env_and_no_event_is_not_judged(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = git_sandbox.root.resolve()
    monkeypatch.setenv(gate.EVENT_ENV, "pull_request")
    monkeypatch.setenv(gate.BASE_REF_ENV, "feature-x")
    monkeypatch.setenv(gate.DEFAULT_BRANCH_ENV, "main")
    assert gate._resolve_target(root) == ("feature-x", "main")
    monkeypatch.delenv(gate.EVENT_ENV)
    assert gate._resolve_target(root) is None


def test_push_is_judged_by_the_pushed_branch(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = git_sandbox.root.resolve()
    monkeypatch.setenv(gate.EVENT_ENV, "push")
    monkeypatch.setenv(gate.DEFAULT_BRANCH_ENV, "main")
    monkeypatch.setenv(gate.REF_ENV, "refs/heads/main")
    assert gate._resolve_target(root) == ("main", "main")
    monkeypatch.setenv(gate.REF_ENV, "refs/heads/feature-x")
    assert gate._resolve_target(root) == ("feature-x", "main")
    monkeypatch.delenv(gate.REF_ENV)
    with pytest.raises(ToolBroken):
        gate._resolve_target(root)


def test_events_other_than_pull_request_and_push_are_red(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = git_sandbox.root.resolve()
    monkeypatch.setenv(gate.EVENT_ENV, "workflow_dispatch")
    assert gate._resolve_target(root) is None
    problems = gate._event_problems(root)
    assert problems
    assert "workflow_dispatch" in problems[0]
    for judged in gate.JUDGED_EVENTS:
        monkeypatch.setenv(gate.EVENT_ENV, judged)
        assert gate._event_problems(root) == []
    monkeypatch.delenv(gate.EVENT_ENV)
    assert gate._event_problems(root) == []


def test_check_reds_a_run_triggered_by_another_event(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """從 ``check`` 打到底：其他事件那一條要真的接在主流程上，不只 ``_event_problems`` 自己對。"""
    root = git_sandbox.root.resolve()
    authors = root / gate.AUTHORS_FILE
    authors.parent.mkdir(parents=True)
    authors.write_text("sandbox@aosr.invalid\n", encoding="utf-8")
    git_sandbox.git("add", gate.AUTHORS_FILE)
    git_sandbox.git("commit", "-q", "-m", "名單")
    for name in (gate.BASE_ENV, gate.HEAD_ENV, gate.BASE_REF_ENV, gate.REF_ENV, gate.EVENT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(gate.DEFAULT_BRANCH_ENV, "main")
    assert gate.check(root, [authors]) == []
    monkeypatch.setenv(gate.EVENT_ENV, "workflow_dispatch")
    assert any("workflow_dispatch" in problem for problem in gate.check(root, [authors]))
