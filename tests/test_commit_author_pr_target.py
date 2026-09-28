"""commit-author-allowlisted 的「合併目標必須是預設分支」那一條（#520 第四支）。

必紅樣本 case-pr-targets-other-branch 守樣本那條路；這裡守真的工作樹那條路：環境變數怎麼讀、
缺了回 2、不是合併請求就不判。暫存 git 樹跟 ``git_sandbox`` 要，不碰真的 repo。
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


def test_pull_request_target_is_read_from_env_and_other_events_are_not_judged(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = git_sandbox.root.resolve()
    monkeypatch.setenv(gate.EVENT_ENV, "pull_request")
    monkeypatch.setenv(gate.BASE_REF_ENV, "feature-x")
    monkeypatch.setenv(gate.DEFAULT_BRANCH_ENV, "main")
    assert gate._resolve_target(root) == ("feature-x", "main")
    monkeypatch.setenv(gate.EVENT_ENV, "push")
    assert gate._resolve_target(root) is None
