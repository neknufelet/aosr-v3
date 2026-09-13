"""主線卡名只從指定的 base 讀；工作樹的新卡因此能被辨認出來。"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tests.conftest import GitSandbox


def _write_card(root: Path, card_id: str) -> None:
    path = root / "governance" / "rules" / f"{card_id}.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'id = "{card_id}"\n', encoding="utf-8")


def test_worktree_card_not_in_base_is_the_only_candidate(git_sandbox: GitSandbox) -> None:
    """若誤讀工作樹而非 base，第三張卡會混進主線名單，候選就消失。"""
    from governance.mainline_cards import mainline_card_ids

    for card_id in ("base-alpha", "base-beta"):
        _write_card(git_sandbox.root, card_id)
    git_sandbox.git("add", "governance/rules")
    git_sandbox.git("commit", "-q", "-m", "base has two cards")
    base = git_sandbox.git("rev-parse", "HEAD").stdout.strip()

    _write_card(git_sandbox.root, "candidate-gamma")
    worktree_ids = frozenset(
        path.stem for path in (git_sandbox.root / "governance" / "rules").glob("*.toml")
    )

    result = mainline_card_ids(git_sandbox.root, base)

    assert result.ids is not None
    assert worktree_ids - result.ids == frozenset({"candidate-gamma"})
    assert result.base == base


def test_missing_base_has_no_candidate_list(git_sandbox: GitSandbox) -> None:
    """base 讀不到時不能用空集合冒充「真的沒有候選」。"""
    from governance.mainline_cards import mainline_card_ids

    result = mainline_card_ids(git_sandbox.root, "refs/heads/not-there")

    assert result.ids is None
    assert "讀不到 base" in result.reason


def test_missing_git_has_no_candidate_list(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """找不到 git 時要明說工具缺席，不能用空集合冒充沒有候選。"""
    from governance import mainline_cards

    monkeypatch.setattr(shutil, "which", lambda _tool: None)

    result = mainline_cards.mainline_card_ids(git_sandbox.root, "main")

    assert result.ids is None
    assert "找不到 git" in result.reason
