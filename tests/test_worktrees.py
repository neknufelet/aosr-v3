"""worktree（第二個工作目錄）的家與名字只有一種；拆樹只拆「PR 已合、頭對得上、樹乾淨」的。"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance import worktrees
from governance.exit_codes import CLEAN, VIOLATION
from governance.worktrees import Linked, PrLookup, PullRequest, WorktreeError
from tests.conftest import GitSandbox


def _seed(sandbox: GitSandbox) -> None:
    (sandbox.root / "a.txt").write_text("a\n", encoding="utf-8")
    sandbox.git("add", "a.txt")
    sandbox.git("commit", "-q", "-m", "one")


def _answers(state: str | None, head: str = "") -> PrLookup:
    """假的 GitHub：每條分支都回同一個答案。``state`` 是 None 代表沒有 PR。"""

    def lookup(_branch: str) -> PullRequest | None:
        return None if state is None else PullRequest(number=7, state=state, head_oid=head)

    return lookup


def _open_tree(sandbox: GitSandbox, root: Path) -> tuple[Path, str]:
    _seed(sandbox)
    name = worktrees.tree_name("feat", 364, "worktree-home")
    path = worktrees.new(name, sandbox.git, root, base="main")
    return path, sandbox.git("rev-parse", "refs/heads/feat/364-worktree-home").stdout.strip()


def _branches(sandbox: GitSandbox) -> list[str]:
    return sandbox.git("branch", "--format=%(refname:short)").stdout.split()


def test_new_tree_lives_in_the_work_folder_under_one_name(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """位置或分支名要是各走各的，這裡的路徑與「沒有毛病」兩句就對不上。"""
    root = tmp_path / "work"

    path, _head = _open_tree(git_sandbox, root)

    assert path == root / "364-worktree-home" / "tree"
    assert (path / "a.txt").is_file()
    (item,) = worktrees.linked(git_sandbox.git)
    assert item == Linked(path=path, branch="feat/364-worktree-home")
    assert worktrees.problems(item, root) == []


def test_new_refuses_a_path_that_already_exists(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    root = tmp_path / "work"
    _open_tree(git_sandbox, root)

    with pytest.raises(WorktreeError, match="已經在了"):
        worktrees.new(worktrees.tree_name("fix", 364, "worktree-home"), git_sandbox.git, root, base="main")


def test_tree_outside_the_work_folder_is_flagged(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """2026-09-19 量到的現況：樹丟在別的地方、名字跟分支對不起來。"""
    _seed(git_sandbox)
    stray = tmp_path / "elsewhere" / "aosr-v3-345-timbre"
    git_sandbox.git("worktree", "add", "-q", "-b", "feat/345-timbre-evaluator", str(stray), "main")

    (item,) = worktrees.linked(git_sandbox.git)

    assert any("放錯地方" in line for line in worktrees.problems(item, tmp_path / "work"))


def test_folder_and_branch_names_must_match(tmp_path: Path) -> None:
    root = tmp_path / "work"
    item = Linked(path=root / "345-timbre" / "tree", branch="feat/345-timbre-evaluator")

    assert any("名字對不上" in line for line in worktrees.problems(item, root))


@pytest.mark.parametrize(
    ("kind", "issue", "slug"),
    [("wip", 1, "ok"), ("feat", 0, "ok"), ("feat", 1, "Has_Upper"), ("feat", 1, "-lead"), ("feat", 1, "")],
)
def test_bad_names_are_refused(kind: str, issue: int, slug: str) -> None:
    with pytest.raises(WorktreeError):
        worktrees.tree_name(kind, issue, slug)


@pytest.mark.parametrize(
    ("state", "head", "why"),
    [(None, "", "沒有 PR"), ("OPEN", "", "還沒合"), ("MERGED", "0" * 40, "沒送出去的提交")],
)
def test_remove_refuses_unless_merged_at_the_local_head(
    git_sandbox: GitSandbox, tmp_path: Path, state: str | None, head: str, why: str
) -> None:
    """拆錯一次就是丟工作：沒 PR、沒合、合了之後本機又多提交，樹與分支都要原封不動。"""
    path, _head = _open_tree(git_sandbox, tmp_path / "work")

    with pytest.raises(WorktreeError, match=why):
        worktrees.remove("364-worktree-home", git_sandbox.git, _answers(state, head))

    assert path.is_dir()
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def test_remove_leaves_a_dirty_tree_alone(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """髒樹由 git 自己擋（這支工具不加 --force）；sandbox 的 git 非零退出就炸成 AssertionError。"""
    path, head = _open_tree(git_sandbox, tmp_path / "work")
    (path / "unsaved.txt").write_text("還沒提交的東西\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="worktree remove"):
        worktrees.remove("364-worktree-home", git_sandbox.git, _answers("MERGED", head))

    assert (path / "unsaved.txt").is_file()
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def test_remove_takes_the_tree_and_branch_but_keeps_the_work_folder(
    git_sandbox: GitSandbox, tmp_path: Path
) -> None:
    path, head = _open_tree(git_sandbox, tmp_path / "work")
    brief = path.parent / "brief.md"
    brief.write_text("工單\n", encoding="utf-8")

    gone = worktrees.remove("feat/364-worktree-home", git_sandbox.git, _answers("MERGED", head))

    assert gone.path == path
    assert not path.exists()
    assert "feat/364-worktree-home" not in _branches(git_sandbox)
    assert worktrees.linked(git_sandbox.git) == []
    assert brief.is_file()


def test_report_goes_red_for_a_merged_tree_still_standing(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    root = tmp_path / "work"
    _path, head = _open_tree(git_sandbox, root)
    items = worktrees.linked(git_sandbox.git)

    assert worktrees.report(items, root, _answers("OPEN")) == CLEAN
    assert worktrees.report(items, root, _answers("MERGED", head)) == VIOLATION


def test_parse_pull_request_tells_no_pr_from_unreadable() -> None:
    """「沒有 PR」與「看不懂 GitHub 回什麼」是兩件事，後者不准當成前者。"""
    assert worktrees.parse_pull_request("[]") is None
    assert worktrees.parse_pull_request('[{"number": 9, "state": "MERGED", "headRefOid": "abc"}]') == PullRequest(
        number=9, state="MERGED", head_oid="abc"
    )
    for raw in ("not json", "{}", '[{"number": "9"}]'):
        with pytest.raises(WorktreeError):
            worktrees.parse_pull_request(raw)
