"""worktree（第二個工作目錄）的家與名字只有一種；拆樹只拆「PR 已合進主線、頭對得上」或「沒開過 PR、沒有主線之外的提交」（#442），而且樹乾淨、沒有會跟著消失的被忽略檔的。"""
from __future__ import annotations

import stat
import subprocess
from pathlib import Path

import pytest

from governance import worktrees
from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION
from governance.worktrees import Linked, PrLookup, PullRequest, WorktreeError
from tests.conftest import GitSandbox


def _seed(sandbox: GitSandbox) -> None:
    (sandbox.root / "a.txt").write_text("a\n", encoding="utf-8")
    sandbox.git("add", "a.txt")
    sandbox.git("commit", "-q", "-m", "one")


def _answers(state: str | None, head: str = "", base: str = "main") -> PrLookup:
    """假的 GitHub：每條分支都回同一個答案。``state`` 是 None 代表沒有 PR。"""

    def lookup(_branch: str) -> PullRequest | None:
        return None if state is None else PullRequest(number=7, state=state, head_oid=head, base=base)

    return lookup


def _open_tree(sandbox: GitSandbox, root: Path) -> tuple[Path, str]:
    _seed(sandbox)
    name = worktrees.tree_name("feat", 364, "worktree-home")
    path = worktrees.new(name, sandbox.git, root, base="main")
    return path, sandbox.git("rev-parse", "refs/heads/feat/364-worktree-home").stdout.strip()


def _branches(sandbox: GitSandbox) -> list[str]:
    return sandbox.git("branch", "--format=%(refname:short)").stdout.split()


def _origin(sandbox: GitSandbox) -> None:
    """沙箱自己當 origin，fetch 一次，``refs/remotes/origin/main`` 就是現在的主線。"""
    if "origin" not in sandbox.git("remote").stdout.split():
        sandbox.git("remote", "add", "origin", str(sandbox.root))
    sandbox.git("fetch", "--quiet", "origin", "main")


def _advance_main(sandbox: GitSandbox) -> None:
    """主線往前走一顆，再更新 origin。用底層指令接一顆新提交、不碰索引：沙箱把 GIT_INDEX_FILE 釘在主樹，
    改了主樹的索引，量那棵樹時會量到主樹的索引、假報它髒（正式的 real_git 會先拿掉那個變數）。"""
    head = sandbox.git("rev-parse", "refs/heads/main").stdout.strip()
    tree = sandbox.git("rev-parse", f"{head}^{{tree}}").stdout.strip()
    later = sandbox.git("commit-tree", tree, "-p", head, "-m", "later").stdout.strip()
    sandbox.git("update-ref", "refs/heads/main", later)
    _origin(sandbox)


def _own_commit(sandbox: GitSandbox, branch: str) -> str:
    """在分支上多一顆主線沒有的提交，不碰那棵樹的工作目錄（內容跟原本一樣，樹照樣乾淨）。"""
    head = sandbox.git("rev-parse", f"refs/heads/{branch}").stdout.strip()
    tree = sandbox.git("rev-parse", f"{head}^{{tree}}").stdout.strip()
    commit = sandbox.git("commit-tree", tree, "-p", head, "-m", "own").stdout.strip()
    sandbox.git("update-ref", f"refs/heads/{branch}", commit)
    return commit


def _strict(sandbox: GitSandbox) -> worktrees.Git:
    """跟 real_git 一樣：git 非零退出就丟 WorktreeError（沙箱那一支丟的是 AssertionError）。"""

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        try:
            return sandbox.git(*args)
        except AssertionError as exc:
            raise WorktreeError(str(exc)) from exc

    return git


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


def test_tree_in_the_work_folder_but_not_named_tree_is_flagged(tmp_path: Path) -> None:
    """住對資料夾、最後一層不叫 tree 的也算放錯；只比上兩層的話這一棵會被判成沒事。"""
    root = tmp_path / "work"
    item = Linked(path=root / "364-worktree-home" / "checkout", branch="feat/364-worktree-home")

    assert any("放錯地方" in line for line in worktrees.problems(item, root))


def test_work_folder_reached_through_a_symlink_is_not_flagged(tmp_path: Path) -> None:
    """git 印的是解過符號連結的真實路徑；root 不解就比的話，每一棵都假紅。"""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    item = Linked(path=real / "364-worktree-home" / "tree", branch="feat/364-worktree-home")

    assert worktrees.problems(item, link) == []


def test_detached_tree_is_flagged_and_never_removed(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    _seed(git_sandbox)
    path = tmp_path / "work" / "364-worktree-home" / "tree"
    git_sandbox.git("worktree", "add", "-q", "--detach", str(path), "main")

    (item,) = worktrees.linked(git_sandbox.git)

    assert item.branch is None
    assert any("沒掛在分支上" in line for line in worktrees.problems(item, tmp_path / "work"))
    with pytest.raises(WorktreeError, match="沒掛在分支上"):
        worktrees.remove("364-worktree-home", git_sandbox.git, _answers("MERGED"))
    assert path.is_dir()


def test_key_that_matches_two_trees_is_refused(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """一個名字對到兩棵（一棵靠資料夾名、一棵靠分支名）就不猜。"""
    _open_tree(git_sandbox, tmp_path / "work")
    git_sandbox.git("worktree", "add", "-q", "-b", "364-worktree-home", str(tmp_path / "stray"), "main")

    with pytest.raises(WorktreeError, match="不只一棵"):
        worktrees.find("364-worktree-home", git_sandbox.git)


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
    ("state", "head", "base", "why"),
    [
        ("OPEN", "", "main", "還沒合"),
        ("MERGED", "0" * 40, "main", "沒送出去的提交"),
        ("MERGED", "LOCAL", "feat/some-other-branch", "不是主線"),
    ],
)
def test_remove_refuses_unless_merged_into_mainline_at_the_local_head(
    git_sandbox: GitSandbox, tmp_path: Path, state: str | None, head: str, base: str, why: str
) -> None:
    """拆錯一次就是丟工作：沒 PR、沒合、合了之後本機又多提交、合進的不是主線，樹與分支都要原封不動。"""
    path, local = _open_tree(git_sandbox, tmp_path / "work")
    answers = _answers(state, local if head == "LOCAL" else head, base)

    with pytest.raises(WorktreeError, match=why):
        worktrees.remove("364-worktree-home", git_sandbox.git, answers)

    assert path.is_dir()
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def test_remove_leaves_a_dirty_tree_alone(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """髒樹工具自己先擋、把檔名印出來（git worktree remove 不加 --force 也會擋，是第二道）。"""
    path, head = _open_tree(git_sandbox, tmp_path / "work")
    (path / "unsaved.txt").write_text("還沒提交的東西\n", encoding="utf-8")

    with pytest.raises(WorktreeError, match="unsaved.txt"):
        worktrees.remove("364-worktree-home", git_sandbox.git, _answers("MERGED", head))

    assert (path / "unsaved.txt").is_file()
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def _ignore(sandbox: GitSandbox, *patterns: str) -> None:
    (sandbox.root / ".gitignore").write_text("".join(f"{p}\n" for p in patterns), encoding="utf-8")
    sandbox.git("add", ".gitignore")


def test_remove_refuses_when_ignored_files_would_vanish(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """git 的「乾淨」不看被忽略的檔，拆樹會無聲把它們刪掉；不是重建得回來的那幾種就要擋。"""
    _ignore(git_sandbox, "notes/", ".venv/")
    path, head = _open_tree(git_sandbox, tmp_path / "work")
    (path / "notes").mkdir()
    (path / "notes" / "measured.md").write_text("量到的數字\n", encoding="utf-8")

    with pytest.raises(WorktreeError, match="notes/"):
        worktrees.remove("364-worktree-home", git_sandbox.git, _answers("MERGED", head))

    assert (path / "notes" / "measured.md").is_file()
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def test_a_cache_name_deep_inside_another_folder_is_not_a_free_pass() -> None:
    """只有根層的 .venv 算環境；notes 底下剛好叫 .venv 的資料夾裡的東西照樣要擋。"""
    assert worktrees._rebuildable(".venv/")
    assert worktrees._rebuildable("governance/__pycache__/")
    assert not worktrees._rebuildable("notes/.venv/")
    assert not worktrees._rebuildable("notes/")


def test_remove_does_not_mind_rebuildable_ignored_files(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """.venv 每一棵都有；這種也擋的話就沒有一棵拆得掉。"""
    _ignore(git_sandbox, ".venv/", "__pycache__/", "/governance/receipts/*")
    path, head = _open_tree(git_sandbox, tmp_path / "work")
    # 一格一格寫、不寫成一條路徑字串：refs-and-links-resolve 會把路徑字串當成對這棵樹的引用去解析。
    for parts in ((".venv", "bin", "python"), ("governance", "__pycache__", "m.pyc"), ("governance", "receipts", "j.xml")):
        target = path.joinpath(*parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x\n", encoding="utf-8")

    worktrees.remove("364-worktree-home", git_sandbox.git, _answers("MERGED", head))

    assert not path.exists()


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

    assert worktrees.report(items, root, _answers("OPEN"), git_sandbox.git) == CLEAN
    assert worktrees.report(items, root, _answers("MERGED", head), git_sandbox.git) == VIOLATION


def test_report_never_says_remove_when_remove_itself_would_refuse(
    git_sandbox: GitSandbox, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """合了之後本機又多提交的那一棵：list 要是照樣寫「該拆」，就是在叫人硬拆、丟掉那幾顆提交。"""
    root = tmp_path / "work"
    _open_tree(git_sandbox, root)

    code = worktrees.report(worktrees.linked(git_sandbox.git), root, _answers("MERGED", "0" * 40), git_sandbox.git)

    said = capsys.readouterr().err
    assert code == VIOLATION
    assert "別硬拆" in said
    assert "該拆" not in said


def test_report_does_not_say_remove_for_a_tree_with_ignored_keepsakes(
    git_sandbox: GitSandbox, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """被忽略檔那一格 list 也要量；不量的話 list 寫「該拆」、remove 卻拒絕，兩邊又不是同一組條件了。"""
    _ignore(git_sandbox, "notes/")
    root = tmp_path / "work"
    path, head = _open_tree(git_sandbox, root)
    (path / "notes").mkdir()
    (path / "notes" / "measured.md").write_text("量到的數字\n", encoding="utf-8")

    worktrees.report(worktrees.linked(git_sandbox.git), root, _answers("MERGED", head), git_sandbox.git)

    said = capsys.readouterr().err
    assert "notes/" in said
    assert "該拆" not in said


def test_one_unmeasurable_tree_does_not_cut_the_list_short(
    git_sandbox: GitSandbox, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """一棵壞掉的樹（量被忽略檔時 git 直接 fatal）不准讓後面幾棵一行都不印，也不准讓這一跑看起來乾淨。"""
    root = tmp_path / "work"
    _path, head = _open_tree(git_sandbox, root)
    other = worktrees.new(worktrees.tree_name("fix", 365, "second"), git_sandbox.git, root, base="main")

    def flaky(*args: str) -> subprocess.CompletedProcess[str]:
        if "status" in args and any("364-worktree-home" in arg for arg in args):
            raise WorktreeError("fatal: not a git repository")
        return git_sandbox.git(*args)

    code = worktrees.report(worktrees.linked(git_sandbox.git), root, _answers("MERGED", head), flaky)

    said = capsys.readouterr().err
    assert code == TOOL_BROKEN
    assert "量不到" in said
    assert str(other) in said


def test_unmeasurable_tree_still_shows_what_needs_no_git(
    git_sandbox: GitSandbox, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """放錯地方是純路徑算出來的，量不到被忽略檔不是不印它的理由；檔名解不開（ValueError）也算量不到。"""
    _seed(git_sandbox)
    stray = tmp_path / "elsewhere" / "stray"
    git_sandbox.git("worktree", "add", "-q", "-b", "feat/9-stray", str(stray), "main")
    head = git_sandbox.git("rev-parse", "refs/heads/feat/9-stray").stdout.strip()

    def undecodable(*args: str) -> subprocess.CompletedProcess[str]:
        if "status" in args:
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
        return git_sandbox.git(*args)

    items = worktrees.linked(git_sandbox.git)
    code = worktrees.report(items, tmp_path / "work", _answers("MERGED", head), undecodable)

    said = capsys.readouterr().err
    assert code == TOOL_BROKEN
    assert "量不到" in said
    assert "放錯地方" in said


def test_main_tree_refuses_empty_output() -> None:
    def silent(*_args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with pytest.raises(WorktreeError, match="什麼都沒印"):
        worktrees.main_tree(silent)


def test_main_turns_an_undecodable_git_answer_into_tool_broken(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """git 印出來的路徑不是合法 UTF-8 時是工具做不下去（2），不是抓到違規（traceback 的 1）。"""

    def undecodable(_repo: Path) -> worktrees.Git:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(worktrees, "repo_root", lambda: git_sandbox.root)
    monkeypatch.setattr(worktrees, "real_git", undecodable)

    assert worktrees.main(["list"]) == TOOL_BROKEN


def _wire_main(
    monkeypatch: pytest.MonkeyPatch, sandbox: GitSandbox, root: Path, answers: PrLookup, start: Path | None = None
) -> list[Path]:
    """把 main 接到沙箱上。回「main 拿哪幾個目錄去要 git」的紀錄。"""
    asked: list[Path] = []

    def fake_git(repo: Path) -> worktrees.Git:
        asked.append(repo)
        return sandbox.git

    _origin(sandbox)
    monkeypatch.setattr(worktrees, "repo_root", lambda: start or sandbox.root)
    monkeypatch.setattr(worktrees, "real_git", fake_git)
    monkeypatch.setattr(worktrees, "gh_pull_request", lambda _repo: answers)
    monkeypatch.setattr(worktrees, "work_root", lambda: root)
    return asked


def test_main_list_exit_code_follows_the_report(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """這支工具守得到的只有「list 回非零」這一件；main 把它吞成 0 就什麼都沒守。"""
    root = tmp_path / "work"
    _path, head = _open_tree(git_sandbox, root)

    _wire_main(monkeypatch, git_sandbox, root, _answers("OPEN"))
    assert worktrees.main(["list"]) == CLEAN
    _wire_main(monkeypatch, git_sandbox, root, _answers("MERGED", head))
    assert worktrees.main(["list"]) == VIOLATION


def test_main_says_tool_broken_not_violation_when_it_cannot_go_on(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(git_sandbox)
    _wire_main(monkeypatch, git_sandbox, tmp_path / "work", _answers(None))

    assert worktrees.main(["remove", "no-such-tree"]) == TOOL_BROKEN


def test_main_tree_reads_the_first_block(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """有別棵樹在的時候，主樹還是第一段那一個。"""
    _open_tree(git_sandbox, tmp_path / "work")

    assert worktrees.main_tree(git_sandbox.git).resolve() == git_sandbox.root.resolve()


def test_main_runs_git_from_the_main_tree_even_when_started_inside_the_tree_to_remove(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """站在要拆的那一棵裡拆自己：git 的 cwd 要是還釘在那一棵，樹拆掉之後刪分支那一步就死在半路。"""
    root = tmp_path / "work"
    path, head = _open_tree(git_sandbox, root)
    asked = _wire_main(monkeypatch, git_sandbox, root, _answers("MERGED", head), start=path)

    assert worktrees.main(["remove", "364-worktree-home"]) == CLEAN

    assert asked[-1].resolve() == git_sandbox.root.resolve()
    assert "feat/364-worktree-home" not in _branches(git_sandbox)


def test_real_git_raises_on_nonzero_and_ignores_ambient_git_env(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """非零退出不炸的話，remove 會一路走完、印「拆掉了」回 0——一張說謊的收據。"""
    _seed(git_sandbox)
    monkeypatch.setenv("GIT_INDEX_FILE", str(tmp_path / "not-an-index"))
    git = worktrees.real_git(git_sandbox.root)

    assert git("status", "--porcelain").stdout == ""
    with pytest.raises(WorktreeError, match="rev-parse"):
        git("rev-parse", "--verify", "refs/heads/not-there")


def _fake_gh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool = bin_dir / "gh"
    tool.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", str(bin_dir))


def test_gh_lookup_asks_for_every_state_of_that_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """只問開著的 PR 的話，合掉的那一筆永遠問不到，就沒有一棵拆得掉。"""
    row = '[{"number": 5, "state": "MERGED", "headRefOid": "abc", "baseRefName": "main"}]'
    _fake_gh(tmp_path, monkeypatch, f"echo \"$@\" > {tmp_path / 'argv.txt'}; echo '{row}'")

    found = worktrees.gh_pull_request(tmp_path)("feat/364-worktree-home")

    assert found == PullRequest(number=5, state="MERGED", head_oid="abc", base="main")
    argv = (tmp_path / "argv.txt").read_text(encoding="utf-8")
    assert "--state all" in argv
    assert "--head feat/364-worktree-home" in argv


def test_gh_failure_is_not_read_as_no_pull_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """問不到 GitHub 跟「沒有 PR」是兩件事。"""
    _fake_gh(tmp_path, monkeypatch, "echo 'no network' >&2; exit 1")

    with pytest.raises(WorktreeError, match="no network"):
        worktrees.gh_pull_request(tmp_path)("feat/364-worktree-home")


def test_parse_pull_request_tells_no_pr_from_unreadable() -> None:
    """「沒有 PR」與「看不懂 GitHub 回什麼」是兩件事，後者不准當成前者。"""
    assert worktrees.parse_pull_request("[]") is None
    merged = '{"number": 9, "state": "MERGED", "headRefOid": "abc", "baseRefName": "main"}'
    assert worktrees.parse_pull_request(f"[{merged}]") == PullRequest(
        number=9, state="MERGED", head_oid="abc", base="main"
    )
    for raw in ("not json", "{}", '[{"number": "9"}]', '[{"number": 9, "state": "MERGED", "headRefOid": "abc"}]'):
        with pytest.raises(WorktreeError):
            worktrees.parse_pull_request(raw)


def test_an_open_pull_request_wins_over_an_older_merged_one() -> None:
    """同一個分支名用過兩輪：舊的合了、新的還開著。拿到舊的那一筆就會把還在做的樹判成可以拆。"""
    rows = (
        '[{"number": 9, "state": "MERGED", "headRefOid": "old", "baseRefName": "main"},'
        ' {"number": 12, "state": "OPEN", "headRefOid": "new", "baseRefName": "main"},'
        ' {"number": 11, "state": "CLOSED", "headRefOid": "mid", "baseRefName": "main"}]'
    )

    found = worktrees.parse_pull_request(rows)

    assert found is not None
    assert (found.number, found.state) == (12, "OPEN")


# ---- #442：沒開過 PR 的安全拆樹路徑 ----


def test_an_empty_tree_without_pr_can_go_after_mainline_moved_on(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """從舊主線開出來、什麼都沒做的樹：主線往前走了也照樣能拆，不要求分支的頭等於最新主線。"""
    path, head = _open_tree(git_sandbox, tmp_path / "work")
    brief = path.parent / "brief.md"
    brief.write_text("工單\n", encoding="utf-8")
    _advance_main(git_sandbox)
    assert git_sandbox.git("rev-parse", "refs/remotes/origin/main").stdout.strip() != head

    gone = worktrees.remove("364-worktree-home", git_sandbox.git, _answers(None))

    assert gone.path == path
    assert not path.exists()
    assert "feat/364-worktree-home" not in _branches(git_sandbox)
    assert brief.is_file()


def test_a_tree_without_pr_that_has_its_own_commit_stays(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    path, _head = _open_tree(git_sandbox, tmp_path / "work")
    own = _own_commit(git_sandbox, "feat/364-worktree-home")
    _advance_main(git_sandbox)

    with pytest.raises(WorktreeError, match="主線之外的提交"):
        worktrees.remove("364-worktree-home", git_sandbox.git, _answers(None))

    assert path.is_dir()
    assert git_sandbox.git("rev-parse", "refs/heads/feat/364-worktree-home").stdout.strip() == own


@pytest.mark.parametrize("unsaved", ["untracked", "modified", "ignored"])
def test_a_tree_without_pr_with_anything_unsaved_stays(
    git_sandbox: GitSandbox, tmp_path: Path, unsaved: str
) -> None:
    """沒有主線之外的提交，但樹裡有沒保存的東西：拆了就丟，一律不拆，東西原封不動。"""
    _ignore(git_sandbox, "notes/")
    path, _head = _open_tree(git_sandbox, tmp_path / "work")
    _origin(git_sandbox)
    target = {"untracked": path / "draft.md", "modified": path / "a.txt", "ignored": path / "notes" / "measured.md"}[unsaved]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("還沒保存的東西\n", encoding="utf-8")

    with pytest.raises(WorktreeError, match="不拆"):
        worktrees.remove("364-worktree-home", git_sandbox.git, _answers(None))

    assert target.read_text(encoding="utf-8") == "還沒保存的東西\n"
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def test_a_failed_pull_request_lookup_stops_removal(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    path, _head = _open_tree(git_sandbox, tmp_path / "work")
    _origin(git_sandbox)

    def broken(_branch: str) -> PullRequest | None:
        raise WorktreeError("gh pr list 回 1：no network")

    with pytest.raises(WorktreeError, match="no network"):
        worktrees.remove("364-worktree-home", git_sandbox.git, broken)

    assert path.is_dir()
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def test_a_failed_git_check_is_not_read_as_zero_commits(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """量不到主線（沒有 origin/main）：git 量不下去就停，不當成「沒有主線之外的提交」。"""
    path, _head = _open_tree(git_sandbox, tmp_path / "work")

    with pytest.raises(WorktreeError, match="rev-list"):
        worktrees.remove("364-worktree-home", _strict(git_sandbox), _answers(None))

    assert path.is_dir()
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def test_main_stops_when_mainline_cannot_be_updated(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """更新不了 origin/main 就不拆、list 也不算數；沙箱沒有 origin 時 fetch 會失敗。"""
    root = tmp_path / "work"
    path, _head = _open_tree(git_sandbox, root)
    monkeypatch.setattr(worktrees, "repo_root", lambda: git_sandbox.root)
    monkeypatch.setattr(worktrees, "real_git", lambda _repo: _strict(git_sandbox))
    monkeypatch.setattr(worktrees, "gh_pull_request", lambda _repo: _answers(None))
    monkeypatch.setattr(worktrees, "work_root", lambda: root)

    assert worktrees.main(["remove", "364-worktree-home"]) == TOOL_BROKEN
    assert worktrees.main(["list"]) == TOOL_BROKEN
    assert path.is_dir()
    assert "feat/364-worktree-home" in _branches(git_sandbox)


def test_report_says_eligible_for_an_empty_tree_without_pr_but_never_goes_red_or_removes(
    git_sandbox: GitSandbox, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """「符合清理條件」只是印給人看：剛開好的樹也長這樣，所以不標紅、不算該拆沒拆，也不自動拆。"""
    root = tmp_path / "work"
    path, _head = _open_tree(git_sandbox, root)
    _advance_main(git_sandbox)

    code = worktrees.report(worktrees.linked(git_sandbox.git), root, _answers(None), git_sandbox.git)

    said = capsys.readouterr().err
    assert code == CLEAN
    assert "符合清理條件" in said
    assert "紅：" not in said
    assert path.is_dir()


def test_report_does_not_call_a_tree_with_its_own_commit_eligible(
    git_sandbox: GitSandbox, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "work"
    _open_tree(git_sandbox, root)
    _own_commit(git_sandbox, "feat/364-worktree-home")
    _origin(git_sandbox)

    code = worktrees.report(worktrees.linked(git_sandbox.git), root, _answers(None), git_sandbox.git)

    assert code == CLEAN
    assert "符合清理條件" not in capsys.readouterr().err


def test_a_merged_branch_is_not_judged_by_commits_outside_mainline(git_sandbox: GitSandbox, tmp_path: Path) -> None:
    """主線用 squash 合併：合過的分支上原本那幾顆提交本來就不在主線裡。已合 PR 的路徑不准被新那一格擋住。"""
    path, _head = _open_tree(git_sandbox, tmp_path / "work")
    own = _own_commit(git_sandbox, "feat/364-worktree-home")
    _origin(git_sandbox)
    assert worktrees.commits_outside_mainline("feat/364-worktree-home", git_sandbox.git) > 0

    worktrees.remove("364-worktree-home", git_sandbox.git, _answers("MERGED", own))

    assert not path.exists()
    assert "feat/364-worktree-home" not in _branches(git_sandbox)


def test_main_updates_mainline_before_judging_a_tree_without_pr(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本機的 origin/main 舊了：分支上那顆提交其實已經在主線裡，只是還沒更新。先更新才判得對。"""
    root = tmp_path / "work"
    path, _head = _open_tree(git_sandbox, root)
    _origin(git_sandbox)
    stale = git_sandbox.git("rev-parse", "refs/remotes/origin/main").stdout.strip()
    own = _own_commit(git_sandbox, "feat/364-worktree-home")
    git_sandbox.git("update-ref", "refs/heads/main", own)
    assert worktrees.commits_outside_mainline("feat/364-worktree-home", git_sandbox.git) > 0
    monkeypatch.setattr(worktrees, "repo_root", lambda: git_sandbox.root)
    monkeypatch.setattr(worktrees, "real_git", lambda _repo: _strict(git_sandbox))
    monkeypatch.setattr(worktrees, "gh_pull_request", lambda _repo: _answers(None))
    monkeypatch.setattr(worktrees, "work_root", lambda: root)

    assert worktrees.main(["remove", "364-worktree-home"]) == CLEAN

    assert git_sandbox.git("rev-parse", "refs/remotes/origin/main").stdout.strip() not in (stale, "")
    assert not path.exists()
