"""``governance/git_range.py`` 的考卷：量一段提交範圍的差異，兩組各咬一件事。

1. **讀差異**（``parse_diff``）——``git diff --name-status -z`` 的輸出怎麼讀。檔名裡有空白
   或換行照樣要讀對，改名的兩側都要收下來，截斷或認不出的狀態一律「這一跑不算數」。
2. **範圍**（``resolve_range``／``diff_entries``）——要量哪一段。這一組真的開 git，所以每一支
   都跟 ``git_sandbox`` 要一棵暫存樹（唯一准 spawn 版控工具的那支 fixture）。
   兩點差異在這裡會出錯：主線在分支之後合進來的變更會被算到候選身上。

這套零件原本是規矩卡 tests-land-with-code 檢查程式的一部分；那張卡 2026-10-29 退掉（#977），
零件留給 precision-contracts-live-in-one-registry 用，考卷只留零件那兩組（判準那幾組跟卡一起退）。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance import git_range
from governance.exit_codes import ToolBroken
from tests.conftest import GitSandbox

# 這幾條刻意指**今天真的在樹上**的檔：差異的輸入拿真路徑，就不會有「引用解析不到」那種假問題
# （規矩卡 refs-and-links-resolve 會咬）。
PRODUCT = "src/aosr/config/paths.py"
PRODUCT_TWO = "src/aosr/config/early_reflection.py"
EXAM = "tests/engine/test_config_cut1_paths.py"
OUTSIDE = "governance/checks/style_guard.py"
DOC = "docs/decisions/ascii-filenames.md"


# ── 第一組：讀差異 ──────────────────────────────────────────────────────────


def test_parser_reads_a_plain_modification() -> None:
    entries = git_range.parse_diff(f"M\0{PRODUCT}\0")
    assert [(e.status, e.old, e.new) for e in entries] == [("M", PRODUCT, PRODUCT)]


def test_parser_reads_a_path_with_a_space() -> None:
    """檔名裡有空白：``-z`` 之下照樣是一個完整路徑，不會被切成兩半。"""
    spaced = "src/aosr/config/two words.py"  # 假設的檔名，只餵給剖析器，不是引用
    entries = git_range.parse_diff(f"M\0{spaced}\0")
    assert [e.new for e in entries] == [spaced]


def test_parser_reads_a_path_with_a_newline() -> None:
    """檔名裡有換行：逐行剖析會漏，``-z`` 的 NUL 界線不會。"""
    odd = "src/aosr/config/two\nlines.py"
    entries = git_range.parse_diff(f"A\0{odd}\0")
    assert [e.new for e in entries] == [odd]


def test_parser_reads_both_sides_of_a_rename() -> None:
    """改名一筆吃三格：狀態、舊路徑、新路徑。"""
    entries = git_range.parse_diff(f"R100\0{PRODUCT}\0{OUTSIDE}\0")
    assert [(e.status, e.old, e.new) for e in entries] == [("R", PRODUCT, OUTSIDE)]


def test_parser_reads_a_copy_as_two_sided() -> None:
    """複製跟改名同一個形狀（兩側），狀態字母不同。"""
    entries = git_range.parse_diff(f"C75\0{PRODUCT}\0{PRODUCT_TWO}\0")
    assert [(e.status, e.old, e.new) for e in entries] == [("C", PRODUCT, PRODUCT_TWO)]


def test_parser_reads_several_records_in_a_row() -> None:
    raw = f"M\0{PRODUCT}\0R100\0{EXAM}\0{OUTSIDE}\0D\0{PRODUCT_TWO}\0"
    entries = git_range.parse_diff(raw)
    assert [(e.status, e.old, e.new) for e in entries] == [
        ("M", PRODUCT, PRODUCT),
        ("R", EXAM, OUTSIDE),
        ("D", PRODUCT_TWO, ""),
    ]


def test_parser_sides_of_an_entry_skip_the_empty_side() -> None:
    """新增只有新路徑、刪除只有舊路徑、改名兩側都算。"""
    added, deleted, renamed = git_range.parse_diff(f"A\0{PRODUCT}\0D\0{EXAM}\0R90\0{OUTSIDE}\0{DOC}\0")
    assert added.sides == (PRODUCT,)
    assert deleted.sides == (EXAM,)
    assert renamed.sides == (OUTSIDE, DOC)


def test_parser_rejects_a_truncated_record() -> None:
    """狀態後面沒有路徑：讀不完整就不出結論。"""
    with pytest.raises(ToolBroken):
        git_range.parse_diff("M\0")


def test_parser_rejects_a_rename_missing_its_second_side() -> None:
    with pytest.raises(ToolBroken):
        git_range.parse_diff(f"R100\0{PRODUCT}\0")


def test_parser_rejects_an_unmerged_status() -> None:
    """``U``（未合併）是「我不知道這個檔怎麼了」，不准當成沒改。"""
    with pytest.raises(ToolBroken):
        git_range.parse_diff(f"U\0{PRODUCT}\0")


def test_parser_rejects_an_unknown_status() -> None:
    with pytest.raises(ToolBroken):
        git_range.parse_diff(f"Z\0{PRODUCT}\0")


def test_parser_rejects_output_that_does_not_end_with_a_nul() -> None:
    """``-z`` 的每一格都以 NUL 收尾；沒有收尾就是輸出被截斷了。

    已被獨立重跑證實的洞：一份「狀態、一個路徑、最後少一個收尾 NUL」的輸出原本照樣被收下，
    等於「讀了半份差異還敢下結論」。
    """
    with pytest.raises(ToolBroken):
        git_range.parse_diff(f"M\0{PRODUCT}")


def test_parser_rejects_a_truncated_rename_second_side() -> None:
    """改名的新路徑被截掉一半（沒有收尾的 NUL）也不准收下。"""
    with pytest.raises(ToolBroken):
        git_range.parse_diff(f"R100\0{PRODUCT}\0{OUTSIDE}")


def test_parser_rejects_a_similarity_score_on_a_one_sided_status() -> None:
    """一側的狀態不帶相似度數字：``A123`` 是我沒看懂的形狀，不是「新增」。"""
    with pytest.raises(ToolBroken):
        git_range.parse_diff(f"A123\0{PRODUCT}\0")


def test_parser_rejects_a_rename_without_a_similarity_score() -> None:
    """兩側的狀態一定帶相似度數字；裸 ``R`` 是我沒看懂的形狀。"""
    with pytest.raises(ToolBroken):
        git_range.parse_diff(f"R\0{PRODUCT}\0{OUTSIDE}")


# ── 第二組：範圍（真的開 git，所以每一支都跟 git_sandbox 要樹）─────────────


def _write(root: Path, rel: str, body: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


def _commit(sandbox: GitSandbox, message: str, *touched: str) -> str:
    for rel in touched:
        _write(sandbox.root, rel, f"# {rel}\n# {message}\n")
    sandbox.git("add", "-A")
    sandbox.git("commit", "-q", "--allow-empty", "-m", message)
    return sandbox.git("rev-parse", "HEAD").stdout.strip()


def _no_range_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(git_range.BASE_ENV, raising=False)
    monkeypatch.delenv(git_range.HEAD_ENV, raising=False)
    monkeypatch.delenv(git_range.EVENT_ENV, raising=False)


def _seeded(sandbox: GitSandbox) -> str:
    """一棵有產品程式與考卷的樹，回傳分支點那一筆。"""
    return _commit(sandbox, "base", PRODUCT, EXAM)


def _touched(sandbox: GitSandbox, rng: git_range.Range) -> set[str]:
    return {e.new for e in git_range.diff_entries(sandbox.root, rng)}


@pytest.fixture
def sandbox_with_a_moving_main(git_sandbox: GitSandbox) -> tuple[GitSandbox, str, str]:
    """分支點之後主線改了考卷、候選分支只改產品程式。"""
    _seeded(git_sandbox)
    _commit(git_sandbox, "主線上另一支 PR 改了考卷", EXAM)
    main_tip = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    git_sandbox.git("switch", "-q", "-c", "candidate", "HEAD~1")
    head_tip = _commit(git_sandbox, "候選只改產品程式", PRODUCT)
    return git_sandbox, main_tip, head_tip


def test_a_change_that_landed_on_main_is_not_credited_to_the_candidate(
    sandbox_with_a_moving_main: tuple[GitSandbox, str, str],
) -> None:
    """三點差異：主線在分支之後改了考卷，候選只改產品——那筆考卷不算候選的。

    兩點差異（``base..head``）在這裡會把主線那筆考卷列出來。
    """
    sandbox, main_tip, head_tip = sandbox_with_a_moving_main
    touched = _touched(sandbox, git_range.Range.three_dot(main_tip, head_tip))
    assert PRODUCT in touched, f"候選自己改的產品程式沒被列出來：{touched}"
    assert EXAM not in touched, f"主線上那筆考卷被算到候選身上了：{touched}"


def test_two_dot_would_have_credited_the_mainline_change(
    sandbox_with_a_moving_main: tuple[GitSandbox, str, str],
) -> None:
    """對照組：同一棵樹用兩點差異量，主線那筆考卷真的會被算進來。

    這一支是上一支的證據——沒有它，「三點是必要的」只是一句話。
    """
    sandbox, main_tip, head_tip = sandbox_with_a_moving_main
    assert EXAM in _touched(sandbox, git_range.Range.two_dot(main_tip, head_tip)), (
        "兩點差異竟然沒列出主線那筆考卷，這個對照組不成立"
    )


def test_the_whole_range_is_measured_not_just_the_last_commit(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """產品改動在較早那一筆、最後一筆只改文件——量的是整段範圍。"""
    base = _seeded(git_sandbox)
    _commit(git_sandbox, "先改產品", PRODUCT)
    head = _commit(git_sandbox, "最後一筆只改文件", DOC)
    monkeypatch.setenv(git_range.BASE_ENV, base)
    monkeypatch.setenv(git_range.HEAD_ENV, head)
    monkeypatch.delenv(git_range.EVENT_ENV, raising=False)
    touched = _touched(git_sandbox, git_range.resolve_range(git_sandbox.root))
    assert touched == {PRODUCT, DOC}, f"只看最後一筆就漏了較早那筆：{touched}"


def test_a_pull_request_without_a_base_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR 事件卻拿不到分支點：沒有範圍就沒有結論，回 2 不回 0。"""
    _seeded(git_sandbox)
    _no_range_env(monkeypatch)
    monkeypatch.setenv(git_range.EVENT_ENV, "pull_request")
    with pytest.raises(ToolBroken):
        git_range.resolve_range(git_sandbox.root)


def test_an_unresolvable_base_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """base 那顆物件不在（淺 clone 的形狀）：回 2，不准當成沒改檔。"""
    head = _seeded(git_sandbox)
    monkeypatch.setenv(git_range.BASE_ENV, "0" * len(head))
    monkeypatch.setenv(git_range.HEAD_ENV, head)
    monkeypatch.delenv(git_range.EVENT_ENV, raising=False)
    with pytest.raises(ToolBroken):
        git_range.resolve_range(git_sandbox.root)


def test_an_unresolvable_head_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _seeded(git_sandbox)
    monkeypatch.setenv(git_range.BASE_ENV, base)
    monkeypatch.setenv(git_range.HEAD_ENV, "no-such-ref-at-all")
    monkeypatch.delenv(git_range.EVENT_ENV, raising=False)
    with pytest.raises(ToolBroken):
        git_range.resolve_range(git_sandbox.root)


def test_bases_with_no_common_ancestor_are_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """兩條不相干的歷史算不出分支點——算不出可信範圍就回 2。"""
    head = _seeded(git_sandbox)
    git_sandbox.git("switch", "-q", "--orphan", "unrelated")
    stranger = _commit(git_sandbox, "另一條歷史", "README.md")
    monkeypatch.setenv(git_range.BASE_ENV, stranger)
    monkeypatch.setenv(git_range.HEAD_ENV, head)
    monkeypatch.delenv(git_range.EVENT_ENV, raising=False)
    with pytest.raises(ToolBroken):
        git_range.resolve_range(git_sandbox.root)


def test_a_local_branch_falls_back_to_the_merge_base(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本機分支沒給 base：從跟主線的共同祖先算起，不是只看最後一筆。

    最後一筆只改文件，所以退回 ``HEAD~1..HEAD`` 的寫法只會列出文件；從共同祖先算才列得出產品。
    """
    _seeded(git_sandbox)
    git_sandbox.git("switch", "-q", "-c", "candidate")
    _commit(git_sandbox, "先改產品", PRODUCT)
    _commit(git_sandbox, "最後一筆只改文件", DOC)
    _no_range_env(monkeypatch)
    touched = _touched(git_sandbox, git_range.resolve_range(git_sandbox.root))
    assert touched == {PRODUCT, DOC}, f"本機分支上只量了最後一筆：{touched}"


def test_on_the_mainline_the_range_is_the_previous_mainline_commit(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """主線 push：共同祖先就是 HEAD 自己，範圍改成上一個主線提交（合併進來的那一整支）。"""
    _seeded(git_sandbox)
    _commit(git_sandbox, "主線上合進來的一筆，只改產品", PRODUCT)
    _no_range_env(monkeypatch)
    touched = _touched(git_sandbox, git_range.resolve_range(git_sandbox.root))
    assert touched == {PRODUCT}, f"主線那一筆沒被量到：{touched}"


def test_a_root_commit_has_no_range(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """根提交沒有父節點、也沒有分支點：回 2，不准拿「什麼都沒改」交差。"""
    _commit(git_sandbox, "根提交", PRODUCT)
    _no_range_env(monkeypatch)
    with pytest.raises(ToolBroken):
        git_range.resolve_range(git_sandbox.root)


def test_an_empty_commit_is_a_valid_range(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """空提交是有效範圍：差異是空的，不是工具壞。"""
    base = _seeded(git_sandbox)
    head = _commit(git_sandbox, "空提交")
    monkeypatch.setenv(git_range.BASE_ENV, base)
    monkeypatch.setenv(git_range.HEAD_ENV, head)
    monkeypatch.delenv(git_range.EVENT_ENV, raising=False)
    assert git_range.diff_entries(git_sandbox.root, git_range.resolve_range(git_sandbox.root)) == []


def test_git_missing_from_path_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """抽掉外部工具就回 2：這一跑沒量到東西，不是沒有違規。"""
    base = _seeded(git_sandbox)
    head = _commit(git_sandbox, "改產品", PRODUCT)
    monkeypatch.setenv("PATH", str(git_sandbox.root / "no-such-bin-dir"))
    with pytest.raises(ToolBroken):
        git_range.diff_entries(git_sandbox.root, git_range.Range.three_dot(base, head))


def test_rename_detection_is_on(git_sandbox: GitSandbox) -> None:
    """改名要被認出來是改名，不是「刪一個、加一個」——兩側都要收下來。"""
    base = _seeded(git_sandbox)
    body = (git_sandbox.root / PRODUCT).read_text(encoding="utf-8")
    _write(git_sandbox.root, OUTSIDE, body)
    (git_sandbox.root / PRODUCT).unlink()
    git_sandbox.git("add", "-A")
    git_sandbox.git("commit", "-q", "-m", "把引擎的檔改名搬出去")
    head = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    entries = git_range.diff_entries(git_sandbox.root, git_range.Range.three_dot(base, head))
    assert [(e.status, e.old, e.new) for e in entries] == [("R", PRODUCT, OUTSIDE)]


def test_no_mainline_ref_at_all_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """沒給 base、這棵樹裡又找不到主線參照：拿不到可信範圍就回 2。

    不准默認「任何本機分支都可以退成最後一筆」——那會讓答案取決於跑的時候 HEAD 剛好是哪一筆。
    """
    _seeded(git_sandbox)
    # 刻意多一筆：HEAD~1 解得出來，所以「根提交沒有父節點」那條路被排除掉，
    # 唯一還能讓它回 2 的理由只剩「找不到主線參照」。
    _commit(git_sandbox, "第二筆", PRODUCT)
    git_sandbox.git("branch", "-m", "main", "somewhere-else")
    _no_range_env(monkeypatch)
    with pytest.raises(ToolBroken):
        git_range.resolve_range(git_sandbox.root)


def test_a_mainline_ref_without_a_common_ancestor_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """有主線參照、卻跟它算不出共同祖先：回 2，不准退回「量最後一筆」。"""
    _seeded(git_sandbox)
    git_sandbox.git("switch", "-q", "--orphan", "stranger")
    _commit(git_sandbox, "另一條歷史", "README.md")
    # 同上，多一筆讓 HEAD~1 解得出來：排除「根提交」那條路，只剩「算不出共同祖先」。
    _commit(git_sandbox, "另一條歷史的第二筆", PRODUCT)
    _no_range_env(monkeypatch)
    with pytest.raises(ToolBroken):
        git_range.resolve_range(git_sandbox.root)


def test_a_fixture_env_does_not_leak_the_callers_git_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """建暫存 repo 的環境要把呼叫端的 GIT_DIR 那幾格丟掉，否則暫存 repo 會指到別人的 .git。"""
    for name in git_range.DROPPED_GIT_ENV:
        monkeypatch.setenv(name, "/somewhere/else")
    env = git_range._fixture_env(tmp_path)
    assert not (set(env) & set(git_range.DROPPED_GIT_ENV)), "呼叫端的 git 環境變數漏進暫存 repo 的環境"
    assert env["GIT_AUTHOR_EMAIL"] == env["GIT_COMMITTER_EMAIL"] == git_range.FIXTURE_IDENTITY
    assert not Path(env["GIT_CONFIG_GLOBAL"]).exists(), "暫存 repo 的全域設定檔應該指到一個不存在的地方"
