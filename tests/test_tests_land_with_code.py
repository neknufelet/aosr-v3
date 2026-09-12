"""規矩卡 tests-land-with-code 的聚焦反例：產品程式與測試要同一支合併請求落地。

分三組，每一組咬一件事：

1. **判決**（``problems``）——給一組差異紀錄，判「這支 PR 有沒有讓產品程式獨自落地」。
   這一組不碰版控，純粹是判準本身：測試側只認 ``.py``、只刪測試不算、改名兩側都算。
2. **讀差異**（``parse_diff``）——``git diff --name-status -z`` 的輸出怎麼讀。檔名裡有空白
   或換行照樣要讀對，改名的兩側都要收下來，截斷或認不出的狀態一律「這一跑不算數」。
3. **範圍**（``resolve_range``／``diff_entries``）——要量哪一段。這一組真的開 git，所以每一支
   都跟 ``git_sandbox`` 要一棵暫存樹（唯一准 spawn 版控工具的那支 fixture）。
   兩點差異在這裡會出錯：主線在分支之後合進來的測試變更會被算到候選身上。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance.checks import tests_land_with_code as card
from governance.exit_codes import ToolBroken
from tests.conftest import GitSandbox

# 這幾條刻意指**今天真的在樹上**的檔：判準判的是路徑，拿真路徑當輸入就不會有「引用解析不到」
# 那種假問題（規矩卡 refs-and-links-resolve 會咬），順手也證明宣告的兩側真的有對象（准入第四條）。
PRODUCT = "src/aosr/config/paths.py"
PRODUCT_TWO = "src/aosr/config/early_reflection.py"
PRODUCT_DATA = "src/aosr/config/data/perceptual.toml"
EXAM = "tests/engine/test_config_cut1_paths.py"
EXAM_TWO = "tests/engine/test_config_cut1_early_reflection.py"
# 兩側之外的家。改名搬進來／搬出去用它當對面。
OUTSIDE = "governance/checks/style_guard.py"
DOC = "docs/decisions/ascii-filenames.md"


def hypothetical(*segments: str) -> str:
    """組一條**假設的**路徑（差異裡的輸入，不是對這棵樹的引用）。

    為什麼不寫成字面值：這幾條路徑本來就必須在這棵樹裡解析不到——那正是它們要證明的事
    （考卷樹底下沒有非 .py、爬出樹的路徑不准過）。字面值會被規矩卡 refs-and-links-resolve
    當成一個死引用，而它不是引用；與其在那張卡上開一條放行，不如在這裡一次講清楚它是什麼。
    """
    return "/".join(segments)


# 測試側的非 .py。今天這棵樹的 tests/ 底下一個非 .py 都沒有（`git ls-files tests` 全是 .py），
# 所以這兩條只能是假設的路徑。
EXAM_README = hypothetical("tests", "README.md")
EXAM_DATA = hypothetical("tests", "engine", "answers.json")

# 攻擊用的宣告路徑。刻意**從檢查程式自己的禁止段清單拼出來**，不是抄一份字面值：
# 「哪幾段一律不准」只有一個家（那份清單），改了它這幾支反例跟著改。
DOT_DOT, _DOT, GIT_DIR = card.FORBIDDEN_SEGMENTS
CLIMB_OUT = hypothetical(DOT_DOT, "outside.py")
INTO_GIT_DIR = hypothetical(GIT_DIR, "hooks", "pre-commit.py")
# 樣本樹裡那條指到樹外面的連結，以及「卡上把兩側登記成別的目錄」那一支用的別家路徑。
LINK_IN_TREE = hypothetical("src", "aosr", "link.py")
OTHER_PRODUCT = hypothetical("engine", "thing.py")

# 產品側／測試側住在哪，是**卡上登記**的資料（`[settings]`），不是程式裡的預設值。
# 這一組考卷把它明寫出來當輸入，好讓判準那幾支的反例不跟今天那張卡的內容綁在一起；
# 「今天那張卡真的登記了 src/ 與 tests/」由 test_the_real_card_registers_both_sides 咬。
RULES = card.Rules(product_paths=("src/",), test_paths=("tests/",), code_suffixes=(".py",))


def verdict(entries: list[card.DiffEntry]) -> list[str]:
    return card.problems(entries, RULES)


def added(path: str) -> card.DiffEntry:
    return card.DiffEntry(status="A", old="", new=path)


def modified(path: str) -> card.DiffEntry:
    return card.DiffEntry(status="M", old=path, new=path)


def deleted(path: str) -> card.DiffEntry:
    return card.DiffEntry(status="D", old=path, new="")


def renamed(old: str, new: str) -> card.DiffEntry:
    return card.DiffEntry(status="R", old=old, new=new)


# ── 第一組：判決 ────────────────────────────────────────────────────────────


def test_product_change_alone_is_a_violation() -> None:
    """只改產品程式、一個字都沒動測試——這張卡要擋的就是這個。"""
    hits = verdict([modified(PRODUCT)])
    assert any(PRODUCT in hit for hit in hits), f"只改產品卻沒判違規：{hits}"


def test_product_change_with_a_test_change_is_clean() -> None:
    """產品與測試同一支 PR 落地，回乾淨。"""
    assert verdict([modified(PRODUCT), modified(EXAM)]) == []


def test_a_readme_under_tests_is_not_a_test_change() -> None:
    """測試側只認登記的副檔名：改考卷樹裡的說明文件不算測試同行。"""
    hits = verdict([modified(PRODUCT), modified(EXAM_README)])
    assert any(PRODUCT in hit for hit in hits), f"{EXAM_README} 被當成測試變更了：{hits}"


def test_a_json_fixture_under_tests_is_not_a_test_change() -> None:
    """測試側只認 .py：換一份資料檔也不算測試同行。"""
    hits = verdict([modified(PRODUCT), modified(EXAM_DATA)])
    assert any(PRODUCT in hit for hit in hits), f"資料檔被當成測試變更了：{hits}"


def test_deleting_a_test_is_not_enough_when_product_is_modified() -> None:
    """產品被改了，測試只被刪掉——刪考卷不是寫考卷。"""
    hits = verdict([modified(PRODUCT), deleted(EXAM)])
    assert any(PRODUCT in hit for hit in hits), f"只刪測試被當成測試同行了：{hits}"


def test_pure_product_deletion_with_a_test_deletion_is_clean() -> None:
    """整塊撤除：產品側全是刪除的時候，同行的測試刪除算同行。"""
    assert verdict([deleted(PRODUCT), deleted(EXAM)]) == []


def test_pure_product_deletion_without_any_test_change_is_a_violation() -> None:
    """整塊撤除也要帶著它的考卷走：產品刪了、測試一個字沒動，紅。"""
    hits = verdict([deleted(PRODUCT)])
    assert any(PRODUCT in hit for hit in hits), f"純刪產品卻沒判違規：{hits}"


def test_renaming_a_product_file_out_of_src_needs_a_test_change() -> None:
    """把引擎的檔改名搬出 src：舊路徑命中產品側，照樣要動測試。"""
    hits = verdict([renamed(PRODUCT, OUTSIDE)])
    assert any(PRODUCT in hit for hit in hits), f"改名出去漏掉了：{hits}"


def test_renaming_a_file_into_src_needs_a_test_change() -> None:
    """把別處的檔改名搬進 src：新路徑命中產品側，也要動測試。"""
    hits = verdict([renamed(OUTSIDE, PRODUCT)])
    assert any(PRODUCT in hit for hit in hits), f"改名進來漏掉了：{hits}"


def test_renaming_a_file_into_tests_counts_as_a_test_change() -> None:
    """改名**進來** tests 底下算測試變更（考卷換了家，還在）。"""
    assert verdict([modified(PRODUCT), renamed(OUTSIDE, EXAM)]) == []


def test_renaming_a_test_out_of_tests_is_not_a_test_change() -> None:
    """改名**出去**不算：考卷離開了 tests，等於被撤掉。"""
    hits = verdict([modified(PRODUCT), renamed(EXAM, OUTSIDE)])
    assert any(PRODUCT in hit for hit in hits), f"改名出去被當成測試同行了：{hits}"


def test_a_governance_only_change_is_clean() -> None:
    """只改治理程式：產品側一筆都沒有，不強求產品測試。"""
    assert verdict([modified(OUTSIDE)]) == []


def test_a_non_python_file_under_src_is_not_a_product_change() -> None:
    """產品側只認 .py：搬一份設定檔不算動產品程式。"""
    assert verdict([modified(PRODUCT_DATA)]) == []


def test_an_empty_diff_is_clean() -> None:
    """一個檔都沒改（空提交、只改別處）是有效範圍，不是工具壞。"""
    assert verdict([]) == []


def test_one_product_file_with_a_test_change_covers_the_others() -> None:
    """判的是同行、不是覆蓋率：一份測試變更就讓整支 PR 的產品變更過關。"""
    assert verdict([modified(PRODUCT), added(PRODUCT_TWO), modified(EXAM)]) == []


# ── 第二組：讀差異 ──────────────────────────────────────────────────────────


def test_parser_reads_a_plain_modification() -> None:
    entries = card.parse_diff(f"M\0{PRODUCT}\0")
    assert [(e.status, e.old, e.new) for e in entries] == [("M", PRODUCT, PRODUCT)]


def test_parser_reads_a_path_with_a_space() -> None:
    """檔名裡有空白：``-z`` 之下照樣是一個完整路徑，不會被切成兩半。"""
    spaced = "src/aosr/config/two words.py"  # 假設的檔名，只餵給剖析器，不是引用
    entries = card.parse_diff(f"M\0{spaced}\0")
    assert [e.new for e in entries] == [spaced]


def test_parser_reads_a_path_with_a_newline() -> None:
    """檔名裡有換行：逐行剖析會漏，``-z`` 的 NUL 界線不會。"""
    odd = "src/aosr/config/two\nlines.py"
    entries = card.parse_diff(f"A\0{odd}\0")
    assert [e.new for e in entries] == [odd]


def test_parser_reads_both_sides_of_a_rename() -> None:
    """改名一筆吃三格：狀態、舊路徑、新路徑。"""
    entries = card.parse_diff(f"R100\0{PRODUCT}\0{OUTSIDE}\0")
    assert [(e.status, e.old, e.new) for e in entries] == [("R", PRODUCT, OUTSIDE)]


def test_parser_reads_a_copy_as_two_sided() -> None:
    """複製跟改名同一個形狀（兩側），狀態字母不同。"""
    entries = card.parse_diff(f"C75\0{PRODUCT}\0{PRODUCT_TWO}\0")
    assert [(e.status, e.old, e.new) for e in entries] == [("C", PRODUCT, PRODUCT_TWO)]


def test_parser_reads_several_records_in_a_row() -> None:
    raw = f"M\0{PRODUCT}\0R100\0{EXAM}\0{OUTSIDE}\0D\0{PRODUCT_TWO}\0"
    entries = card.parse_diff(raw)
    assert [(e.status, e.old, e.new) for e in entries] == [
        ("M", PRODUCT, PRODUCT),
        ("R", EXAM, OUTSIDE),
        ("D", PRODUCT_TWO, ""),
    ]


def test_parser_rejects_a_truncated_record() -> None:
    """狀態後面沒有路徑：讀不完整就不出結論。"""
    with pytest.raises(ToolBroken):
        card.parse_diff("M\0")


def test_parser_rejects_a_rename_missing_its_second_side() -> None:
    with pytest.raises(ToolBroken):
        card.parse_diff(f"R100\0{PRODUCT}\0")


def test_parser_rejects_an_unmerged_status() -> None:
    """``U``（未合併）是「我不知道這個檔怎麼了」，不准當成沒改。"""
    with pytest.raises(ToolBroken):
        card.parse_diff(f"U\0{PRODUCT}\0")


def test_parser_rejects_an_unknown_status() -> None:
    with pytest.raises(ToolBroken):
        card.parse_diff(f"Z\0{PRODUCT}\0")


# ── 第三組：範圍（真的開 git，所以每一支都跟 git_sandbox 要樹）─────────────


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
    monkeypatch.delenv(card.BASE_ENV, raising=False)
    monkeypatch.delenv(card.HEAD_ENV, raising=False)
    monkeypatch.delenv(card.EVENT_ENV, raising=False)


def _seeded(sandbox: GitSandbox) -> str:
    """一棵有產品程式與考卷的樹，回傳分支點那一筆。"""
    return _commit(sandbox, "base", PRODUCT, EXAM)


def test_a_test_change_that_landed_on_main_is_not_credited_to_the_candidate(
    sandbox_with_a_moving_main: tuple[GitSandbox, str, str],
) -> None:
    """三點差異：主線在分支之後改了考卷，候選只改產品——那筆考卷不算候選的。

    兩點差異（``base..head``）在這裡會把主線那筆考卷列出來，於是只改產品的 PR 照樣綠。
    """
    sandbox, main_tip, head_tip = sandbox_with_a_moving_main
    entries = card.diff_entries(sandbox.root, card.Range.three_dot(main_tip, head_tip))
    touched = {e.new for e in entries}
    assert PRODUCT in touched, f"候選自己改的產品程式沒被列出來：{touched}"
    assert EXAM not in touched, f"主線上那筆考卷被算到候選身上了：{touched}"


def test_two_dot_would_have_credited_the_mainline_test_change(
    sandbox_with_a_moving_main: tuple[GitSandbox, str, str],
) -> None:
    """對照組：同一棵樹用兩點差異量，主線那筆考卷真的會被算進來。

    這一支是上一支的證據——沒有它，「三點是必要的」只是一句話。
    """
    sandbox, main_tip, head_tip = sandbox_with_a_moving_main
    entries = card.diff_entries(sandbox.root, card.Range.two_dot(main_tip, head_tip))
    assert EXAM in {e.new for e in entries}, "兩點差異竟然沒列出主線那筆考卷，這個對照組不成立"


@pytest.fixture
def sandbox_with_a_moving_main(git_sandbox: GitSandbox) -> tuple[GitSandbox, str, str]:
    """分支點之後主線改了考卷、候選分支只改產品程式。"""
    _seeded(git_sandbox)
    _commit(git_sandbox, "主線上另一支 PR 改了考卷", EXAM)
    main_tip = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    git_sandbox.git("switch", "-q", "-c", "candidate", "HEAD~1")
    head_tip = _commit(git_sandbox, "候選只改產品程式", PRODUCT)
    return git_sandbox, main_tip, head_tip


def test_the_whole_range_is_judged_not_just_the_last_commit(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """產品改動在較早那一筆、最後一筆只改文件——判的是整段範圍。"""
    base = _seeded(git_sandbox)
    _commit(git_sandbox, "先改產品", PRODUCT)
    head = _commit(git_sandbox, "最後一筆只改文件", DOC)
    monkeypatch.setenv(card.BASE_ENV, base)
    monkeypatch.setenv(card.HEAD_ENV, head)
    monkeypatch.delenv(card.EVENT_ENV, raising=False)
    rng = card.resolve_range(git_sandbox.root)
    hits = verdict(card.diff_entries(git_sandbox.root, rng))
    assert any(PRODUCT in hit for hit in hits), f"只看最後一筆就放行了：{hits}"


def test_a_pull_request_without_a_base_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PR 事件卻拿不到分支點：沒有範圍就沒有結論，回 2 不回 0。"""
    _seeded(git_sandbox)
    _no_range_env(monkeypatch)
    monkeypatch.setenv(card.EVENT_ENV, "pull_request")
    with pytest.raises(ToolBroken):
        card.resolve_range(git_sandbox.root)


def test_an_unresolvable_base_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """base 那顆物件不在（淺 clone 的形狀）：回 2，不准當成沒改檔。"""
    head = _seeded(git_sandbox)
    monkeypatch.setenv(card.BASE_ENV, "0" * len(head))
    monkeypatch.setenv(card.HEAD_ENV, head)
    monkeypatch.delenv(card.EVENT_ENV, raising=False)
    with pytest.raises(ToolBroken):
        card.resolve_range(git_sandbox.root)


def test_an_unresolvable_head_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = _seeded(git_sandbox)
    monkeypatch.setenv(card.BASE_ENV, base)
    monkeypatch.setenv(card.HEAD_ENV, "no-such-ref-at-all")
    monkeypatch.delenv(card.EVENT_ENV, raising=False)
    with pytest.raises(ToolBroken):
        card.resolve_range(git_sandbox.root)


def test_bases_with_no_common_ancestor_are_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """兩條不相干的歷史算不出分支點——算不出可信範圍就回 2。"""
    head = _seeded(git_sandbox)
    git_sandbox.git("switch", "-q", "--orphan", "unrelated")
    stranger = _commit(git_sandbox, "另一條歷史", "README.md")
    monkeypatch.setenv(card.BASE_ENV, stranger)
    monkeypatch.setenv(card.HEAD_ENV, head)
    monkeypatch.delenv(card.EVENT_ENV, raising=False)
    with pytest.raises(ToolBroken):
        card.resolve_range(git_sandbox.root)


def test_a_local_branch_falls_back_to_the_merge_base(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """本機分支沒給 base：從跟主線的共同祖先算起，不是只看最後一筆。

    最後一筆只改文件，所以退回 ``HEAD~1..HEAD`` 的寫法會回乾淨；從共同祖先算才判得出紅。
    """
    _seeded(git_sandbox)
    git_sandbox.git("switch", "-q", "-c", "candidate")
    _commit(git_sandbox, "先改產品", PRODUCT)
    _commit(git_sandbox, "最後一筆只改文件", DOC)
    _no_range_env(monkeypatch)
    rng = card.resolve_range(git_sandbox.root)
    hits = verdict(card.diff_entries(git_sandbox.root, rng))
    assert any(PRODUCT in hit for hit in hits), f"本機分支上只量了最後一筆：{hits}"


def test_on_the_mainline_the_range_is_the_previous_mainline_commit(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """主線 push：共同祖先就是 HEAD 自己，範圍改成上一個主線提交（合併進來的那一整支）。"""
    _seeded(git_sandbox)
    _commit(git_sandbox, "主線上合進來的一筆，只改產品", PRODUCT)
    _no_range_env(monkeypatch)
    rng = card.resolve_range(git_sandbox.root)
    hits = verdict(card.diff_entries(git_sandbox.root, rng))
    assert any(PRODUCT in hit for hit in hits), f"主線那一筆沒被判到：{hits}"


def test_a_root_commit_has_no_range(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """根提交沒有父節點、也沒有分支點：回 2，不准拿「什麼都沒改」交差。"""
    _commit(git_sandbox, "根提交", PRODUCT)
    _no_range_env(monkeypatch)
    with pytest.raises(ToolBroken):
        card.resolve_range(git_sandbox.root)


def test_an_empty_commit_is_a_valid_range(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """空提交是有效範圍：差異是空的，回乾淨，不是工具壞。"""
    base = _seeded(git_sandbox)
    head = _commit(git_sandbox, "空提交")
    monkeypatch.setenv(card.BASE_ENV, base)
    monkeypatch.setenv(card.HEAD_ENV, head)
    monkeypatch.delenv(card.EVENT_ENV, raising=False)
    rng = card.resolve_range(git_sandbox.root)
    assert verdict(card.diff_entries(git_sandbox.root, rng)) == []


def test_git_missing_from_path_is_tool_broken(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """抽掉外部工具就回 2：這一跑沒量到東西，不是沒有違規。"""
    base = _seeded(git_sandbox)
    head = _commit(git_sandbox, "改產品", PRODUCT)
    monkeypatch.setenv("PATH", str(git_sandbox.root / "no-such-bin-dir"))
    with pytest.raises(ToolBroken):
        card.diff_entries(git_sandbox.root, card.Range.three_dot(base, head))


def test_rename_detection_is_on(git_sandbox: GitSandbox) -> None:
    """改名要被認出來是改名，不是「刪一個、加一個」——兩側都要判得到。"""
    base = _seeded(git_sandbox)
    body = (git_sandbox.root / PRODUCT).read_text(encoding="utf-8")
    _write(git_sandbox.root, OUTSIDE, body)
    (git_sandbox.root / PRODUCT).unlink()
    git_sandbox.git("add", "-A")
    git_sandbox.git("commit", "-q", "-m", "把引擎的檔改名搬出去")
    head = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    entries = card.diff_entries(git_sandbox.root, card.Range.three_dot(base, head))
    assert [(e.status, e.old, e.new) for e in entries] == [("R", PRODUCT, OUTSIDE)]


def test_a_real_work_tree_may_not_carry_a_fixture_declaration(
    git_sandbox: GitSandbox,
) -> None:
    """真的 git 工作樹的根放著樣本用的宣告檔一律回 2——不然擺一個檔就繞過真歷史。"""
    _seeded(git_sandbox)
    _write(git_sandbox.root, card.FIXTURE_PLAN_FILE, "commit base\nadd " + PRODUCT + "\n")
    with pytest.raises(ToolBroken):
        card.resolve_source(git_sandbox.root)


def test_a_tree_that_is_neither_a_work_tree_nor_a_fixture_is_tool_broken(
    tmp_path: Path,
) -> None:
    """既不是版控樹的根、也沒有宣告檔：拿不到範圍，回 2。"""
    with pytest.raises(ToolBroken):
        card.resolve_source(tmp_path)


# ── 第四組：初讀找出來的窟窿（interim-review.md；前兩條是別人在自己的 scratch 裡重跑過的）──


def test_parser_rejects_output_that_does_not_end_with_a_nul() -> None:
    """``-z`` 的每一格都以 NUL 收尾；沒有收尾就是輸出被截斷了。

    已被獨立重跑證實的洞：一份「狀態、一個路徑、最後少一個收尾 NUL」的輸出原本照樣被收下，
    等於「讀了半份差異還敢下結論」。
    """
    with pytest.raises(ToolBroken):
        card.parse_diff(f"M\0{PRODUCT}")


def test_parser_rejects_a_truncated_rename_second_side() -> None:
    """改名的新路徑被截掉一半（沒有收尾的 NUL）也不准收下。"""
    with pytest.raises(ToolBroken):
        card.parse_diff(f"R100\0{PRODUCT}\0{OUTSIDE}")


def test_parser_rejects_a_similarity_score_on_a_one_sided_status() -> None:
    """一側的狀態不帶相似度數字：``A123`` 是我沒看懂的形狀，不是「新增」。"""
    with pytest.raises(ToolBroken):
        card.parse_diff(f"A123\0{PRODUCT}\0")


def test_parser_rejects_a_rename_without_a_similarity_score() -> None:
    """兩側的狀態一定帶相似度數字；裸 ``R`` 是我沒看懂的形狀。"""
    with pytest.raises(ToolBroken):
        card.parse_diff(f"R\0{PRODUCT}\0{OUTSIDE}")


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
        card.resolve_range(git_sandbox.root)


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
        card.resolve_range(git_sandbox.root)


# ── 第五組：樣本宣告的路徑不准逃出樣本樹 ───────────────────────────────────
#
# 這一組把樣本樹開在 pytest 的 tmp_path（真 repo 外面），所以連「有沒有寫進受檢的樹」
# 這件事都量得到：宣告檔照著建歷史的那一步只准讀樣本樹裡的檔、只准寫暫存的那棵 repo。


def _scratch_fixture(root: Path, plan: str, files: dict[str, str]) -> Path:
    """在真 repo 外面搭一棵樣本樹：一份宣告檔，加幾個真的檔。"""
    for rel, body in files.items():
        _write(root, rel, body)
    _write(root, card.FIXTURE_PLAN_FILE, plan)
    _write(
        root,
        f"governance/rules/{card.CARD_ID}.toml",
        f'id = "{card.CARD_ID}"\n'
        "[settings]\n"
        'product_paths = ["src/"]\n'
        'test_paths = ["tests/"]\n'
        'code_suffixes = [".py"]\n',
    )
    return root


def test_a_plan_builds_a_history_from_the_real_files_in_the_tree(tmp_path: Path) -> None:
    """正向對照：合法的宣告真的建出歷史，量到的就是宣告的那一筆。

    沒有這一支，下面那幾支「逃出去要回 2」證明不了「合法的路還走得通」。
    """
    root = _scratch_fixture(
        tmp_path,
        f"commit base\nadd {PRODUCT}\nadd {EXAM}\ncommit head\nmodify {PRODUCT}\n",
        {PRODUCT: "# 產品\n", EXAM: "# 考卷\n"},
    )
    source = card.resolve_source(root)
    try:
        entries = card.diff_entries(source.work_tree, source.rng)
    finally:
        source.cleanup()
    assert [(e.status, e.new) for e in entries] == [("M", PRODUCT)]


def test_a_plan_does_not_write_into_the_fixture_tree(tmp_path: Path) -> None:
    """建歷史那一步不准動受檢的樣本樹：樣本是證據，只准讀。"""
    root = _scratch_fixture(
        tmp_path,
        f"commit base\nadd {PRODUCT}\nadd {EXAM}\ncommit head\nmodify {PRODUCT}\n",
        {PRODUCT: "# 產品\n", EXAM: "# 考卷\n"},
    )
    before = sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))
    source = card.resolve_source(root)
    try:
        assert not source.work_tree.is_relative_to(root), (
            f"暫存的歷史 {source.work_tree} 建在樣本樹裡面了——那是往受檢的樹寫檔"
        )
        card.diff_entries(source.work_tree, source.rng)
    finally:
        source.cleanup()
    assert sorted(p.relative_to(root).as_posix() for p in root.rglob("*")) == before


def test_a_plan_may_not_use_an_absolute_path(tmp_path: Path) -> None:
    """已被獨立重跑證實的洞：宣告絕對路徑，原本會去改寫樣本樹外面的檔。

    外面那個檔**刻意先建好**：不建的話這一支會因為「樣本樹裡沒有這個檔」而紅，
    那是對的結論配錯的理由，證明不了逃出去這條路被封住。
    """
    outside = tmp_path / "outside.py"
    outside.write_text("# 樣本樹外面的檔，不准被動到\n", encoding="utf-8")
    root = _scratch_fixture(
        tmp_path / "tree",
        f"commit base\nadd {PRODUCT}\ncommit head\nadd {outside}\n",
        {PRODUCT: "# 產品\n"},
    )
    with pytest.raises(ToolBroken):
        card.resolve_source(root)
    assert outside.read_text(encoding="utf-8") == "# 樣本樹外面的檔，不准被動到\n"


def test_a_plan_may_not_climb_out_with_dot_dot(tmp_path: Path) -> None:
    """``..`` 同一件事，而且外面那個檔也先建好，免得因為「檔不在」而誤紅。"""
    outside = tmp_path / "outside.py"
    outside.write_text("# 樣本樹外面的檔\n", encoding="utf-8")
    root = _scratch_fixture(
        tmp_path / "tree",
        f"commit base\nadd {PRODUCT}\ncommit head\nadd {CLIMB_OUT}\n",
        {PRODUCT: "# 產品\n"},
    )
    with pytest.raises(ToolBroken):
        card.resolve_source(root)
    assert outside.read_text(encoding="utf-8") == "# 樣本樹外面的檔\n"


def test_a_plan_may_not_touch_the_git_directory(tmp_path: Path) -> None:
    """宣告碰 ``.git`` 底下的東西一律回 2——建歷史那一步不准去動版控自己的內臟。

    那個檔在樣本樹裡**真的存在**，所以唯一還能讓它紅的理由只剩「路徑形狀不准」。
    """
    root = _scratch_fixture(
        tmp_path,
        f"commit base\nadd {PRODUCT}\ncommit head\nadd {INTO_GIT_DIR}\n",
        {PRODUCT: "# 產品\n", INTO_GIT_DIR: "# 鉤子\n"},
    )
    with pytest.raises(ToolBroken):
        card.resolve_source(root)


def test_a_plan_may_not_rename_out_of_the_fixture_tree(tmp_path: Path) -> None:
    """改名的**新**路徑也要封：只封舊路徑等於留著寫出去那一側。"""
    root = _scratch_fixture(
        tmp_path / "tree",
        f"commit base\nadd {PRODUCT}\ncommit head\nrename {PRODUCT} {CLIMB_OUT}\n",
        {PRODUCT: "# 產品\n"},
    )
    with pytest.raises(ToolBroken):
        card.resolve_source(root)


def test_a_plan_may_not_read_through_a_symlink_out_of_the_tree(tmp_path: Path) -> None:
    """宣告一條指到樣本樹外面的連結：解析之後跑出樣本樹就回 2。"""
    elsewhere = tmp_path / "outsider.py"
    elsewhere.write_text("# 樣本樹外面的檔\n", encoding="utf-8")
    root = _scratch_fixture(
        tmp_path / "tree",
        f"commit base\nadd {PRODUCT}\ncommit head\nadd {LINK_IN_TREE}\n",
        {PRODUCT: "# 產品\n"},
    )
    (root / LINK_IN_TREE).symlink_to(elsewhere)
    with pytest.raises(ToolBroken):
        card.resolve_source(root)


# ── 第六組：產品側／測試側住在哪，是卡上登記的 ─────────────────────────────


def test_the_prefixes_come_from_the_card_not_from_the_program(tmp_path: Path) -> None:
    """卡上把產品側登記成別的目錄，判準就跟著換家——程式裡沒有預設值。"""
    _write(
        tmp_path,
        f"governance/rules/{card.CARD_ID}.toml",
        f'id = "{card.CARD_ID}"\n'
        "[settings]\n"
        'product_paths = ["engine/"]\n'
        'test_paths = ["exams/"]\n'
        'code_suffixes = [".py"]\n',
    )
    files = [tmp_path / f"governance/rules/{card.CARD_ID}.toml"]
    rules = card.read_rules(tmp_path, files)
    assert card.problems([modified(OTHER_PRODUCT)], rules), "卡上登記的產品側沒被採用"
    assert card.problems([modified(PRODUCT)], rules) == [], "程式裡還留著 src/ 的預設值"


def test_a_missing_card_is_tool_broken(tmp_path: Path) -> None:
    """讀不到卡上的登記簿就回 2：判準沒有第二個家。"""
    with pytest.raises(ToolBroken):
        card.read_rules(tmp_path, [])


def test_a_card_with_an_incomplete_registry_is_tool_broken(tmp_path: Path) -> None:
    """登記簿缺一格（打錯字的名單等於沒名單）也回 2。"""
    _write(
        tmp_path,
        f"governance/rules/{card.CARD_ID}.toml",
        f'id = "{card.CARD_ID}"\n[settings]\nproduct_paths = ["src/"]\n',
    )
    with pytest.raises(ToolBroken):
        card.read_rules(tmp_path, [tmp_path / f"governance/rules/{card.CARD_ID}.toml"])


def test_the_real_card_registers_both_sides() -> None:
    """今天樹上那張卡真的把 src/ 與 tests/ 登記進去了（上面那一組的前提）。"""
    from governance.exit_codes import enumerate_files, repo_root

    root = repo_root()
    rules = card.read_rules(root, enumerate_files(root))
    assert (rules.product_paths, rules.test_paths) == (("src/",), ("tests/",))


# ── 第七組：真樹裡的宣告檔只准住在卡宣告的樣本樹底下 ───────────────────────


def test_a_declaration_inside_a_declared_fixture_tree_is_fine(tmp_path: Path) -> None:
    home = "governance/fixtures/tests-land-with-code"
    _write(
        tmp_path,
        f"governance/rules/{card.CARD_ID}.toml",
        f'id = "{card.CARD_ID}"\nnegative_fixture = "{home}"\n',
    )
    plan = tmp_path / home / "control" / card.FIXTURE_PLAN_FILE
    _write(tmp_path, f"{home}/control/{card.FIXTURE_PLAN_FILE}", "commit base\n")
    files = [tmp_path / f"governance/rules/{card.CARD_ID}.toml", plan]
    assert card.stray_declarations(tmp_path, files) == []


def test_a_declaration_parked_outside_the_fixture_trees_is_a_violation(tmp_path: Path) -> None:
    """把宣告檔擺在別處，就等於真歷史被一個檔案繞過。"""
    home = "governance/fixtures/tests-land-with-code"
    _write(
        tmp_path,
        f"governance/rules/{card.CARD_ID}.toml",
        f'id = "{card.CARD_ID}"\nnegative_fixture = "{home}"\n',
    )
    stray = tmp_path / "somewhere" / card.FIXTURE_PLAN_FILE
    _write(tmp_path, f"somewhere/{card.FIXTURE_PLAN_FILE}", "commit base\n")
    files = [tmp_path / f"governance/rules/{card.CARD_ID}.toml", stray]
    assert card.stray_declarations(tmp_path, files), "擺在樣本樹外面的宣告檔沒被咬"
