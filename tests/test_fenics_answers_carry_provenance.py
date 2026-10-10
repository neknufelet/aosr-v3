"""FEniCS 答案卡不能讓樣本宣告檔接管真的提交範圍。"""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from governance import exit_codes
from governance.checks import fenics_answers_carry_provenance as card
from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION

from tests.conftest import GitSandbox


REPO = Path(__file__).resolve().parents[1]
NEEDED = (
    "blueprint/fem_fenics_answers.json",
    "blueprint/fem_fenics_problem.json",
    "governance/rules/fenics-answers-carry-provenance.toml",
)


def _copy(root: Path, rel: str) -> None:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((REPO / rel).read_bytes())


@pytest.mark.parametrize(
    "declaration_rel",
    (
        "governance/fixture-fenics-answer-range.txt",
        "governance/fixture-fenics-answer-base-cases.json",
    ),
)
@pytest.mark.parametrize("content", ["same\n", "same\nbase_card = '''[settings]'''\n"])
def test_real_work_tree_fixture_declaration_is_tool_broken(
    git_sandbox: GitSandbox,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    declaration_rel: str,
    content: str,
) -> None:
    """真工作樹擺任一份樣本宣告都不能關掉規則 5，外殼必須回 2。"""
    for rel in NEEDED:
        _copy(git_sandbox.root, rel)
    declaration = git_sandbox.root / declaration_rel
    declaration.write_text(content, encoding="utf-8")
    monkeypatch.setattr(exit_codes, "repo_root", lambda: git_sandbox.root)

    result = exit_codes.run(
        card.check,
        ["--scan-root", str(git_sandbox.root)],
        targets=card.targets,
    )

    assert result == TOOL_BROKEN
    assert "宣告檔只准活在樣本迷你樹" in capsys.readouterr().err


def _rules_for_range_test() -> card.Rules:
    return card.Rules(
        patterns=("fem_fenics_answers*.json",),
        schema_field="schema",
        schema_prefix="fem-fenics-answers/",
        provenance_field="provenance",
        cases_field="cases",
        required_fields=("producer",),
        image_digest="sha256:0",
        rerun_fields=("generated_at",),
        problem_file_field="problem_file",
        problem_sha256_field="problem_sha256",
        image_digest_field="image_digest",
        range_base_env="AOSR_RANGE_BASE",
        range_head_env="AOSR_RANGE_HEAD",
        fixture_range_file="governance/fixture-fenics-answer-range.txt",
        fixture_base_cases_file="governance/fixture-fenics-answer-base-cases.json",
    )


def test_range_ignores_non_json_files_under_blueprint(git_sandbox: GitSandbox) -> None:
    """提交範圍裡改到 blueprint/ 的 .py（產生器、獨立檢查程式）不是改答案，不准拿去剖 JSON 判成 2。"""
    root = git_sandbox.root
    (root / "blueprint").mkdir()
    tool = root / "blueprint" / "reference_something_check.py"
    tool.write_text("X = 1\n", encoding="utf-8")
    git_sandbox.git("add", "blueprint")
    git_sandbox.git("commit", "-q", "-m", "base")
    base = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    tool.write_text("X = 2\n", encoding="utf-8")
    git_sandbox.git("commit", "-q", "-am", "head")
    head = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    rng = card.CommitRange(work_tree=root, base=base, head=head, label="sandbox")
    assert card._range_hits(rng, _rules_for_range_test()) == []



def _rel(*parts: str) -> str:
    """沙箱裡的路徑分段寫：整串寫在字串裡會被引用檢查當成這棵樹裡的真引用（它解析不到就紅）。"""
    return "/".join(parts)


def _answer(cases: list[float], generated_at: str = "t0", schema: str = "fem-fenics-answers/v1",
            problem: str = "p") -> str:
    """整份一行的答案（跟真的答案檔一樣），題目身分由 problem 決定：改一個數字 git 的相似度就認不出改名。"""
    provenance = {"generated_at": generated_at, "problem_file": _rel("blueprint", f"{problem}.json"),
                  "problem_sha256": problem.encode().hex().ljust(64, "0"), "padding": "x" * 64}
    return json.dumps({"schema": schema, "cases": cases, "provenance": provenance})


def _range_after(git_sandbox: GitSandbox, base_files: dict[str, str],
                 change: dict[str, str | bytes | None], rules: card.Rules | None = None) -> list[str]:
    """base 提交 base_files；head 照 change 改（值是新內容，bytes 照原位元組寫，None 是刪掉），回規則 5 的範圍命中。"""
    root = git_sandbox.root
    for rel, text in base_files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    git_sandbox.git("add", "-A")
    git_sandbox.git("commit", "-q", "-m", "base")
    base = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    for rel, new_text in change.items():
        if new_text is None:
            (root / rel).unlink()
            continue
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(new_text, bytes):
            (root / rel).write_bytes(new_text)
        else:
            (root / rel).write_text(new_text, encoding="utf-8")
    git_sandbox.git("add", "-A")
    git_sandbox.git("commit", "-q", "-m", "head")
    head = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    return card._range_hits(card.CommitRange(work_tree=root, base=base, head=head, label="sandbox"),
                            rules or _rules_for_range_test(), card._fixture_trees(CARD))


OLD = _rel("blueprint", "fem_fenics_answers.json")
SECOND = _rel("blueprint", "fem_fenics_answers_b.json")
CARD = REPO / "governance" / "rules" / "fenics-answers-carry-provenance.toml"
ESCAPED = "{old} 是受管答案，在這個範圍裡搬成 {new} 就脫管了（受管的是 blueprint 底下的 .json）；要退休就刪掉，要留就留在 blueprint 底下的 .json"
RERUN = "{rel} 的 cases 已變，但重錄身分三格全都沒變"
IN_PLACE = ("{rel} 原地改成不再是受管答案（schema 改掉、檔名又不命中樣式），題目身分還在，就脫管了；"
            "要退休就刪掉，要留就留受管的 schema")
SOLVER = _rel("blueprint", "solver_notes.json")
SOLVER_B = _rel("blueprint", "solver_notes_b.json")


@pytest.mark.parametrize("change,expected", [
    # 改一個數字、重錄身分沒動，同時改名或搬家：git 認不出改名，以前會變成「刪一份、加一份」溜過去（#322 複查）。
    ({OLD: None, _rel("blueprint", "fem_fenics_answers_v2.json"): _answer([1.25])}, [RERUN.format(rel=_rel("blueprint", "fem_fenics_answers_v2.json"))]),
    ({OLD: None, _rel("blueprint", "sub", "fem_fenics_answers.json"): _answer([1.25])}, [RERUN.format(rel=_rel("blueprint", "sub", "fem_fenics_answers.json"))]),
    # 改名成非 .json、搬出 blueprint：題目身分配得到就是脫管。
    ({OLD: None, _rel("blueprint", "fem_fenics_answers.txt"): _answer([1.0])}, [ESCAPED.format(old=OLD, new=_rel("blueprint", "fem_fenics_answers.txt"))]),
    ({OLD: None, _rel("data", "fem_fenics_answers.json"): _answer([1.0])}, [ESCAPED.format(old=OLD, new=_rel("data", "fem_fenics_answers.json"))]),
    # 照規矩重錄（重錄身分有動）就不紅；只改名不改內容也不紅；刪掉就是退休。
    ({OLD: None, _rel("blueprint", "fem_fenics_answers_v2.json"): _answer([1.25], generated_at="t1")}, []),
    ({OLD: None, _rel("blueprint", "fem_fenics_answers_v2.json"): _answer([1.0])}, []),
    ({OLD: None}, []),
    # 原地改數字、重錄身分沒動：原本的規則 5。
    ({OLD: _answer([1.25])}, [RERUN.format(rel=OLD)]),
    # 舊的留著、另開一份 v2 改數字不重錄：一樣配得到（複查）。
    ({_rel("blueprint", "fem_fenics_answers_v2.json"): _answer([1.25])},
     [RERUN.format(rel=_rel("blueprint", "fem_fenics_answers_v2.json"))]),
    # 退休答案時同一個範圍加了必紅樣本（道具答案寫同一個題目檔）：不是搬家（複查抓到的誤紅）。
    ({OLD: None, _rel("governance", "fixtures", "fenics-answers-carry-provenance", "case-new", "blueprint",
                      "fem_fenics_answers.json"): _answer([1.0])}, []),
    # 只排除這張卡自己的樣本樹：搬進別張卡的樣本資料夾一樣是脫管（複查）。
    ({OLD: None, _rel("governance", "fixtures", "other-card", "case", "fem_fenics_answers.json"): _answer([1.0])},
     [ESCAPED.format(old=OLD, new=_rel("governance", "fixtures", "other-card", "case", "fem_fenics_answers.json"))]),
    # 路徑帶引號（git 不加 -z 會把它加引號）：照樣配得到、照樣比。
    ({OLD: None, _rel("blueprint", 'we"ird.json'): _answer([1.25])}, [RERUN.format(rel=_rel("blueprint", 'we"ird.json'))]),
], ids=["rename-json-with-new-number", "move-to-subdir-with-new-number", "rename-to-txt", "move-out-of-blueprint",
        "rerecorded", "pure-rename", "retired", "modified-in-place", "copy-without-delete",
        "retire-while-adding-a-fixture", "move-into-another-cards-fixtures", "quoted-path"])
def test_moved_or_renamed_answers_are_paired_by_problem_identity(
        git_sandbox: GitSandbox, change: dict[str, str | bytes | None], expected: list[str]) -> None:
    assert _range_after(git_sandbox, {OLD: _answer([1.0])}, change) == expected


def test_changing_the_schema_to_escape_is_still_compared(git_sandbox: GitSandbox) -> None:
    """只靠 schema 認的受管答案，範圍裡把 schema 改掉、同時改數字：base 那邊受管就照規則 5 比；出身還在，另外報原地脫管。"""
    rel = _rel("blueprint", "solver_notes.json")
    hits = _range_after(git_sandbox, {rel: _answer([1.0])}, {rel: _answer([1.25], schema="something-else/v1")})
    assert hits == [RERUN.format(rel=rel), IN_PLACE.format(rel=rel)]


def test_list_shaped_blueprint_json_is_not_an_answer(git_sandbox: GitSandbox) -> None:
    """blueprint 底下頂層是清單的 json（rules-436 那類）被改到：不是答案，不准把整跑判成 2。"""
    rel = _rel("blueprint", "rules-436.json")
    assert _range_after(git_sandbox, {rel: json.dumps([1, 2])}, {rel: json.dumps([1, 2, 3])}) == []


def test_unrelated_added_files_are_not_paired(git_sandbox: GitSandbox) -> None:
    """退休一份答案、同一個範圍另外加了不相干的檔（二進位、別的題目）：不配對、不紅。"""
    other = json.dumps({"schema": "fem-fenics-answers/v1", "cases": [9.0],
                        "provenance": {"generated_at": "t0", "problem_file": _rel("blueprint", "q.json"), "problem_sha256": "cd" * 32}})
    change: dict[str, str | bytes | None] = {OLD: None, _rel("blueprint", "fem_fenics_answers_q.json"): other,
                                             _rel("docs", "x.bin"): b"\x00\xff\xfe"}
    assert _range_after(git_sandbox, {OLD: _answer([1.0])}, change) == []


def test_two_answers_of_one_problem_moved_together_are_not_cross_compared(git_sandbox: GitSandbox) -> None:
    """同一題、同一次跑的兩份答案原封不動一起搬：各自配得到 cases 相同的那一份，不准拿去跟另一份比（複查）。"""
    second = SECOND
    change: dict[str, str | bytes | None] = {
        OLD: None, second: None,
        _rel("blueprint", "sub", "fem_fenics_answers.json"): _answer([1.0]),
        _rel("blueprint", "sub", "fem_fenics_answers_b.json"): _answer([2.0]),
    }
    assert _range_after(git_sandbox, {OLD: _answer([1.0]), second: _answer([2.0])}, change) == []


def test_moving_an_answer_into_an_existing_file_outside_is_an_escape(git_sandbox: GitSandbox) -> None:
    """刪掉答案、把內容寫進範圍外一份已經存在的檔（改動不是新增）：一樣是搬家脫管（複查）。"""
    existing = _rel("data", "x.json")
    hits = _range_after(git_sandbox, {OLD: _answer([1.0]), existing: json.dumps({"k": 1})},
                        {OLD: None, existing: _answer([1.0])})
    assert hits == [ESCAPED.format(old=OLD, new=existing)]


def test_rerecording_in_place_is_not_compared_with_a_same_day_sibling(git_sandbox: GitSandbox) -> None:
    """原地重錄只跟同一路徑那一份比：同一題另一份剛好也是今天錄的（重錄時間只記到日期），不准拿它把正式重錄判紅。"""
    sibling = _rel("blueprint", "fem_fenics_answers_b.json")
    base = {OLD: _answer([1.0], generated_at="t0"), sibling: _answer([2.0], generated_at="t1")}
    assert _range_after(git_sandbox, base, {OLD: _answer([1.5], generated_at="t1")}) == []



@pytest.mark.parametrize("second_problem", ["p", "q"], ids=["same-problem", "other-problem"])
@pytest.mark.parametrize("swap_back", [True, False], ids=["swapped", "deleted-and-overwritten"])
def test_swapping_answers_cannot_hide_new_numbers(
        git_sandbox: GitSandbox, second_problem: str, swap_back: bool) -> None:
    """兩份受管答案對調、或刪一份把它的出身蓋到另一份上，再順手改數字不重錄：原地改的也要跟「內容離開原位」的那幾份比，
    不然同路徑那份的重錄時間不同，就把只換數字藏過去了（複查抓到上一版的退步）。"""
    first_base = _answer([1.0], generated_at="t0")
    second_base = _answer([2.0], generated_at="t1", problem=second_problem)
    change: dict[str, str | bytes | None] = {OLD: second_base if swap_back else None,
                                             SECOND: _answer([1.5], generated_at="t0")}
    assert _range_after(git_sandbox, {OLD: first_base, SECOND: second_base}, change) == [RERUN.format(rel=SECOND)]


THIRD = _rel("blueprint", "fem_fenics_answers_c.json")


def test_swapping_only_the_numbers_is_red(git_sandbox: GitSandbox) -> None:
    """只對調數字、出身各自留在原位：數字對得上兄弟、出身對得上自己，但沒有一份候選兩樣都相同，就是只換數字（複查）。"""
    base = {OLD: _answer([1.0], generated_at="t0"), SECOND: _answer([2.0], generated_at="t1")}
    change: dict[str, str | bytes | None] = {OLD: _answer([2.0], generated_at="t0"),
                                             SECOND: _answer([1.0], generated_at="t1")}
    assert _range_after(git_sandbox, base, change) == [RERUN.format(rel=OLD), RERUN.format(rel=SECOND)]


def test_retiring_one_and_moving_its_numbers_into_another_is_red(git_sandbox: GitSandbox) -> None:
    base = {OLD: _answer([1.0], generated_at="t0"), SECOND: _answer([2.0], generated_at="t1")}
    change: dict[str, str | bytes | None] = {SECOND: None, OLD: _answer([2.0], generated_at="t0")}
    assert _range_after(git_sandbox, base, change) == [RERUN.format(rel=OLD)]


def test_rotating_only_the_numbers_of_three_answers_is_red(git_sandbox: GitSandbox) -> None:
    base = {OLD: _answer([1.0], generated_at="t0"), SECOND: _answer([2.0], generated_at="t1"),
            THIRD: _answer([3.0], generated_at="t2")}
    change: dict[str, str | bytes | None] = {OLD: _answer([2.0], generated_at="t0"), SECOND: _answer([3.0], generated_at="t1"),
                                             THIRD: _answer([1.0], generated_at="t2")}
    assert _range_after(git_sandbox, base, change) == [RERUN.format(rel=rel) for rel in (OLD, SECOND, THIRD)]


def test_touching_a_sibling_without_changing_it_does_not_make_it_a_source(git_sandbox: GitSandbox) -> None:
    """今天重錄 A，同一支合併請求順手碰了今天錄的兄弟 B、但它的數字與重錄身分都沒變（只加一個欄位）：B 的內容沒離開原位，
    不准拿它把 A 的正式重錄判紅（複查）。"""
    sibling = json.loads(_answer([2.0], generated_at="t1"))
    base = {OLD: _answer([1.0], generated_at="t0"), SECOND: json.dumps(sibling)}
    change: dict[str, str | bytes | None] = {OLD: _answer([1.5], generated_at="t1"),
                                             SECOND: json.dumps(sibling | {"note": "加註"})}
    assert _range_after(git_sandbox, base, change) == []



def test_an_answer_rewritten_as_a_list_and_moved_out_is_an_escape(git_sandbox: GitSandbox) -> None:
    """只靠 schema 認的受管答案（檔名不命中樣式）原地改成清單，內容配新數字搬到範圍外：一樣是搬家脫管（#643）。"""
    outside = _rel("data", "solver.json")
    hits = _range_after(git_sandbox, {SOLVER: _answer([1.0])}, {SOLVER: json.dumps([1, 2]), outside: _answer([1.25])})
    assert hits == [ESCAPED.format(old=SOLVER, new=outside)]


@pytest.mark.parametrize("moved_out", [False, True], ids=["stays", "also-moved-out"])
def test_changing_the_schema_in_place_with_identity_kept_is_an_escape(git_sandbox: GitSandbox, moved_out: bool) -> None:
    """原地把 schema 改掉、出身還在：檔還在原位、下一支合併請求就能隨便改數字，本身就是脫管（#643 複查）。"""
    change: dict[str, str | bytes | None] = {SOLVER: _answer([1.0], schema="archive/v1", generated_at="t9")}
    if moved_out:
        change[_rel("data", "solver.json")] = _answer([1.25])
    assert _range_after(git_sandbox, {SOLVER: _answer([1.0])}, change) == [IN_PLACE.format(rel=SOLVER)]


def test_same_problem_answers_changed_in_place_are_not_blamed_on_each_other(git_sandbox: GitSandbox) -> None:
    """兩份同題答案各自原地改 schema：各報原地脫管，不准互指「搬成」對方（#643 複查）。"""
    base = {SOLVER: _answer([1.0]), SOLVER_B: _answer([2.0])}
    change: dict[str, str | bytes | None] = {SOLVER: _answer([1.0], schema="archive/v1"),
                                             SOLVER_B: _answer([2.0], schema="archive/v1")}
    assert _range_after(git_sandbox, base, change) == [IN_PLACE.format(rel=SOLVER), IN_PLACE.format(rel=SOLVER_B)]


def test_an_answer_rewritten_as_a_list_with_nothing_moved_is_not_an_escape(git_sandbox: GitSandbox) -> None:
    """原地改成清單、內容沒搬到任何地方：沒有脫管可抓（靜態那一層管不管得到另論）。"""
    assert _range_after(git_sandbox, {SOLVER: _answer([1.0])}, {SOLVER: json.dumps([1, 2])}) == []


def test_a_list_rewrite_is_not_blamed_on_a_sibling_changed_in_place(git_sandbox: GitSandbox) -> None:
    """A 原地改成清單、同題兄弟 B 原地改 schema（檔裡還帶題目身分）：B 報原地脫管，不准把 A 說成「搬成 B」（#643 複查）。"""
    base = {SOLVER: _answer([1.0]), SOLVER_B: _answer([2.0])}
    change: dict[str, str | bytes | None] = {SOLVER: json.dumps([1, 2]), SOLVER_B: _answer([2.0], schema="archive/v1")}
    assert _range_after(git_sandbox, base, change) == [IN_PLACE.format(rel=SOLVER_B)]


CARD_REL = "governance/rules/fenics-answers-carry-provenance.toml"
FIXTURES = REPO / "governance" / "fixtures" / "fenics-answers-carry-provenance"
PREFIX_CASE = FIXTURES / "case-card-prefix-and-cases-changed"
PREFIX_CONTROL = FIXTURES / "control" / "schema-migration"


@pytest.mark.parametrize("registration", ["prefix", "patterns"])
@pytest.mark.parametrize("answer_change", ["numbers", "escape", "card-only"])
def test_base_classification_uses_the_base_card(
        git_sandbox: GitSandbox, registration: str, answer_change: str) -> None:
    """把 base 也交給 head 卡分類就漏抓；改前綴、改檔名樣式兩邊都考，包含只改卡造成的脫管。"""
    base_card = (PREFIX_CONTROL / CARD_REL).read_text(encoding="utf-8")
    base_card = base_card.replace('answer_schema_prefix = "migrated-fenics/"',
                                  'answer_schema_prefix = "fem-fenics-answers/"')
    if registration == "prefix":
        rel, base_schema = SOLVER, "fem-fenics-answers/v1"
        head_rules = replace(_rules_for_range_test(), schema_prefix="migrated-fenics/")
        head_card = base_card.replace('answer_schema_prefix = "fem-fenics-answers/"',
                                      'answer_schema_prefix = "migrated-fenics/"')
    else:
        rel, base_schema = OLD, "archive/v1"
        head_rules = replace(_rules_for_range_test(), patterns=("new_answers*.json",))
        head_card = base_card.replace('answer_file_patterns = ["fem_fenics_answers*.json"]',
                                      'answer_file_patterns = ["new_answers*.json"]')
    change: dict[str, str | bytes | None] = {CARD_REL: head_card}
    expected = [IN_PLACE.format(rel=rel)]
    if answer_change == "numbers":
        change[rel] = _answer([1.25], schema=base_schema)
        expected.insert(0, RERUN.format(rel=rel))
    elif answer_change == "escape":
        outside = _rel("data", "answer.json")
        change.update({rel: None, outside: _answer([1.25], schema=base_schema)})
        expected = [ESCAPED.format(old=rel, new=outside)]
    assert _range_after(git_sandbox, {CARD_REL: base_card, rel: _answer([1.0], schema=base_schema)},
                        change, head_rules) == expected


def test_missing_base_card_falls_back_to_head_rules(git_sandbox: GitSandbox) -> None:
    """base 沒卡也有舊答案；若當成沒有受管答案，新增卡同時改數字就漏抓。"""
    assert _range_after(git_sandbox, {SOLVER: _answer([1.0])},
                        {CARD_REL: CARD.read_text(encoding="utf-8"), SOLVER: _answer([1.25])}) == [RERUN.format(rel=SOLVER)]


@pytest.mark.parametrize("base_card", [
    '[settings\n', '[settings]\n', 'settings = []\n',
    CARD.read_text(encoding="utf-8").replace('answer_schema_prefix = "fem-fenics-answers/"',
                                             'answer_schema_prefix = 7'),
    b'\xff',
], ids=["invalid-toml", "missing-fields", "settings-not-table", "invalid-prefix-type", "invalid-utf8"])
def test_unreadable_base_card_is_tool_broken_even_without_answer_changes(
        git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch, base_card: str | bytes) -> None:
    """base 卡存在卻解析不了、缺欄位或型別錯，不准在沒有答案差異時跳過，也不准回 0。"""
    for rel in NEEDED:
        _copy(git_sandbox.root, rel)
    path = git_sandbox.root / CARD_REL
    path.write_bytes(base_card.encode() if isinstance(base_card, str) else base_card)
    git_sandbox.git("add", "-A")
    git_sandbox.git("commit", "-q", "-m", "base")
    base = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    _copy(git_sandbox.root, CARD_REL)
    git_sandbox.git("commit", "-q", "-am", "head")
    monkeypatch.setenv("AOSR_RANGE_BASE", base)
    monkeypatch.setenv("AOSR_RANGE_HEAD", "HEAD")
    monkeypatch.setattr(exit_codes, "repo_root", lambda: git_sandbox.root)
    assert exit_codes.run(card.check, ["--scan-root", str(git_sandbox.root)], targets=card.targets) == TOOL_BROKEN


def test_prefix_change_with_new_numbers_fixture_is_red() -> None:
    assert exit_codes.run(card.check, ["--scan-root", str(PREFIX_CASE)], targets=card.targets) == VIOLATION


def test_prefix_migration_control_is_green() -> None:
    """只遷移前綴與 schema，數字與重錄身分都保留，必須回 0。"""
    assert exit_codes.run(card.check, ["--scan-root", str(PREFIX_CONTROL)], targets=card.targets) == CLEAN


def _copy_prefix_control(git_sandbox: GitSandbox) -> Path:
    target = git_sandbox.root / "fixture"
    for source in PREFIX_CONTROL.rglob("*"):
        if source.is_file():
            dest = target / source.relative_to(PREFIX_CONTROL)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
    return target


def test_prefix_migration_control_with_changed_numbers_is_red(
        git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    """綠色控制組植入新數字，不能仍然綠。"""
    root = _copy_prefix_control(git_sandbox)
    answer = root / SOLVER
    data = json.loads(answer.read_text(encoding="utf-8"))
    data["cases"] = {"flat": [2], "lowabs": [2]}
    answer.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(exit_codes, "repo_root", lambda: git_sandbox.root)
    assert exit_codes.run(card.check, ["--scan-root", str(root)], targets=card.targets) == VIOLATION


@pytest.mark.parametrize("extra", ['base_card = 1\n', 'base_schema = []\n', 'unknown = "x"\n',
                                    'base_card = ""\n', 'base_schema = ""\n', '[broken\n'])
def test_malformed_extended_fixture_declaration_is_tool_broken(
        git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch, extra: str) -> None:
    root = _copy_prefix_control(git_sandbox)
    declaration = root / "governance" / "fixture-fenics-answer-range.txt"
    declaration.write_text("same\n" + extra, encoding="utf-8")
    monkeypatch.setattr(exit_codes, "repo_root", lambda: git_sandbox.root)
    assert exit_codes.run(card.check, ["--scan-root", str(root)], targets=card.targets) == TOOL_BROKEN


def test_invalid_base_cases_declaration_is_tool_broken(
        git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _copy_prefix_control(git_sandbox)
    declaration = root / "governance" / "fixture-fenics-answer-range.txt"
    declaration.write_text("base-cases\n", encoding="utf-8")
    base_cases = root / "governance" / "fixture-fenics-answer-base-cases.json"
    base_cases.write_text("{\n", encoding="utf-8")
    monkeypatch.setattr(exit_codes, "repo_root", lambda: git_sandbox.root)
    assert exit_codes.run(card.check, ["--scan-root", str(root)], targets=card.targets) == TOOL_BROKEN
