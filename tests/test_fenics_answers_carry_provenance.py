"""FEniCS 答案卡不能讓樣本宣告檔接管真的提交範圍。"""
import json
from pathlib import Path

import pytest

from governance import exit_codes
from governance.checks import fenics_answers_carry_provenance as card
from governance.exit_codes import TOOL_BROKEN

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
def test_real_work_tree_fixture_declaration_is_tool_broken(
    git_sandbox: GitSandbox,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    declaration_rel: str,
) -> None:
    """真工作樹擺任一份樣本宣告都不能關掉規則 5，外殼必須回 2。"""
    for rel in NEEDED:
        _copy(git_sandbox.root, rel)
    declaration = git_sandbox.root / declaration_rel
    declaration.write_text("same\n", encoding="utf-8")
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
                 change: dict[str, str | bytes | None]) -> list[str]:
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
                            _rules_for_range_test(), card._fixture_trees(CARD))


OLD = _rel("blueprint", "fem_fenics_answers.json")
SECOND = _rel("blueprint", "fem_fenics_answers_b.json")
CARD = REPO / "governance" / "rules" / "fenics-answers-carry-provenance.toml"
ESCAPED = "{old} 是受管答案，在這個範圍裡搬成 {new} 就脫管了（受管的是 blueprint 底下的 .json）；要退休就刪掉，要留就留在 blueprint 底下的 .json"
RERUN = "{rel} 的 cases 已變，但重錄身分三格全都沒變"


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
    """只靠 schema 認的受管答案，範圍裡把 schema 改掉、同時改數字：base 那邊受管就照規則 5 比。"""
    rel = _rel("blueprint", "solver_notes.json")
    hits = _range_after(git_sandbox, {rel: _answer([1.0])}, {rel: _answer([1.25], schema="something-else/v1")})
    assert hits == [RERUN.format(rel=rel)]


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
