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


def _answer(cases: list[int]) -> str:
    return json.dumps({"schema": "fem-fenics-answers/v1", "cases": cases, "provenance": {"generated_at": "t0"}})


@pytest.mark.parametrize("old_name,new_name,managed", [
    ("fem_fenics_answers.json", "fem_fenics_answers.txt", True),
    ("fem_fenics_answers.json", "fem_fenics_answers.json.bak", True),
    ("solver_notes.json", "solver_notes.json.bak", True),
    ("notes.json", "notes.txt", False),
    # 改名成另一個 .json 還在管，不算脫管：照原本的「cases 變了身分要跟著變」那條判。
    ("fem_fenics_answers.json", "fem_fenics_answers_v2.json", False),
], ids=["managed-by-name-to-txt", "managed-by-name-to-bak", "managed-by-schema-to-bak", "unmanaged-json",
        "managed-json-to-json"])
def test_renaming_a_managed_answer_away_from_json_is_red(
        git_sandbox: GitSandbox, old_name: str, new_name: str, managed: bool) -> None:
    """受管答案在同一個範圍裡改名成非 .json，「只讀 .json」那條會放掉它，等於改名脫管（#322）。"""
    root = git_sandbox.root
    (root / "blueprint").mkdir()
    old = root / "blueprint" / old_name
    old.write_text(_answer([1]) if managed else json.dumps({"schema": "something-else/v1"}), encoding="utf-8")
    git_sandbox.git("add", "blueprint")
    git_sandbox.git("commit", "-q", "-m", "base")
    base = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    git_sandbox.git("mv", f"blueprint/{old_name}", f"blueprint/{new_name}")
    git_sandbox.git("commit", "-q", "-m", "head")
    head = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    rng = card.CommitRange(work_tree=root, base=base, head=head, label="sandbox")
    hits = card._range_hits(rng, _rules_for_range_test())
    assert hits == ([f"blueprint/{old_name} 是受管答案，改名成 blueprint/{new_name} 就脫管了；"
                     "要退休就刪掉，要留就留 .json"] if managed else [])
