"""FEniCS 答案卡不能讓樣本宣告檔接管真的提交範圍。"""
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
