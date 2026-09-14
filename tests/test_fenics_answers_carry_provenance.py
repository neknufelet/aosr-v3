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
