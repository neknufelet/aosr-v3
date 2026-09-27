"""必要檢查的觸發與跳過條件必須由卡上設定守住。"""
from __future__ import annotations

from pathlib import Path

import pytest

from governance.checks import ci_jobs_cannot_die_quietly as ci

ROOT = Path(__file__).resolve().parents[1]


def _case(tmp_path: Path, mutation: str) -> list[str]:
    root = tmp_path / "tree"
    (root / "governance/rules").mkdir(parents=True)
    (root / "governance/rules/ci-jobs-cannot-die-quietly.toml").write_text(
        (ROOT / "governance/rules/ci-jobs-cannot-die-quietly.toml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "governance/required-status-checks.txt").write_text("verify\nacceptance\n", encoding="utf-8")
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/acceptance.yml").write_text(
        (ROOT / ".github/workflows/acceptance.yml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    workflow = root / ".github/workflows/verify.yml"
    text = """name: verify
on:
  pull_request:
    types: [opened, synchronize, reopened]
jobs:
  verify:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - run: uv run python -m governance.status.record_step --name pytest --out-dir governance/receipts/steps -- uv run pytest
"""
    (root / ".github/workflows/verify.yml").write_text(text, encoding="utf-8")
    if mutation == "missing-edited":
        workflow = root / ".github/workflows/acceptance.yml"
        text = workflow.read_text(encoding="utf-8")
    if mutation == "missing-edited":
        text = text.replace("reopened, edited", "reopened")
    elif mutation == "job-if":
        text = text.replace("    runs-on:", "    if: false\n    runs-on:", 1)
    elif mutation == "paths":
        text = text.replace("  pull_request:\n", "  pull_request:\n    paths: ['src/**']\n", 1)
    elif mutation == "branches":
        text = text.replace("  pull_request:\n", "  pull_request:\n    branches: [main]\n", 1)
    workflow.write_text(text, encoding="utf-8")
    return ci.check(root, list(root.rglob("*")))


@pytest.mark.parametrize("mutation", ["missing-edited", "job-if", "paths", "branches"])
def test_required_job_cannot_skip_pull_request(tmp_path: Path, mutation: str) -> None:
    assert _case(tmp_path, mutation)


def test_required_jobs_with_registered_events_pass(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "governance/rules").mkdir(parents=True)
    (root / "governance/rules/ci-jobs-cannot-die-quietly.toml").write_text(
        (ROOT / "governance/rules/ci-jobs-cannot-die-quietly.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "governance/required-status-checks.txt").write_text("acceptance\n", encoding="utf-8")
    (root / ".github/workflows").mkdir(parents=True)
    (root / ".github/workflows/acceptance.yml").write_text(
        (ROOT / ".github/workflows/acceptance.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert ci.check(root, list(root.rglob("*"))) == []


def test_acceptance_card_stays_visible_on_status_page() -> None:
    from governance.status.model import RuleCard
    from governance.status.render import cards_block

    page = cards_block((RuleCard("acceptance-verdict-points-at-head", "驗收行由雲端核。", (), "", "acceptance"),), 0)
    assert "acceptance-verdict-points-at-head" in page
    assert "acceptance job 判，只跑合併請求" in page
