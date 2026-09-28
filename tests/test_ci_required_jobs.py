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
    from governance.loader import load_all_cards
    from governance.status.collect import read_rule_cards
    from governance.status.render import cards_block

    source = next(
        card for card in load_all_cards(ROOT)
        if card.check_module == "governance.checks.acceptance_verdict_points_at_head"
    )
    cards = read_rule_cards(ROOT, {})
    card = next(card for card in cards if card.card_id == source.id)
    page = cards_block(cards, 0)
    row = next(line for line in page.split("<li>") if card.card_id in line)
    assert f"在 {card.job} job 判" in row


def test_ci_jobs_green_control_is_green() -> None:
    from governance.loader import load_all_cards
    from governance.exit_codes import CLEAN
    from tests.test_fixture_runner import _assert_report_line, _run, _tail

    card = next(card for card in load_all_cards(ROOT) if card.id == "ci-jobs-cannot-die-quietly")
    tree = ROOT / "governance/fixtures/ci-jobs-cannot-die-quietly-green/control"
    proc = _run(card, tree)
    _assert_report_line(proc)
    assert proc.returncode == CLEAN, f"綠控制樣本回 {proc.returncode}：{_tail(proc)}"


def test_unregistered_required_job_returns_two_before_on_shape() -> None:
    from governance.checks.ci_required_gate import required_trigger_problems
    from governance.exit_codes import ToolBroken

    with pytest.raises(ToolBroken, match="沒在卡"):
        required_trigger_problems(
            "workflow.yml", {"on": "pull_request"}, {"acceptance": {}},
            frozenset({"acceptance"}),
            {"required_pull_request_events": {}, "forbidden_pull_request_filters": ["paths"]},
        )
