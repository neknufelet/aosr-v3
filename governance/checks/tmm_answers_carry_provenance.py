"""TMM 凍結答案守門入口；判準與提交範圍比對交給共用核心。"""
import sys
from pathlib import Path

from governance import frozen_answers
from governance.exit_codes import run
from governance.frozen_answers import CommitRange as CommitRange, Rules as Rules
from governance.frozen_answers import _fixture_trees as _fixture_trees, _run_git as _run_git
from governance.frozen_answers import read_rules as read_rules


CHECK_REL = "governance/checks/tmm_answers_carry_provenance.py"
CARD_REL = "governance/rules/tmm-answers-carry-provenance.toml"
IDENTITY = frozen_answers.AnswerCheck(CHECK_REL, CARD_REL, "TMM", "原套件 commit")


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    return frozen_answers.targets(scan_root, files, IDENTITY)


def _range_hits(rng: CommitRange, rules: Rules, fixture_trees: tuple[str, ...] = ()) -> list[str]:
    return frozen_answers._range_hits(rng, rules, IDENTITY, fixture_trees)


def check(scan_root: Path, files: list[Path]) -> list[str]:
    return frozen_answers.check(scan_root, files, IDENTITY, _run_git)


if __name__ == "__main__":
    sys.exit(run(check, description="TMM 凍結答案要帶出身且不准只換數字", targets=targets))
