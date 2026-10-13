"""四卡的 schema 3 正向樣本，讀副本以免鏡像鎖寫進真樣本樹。"""
from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from governance.checks import (
    four_roles_different_actors,
    identity_strings_generated,
    receipt_authority_is_the_cloud_run,
    receipt_schema_complete,
)

FIXTURES = Path(__file__).resolve().parents[1] / "governance/fixtures"
CHECKERS: list[tuple[str, Callable[[Path, list[Path]], list[str]]]] = [
    ("receipt-schema-complete", receipt_schema_complete.check),
    ("receipt-authority-is-the-cloud-run", receipt_authority_is_the_cloud_run.check),
    ("four-roles-different-actors", four_roles_different_actors.check),
    ("identity-strings-generated", identity_strings_generated.check),
]


@pytest.mark.parametrize("card,check", CHECKERS, ids=[card for card, _ in CHECKERS])
def test_schema_three_single_job_control_is_green(
    tmp_path: Path, card: str, check: Callable[[Path, list[Path]], list[str]],
) -> None:
    root = tmp_path / "case"
    shutil.copytree(FIXTURES / (card + "-green") / "schema-3", root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    assert check(root, files) == []
