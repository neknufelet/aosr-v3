"""從指定的 Git base 讀主線規矩卡名。"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MainlineCards:
    ids: frozenset[str] | None
    reason: str
    base: str


def mainline_card_ids(repo_root: Path, base_ref: str) -> MainlineCards:
    """回傳 ``base_ref`` 上的卡 id；工具或 base 不可讀時不捏造空集合。"""
    git = shutil.which("git")
    if git is None:
        return MainlineCards(ids=None, reason="找不到 git", base=base_ref)
    try:
        proc = subprocess.run(
            [git, "ls-tree", "--name-only", base_ref, "governance/rules/"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return MainlineCards(ids=None, reason=f"讀不到 base={base_ref}", base=base_ref)
    if proc.returncode != 0:
        return MainlineCards(ids=None, reason=f"讀不到 base={base_ref}", base=base_ref)
    ids = frozenset(
        Path(line).stem
        for line in proc.stdout.splitlines()
        if line.startswith("governance/rules/") and line.endswith(".toml")
    )
    return MainlineCards(ids=ids, reason="", base=base_ref)
