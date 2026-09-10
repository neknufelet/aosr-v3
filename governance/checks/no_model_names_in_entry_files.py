#!/usr/bin/env python3
"""每次開工會載入的檔，全文不准出現字典裡的那些名字（骨架，還沒判）。

這一版刻意只有掃描面與門檻，check() 一律回空 list——先讓必紅樣本與控制樣本紅在那裡，
再把判準寫進來。
"""
from __future__ import annotations

import sys
import tomllib
from pathlib import Path

from governance.exit_codes import ToolBroken, run
from governance.loader import EXEMPTION_KEYS, RULES_DIR, setting_strings

CHECK_REL = "governance/checks/no_model_names_in_entry_files.py"

WORD_KEYS = ("model_words", "cli_words", "quota_words")
LIST_KEYS = ("entry_files", "scan_dirs", *WORD_KEYS)
ALLOW_KEY = "allow"
ALLOW_ENTRY_KEYS = ("text", *EXEMPTION_KEYS)
SETTINGS_KEYS = (*LIST_KEYS, ALLOW_KEY)


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 governance/rules/ 讀這支檢查的字典與掃描面宣告。"""
    rules_dir = scan_root / RULES_DIR
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——字典只寫在卡上，讀不到字典這一跑就不算數"
        )
    path, data = mine[0]
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{path.relative_to(scan_root)} 沒有 [settings] 表（字典與掃描面都寫在那裡）")
    return settings


def targets(scan_root: Path, files: list[Path]) -> list[Path]:
    """掃描面：卡宣告的入口檔，加上卡宣告的目錄底下的每一個檔。"""
    settings = _card_settings(scan_root, files)
    entry = set(setting_strings(settings, "entry_files"))
    dirs = [d.rstrip("/") + "/" for d in setting_strings(settings, "scan_dirs")]
    picked = [
        f
        for f in files
        if (rel := f.relative_to(scan_root).as_posix()) in entry
        or any(rel.startswith(prefix) for prefix in dirs)
    ]
    return sorted(picked)


def check(scan_root: Path, files: list[Path]) -> list[str]:
    """骨架：還沒有判準，一律回空 list（必紅樣本因此會紅在後設測試第 2、5 回）。"""
    targets(scan_root, files)
    return []


if __name__ == "__main__":
    sys.exit(
        run(
            check,
            description="每次開工會載入的檔全文不准出現模型名、模型家的 CLI 名、用量字樣",
            targets=targets,
        )
    )
