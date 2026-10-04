"""開不了系統暫存目錄時，檢查程式回 2（這一跑不算數），不准炸成離開碼 1 冒充「抓到違規」（#318）。

在寫不了暫存區的沙箱裡跑型別警衛，它建 mypy 快取目錄時直接炸出 FileNotFoundError、離開碼 1。現在每支檢查
開暫存目錄都走共用外殼的 ``make_temp_dir``，開不了就包成 ToolBroken；這裡考那支函式、考型別警衛整支跑一次，
再掃一遍每支檢查程式不准自己 import tempfile（繞過外殼就又會炸成 1）。
"""
from __future__ import annotations

import ast
import re
import shutil
import tempfile
from pathlib import Path

import pytest

from governance.checks import type_guard
from governance.exit_codes import TOOL_BROKEN, ToolBroken, make_temp_dir, run

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "governance" / "checks"
# 型別對不上的那份必紅樣本：暫存區正常時它回 1（後設測試在考），這裡要它回 2。
TYPE_MISMATCH = REPO / "governance" / "fixtures" / "type-guard" / "case-type-mismatch"
UNUSABLE = "開不了系統暫存目錄"


def _unusable_temp_area(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(suffix: str | None = None, prefix: str | None = None, dir: str | None = None) -> str:
        raise FileNotFoundError("No usable temporary directory found")

    monkeypatch.setattr(tempfile, "mkdtemp", refuse)


def test_an_unusable_temp_area_is_tool_broken(monkeypatch: pytest.MonkeyPatch) -> None:
    _unusable_temp_area(monkeypatch)
    with pytest.raises(ToolBroken, match=UNUSABLE):
        make_temp_dir("aosr-probe-")


def test_type_guard_exits_2_not_1_when_it_cannot_open_its_cache(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _unusable_temp_area(monkeypatch)
    code = run(type_guard.check, ["--scan-root", str(TYPE_MISMATCH)], targets=type_guard.targets)
    assert code == TOOL_BROKEN
    assert re.search(f"FAIL\\(2\\) 工具自壞：{UNUSABLE}", capsys.readouterr().err)


def _imports_tempfile(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "tempfile" for alias in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and node.module == "tempfile":
            return True
    return False


def test_no_check_opens_temp_dirs_behind_the_shared_shell() -> None:
    checks = sorted(CHECKS.glob("*.py"))
    assert checks
    assert [path.name for path in checks if _imports_tempfile(path)] == []


def test_type_guard_exits_2_when_it_cannot_clean_its_cache(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """快取清不掉就是這一跑被汙染了，跟其他檢查同一個形狀回 2、不吞（改寫前的暫存目錄寫法清不掉也會報錯）。"""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def stuck(path: str | Path) -> None:
        raise PermissionError(f"清不掉 {path}")

    monkeypatch.setattr(shutil, "rmtree", stuck)
    code = run(type_guard.check, ["--scan-root", str(TYPE_MISMATCH)], targets=type_guard.targets)
    assert code == TOOL_BROKEN
    assert "FAIL(2) 工具自壞：清不掉自己開的暫存目錄" in capsys.readouterr().err
