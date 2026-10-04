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

from governance.checks import fenics_answers_carry_provenance as fenics
from governance.checks import type_guard
from governance.exit_codes import TOOL_BROKEN, CheckFn, TargetFn, ToolBroken, make_temp_dir, run

REPO = Path(__file__).resolve().parents[1]
CHECKS = REPO / "governance" / "checks"
# 型別對不上的那份必紅樣本：暫存區正常時它回 1（後設測試在考），這裡要它回 2。
TYPE_MISMATCH = REPO / "governance" / "fixtures" / "type-guard" / "case-type-mismatch"
# 帶樣本範圍宣告的 FEniCS 必紅樣本：會在暫存目錄建一段 git 歷史。
FENICS_RANGE = REPO / "governance" / "fixtures" / "fenics-answers-carry-provenance" / "case-digest-mismatch"
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


def _stuck_cleanup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """暫存區指到考卷自己的 tmp_path（清不掉的東西不落在真的系統暫存區），刪目錄一律失敗。"""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def stuck(path: str | Path) -> None:
        raise PermissionError(f"清不掉 {path}")

    monkeypatch.setattr(shutil, "rmtree", stuck)


@pytest.mark.parametrize("check,targets,case", [
    (type_guard.check, type_guard.targets, TYPE_MISMATCH),
    (fenics.check, fenics.targets, FENICS_RANGE),
], ids=["type-guard", "fenics"])
def test_a_check_exits_2_when_it_cannot_clean_its_temp_dir(
        check: CheckFn, targets: TargetFn, case: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """清不掉就是這一跑被汙染了，回 2、不吞，也不炸成 1（FEniCS 那支原本直接 rmtree，複查抓到回 1）。"""
    _stuck_cleanup(monkeypatch, tmp_path)
    assert run(check, ["--scan-root", str(case)], targets=targets) == TOOL_BROKEN
    assert "FAIL(2) 工具自壞：清不掉自己開的暫存目錄" in capsys.readouterr().err


def test_fenics_removes_its_temp_history_when_building_it_fails(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """暫存歷史建到一半失敗（還沒交回給 check()）：先清掉自己開的暫存目錄再往外丟，不留 aosr-fenics-range-*。"""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    real = fenics._run_git

    def failing(argv: list[str], cwd: Path, what: str, env: dict[str, str] | None = None) -> str:
        if what == "提交 base":
            raise ToolBroken("提交 base 失敗（考卷造的）")
        return real(argv, cwd, what, env)

    monkeypatch.setattr(fenics, "_run_git", failing)
    assert run(fenics.check, ["--scan-root", str(FENICS_RANGE)], targets=fenics.targets) == TOOL_BROKEN
    assert list(tmp_path.glob("aosr-fenics-range-*")) == []
