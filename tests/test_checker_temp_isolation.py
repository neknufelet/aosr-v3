"""檢查程式自己開的暫存樹不准落在被掃的那棵樹裡。

三處 ``tempfile.mkdtemp(dir=scan_root)`` 會在檢查執行途中把暫存目錄開進**被掃的樹**：
就算跑完清乾淨（rmdir／rmtree），執行當下那棵樹裡多出一顆目錄，就會被同一批檢查看到
（例如 file-placement-allowlist、scan-scope-has-no-holes 那類數目／路徑的掃描面），
也等於施工者可藉由「開暫存目錄」污染 scan_root。所以測試不能只斷言「跑完沒殘留」，
要捕捉執行途中暫存目錄真的落在哪裡。

做法：把版控 fixtures copytree 到 ``git_sandbox.root`` 的子目錄（受測操作全部對副本，
不碰 REPO 的 fixture），再把 ``tempfile.tempdir``（mkdtemp 缺省落點）指到 ``tmp_path`` 下
另一個獨立的 system-temp 目錄，最後包一層 ``tempfile.mkdtemp`` 把每一個真的開出來的目錄
記下來，斷言它們沒有一個落在 scan_root 底下、而且跑完都被清掉。還沒修之前，
``dir=scan_root`` 會無視 ``tempdir`` 直落被掃的樹，這一題就 RED。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from governance.checks import check_exit_code_honest, commit_author_allowlisted
from governance.exit_codes import ToolBroken
from governance.loader import load_card
from tests.conftest import GitSandbox

REPO = Path(__file__).resolve().parents[1]

# 三處會開暫存目錄的程式點，各配一份已知會走那條路的樣本（唯讀，只當 copytree 的來源）。
COMMIT_AUTHOR_CASE = REPO / "governance" / "fixtures" / "commit-author-allowlisted" / "control"
EXIT_CODE_CASE = (
    REPO
    / "governance"
    / "fixtures"
    / "check-exit-code-honest"
    / "case-empty-enumeration-not-fail-closed"
)


def _system_temp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """一個獨立於 scan_root 的系統暫存根，指給 ``tempfile.tempdir``。"""
    system_temp = tmp_path / "system-temp"
    system_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(system_temp))
    return system_temp


def _record_mkdtemp(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    """把 ``tempfile.mkdtemp`` 包一層，記下每一個真的開出來的路徑，行為不變。"""
    real = tempfile.mkdtemp
    recorded: list[Path] = []

    def wrapper(
        suffix: str | None = None,
        prefix: str | None = None,
        dir: str | os.PathLike[str] | None = None,
    ) -> str:
        path = real(suffix=suffix, prefix=prefix, dir=dir)
        recorded.append(Path(path))
        return path

    monkeypatch.setattr(tempfile, "mkdtemp", wrapper)
    return recorded


def _inside_scan_root(recorded: list[Path], scan_root: Path) -> list[Path]:
    return [p for p in recorded if p.is_relative_to(scan_root)]


def _left_behind(recorded: list[Path]) -> list[Path]:
    return [p for p in recorded if p.exists()]


def test_commit_author_temp_tree_is_outside_scan_root(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """commit_author 的樣本歷史暫存 tree 不准開在被掃的樣本樹裡。"""
    scan_root = git_sandbox.root / "commit-author-case"
    shutil.copytree(COMMIT_AUTHOR_CASE, scan_root)
    _system_temp(tmp_path, monkeypatch)
    recorded = _record_mkdtemp(monkeypatch)

    work_tree, _rng, cleanup = commit_author_allowlisted._resolve_range(scan_root)
    try:
        assert recorded, "mkdtemp 沒有被叫到——這支檢查應該在暫存樹重建樣本歷史"
        inside = _inside_scan_root(recorded, scan_root)
        assert not inside, (
            f"樣本歷史的暫存目錄落在被掃的樹裡：{[str(p) for p in inside]}"
            "——檢查執行途中腳下多出一顆目錄，會污染 scan_root"
        )
        assert work_tree.is_dir(), "cleanup 前，重建樣本歷史的實際工作樹必須還在磁碟上"
    finally:
        cleanup()
    leftover = _left_behind(recorded)
    assert not leftover, (
        f"cleanup 之後還殘留這些暫存目錄：{[str(p) for p in leftover]}"
        "——自己開的暫存樹跑完必須清乾淨"
    )


def test_empty_probe_temp_is_outside_scan_root(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check_exit_code_honest 的空集合探針暫存目錄不准開在被掃的樹裡。"""
    scan_root = git_sandbox.root / "exit-code-case"
    shutil.copytree(EXIT_CODE_CASE, scan_root)
    _system_temp(tmp_path, monkeypatch)
    recorded = _record_mkdtemp(monkeypatch)

    card = load_card(scan_root / "governance" / "rules" / "sample-card.toml", scan_root)
    check_exit_code_honest._probe_problems(card, scan_root, card.check, 0)

    assert recorded, "mkdtemp 沒有被叫到——空集合探針應該自己開一顆暫存目錄"
    inside = _inside_scan_root(recorded, scan_root)
    assert not inside, (
        f"空集合探針的暫存目錄落在被掃的樹裡：{[str(p) for p in inside]}"
        "——檢查執行途中腳下多出一顆目錄，會污染 scan_root"
    )
    leftover = _left_behind(recorded)
    assert not leftover, (
        f"探針跑完還殘留這些暫存目錄：{[str(p) for p in leftover]}"
        "——自己開的暫存目錄跑完必須清乾淨"
    )


def test_shell_probe_temp_is_outside_scan_root(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """check_exit_code_honest 的外殼探針暫存目錄不准開在被掃的樹裡。"""
    scan_root = git_sandbox.root / "exit-code-case"
    shutil.copytree(EXIT_CODE_CASE, scan_root)
    _system_temp(tmp_path, monkeypatch)
    recorded = _record_mkdtemp(monkeypatch)

    check_exit_code_honest._shell_problems(scan_root, "governance/exit_codes.py", 0)

    assert recorded, "mkdtemp 沒有被叫到——外殼探針應該自己開一顆暫存目錄"
    inside = _inside_scan_root(recorded, scan_root)
    assert not inside, (
        f"外殼探針的暫存目錄落在被掃的樹裡：{[str(p) for p in inside]}"
        "——檢查執行途中腳下多出一顆目錄，會污染 scan_root"
    )
    leftover = _left_behind(recorded)
    assert not leftover, (
        f"探針跑完還殘留這些暫存目錄：{[str(p) for p in leftover]}"
        "——自己開的暫存目錄跑完必須清乾淨"
    )


def test_empty_probe_real_shell_measures_empty_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """空集合探針要真的量到「列舉出來是空集合」，不是被「落在 repo 外面」擋下（#234 的 F1）。

    用真 repo 的 governance/ 當「被測掃描根」——它帶的是真正的共用外殼（``exit_codes.py`` 有
    ``resolve_scan_root``／``repo_root`` 那道「掃描根要在 repo 裡」的門）。_empty_git_repo 把那一份
    governance 原封不動複製進獨立 git 樹，讓檢查副本的 repo_root 指到獨立樹，空集合子目錄才過得
    了那道門、真正撞進「列舉出來是空集合」。跑的是 subprocess(副本)，全程不寫真 repo。
    """
    _system_temp(tmp_path, monkeypatch)
    probe, cleanup, run_cwd = check_exit_code_honest._empty_git_repo(
        "aosr-behavior-probe-", copy_governance_from=REPO
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "governance.checks.style_guard", "--scan-root", str(probe)],
            cwd=run_cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=check_exit_code_honest.PROBE_TIMEOUT,
        )
    finally:
        cleanup()
    assert proc.returncode == 2, f"應回 2，實際 {proc.returncode}：{proc.stderr.strip()}"
    assert "列舉出來是空集合" in proc.stderr, (
        f"回 2 的原因不是「列舉出來是空集合」，而是別的事：{proc.stderr.strip()}"
    )
    assert "落在 repo" not in proc.stderr, (
        f"回 2 是被「落在 repo 外面」擋下，量到的不是空集合那份守門：{proc.stderr.strip()}"
    )


def test_copytree_oserror_becomes_toolbroken_and_cleans_container(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """複製 governance 副本時炸出 OSError（含 shutil.Error）要轉 ToolBroken，容器也要清掉。

    seam 注入的是**真的** ``shutil.Error``（它是 ``OSError`` 的子類），不是換掉整個函式的假物件：
    ``shutil.copytree`` 在來源讀不到、目的地寫不進去時真的會丟它，這條路徑得回 2 而不是讓
    ``shutil.Error`` 直穿——直穿的話外殼看不到 ToolBroken，離開碼變成 1（被讀成「抓到違規」）。
    這一題同時檢查清理是真的：容器是這個測試自己錄下來的 ``mkdtemp`` 落點，跑完必須不存在。
    """
    scan_root = git_sandbox.root / "exit-code-case"
    shutil.copytree(EXIT_CODE_CASE, scan_root)
    _system_temp(tmp_path, monkeypatch)
    recorded = _record_mkdtemp(monkeypatch)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise shutil.Error("injected unreadable source")

    monkeypatch.setattr(shutil, "copytree", _boom)
    with pytest.raises(ToolBroken) as excinfo:
        check_exit_code_honest._empty_git_repo(
            "aosr-copy-error-probe-", copy_governance_from=scan_root
        )

    assert recorded, "mkdtemp 沒有被叫到——空 repo 探針應該自己開暫存目錄"
    assert isinstance(excinfo.value.__cause__, shutil.Error), (
        f"ToolBroken 沒有把原本的 shutil.Error 留在 __cause__：{excinfo.value.__cause__!r}"
        "——轉碼時不准把原錯誤吞掉"
    )
    leftover = _left_behind(recorded)
    assert not leftover, (
        f"copytree 失敗之後還殘留這些暫存目錄：{[str(p) for p in leftover]}"
        "——開一半的暫存容器不能留在系統暫存區"
    )


def test_git_init_failure_cleans_temp_container(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """開空 repo 的 git init 半路失敗時，暫存容器要被清乾淨，不留自開的目錄。"""
    _system_temp(tmp_path, monkeypatch)
    recorded = _record_mkdtemp(monkeypatch)

    def _boom(_repo: Path, _env: dict[str, str], _template: Path) -> None:
        raise ToolBroken("故意讓 git init 失敗")

    monkeypatch.setattr(check_exit_code_honest, "_git_init", _boom)
    with pytest.raises(ToolBroken):
        check_exit_code_honest._empty_git_repo("aosr-init-fail-probe-")

    assert recorded, "mkdtemp 沒有被叫到——空 repo 探針應該自己開暫存目錄"
    leftover = _left_behind(recorded)
    assert not leftover, (
        f"init 失敗之後還殘留這些暫存目錄：{[str(p) for p in leftover]}"
        "——開一半的暫存容器不能留在系統暫存區"
    )


def test_probe_written_into_still_cleans_container_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """探針目錄被寫進東西（probe.rmdir 失敗）時，容器仍要清掉、而且仍報原 ToolBroken。"""
    _system_temp(tmp_path, monkeypatch)
    recorded = _record_mkdtemp(monkeypatch)

    probe, cleanup, _run_cwd = check_exit_code_honest._empty_git_repo("aosr-written-probe-")
    (probe / "junk").write_text("x", encoding="utf-8")

    with pytest.raises(ToolBroken) as excinfo:
        cleanup()
    assert "被寫入" in str(excinfo.value), f"例外沒說清是「被寫入」：{excinfo.value}"
    leftover = _left_behind(recorded)
    assert not leftover, (
        f"「被寫入」的例外之後還殘留暫存容器：{[str(p) for p in leftover]}"
        "——探針目錄刪不掉，容器本身還是要清掉"
    )
