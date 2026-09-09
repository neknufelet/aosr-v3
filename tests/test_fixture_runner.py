"""後設測試：每張規矩卡都要跑完五回合，任一回合不符就紅。

五回合（退出碼約定見 governance/exit_codes.py）：
  1. 乾淨樹（真 repo 根）        = 0
  2. 卡宣告的每一份必紅樣本      = 1
  3. 掃描根換成不存在的路徑      = 2
  4. 抽掉卡宣告的外部工具        = 2（卡宣告 external_tools = [] 時改為斷言「宣告為空」並記一行，不 skip）
  5. 控制樣本（已知會咬的最小輸入）= 1

這支測試自己不認識任何一張卡的內容，全部從卡的欄位讀出來，所以加卡不用改它。
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from governance.exit_codes import CLEAN, TOOL_BROKEN, VIOLATION  # noqa: E402
from governance.loader import Card, load_all_cards  # noqa: E402

CARDS = load_all_cards(REPO)
CARD_IDS = [c.id for c in CARDS]


def _run(card: Card, scan_root: Path | str, *, path: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["AOSR_BITE_DEPTH"] = "0"
    if path is not None:
        env["PATH"] = path
    return subprocess.run(
        [sys.executable, "-m", card.check_module, "--scan-root", str(scan_root)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
    )


def _path_without(tool: str) -> str:
    """把某個外部工具從 PATH 裡拿掉（只拿掉裝著它的那幾個目錄，不是清空 PATH）。"""
    kept = [
        d
        for d in os.environ.get("PATH", "").split(os.pathsep)
        if d and not (Path(d) / tool).exists()
    ]
    return os.pathsep.join(kept) or str(REPO / "governance" / "no-such-bin-dir")


def _tail(proc: subprocess.CompletedProcess[str]) -> str:
    out = (proc.stdout + proc.stderr).strip().splitlines()
    return " | ".join(out[-6:])


def _assert_report_line(proc: subprocess.CompletedProcess[str]) -> None:
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("scan_root=")]
    assert lines, f"檢查沒有印出 scan_root=... files=... hits=... 那一行：{_tail(proc)}"
    parts = lines[-1].split()
    assert len(parts) == 3 and parts[1].startswith("files=") and parts[2].startswith("hits="), (
        f"報告行格式不對：{lines[-1]!r}"
    )
    int(parts[1].removeprefix("files="))
    int(parts[2].removeprefix("hits="))


def test_there_is_at_least_one_card() -> None:
    """第 0 關：governance/rules/ 裡至少要有一張卡，不然這支後設測試會變成永遠回綠的空跑。"""
    assert CARDS, "governance/rules/ 裡一張卡都沒有——後設測試會變成永遠回綠的空跑"


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round1_clean_tree_is_green(card: Card) -> None:
    """第 1 回：乾淨樹（真 repo 根）必須回 0。"""
    proc = _run(card, REPO)
    _assert_report_line(proc)
    assert proc.returncode == CLEAN, f"{card.id} 在乾淨樹回 {proc.returncode}，應為 0：{_tail(proc)}"


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round2_negative_fixtures_are_red(card: Card) -> None:
    """第 2 回：卡宣告的每一份必紅樣本都必須回 1。"""
    cases = card.negative_cases(REPO)
    assert cases, f"{card.id} 的 negative_fixture 底下沒有任何樣本目錄"
    for case in cases:
        proc = _run(card, case)
        _assert_report_line(proc)
        assert proc.returncode == VIOLATION, (
            f"{card.id} 餵樣本 {case.relative_to(REPO)} 回 {proc.returncode}，應為 1：{_tail(proc)}"
        )


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round3_missing_scan_root_is_tool_broken(card: Card) -> None:
    """第 3 回：掃描根不存在必須回 2（工具自壞），不准回 0。"""
    missing = REPO / "governance" / "no-such-scan-root-ROUND3"
    assert not missing.exists()
    proc = _run(card, missing)
    _assert_report_line(proc)
    assert proc.returncode == TOOL_BROKEN, (
        f"{card.id} 掃描根不存在時回 {proc.returncode}，應為 2（工具自壞）：{_tail(proc)}"
    )


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round4_external_tool_removed_is_tool_broken(card: Card, capsys: pytest.CaptureFixture[str]) -> None:
    """第 4 回：把卡宣告的外部工具從 PATH 拿掉必須回 2；卡宣告 external_tools = [] 時斷言宣告為空並記一行，不 skip。"""
    if not card.external_tools:
        # 不 skip：卡必須明寫 external_tools = []，這裡斷言它真的是空的並記一行。
        assert card.external_tools == [], f"{card.id} 的 external_tools 不是空 list"
        with capsys.disabled():
            print(f"\n[第4回] {card.id} 宣告 external_tools = []（無外部工具可抽，已斷言宣告為空）")
        return
    for tool in card.external_tools:
        proc = _run(card, REPO, path=_path_without(tool))
        _assert_report_line(proc)
        assert proc.returncode == TOOL_BROKEN, (
            f"{card.id} 抽掉外部工具 {tool} 後回 {proc.returncode}，應為 2：{_tail(proc)}"
        )


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round5_control_fixture_is_red(card: Card) -> None:
    """第 5 回：控制樣本（已知會咬的最小輸入）必須回 1。"""
    control = card.control_path(REPO)
    proc = _run(card, control)
    _assert_report_line(proc)
    assert proc.returncode == VIOLATION, (
        f"{card.id} 餵控制樣本 {control.relative_to(REPO)} 回 {proc.returncode}，應為 1：{_tail(proc)}"
    )
