"""後設測試：每張規矩卡都要跑完五回合，任一回合不符就紅。

五回合（退出碼約定見 governance/exit_codes.py）：
  1. 乾淨樹（真 repo 根）        = 0（宣告了 [junit] 的卡在「產收據那一跑」裡改為 = 1，見下）
  2. 卡宣告的每一份必紅樣本      = 1
  3. 掃描根換成不存在的路徑      = 2
  4. 抽掉卡宣告的外部工具        = 2（卡宣告 external_tools = [] 時改為斷言「宣告為空」並記一行，不 skip）
  5. 控制樣本（已知會咬的最小輸入）= 1

這支測試自己不認識任何一張卡的內容，全部從卡的欄位讀出來，所以加卡不用改它。

第 1 回有一個由卡的欄位決定的分支：卡宣告了 `[junit]`（要判一份 pytest 收據）時，
`tests/conftest.py` 會在開跑前先跑一次真的全套把收據產出來。**在那一跑裡**收據還不存在，
這時候正確答案是紅不是綠，所以第 1 回改為斷言「檢查必須回 1」——比平常更凶，不是放水。
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
from tests.conftest import SEED_ENV  # noqa: E402

CARDS = load_all_cards(REPO)
CARD_IDS = [c.id for c in CARDS]


def _run(card: Card, scan_root: Path | str, *, path: str | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # 不要在被掃的樹裡留 __pycache__——樣本樹的檔案清單就是證據，不該被跑測試這件事改變。
    env["PYTHONDONTWRITEBYTECODE"] = "1"
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
    # 逐項具名比對，不寫 len(parts) == 3——鎖死數量的斷言由 assertions-not-pinned-to-counts 咬。
    # 這樣寫也比較有用：格式錯的時候直接印出「實際是哪幾個欄位」。
    keys = [p.split("=", 1)[0] for p in parts]
    assert keys == ["scan_root", "files", "hits"], f"報告行的欄位不對：{keys}（{lines[-1]!r}）"
    int(parts[1].removeprefix("files="))
    int(parts[2].removeprefix("hits="))


def test_there_is_at_least_one_card() -> None:
    """第 0 關：governance/rules/ 裡至少要有一張卡，不然這支後設測試會變成永遠回綠的空跑。"""
    assert CARDS, "governance/rules/ 裡一張卡都沒有——後設測試會變成永遠回綠的空跑"


@pytest.mark.parametrize("card", CARDS, ids=CARD_IDS)
def test_round1_clean_tree_is_green(card: Card) -> None:
    """第 1 回：乾淨樹（真 repo 根）必須回 0。

    例外：卡宣告了 `[junit]`、而且這一跑就是 `tests/conftest.py` 派去產收據的那一跑——
    收據此刻還不存在，正確答案是紅。那一回合改為斷言「檢查必須回 1」。
    """
    if card.junit and os.environ.get(SEED_ENV):
        receipt = REPO / card.junit_path
        assert not receipt.exists(), (
            f"產收據那一跑裡 {card.junit_path} 竟然已經存在，這個斷言就證明不了什麼了"
        )
        proc = _run(card, REPO)
        _assert_report_line(proc)
        assert proc.returncode == VIOLATION, (
            f"{card.id} 在收據不存在時回 {proc.returncode}，應為 1"
            f"——沒有收據就是沒有綠，不准當乾淨：{_tail(proc)}"
        )
        return
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
