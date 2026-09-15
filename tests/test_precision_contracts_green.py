"""精度契約卡的綠樹：主線在分支點之後自己改了登記簿並帶紙，候選分支沒動——餵下去必須回 0。

後設測試八回合裡沒有「餵一棵樹必須回 0」這一格（第 1 回的乾淨樹是真 repo），所以「舊登記簿要從共同分支點讀、
差異要用三點」這件事得另開一棵綠樹證明（第二輪找碴點的洞）。
"""
from __future__ import annotations

from pathlib import Path

from governance.exit_codes import CLEAN
from governance.loader import Card, load_all_cards
from tests.test_fixture_runner import _assert_report_line, _run, _tail

REPO = Path(__file__).resolve().parents[1]
CARD_ID = "precision-contracts-live-in-one-registry"
GREEN_TREE = "governance/fixtures/precision-contracts-live-in-one-registry-green/main-changed-value-with-paper"


def _card() -> Card:
    cards = [c for c in load_all_cards(REPO) if c.id == CARD_ID]
    assert cards, f"governance/rules/ 裡找不到 id 是 {CARD_ID} 的卡"
    return cards[0]


def test_mainline_change_after_fork_is_not_charged_to_the_candidate() -> None:
    """主線分支點之後自己改值帶紙、候選沒動登記簿：回 0、hits 是零。"""
    tree = REPO / GREEN_TREE
    assert tree.is_dir(), f"{GREEN_TREE} 不在——沒有樹可餵"
    proc = _run(_card(), tree)
    _assert_report_line(proc)
    assert proc.returncode == CLEAN, f"主線自己的改值被算到候選身上（回 {proc.returncode}）：{_tail(proc)}"
    report = [ln for ln in proc.stdout.splitlines() if ln.startswith("scan_root=")][-1]
    assert report.split()[-1] == "hits=0", f"回 0 但收據行說有違規：{report!r}"
