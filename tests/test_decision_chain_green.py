"""合法的取代鏈不准被誤咬：餵一棵綠樹下去，必須回 0、而且一筆違規都沒有。

後設測試的六回合裡沒有「餵一棵樹必須回 0」這一格：第 1 回的乾淨樹是真 repo，而真 repo
的決策紙一條取代鏈都沒有；控制樣本那一回合的正確答案是 1（已知會咬的最小輸入），
所以「走鏈那一段不會誤咬合法的多跳鏈」在那六回合裡證不出來。

找碴席（blueprint/cards-38.json 的 critic_v2）點的正是這個洞：逐跳要求「還在生效」會誤咬
合法的三跳鏈，而第一輪的對照組沒有覆蓋多跳鏈，所以那個誤咬不會被測出來。這一支就是那個
對照組：綠樹裡有一條三跳鏈（鏈尾停在一份 accepted，中途每一跳都是 superseded、雙向指標
都對），加一張沒有取代關係的紙。

跑法刻意跟後設測試共用同一支 ``_run``（同一份環境衛生：不帶 PYTHONPATH、不寫 .pyc），
斷言離開碼與那一行收據——回 0 而印不出收據，等於「什麼都沒掃到卻說乾淨」。
"""
from __future__ import annotations

from pathlib import Path

from governance.exit_codes import CLEAN
from governance.loader import Card, load_all_cards
from tests.test_fixture_runner import _assert_report_line, _run, _tail

REPO = Path(__file__).resolve().parents[1]

CARD_ID = "decision-paper-structure"
GREEN_TREE = "governance/fixtures/decision-paper-structure-green/legal-multi-hop-chain"
REPORT_PREFIX = "scan_root="
NO_HITS = "hits=0"


def _card() -> Card:
    """卡是從 governance/rules/ 讀出來的，不是這裡抄一份。"""
    cards = [c for c in load_all_cards(REPO) if c.id == CARD_ID]
    assert cards, f"governance/rules/ 裡找不到 id 是 {CARD_ID} 的卡——這一支就沒有對象可測"
    return cards[0]


def test_legal_supersede_chain_is_green() -> None:
    """綠樹（合法的三跳取代鏈）餵下去必須回 0，而且收據行的 hits 是零。"""
    tree = REPO / GREEN_TREE
    assert tree.is_dir(), f"{GREEN_TREE} 不在——沒有樹可餵，這一支證不了任何事"
    proc = _run(_card(), tree)
    _assert_report_line(proc)
    assert proc.returncode == CLEAN, (
        f"合法的取代鏈被咬了（回 {proc.returncode}，應為 0）：{_tail(proc)}"
    )
    report = [ln for ln in proc.stdout.splitlines() if ln.startswith(REPORT_PREFIX)][-1]
    assert report.split()[-1] == NO_HITS, f"回 0 但收據行說有違規：{report!r}"
