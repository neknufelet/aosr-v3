"""合法的新卡不准被誤咬：餵一棵綠樹下去，必須回 0、而且一筆違規都沒有。

後設測試的六回合裡沒有「餵一棵樹必須回 0」這一格：第 1 回的乾淨樹是真 repo，而真 repo
每一張卡的名字都在舊卡名單裡，所以「新卡寫了正整數票號會放行」在那六回合裡證不出來；
第 2 回的必紅樣本一律要求回 1，把這棵放進去會互相打架（同一份樣本被要求同時回 0 又回 1）。

這是「名單不再現算」那個致命洞的對照組：舊版拿 ``blueprint/cards-38.json`` 現算名單，
每一張新卡都會被算成舊卡、自動免票號，於是「新卡要票號」這條對真正的卡永遠不生效；
但當時沒有任何一棵樣本在測「合法新卡」長什麼樣，所以洞不會被測出來。這一支就是那個對照組。

跑法刻意跟後設測試共用同一支 ``_run``（同一份環境衛生：不帶 PYTHONPATH、不寫 .pyc），
斷言離開碼與那一行收據——回 0 而印不出收據，等於「什麼都沒掃到卻說乾淨」。
"""
from __future__ import annotations

from pathlib import Path

from governance.exit_codes import CLEAN
from governance.loader import Card, load_all_cards
from tests.test_fixture_runner import _assert_report_line, _run, _tail

REPO = Path(__file__).resolve().parents[1]

CARD_ID = "rule-card-required-fields"
GREEN_TREE = "governance/fixtures/rule-card-required-fields-green/case-new-card-legal"
REPORT_PREFIX = "scan_root="
NO_HITS = "hits=0"


def _card() -> Card:
    """卡是從 governance/rules/ 讀出來的，不是這裡抄一份。"""
    cards = [c for c in load_all_cards(REPO) if c.id == CARD_ID]
    assert cards, f"governance/rules/ 裡找不到 id 是 {CARD_ID} 的卡——這一支就沒有對象可測"
    return cards[0]


def test_legal_new_card_is_green() -> None:
    """綠樹（卡名不在舊卡名單裡、票號是正整數）餵下去必須回 0，而且收據行的 hits 是零。"""
    tree = REPO / GREEN_TREE
    assert tree.is_dir(), f"{GREEN_TREE} 不在——沒有樹可餵，這一支證不了任何事"
    proc = _run(_card(), tree)
    _assert_report_line(proc)
    assert proc.returncode == CLEAN, (
        f"合法的新卡被咬了（回 {proc.returncode}，應為 0）：{_tail(proc)}"
    )
    report = [ln for ln in proc.stdout.splitlines() if ln.startswith(REPORT_PREFIX)][-1]
    assert report.split()[-1] == NO_HITS, f"回 0 但收據行說有違規：{report!r}"
