"""舊卡名單要跟磁碟上的卡名雙向相等——它是第四關唯一的分界線，加一個名字就是放行一張卡。

``governance/checks/rule_card_required_fields.py`` 的 ``LEGACY_CARD_NAMES`` 決定哪張卡
不必寫票號。它是一份寫死的常數：上一輪找碴席實測，把 ``"sample-card"`` 併進去，
``case-admission-issue-missing`` 那棵樹的票號違規就從 1 筆變 0 筆——檢查沒有回 2，
也沒有留下任何「名單被動過」的痕跡；那一棵樹還紅在別的規矩上，所以整棵看起來照樣是紅的，
「新卡要票號」這條規矩本身卻已經被關掉。同一個 PR 改一行就能讓新卡免票號。

**為什麼不塞進檢查程式。** 第四關要跑得進必紅樣本樹，而樣本樹裡有一張磁碟上的
``sample-card`` 不在常數裡——斷言放進檢查程式的話，每一棵樣本樹都會被這一條咬成紅，
那不是那些樣本想證的事。做成獨立測試才安全：它只讀真 repo 的 ``governance/rules/``，
一個樣本樹都不碰。

兩個方向各一條，各自具名（不是斷言「筆數是幾個」）：

* 常數裡的每一個名字，今天都要在磁碟上有一張卡——幽靈名留著，就是給新卡取那個名字免票號；
* 磁碟上的每一張卡，都要在常數裡——漏一個名字，那張既有卡當場被當成新卡要求補票號。
"""
from __future__ import annotations

from pathlib import Path

from governance.checks.rule_card_required_fields import LEGACY_CARD_NAMES
from governance.loader import load_all_cards

REPO = Path(__file__).resolve().parents[1]


def _cards_on_disk() -> set[str]:
    """``governance/rules/`` 底下每一張卡的 id。第四關比對用的就是這個 id，不是檔名。"""
    return {card.id for card in load_all_cards(REPO)}


def test_no_ghost_name_in_legacy_list() -> None:
    """常數裡的幽靈名不准留：留著就是一個免票號的後門。"""
    ghosts = sorted(LEGACY_CARD_NAMES - _cards_on_disk())
    assert not ghosts, (
        f"LEGACY_CARD_NAMES 裡這幾個名字在 governance/rules/ 底下沒有對應的卡：{ghosts}"
        "——幽靈名留著，任何新卡取那個名字就自動免票號，第四關對它等於不存在；"
        "卡片本體已經不在樹上的名字要從常數裡刪掉"
    )


def test_no_card_missing_from_legacy_list() -> None:
    """磁碟上的每一張卡都要在常數裡：漏一個名字，那張卡當場被要求補票號。"""
    missing = sorted(_cards_on_disk() - LEGACY_CARD_NAMES)
    assert not missing, (
        f"governance/rules/ 底下這幾張卡不在 LEGACY_CARD_NAMES 裡：{missing}"
        "——舊卡漏了一個名字，那張卡會被當成新卡要求 admission_issue，"
        "而它立卡的時候根本還沒有開票制度；真的新卡請照程序填票號，不要併進名單"
    )
