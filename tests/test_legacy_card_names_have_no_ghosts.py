"""舊卡名單裡的幽靈名不准留——留一個名字，就是給新卡開一個免票號的後門。

``governance/checks/rule_card_required_fields.py`` 的 ``LEGACY_CARD_NAMES`` 決定哪張卡
不必寫票號。它是一份寫死的常數：上一輪找碴席實測，把 ``"sample-card"`` 併進去，
``case-admission-issue-missing`` 那棵樹的票號違規就從 1 筆變 0 筆——檢查沒有回 2，
也沒有留下任何「名單被動過」的痕跡；那一棵樹還紅在別的規矩上，所以整棵看起來照樣是紅的，
「新卡要票號」這條規矩本身卻已經被關掉。同一個 PR 改一行就能讓新卡免票號。

這條測試只守**一個方向**：名單裡的每一個名字，今天都要在磁碟上有一張卡。幽靈名留著，
就是給新卡取那個名字、自動免票號的路；卡片本體已經被合併掉的名字要從名單刪掉。
名單是一份**歷史名冊**，不是「今天有哪些卡」的清單。

**另一個方向這條測試擋不住，這是刻意的取捨。** 有人把一張新卡的名字偷偷併進名單，
這條測試看不出來——那個名字今天確實在磁碟上有一張卡，看起來就跟舊卡一樣。要擋它得改
檢查程式（例如讓名單與票號交叉比對），那會走 PR、靠人在 diff 上看到。**反過來也不准**
把「磁碟上每一張卡都要在名單裡」寫成斷言：那會把下一張合法的新卡鎖死——新卡不在名單裡，
斷言 FAIL，而唯一能讓它變綠的動作正好是把它併進名單，也就是關掉票號那顆牙。規矩與自己的
測試互鎖，決策紙 ``docs/decisions/card-admission-threshold.md`` 從沒要求這件事。

**為什麼不塞進檢查程式。** 第四關要跑得進必紅樣本樹，而樣本樹裡有一張磁碟上的
``sample-card`` 不在常數裡——斷言放進檢查程式的話，每一棵樣本樹都會被這一條咬成紅，
那不是那些樣本想證的事。做成獨立測試才安全：它只讀真 repo 的 ``governance/rules/``，
一個樣本樹都不碰。
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
