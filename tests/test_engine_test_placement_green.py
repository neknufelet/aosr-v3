"""考卷位置那一條不准無差別開槍：卡上沒有比較細的籃子時，它沒有對象。

後設測試的第 1 回在真 repo 上驗的是「有兩個籃子時不誤咬」，第 2 回驗的是「放錯位置會紅」
——兩回都在**今天這張卡**（籃子表裡有 `tests.engine.`）上跑。但 `_engine_dirs` 拿的是
最深的籃子，卡上只剩最上層那一個（`tests.`）的時候它回空集合；沒有那一道守衛，空集合會讓
**每一支**載入 `aosr` 的考卷都被判紅，訊息還會說「卡說那一籃的目錄是 []」。

這一支就是那道守衛的對照組：卡是從 `governance/rules/` 讀出來的（不是這裡抄一份），
把它的籃子表換成「只有 `tests.` 那一籃」，餵一支放錯位置、載入 `aosr` 的考卷進去，必須
一筆違規都沒有。故意留著這個例子：**判不出來就沒有對象**，不是「判不出來就一律紅」。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from governance.checks import green_must_be_real_green as green
from governance.loader import Card, load_all_cards

REPO = Path(__file__).resolve().parents[1]

CARD_ID = "green-must-be-real-green"
ONLY_TOP_BUCKET = {"bucket": [{"prefix": "tests.", "collected_floor": 1}]}
# 拼出來的檔名，不寫成完整路徑字面值：這支檔是 .py，裡面出現的路徑 token 會被
# refs-and-links-resolve 拿去解析，而那支檔只活在這一支測試的 tmp 目錄裡、不在版控樹裡。
ENGINE_TEST_NAME = "test_some_engine_thing" + ".py"


def _card() -> Card:
    """卡是從 governance/rules/ 讀出來的，不是這裡抄一份。"""
    cards = [c for c in load_all_cards(REPO) if c.id == CARD_ID]
    assert cards, f"governance/rules/ 裡找不到 id 是 {CARD_ID} 的卡——這一支就沒有對象可測"
    return cards[0]


def _only_top_bucket(card: Card) -> Card:
    """同一張卡、籃子表換成只有最上層那一籃（`tests.`）。"""
    assert card.junit, f"{CARD_ID} 竟然沒有 [junit]，這一支的前提不成立"
    return replace(card, junit={**card.junit, **ONLY_TOP_BUCKET})


def test_single_top_bucket_has_no_engine_directory() -> None:
    """只有 `tests.` 那一籃時，推不出更細的目錄——那就沒有對象。"""
    assert _only_top_bucket(_card()).junit_buckets, "換出來的卡讀不出籃子表，這一支就白跑了"
    assert green._engine_dirs(_only_top_bucket(_card())) == []


def test_engine_import_outside_engine_dir_is_not_flagged_when_no_engine_bucket(tmp_path: Path) -> None:
    """卡上沒有比較細的籃子時，載入 `aosr` 的考卷不准被判紅（判不出來不是判紅）。"""
    source = tmp_path / "tests" / ENGINE_TEST_NAME
    source.parent.mkdir(parents=True)
    source.write_text("import aosr\n", encoding="utf-8")
    bad = green._placement_group_problems(_only_top_bucket(_card()), tmp_path, [source])
    assert bad == [], f"卡上沒有更細的籃子時竟然開槍了：{bad}"
