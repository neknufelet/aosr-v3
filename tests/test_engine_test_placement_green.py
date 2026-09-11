"""考卷位置那一條的兩道底線：真卡算得出引擎籃子，而算不出來時不開槍。

後設測試的第 1 回在真 repo 上驗的是「有兩個籃子時不誤咬」，第 2 回驗的是「放錯位置會紅」
——兩回都在**今天這張卡**（籃子表裡有 `tests.engine.`）上跑。它們守不到兩件事：

1. **真卡把引擎那一籃刪掉**：`_engine_dirs` 拿的是最深的籃子，少了那一籃就回空集合，
   整條守門會**靜靜關掉**（實測：回 0，全樹沒有一處紅）。這一刀在 `tests/` 對**真卡**
   加兩條斷言把這件事釘住——真卡算得出引擎目的地，而且就是那一個。
2. **判不出來的時候**：卡上只剩最上層那一個籃子（`tests.`）時 `_engine_dirs` 回空集合；
   沒有那一道守衛，空集合會讓**每一支**載入 `aosr` 的考卷都被判紅，訊息還會說「卡說那一籃
   的目錄是 []」。判不出來就沒有對象，不是「判不出來就一律紅」。

卡一律從 `governance/rules/` 真的載入（不是這裡抄一份、也不是造一張假的）——造假的守不到
真卡。斷言右邊一律是 list 不是整數：`assertions-not-pinned-to-counts` 咬的是把數量鎖死。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from governance.checks import green_must_be_real_green as green
from governance.loader import Card, load_all_cards, setting_tables

REPO = Path(__file__).resolve().parents[1]

CARD_ID = "green-must-be-real-green"
ONLY_TOP_BUCKET = {"bucket": [{"prefix": "tests.", "collected_floor": 1}]}
# 真卡算出來的引擎目的地。寫成清單比對（不是比長度）：這一條要釘的是「就那一個、而且是
# 那一個」，長度一樣但指到別的目錄（例如多一籃 `tests.tools.`）也要紅。
ENGINE_DIR = "tests/engine"
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


def _with_extra_bucket(card: Card) -> Card:
    """同一張卡、籃子表多一籃（模擬「引擎考卷搬去第三個籃子」的收窄）。"""
    assert card.junit, f"{CARD_ID} 竟然沒有 [junit]，這一支的前提不成立"
    buckets = [*setting_tables(card.junit, "bucket"), {"prefix": "tests.tools.", "collected_floor": 1}]
    return replace(card, junit={**card.junit, "bucket": buckets})


def test_real_card_yields_an_engine_directory() -> None:
    """真卡算得出引擎目的地——刪掉 `tests.engine.` 那一籃，整條守門會靜靜關掉。"""
    assert green._engine_dirs(_card()), (
        f"真卡 {CARD_ID} 算不出引擎目的地（`_engine_dirs` 回空集合）"
        "——這一條守門在真樹上等於關掉了：每一支放錯位置的引擎考卷都不會紅"
    )


def test_real_card_engine_directory_is_the_engine_bucket() -> None:
    """真卡算出來的引擎目的地就是 `tests/engine` 那一個，不多不少。"""
    assert green._engine_dirs(_card()) == [ENGINE_DIR], (
        f"真卡 {CARD_ID} 算出來的引擎目的地是 {green._engine_dirs(_card())}，"
        f"應該是 [{ENGINE_DIR!r}]——多一籃就多一個合法的家，引擎考卷搬過去就不紅了"
    )


def test_extra_bucket_moves_the_engine_directory() -> None:
    """多一籃之後，真卡那一條的比對要咬得到（這一支證上面那一條不是永遠為真）。"""
    assert green._engine_dirs(_with_extra_bucket(_card())) != [ENGINE_DIR]


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
