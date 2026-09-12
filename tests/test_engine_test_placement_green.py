"""「哪一籃是引擎那一籃」由卡面說了算，不是猜出來的（票 #148）。

以前 `_engine_dirs()` 拿「籃子表裡最深的那幾籃」當引擎的家。那是猜的，兩個方向都出過事
（`#143` 唯讀找碴當場實測過）：

1. **把 `tests.engine.` 那一籃刪掉**：猜出來是空集合，整條守門**靜靜關掉**，全樹回 0。
   後設測試看不到——它只斷言離開碼是 1、不問哪一條紅。
2. **卡上多一籃**（例如 `tests.tools.`）：引擎考卷多一個合法的家，搬過去就不紅了，
   而卡的人話寫的是單數的「那一籃」。

改成卡自己說：`[junit]` 的 `engine_bucket` 那一格指名哪一籃是引擎那一籃，檢查程式照著讀、
不再猜。這一支守那一格的權威：真卡指得出來、指到籃子表裡沒有的籃子要回「這一跑不算數」
（回 2，不是靜靜關掉）、多一籃不會多一個家、把那一格刪掉是卡壞掉。

卡一律從 `governance/rules/` 真的載入（不是這裡抄一份、也不是造一張假的）——造假的守不到真卡。
斷言右邊不鎖數量：`assertions-not-pinned-to-counts` 咬的是把數量鎖死。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from governance.checks import green_must_be_real_green as green
from governance.exit_codes import ToolBroken
from governance.loader import Card, card_problems, load_all_cards, setting_tables

REPO = Path(__file__).resolve().parents[1]

CARD_ID = "green-must-be-real-green"
CARD_FILE = "governance/rules/green-must-be-real-green.toml"
# 卡上那一格的名字（`[junit]` 底下指名引擎籃的那一格）。
ENGINE_BUCKET_KEY = "engine_bucket"
# 真卡算出來的引擎目的地。指到別的目錄（例如多一籃之後跟著漂）也要紅。
ENGINE_DIR = "tests/engine"
# 籃子表裡沒有這一籃——卡指到它就是「這一跑不算數」，不是靜靜關掉。
ABSENT_BUCKET = "tests.nowhere."
# 多出來的第三籃：跟引擎那一籃一樣深，取最深的猜法會把它也算成合法的家。
EXTRA_BUCKET = "tests.tools."
# 整棵考卷樹當一籃（16 張只有一籃的必紅樣本卡就是這個形狀）。
WHOLE_TREE_BUCKET = "tests."
FLOOR = 1

# 拼出來的檔名，不寫成完整路徑字面值：這支檔是 .py，裡面形如 `目錄/檔名.副檔名` 的字面值
# 會被 refs-and-links-resolve 拿去版控裡解析，而下面這幾支考卷只活在暫存目錄裡。
ENGINE_TEST_NAME = "test_some_engine_thing" + ".py"
ENGINE_IMPORT = "import aosr\n"


def _card() -> Card:
    """卡是從 governance/rules/ 讀出來的，不是這裡抄一份。"""
    cards = [c for c in load_all_cards(REPO) if c.id == CARD_ID]
    assert cards, f"governance/rules/ 裡找不到 id 是 {CARD_ID} 的卡——這一支就沒有對象可測"
    return cards[0]


def _buckets(card: Card) -> list[dict[str, object]]:
    assert card.junit, f"{CARD_ID} 竟然沒有 [junit]，這一支的前提不成立"
    return setting_tables(card.junit, "bucket")


def _retuned(card: Card, buckets: list[dict[str, object]], engine_bucket: str) -> Card:
    """同一張卡，換一張籃子表與一格「哪一籃是引擎那一籃」。"""
    assert card.junit, f"{CARD_ID} 竟然沒有 [junit]，這一支的前提不成立"
    return replace(card, junit={**card.junit, "bucket": buckets, ENGINE_BUCKET_KEY: engine_bucket})


def _written(root: Path, *parts: str) -> Path:
    """在暫存樹裡放一支載入引擎套件的考卷，回傳它的路徑。"""
    source = root.joinpath(*parts) / ENGINE_TEST_NAME
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(ENGINE_IMPORT, encoding="utf-8")
    return source


def test_real_card_names_a_bucket_that_is_really_in_the_table() -> None:
    """真卡指得出引擎那一籃，而且那一籃真的在它自己的籃子表裡。"""
    card = _card()
    prefixes = [prefix for prefix, _floor in card.junit_buckets]
    assert card.junit_engine_bucket in prefixes, (
        f"真卡 {CARD_ID} 宣告的引擎籃 {card.junit_engine_bucket!r} 不在籃子表 {prefixes} 裡"
    )


def test_real_card_engine_directory_is_the_engine_bucket() -> None:
    """真卡算出來的引擎目的地就是那一個目錄，不多不少。"""
    assert green._engine_dir(_card()) == ENGINE_DIR, (
        f"真卡 {CARD_ID} 算出來的引擎目的地是 {green._engine_dir(_card())!r}，"
        f"應該是 {ENGINE_DIR!r}——指到別處，放錯位置的引擎考卷就不紅了"
    )


def test_declared_bucket_missing_from_the_table_is_tool_broken() -> None:
    """卡指到籃子表裡沒有的籃子：回「這一跑不算數」，不是靜靜關掉。"""
    card = _retuned(_card(), _buckets(_card()), ABSENT_BUCKET)
    with pytest.raises(ToolBroken) as caught:
        green._engine_dir(card)
    assert ABSENT_BUCKET in str(caught.value), f"訊息沒說是哪一籃指不到：{caught.value}"


def test_deleting_the_engine_bucket_does_not_switch_the_rule_off(tmp_path: Path) -> None:
    """把引擎那一籃從籃子表裡刪掉、宣告留著：整條守門不准靜靜關掉。

    這就是找碴實測到的那個洞——舊實作在這個形狀底下回空集合、一筆都不報、全樹回 0。
    """
    kept = [b for b in _buckets(_card()) if b.get("prefix") != _card().junit_engine_bucket]
    assert kept, "籃子表刪到空的，這一支證不了「刪掉引擎那一籃」跟「整張表沒了」是兩件事"
    card = _retuned(_card(), kept, _card().junit_engine_bucket)
    source = _written(tmp_path, "tests")
    with pytest.raises(ToolBroken):
        green._placement_group_problems(card, tmp_path, [source])


def test_extra_bucket_does_not_add_a_second_home(tmp_path: Path) -> None:
    """卡上多一籃不會多一個合法的家：引擎的家還是卡上指名的那一個。"""
    card = _retuned(
        _card(),
        [*_buckets(_card()), {"prefix": EXTRA_BUCKET, "collected_floor": FLOOR}],
        _card().junit_engine_bucket,
    )
    assert green._engine_dir(card) == ENGINE_DIR
    source = _written(tmp_path, "tests", "tools")
    bad = green._placement_group_problems(card, tmp_path, [source])
    assert bad, "引擎考卷住進第三籃竟然不紅——多一籃就多一個家，正是這一票要收的洞"
    # 正面那一半：同一張三籃的卡，考卷住在卡指名的那一籃就不准被咬（多一籃不是「一律紅」）。
    settled = _written(tmp_path, "tests", "engine")
    assert green._placement_group_problems(card, tmp_path, [settled]) == []


def test_engine_test_inside_the_declared_bucket_is_not_flagged(tmp_path: Path) -> None:
    """住對地方的引擎考卷不准被誤咬（這一支證上面那幾條不是永遠為真）。"""
    source = _written(tmp_path, "tests", "engine")
    assert green._placement_group_problems(_card(), tmp_path, [source]) == []


def test_engine_test_outside_the_declared_bucket_is_flagged(tmp_path: Path) -> None:
    """住在治理層那一籃的引擎考卷要紅：放錯籃子不算放外面。"""
    source = _written(tmp_path, "tests")
    assert green._placement_group_problems(_card(), tmp_path, [source]), (
        "tests/ 根層那一支載入引擎套件的考卷沒被報出來——它會進錯籃子、把那一籃的地板灌大"
    )


def test_whole_test_tree_as_the_engine_bucket_flags_nothing(tmp_path: Path) -> None:
    """卡指名的引擎籃就是整棵考卷樹時，哪裡都算住對——那 16 張只有一籃的樣本卡就是這形狀。"""
    card = _retuned(_card(), [{"prefix": WHOLE_TREE_BUCKET, "collected_floor": FLOOR}], WHOLE_TREE_BUCKET)
    source = _written(tmp_path, "tests")
    assert green._placement_group_problems(card, tmp_path, [source]) == []


def test_card_without_the_declaration_is_a_broken_card(tmp_path: Path) -> None:
    """把那一格從卡上刪掉＝卡壞掉（讀不出判準，不是回去猜）。"""
    text = (REPO / CARD_FILE).read_text(encoding="utf-8")
    stripped = "\n".join(
        line for line in text.splitlines() if not line.startswith(ENGINE_BUCKET_KEY)
    )
    assert stripped != text, f"真卡裡根本沒有 {ENGINE_BUCKET_KEY} 那一行，這一支就白跑了"
    target = tmp_path / (CARD_ID + ".toml")
    target.write_text(stripped, encoding="utf-8")
    bad = card_problems(target, REPO)
    assert any(ENGINE_BUCKET_KEY in problem for problem in bad), (
        f"刪掉 {ENGINE_BUCKET_KEY} 之後卡竟然還讀得過：{bad}"
    )
