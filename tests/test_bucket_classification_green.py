"""收據那邊與原始碼那邊分籃必須是同一份判準（票 #149）。

同一張綠卡以前有兩份「一個東西落在哪一籃」的實作：`_bucket_prefix()` 拿收據裡的
`classname`（pytest 給每一題的模組名）去比**字串前綴**，`_bucket_of()` 拿考卷的檔案路徑
換成模組路徑之後**逐段比**。今天答案一致，但「同一件事兩份做法、沒有人是主人」正是這個
repo 一再吃過的病：改一個籃子前綴要記得改兩處，忘了一處就開始漂。

收成一支之後，核心只有一個——「一串模組段落在哪一籃」，逐段比、取最長符合——收據那邊餵
`classname`、原始碼那邊餵檔案路徑換出來的模組路徑，兩邊各自只做自己的正規化。

這一支守三件事：

1. **契約內兩邊同答案。** 同一支考卷的檔案路徑與它的 `classname` 落進同一籃。
2. **模組邊界不准被字串前綴吃掉。** 籃子前綴沒寫尾巴那一點（`tests.engine`）的時候，
   舊的字串前綴實作會把 `tests.engineering.…` 吞進引擎那一籃——那是下一次漂移的起點，
   也是這一支對舊實作的反例。
3. **契約外不准假裝一致。** 兩邊同答案是有條件的：籃子前綴指的必須是**資料夾**。
   前綴撞上同名的**模組檔**時（考卷樹底下有一支叫 `engine` 的 .py，而卡上有 `tests.engine.`
   那一籃），`tests.engine.TestThing` 講不出最後那一段是「那支模組裡的類別」還是
   「那個資料夾底下的模組」——收據那邊算出一個答案、原始碼那邊算出另一個。那時候不准挑
   一邊當答案，也不准猜類別的命名大小寫：**回 2，這一跑不算數**。
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from governance.checks import green_must_be_real_green as green
from governance.exit_codes import ToolBroken
from governance.loader import Card, load_all_cards

REPO = Path(__file__).resolve().parents[1]
CARD_ID = "green-must-be-real-green"

# 考卷樹那一層與撞名的那個名字（拼路徑用，不寫成完整路徑字面值）。
TEST_DIR = "tests"
ENGINE = "engine"
MODULE_SEPARATOR = "."

# 一籃兩格（前綴 ＋ 那一籃的收集數地板）。這一支只判分籃，地板填哪個數都不影響結果。
FLOOR = 1

# 同一張籃子表的兩種寫法。尾巴那一點是人手寫的，卡上寫不寫得出來不該改變分籃的答案。
DOTTED = (("tests.", FLOOR), ("tests.engine.", FLOOR))
UNDOTTED = (("tests", FLOOR), ("tests.engine", FLOOR))

# 把籃子改名之後的同一張表（證「改一個前綴，兩邊一起動」）。
RENAMED = (("tests.", FLOOR), ("tests.motor.", FLOOR))

# 拼出來的副檔名，不寫成完整路徑字面值：refs-and-links-resolve 會把 .py 裡形如
# `目錄/檔名.副檔名` 的字面值拿去版控裡解析，而下面這些路徑只活在這張表上。
PY = "." + "py"


def _source(*parts: str) -> str:
    """一支考卷的路徑（相對掃描根、posix 寫法）。"""
    return "/".join(parts) + PY


def _card() -> Card:
    """卡是從 governance/rules/ 讀出來的，不是這裡抄一份。"""
    cards = [c for c in load_all_cards(REPO) if c.id == CARD_ID]
    assert cards, f"governance/rules/ 裡找不到 id 是 {CARD_ID} 的卡——這一支就沒有對象可測"
    return cards[0]


def _card_with(buckets: tuple[tuple[str, int], ...]) -> Card:
    """真卡換一張籃子表。"""
    card = _card()
    assert card.junit, f"{CARD_ID} 竟然沒有 [junit]，這一支的前提不成立"
    table = [{"prefix": prefix, "collected_floor": floor} for prefix, floor in buckets]
    return replace(card, junit={**card.junit, "bucket": table})


# 一支考卷的兩個身分：它的檔案路徑，與 pytest 在收據裡給它的 `classname`。
# 每一列後面那一格是它該落進的籃子（用 DOTTED 那張表判）。
PAIRS = (
    (_source("tests", "test_build_status"), "tests.test_build_status", "tests."),
    (_source("tests", "engine", "test_aosr_runtime"), "tests.engine.test_aosr_runtime", "tests.engine."),
    (_source("tests", "engine", "deep", "test_thing"), "tests.engine.deep.test_thing", "tests.engine."),
    # 模組邊界：`engineering` 不是 `engine` 底下的東西，前綴比對不准把它吃進去。
    (_source("tests", "engineering", "test_thing"), "tests.engineering.test_thing", "tests."),
    # 考卷樹底下叫 `engine` 的那**一支檔**（不是那個資料夾）：它住在 `tests` 那一層，
    # 不住引擎那一籃——模組與套件同名正是分籃最容易錯的地方。
    (_source("tests", "engine"), "tests.engine", "tests."),
    # 哪一籃都不是的，兩邊都要回「沒有籃子」。
    (_source("legacy", "test_orphan"), "legacy.test_orphan", ""),
)

# 同一件事的 UNDOTTED 版答案（跟上面那張表逐列對應，只有前綴的寫法不同）。
UNDOTTED_BUCKETS = ("tests", "tests.engine", "tests.engine", "tests", "tests", "")

# 收據裡還有一種形狀：類別底下的題，`classname` 後面多接一段類別名（巢狀類別會更多段）。
# 分籃只看前面幾段，所以類別叫什麼、大小寫怎麼寫、巢狀幾層都不該改變答案——這一支不猜
# 哪一段是類別，也就不會去猜別人的命名習慣。
CLASS_BASED = (
    ("tests.engine.test_thing.TestThing", "tests.engine."),
    ("tests.engine.test_thing.TestOuter.TestInner", "tests.engine."),
    ("tests.engine.test_thing.lowercase_class", "tests.engine."),
    ("tests.test_build_status.TestThing", "tests."),
    ("tests.engineering.test_thing.TestThing", "tests."),
)

# 分籃兩邊同答案的**支援契約**：籃子前綴指的是資料夾，不准跟同名的模組檔撞名。
# 撞名的那一種長這樣——考卷樹底下有一支叫 `engine` 的模組檔，而卡上又有 `tests.engine.`
# 那一籃：`tests.engine.TestThing` 講不出最後那一段是「那支模組裡的類別」還是
# 「那個資料夾底下的模組」，兩邊必然各自解讀。那時候的正解是回 2（這一跑不算數），
# 不是挑一邊當答案——由 `_assert_buckets_name_directories` 守著，下面兩支盯它。
COLLIDING_CLASSNAME = "tests.engine.TestThing"

# 撞名不只發生在前綴的**最後一段**：前綴寫得比模組檔還深的時候，撞的是**祖先**那一段。
# 卡上寫 `tests.test_engine.TestThing.` 那一籃，而考卷樹底下真的有一支叫 test_engine 的
# 模組檔——`tests.test_engine.TestThing.TestInner`（巢狀類別）收據那邊會落進那一籃，
# 原始碼那邊那支檔住的是 tests 那一層。只查最後一段查不到這一種。
ANCESTOR_MODULE = "test_engine"
ANCESTOR_CLASS = "TestThing"
ANCESTOR_BUCKETS = (
    ("tests.", FLOOR),
    (MODULE_SEPARATOR.join((TEST_DIR, ANCESTOR_MODULE, ANCESTOR_CLASS)) + MODULE_SEPARATOR, FLOOR),
)
ANCESTOR_CLASSNAME = MODULE_SEPARATOR.join((TEST_DIR, ANCESTOR_MODULE, ANCESTOR_CLASS, "TestInner"))


@pytest.mark.parametrize(("rel", "classname", "bucket"), PAIRS)
def test_both_sides_agree_on_the_same_test(rel: str, classname: str, bucket: str) -> None:
    """同一支考卷的路徑與 classname 落進同一籃，而且就是表上那一籃。"""
    from_source = green._bucket_of_source(rel, DOTTED)
    from_receipt = green._bucket_of_classname(classname, DOTTED)
    assert from_source == from_receipt, (
        f"{rel} 與它的 classname {classname} 分到不同籃子（原始碼那邊 {from_source!r}、"
        f"收據那邊 {from_receipt!r}）——同一件事兩份做法就是這樣開始漂的"
    )
    assert from_source == bucket, f"{rel} 落進 {from_source!r}，應該是 {bucket!r}"


@pytest.mark.parametrize(("pair", "bucket"), tuple(zip(PAIRS, UNDOTTED_BUCKETS, strict=True)))
def test_prefix_without_a_trailing_dot_still_stops_at_the_module_boundary(
    pair: tuple[str, str, str], bucket: str
) -> None:
    """前綴少了尾巴那一點，分籃的答案不准改變——舊的字串前綴實作在這裡會誤咬。

    `tests.engineering.test_thing` 拿去比字串前綴 `tests.engine` 是符合的，逐段比不符合。
    誤咬的代價是那一題被算進引擎那一籃，兩邊的地板同時算錯。
    """
    rel, classname, _dotted_bucket = pair
    assert green._bucket_of_source(rel, UNDOTTED) == bucket
    assert green._bucket_of_classname(classname, UNDOTTED) == bucket


@pytest.mark.parametrize(("classname", "bucket"), CLASS_BASED)
def test_class_based_classname_lands_in_the_module_bucket(classname: str, bucket: str) -> None:
    """類別底下的題（classname 多一段或多段類別名）跟它那支檔同一籃。

    分籃只比前面幾段，所以巢狀類別、小寫的類別名都落在同一籃——不猜哪一段是類別。
    """
    assert green._bucket_of_classname(classname, DOTTED) == bucket


def test_a_bucket_prefix_that_is_really_a_module_file_is_tool_broken(tmp_path: Path) -> None:
    """籃子前綴撞上同名的模組檔：回「這一跑不算數」，不准挑一邊當答案。

    這就是兩邊會各自解讀的那一種——`tests/engine` 那支**模組檔**裡類別底下的題，
    收據那邊算出 `tests.engine.`、原始碼那邊算出 `tests.`。判不出來就不判。
    """
    collision = tmp_path / TEST_DIR / (ENGINE + PY)
    collision.parent.mkdir(parents=True)
    collision.write_text("", encoding="utf-8")
    # 這一支檔真的是那個沒有唯一答案的形狀：兩邊現在就對不起來。
    assert green._bucket_of_classname(COLLIDING_CLASSNAME, DOTTED) != green._bucket_of_source(
        collision.relative_to(tmp_path).as_posix(), DOTTED
    )
    with pytest.raises(ToolBroken) as caught:
        green._assert_buckets_name_directories(_card_with(DOTTED), tmp_path, [collision])
    assert ENGINE in str(caught.value), f"訊息沒說是哪一支檔撞名：{caught.value}"


def test_a_bucket_prefix_crossing_an_ancestor_module_is_tool_broken(tmp_path: Path) -> None:
    """前綴**穿過**一支模組檔（撞的是祖先那一段，不是最後一段）：一樣回「這一跑不算數」。

    只查前綴最後那一段的實作在這裡不開槍：`tests/test_engine/TestThing` 那支 .py 不存在，
    可是 `tests/test_engine` 那支存在——巢狀類別的 classname 照樣會落進那一籃，
    而原始碼那邊算的是那支模組檔住的那一層。
    """
    collision = tmp_path / TEST_DIR / (ANCESTOR_MODULE + PY)
    collision.parent.mkdir(parents=True)
    collision.write_text("", encoding="utf-8")
    rel = collision.relative_to(tmp_path).as_posix()
    # 先證這個形狀真的沒有唯一答案：兩邊現在就對不起來。
    assert green._bucket_of_classname(ANCESTOR_CLASSNAME, ANCESTOR_BUCKETS) != green._bucket_of_source(
        rel, ANCESTOR_BUCKETS
    )
    with pytest.raises(ToolBroken) as caught:
        green._assert_buckets_name_directories(_card_with(ANCESTOR_BUCKETS), tmp_path, [collision])
    assert ANCESTOR_MODULE in str(caught.value), f"訊息沒說是哪一支檔撞名：{caught.value}"


def test_a_deep_prefix_whose_ancestors_are_real_directories_is_not_tool_broken(tmp_path: Path) -> None:
    """對照組：同一張深前綴的籃子表，每一段祖先都是真的資料夾就不准開槍，而且兩邊同答案。"""
    source = tmp_path / TEST_DIR / ANCESTOR_MODULE / ANCESTOR_CLASS / ("test_thing" + PY)
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    green._assert_buckets_name_directories(_card_with(ANCESTOR_BUCKETS), tmp_path, [source])
    rel = source.relative_to(tmp_path).as_posix()
    assert green._bucket_of_source(rel, ANCESTOR_BUCKETS) == green._bucket_of_classname(
        ANCESTOR_CLASSNAME, ANCESTOR_BUCKETS
    )


def test_a_bucket_prefix_pointing_at_a_directory_is_not_tool_broken(tmp_path: Path) -> None:
    """對照組：同一張籃子表、同一棵樹，只差那支撞名的檔換成資料夾底下的模組，就不准開槍。"""
    source = tmp_path / TEST_DIR / ENGINE / ("test_thing" + PY)
    source.parent.mkdir(parents=True)
    source.write_text("", encoding="utf-8")
    green._assert_buckets_name_directories(_card_with(DOTTED), tmp_path, [source])


def test_renaming_a_prefix_moves_both_sides_together() -> None:
    """改一個籃子前綴，收據那邊與原始碼那邊一起改——沒有第二處要記得跟著動。

    籃子表從 `tests.engine.` 改名成 `tests.motor.` 之後：新名字那一層的考卷兩邊都落進新籃，
    舊名字那一層的考卷兩邊都退回上一層那一籃。哪一邊漏改，這兩組斷言就對不起來。
    """
    moved_source = _source("tests", "motor", "test_thing")
    assert green._bucket_of_source(moved_source, RENAMED) == "tests.motor."
    assert green._bucket_of_classname("tests.motor.test_thing", RENAMED) == "tests.motor."

    stale_source = _source("tests", "engine", "test_thing")
    assert green._bucket_of_source(stale_source, RENAMED) == "tests."
    assert green._bucket_of_classname("tests.engine.test_thing", RENAMED) == "tests."
