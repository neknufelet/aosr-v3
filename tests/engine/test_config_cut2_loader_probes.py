"""這一刀那 9 支載入器的考卷：**逐筆跟 donor 標準答案比**（case 表宣告的每一筆）。

**這一支在守什麼。** 答案檔住 ``blueprint/``（``identity-strings-generated`` 與
``refs-and-links-resolve`` 都扣掉那一層），所以「它還有幾筆、內容對不對」只有考卷這一邊
在守。守的方式是三層：

1. ``test_every_declared_case_has_at_least_one_answer``——答案檔的 case id 集合**等於**
   case 表宣告的集合（少一筆紅、多一筆也紅；比的是具名的集合，不是筆數）。
2. 下面的參數化——**每一筆** declared case 都跟答案檔裡那一筆逐格比（載入器回傳的
   物件逐欄比、例外比型別與正規化訊息）。參數化的清單就是 case 表，所以「刪掉一筆 case」
   會讓這一支少一題，而第 1 條會當場紅。
3. 答案檔那一邊是**產生器在唯讀的 v2 工作樹上跑出來的**；這一支只比，不產。要改答案就
   重跑產生器（人不碰裡面的數字）。

**這一支的名字改過（票 #163）。** 那一條原本叫 `…has_exactly_one_answer`，但它比的是
集合相等——集合看不出「一對一」，同一個 id 兩筆照樣過。真正的唯一性由
`tests/engine/test_config_cut1_case_table.py` 的
`test_answer_file_and_case_table_have_unique_ids` 與這裡的
`test_declared_case_ids_are_unique` 各守一半；名字改成它真的在做的事，不然下一個讀的人
會以為這裡已經蓋到唯一性了。
"""
from __future__ import annotations

from typing import cast

import pytest

from blueprint import config_cut1_cases as cases
from tests.engine._config_answers import (
    answer_case_ids,
    as_plain,
    check_case_id_uniqueness,
    declared_case_ids,
    decode,
    expected_block,
    is_approx,
    loader_case_run,
    probe_by_id,
    probe_raised,
)

# 這一刀（票 #127 後半）那 9 支模組宣告的 case id，排序好當參數化清單——**清單就是 case 表**，
# 不是這裡另外抄一份。刪一筆 case 就少一題，少的那一題會讓下面第 1 條紅。
CUT2_CASE_IDS: tuple[str, ...] = tuple(sorted(cases.cut2_case_ids()))


def test_every_declared_case_has_at_least_one_answer() -> None:
    """答案檔的 case id 集合＝case 表宣告的集合（少一筆紅、多一筆也紅）。

    這一條**只比集合**（所以名字不叫 `exactly_one_answer`）：唯一性那一格由
    `test_declared_case_ids_are_unique` 與 case 表那一支的
    `test_answer_file_and_case_table_have_unique_ids` 守。
    """
    missing = sorted(declared_case_ids() - answer_case_ids())
    extra = sorted(answer_case_ids() - declared_case_ids())
    assert not missing, f"答案檔少了 case 表宣告的這幾筆：{missing}"
    assert not extra, f"答案檔有 case 表沒宣告的這幾筆：{extra}"


def test_declared_case_ids_are_unique() -> None:
    """答案檔與 case 表的 id 都不准重複（票 #163：矛盾的雙胞胎可以共存）。

    跟上一條的集合相等是兩件事。控制組（重複紅／少一筆紅／改值紅）住在
    `tests/engine/test_config_judge_message.py`。
    """
    check_case_id_uniqueness()


def test_every_declared_cut2_case_has_a_donor_probe() -> None:
    """這一刀宣告的每一筆 case，答案檔裡都要有（逐筆具名比，不比筆數）。"""
    present = {case_id for case_id in answer_case_ids() if case_id.startswith("cut2_")}
    missing = sorted(set(CUT2_CASE_IDS) - present)
    assert not missing, f"答案檔少了這一刀的這幾筆：{missing}"


def test_cut2_case_ids_are_not_empty_and_cover_every_module() -> None:
    """這一刀每一支模組都有東西被裁判（case 或常數），而且整組不是空的。

    ``cut2_fem_lane`` 是刻意的例外：它整支都是模組層常數（含三個載入期讀檔算出來的），
    沒有公開函式可以餵，所以「每一支都有探針」對它不成立。**它的裁判是另一支考卷**
    （``tests/engine/test_config_cut2_constants.py``：逐個拿 ``getattr(模組, NAME)``
    跟答案檔的凍結值比）——不是這一句話。獨立驗證抓到過一次「話寫在這裡、沒有實作」：
    當時這一支只迭代探針，`fem_lane` 的常數改壞了整套照樣綠。
    """
    assert CUT2_CASE_IDS
    for name in cases.CUT2_MODULES:
        table_name = f"cut2_{name}"
        assert cases.cases_for(table_name) or cases.constants_for(table_name), (
            f"{table_name} 既沒有 case 也沒有常數——這一支沒有裁判"
        )
    without_constants = [
        name for name in cases.CUT2_MODULES if not cases.constants_for(f"cut2_{name}")
    ]
    assert without_constants == [], f"這幾支沒有宣告常數：{without_constants}"


@pytest.mark.parametrize("case_id", CUT2_CASE_IDS)
def test_declared_case_matches_the_donor_answer(case_id: str) -> None:
    """這一筆 case：新家跑出來的東西跟 donor 標準答案那一筆逐格相同。

    成功的那幾筆比整個回傳物件（逐欄、巢狀的也逐欄；浮點逐位元，不用容差）；失敗的
    那幾筆比**例外型別**與**正規化過的訊息**（絕對路徑收掉之後剩下的那一串）。
    """
    expected = expected_block(probe_by_id(case_id))
    with loader_case_run(case_id) as actual:
        raised = probe_raised(expected)
        if raised is not None:
            assert isinstance(actual, dict), f"{case_id}：donor 說會炸，新家卻回了一個值"
            got = cast(dict[str, object], actual)
            assert "raised" in got, f"{case_id}：donor 說會炸，新家回的是 {sorted(got)}"
            got_raised = cast(dict[str, object], got["raised"])
            assert got_raised["type"] == raised["type"], (
                f"{case_id}：donor 炸 {raised['type']}，新家炸 {got_raised['type']}"
            )
            assert got_raised["message"] == raised["message"], (
                f"{case_id}：訊息跟 donor 不一樣\n"
                f"  donor: {raised['message']!r}\n"
                f"  新家 : {got_raised['message']!r}"
            )
            return
        assert not (isinstance(actual, dict) and "raised" in actual), (
            f"{case_id}：donor 說會成功，新家炸了 {actual!r}"
        )
        # 成功那幾筆的 expected **本身就是值記號**（`{"kind": …}`），所以直接解它；
        # 解出來的容器跟新家那一個物件都先走 `as_plain` 再逐欄比。
        want = as_plain(decode(expected))
        assert is_approx(as_plain(actual), want), f"{case_id}：新家跑出來的值跟 donor 不一樣"
