"""case 表與答案檔的**集合關係**：答案檔的份量下限住在這裡。

**這一條在守什麼（PR #150 的找碴 F1）。** 答案檔住 ``blueprint/``，是
``identity-strings-generated`` 與 ``refs-and-links-resolve`` 都扣掉的地方——沒有任何機器
在掃它。原本的形狀是「產生器自己去上一代跑一輪、考卷逐筆跟答案檔比」，那份考卷只保證
那一串**非空**：把答案檔的探針逐筆刪掉，只有少數幾筆刪了會紅，其餘刪掉整套照樣綠。

修法是把「有哪些 case」搬進版控（``blueprint/config_cut1_cases.py``），產生器照它跑、
考卷用這一支斷言**答案檔的 id 集合等於它宣告的集合**：少一筆紅、多一筆也紅。
比的是具名的集合，不是筆數（``assertions-not-pinned-to-counts`` 咬後者）。

票 #127 後半（讀檔那 9 支）沿用同一份表與同一份答案檔：這一支的模組名單跟著長成兩組
（第 1 刀 11 支 ＋ 這一刀 9 支），集合比對那一條照樣看整份答案檔。
"""
from __future__ import annotations

import re

from blueprint import config_cut1_cases as cases
from tests.engine._config_answers import (
    CUT1_MODULES,
    CUT2_TABLE_MODULES,
    answer_case_ids,
    check_case_ids,
    declared_case_ids,
    donor_provenance,
)


def test_answer_file_has_exactly_the_declared_cases() -> None:
    """答案檔的 case id 集合＝case 表宣告的集合（少一筆紅、多一筆也紅）。"""
    check_case_ids()


def test_declared_and_present_sets_are_not_empty() -> None:
    """兩邊都不是空的（空集合會讓上面那一條變成「兩個空集合相等」的假綠）。"""
    assert declared_case_ids()
    assert answer_case_ids()


def test_the_two_halves_are_disjoint_and_cover_the_whole_table() -> None:
    """兩刀的名單互不重疊，合起來就是整張表（少一支＝某一刀漏了模組）。"""
    cut1 = set(CUT1_MODULES)
    cut2 = set(CUT2_TABLE_MODULES)
    assert not (cut1 & cut2), f"兩份名單重疊：{sorted(cut1 & cut2)}"
    assert cut1 | cut2 == set(cases.MODULES), (
        f"漏掉的模組：{sorted(set(cases.MODULES) - (cut1 | cut2))}"
    )


def test_every_module_in_the_case_table_has_constants() -> None:
    """case 表宣告的每一支模組都有常數要凍結（沒有常數的模組是漏填）。"""
    for name in sorted(cases.MODULES):
        assert cases.constants_for(name), f"{name} 在 case 表裡沒有宣告任何常數"


def test_donor_provenance_is_measured_not_declared() -> None:
    """答案檔的 donor 出身有記號、解析出來的完整 sha、以及「那棵樹是乾淨的」。"""
    donor = donor_provenance()
    assert donor["tag"] == "v3-donor"
    commit = str(donor["commit"])
    assert re.fullmatch(r"[0-9a-f]{40}", commit), f"donor commit 不是完整的十六進位 sha：{commit!r}"
    assert donor["clean"] is True, "答案檔說 donor 樹不乾淨——那就不該拿它當標準答案"
