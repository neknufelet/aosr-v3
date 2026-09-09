"""控制樣本用的假測試：三條合規的斷言 ＋ 一條已知會咬的斷言。

已知會咬的最小輸入。用途不是測某個特定寫法，而是「檢查還活著」的活體證明——
這一份餵下去回 0，就代表 assertions-not-pinned-to-counts 自己死了。

同一支檔裡刻意擺三條合規的斷言：逐項具名的集合比對、對照 import 進來的登記值、
兩個 len() 互比。它們必須都不被咬（報告行的 hits 要正好是 1），
不然這支檢查就是在誤咬正常寫法。

只給 assertions-not-pinned-to-counts 的樣本當道具用，不是真的測試。
"""
from collections import Counter

REGISTERED_NAMES = {"a", "b"}


def collect_names():
    return {"a", "b"}


def find_defects():
    return ["missing-field", "bad-level", "orphan-check"]


def test_names_are_named_one_by_one():
    """合規：逐項具名的集合比對。紅的時候直接印出多了誰、少了誰。"""
    names = collect_names()
    assert names == {"a", "b"}


def test_names_match_the_registry():
    """合規：對照別處登記的值，不是在斷言裡寫死一個數字。"""
    assert collect_names() == REGISTERED_NAMES


def test_no_duplicate_names():
    """合規：兩個 len() 互比，右邊不是寫死的數字。"""
    names = list(collect_names())
    assert len(names) == len(set(names))


def test_no_leftovers():
    """合規：直接比對內容，不比數量。"""
    assert Counter(find_defects()) == Counter(["missing-field", "bad-level", "orphan-check"])


def test_defect_count_is_pinned():
    """已知會咬的那一條：把缺陷數鎖死成 3。"""
    findings = find_defects()
    assert len(findings) == 3
