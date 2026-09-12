"""控制樣本用的假測試：幾條合規的寫法 ＋ 一條已知會咬的寫法。

已知會咬的最小輸入。用途不是測某個特定寫法，而是「檢查還活著」的活體證明——
這一份餵下去回 0，就代表 assertions-not-pinned-to-counts 自己死了。

合規的那幾條都是新牙（守門的 if 那半）刻意不該咬的形狀：逐項具名比對、對照 import 進來的
登記值、兩個 len() 互比、對照從資料算出來的值；守門的 if 沒有 raise 也不咬。它們必須都不被
咬（報告行的 hits 要正好是 1），不然這支檢查就是在誤咬正常寫法。

只給 assertions-not-pinned-to-counts 的樣本當道具用，不是真的測試。
"""
from collections import Counter

from registry import EXPECTED_DEFECTS as REGISTERED_DEFECTS

REGISTERED_NAMES = {"a", "b"}


def collect_names():
    return {"a", "b"}


def find_defects():
    return ["missing-field", "bad-level", "orphan-check"]


def expected_defect_count():
    """合規：從資料算出來的值（對照別處登記的清單長度），不是寫死的數字。"""
    return len(["missing-field", "bad-level", "orphan-check"])


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


def test_guard_against_imported_count():
    """合規：守門的 if 拿 len 跟 import 進來的登記值比，本體有 raise 也不咬。"""
    findings = find_defects()
    if len(findings) != REGISTERED_DEFECTS:
        raise AssertionError("跟登記的數量對不上")


def test_guard_against_computed_count():
    """合規：守門的 if 拿 len 跟從資料算出來的值比，本體有 raise 也不咬。"""
    findings = find_defects()
    if len(findings) != expected_defect_count():
        raise AssertionError("算出來的數量對不上")


def test_guard_against_other_len():
    """合規：守門的 if 拿 len 跟另一個 len 比，右邊不是寫死的數字。"""
    findings = find_defects()
    if len(findings) != len(set(findings)):
        raise AssertionError("有重複")


def test_guard_without_raise_is_not_pinned():
    """合規：if 條件拿 len 跟整數比，但本體沒有 raise，不算守門式鎖死。"""
    findings = find_defects()
    if len(findings) != 3:
        findings.append("extra")


def test_defect_count_is_pinned():
    """已知會咬的那一條：把缺陷數鎖死成 3。"""
    findings = find_defects()
    assert len(findings) == 3
