"""樣本用的假測試：先把數字指給同一支檔裡的名字，再拿它鎖死數量。

壞在：``EXPECTED_DEFECTS = 3`` 之後 ``assert len(findings) == EXPECTED_DEFECTS``。
等號右邊不是整數字面值，但數字還是寫死在同一支測試檔裡，行為跟直接寫 3 一模一樣，
是零成本的繞法。真的合規是對照別處登記、import 進來的值。

只給 assertions-not-pinned-to-counts 的樣本當道具用，不是真的測試。
"""

EXPECTED_DEFECTS = 3


def find_defects():
    return ["missing-field", "bad-level", "orphan-check"]


def test_defects():
    findings = find_defects()
    assert len(findings) == EXPECTED_DEFECTS
