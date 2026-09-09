"""樣本用的假測試：把缺陷數鎖死成 3。

壞在：``assert len(findings) == 3``。今天剛好三筆，明天多抓到一筆真缺陷，這條斷言就紅——
斷言在懲罰改善，不是在描述契約。合規寫法是逐項具名比對，或對照別處登記、import 進來的值。

只給 assertions-not-pinned-to-counts 的樣本當道具用，不是真的測試。
"""


def find_defects():
    return ["missing-field", "bad-level", "orphan-check"]


def test_defects():
    findings = find_defects()
    assert len(findings) == 3
