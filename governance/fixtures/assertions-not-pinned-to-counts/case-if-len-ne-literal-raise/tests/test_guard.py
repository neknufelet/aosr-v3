"""樣本用的假測試：用守門的 ``if`` 把缺陷數鎖死成 3。

壞在：``if len(findings) != 3: raise AssertionError(...)``。跟 ``assert len(findings) == 3``
是同一個病——今天剛好三筆，明天多抓到一筆真缺陷，這道閘就紅，守門的閘變成在懲罰改善。
換成 ``assert`` 之外的寫法躲不掉：數量寫死在測試檔裡，行為一模一樣。

只給 assertions-not-pinned-to-counts 的樣本當道具用，不是真的測試。
"""


def find_defects():
    return ["missing-field", "bad-level", "orphan-check"]


def test_defects():
    findings = find_defects()
    if len(findings) != 3:
        raise AssertionError("缺陷數應該是 3")
