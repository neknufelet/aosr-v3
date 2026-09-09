"""樣本用的假測試：把數量鎖死成 0。

壞在：``assert len(errors) == 0``。它不會懲罰改善（0 底下沒有更好的），但一樣把
「是哪幾筆」藏起來——紅的時候只印得出 1 != 0，印不出那一筆是什麼。合規寫法
``assert errors == []`` 或 ``assert not errors``，改起來零成本而且紅得比較有用。

只給 assertions-not-pinned-to-counts 的樣本當道具用，不是真的測試。
"""


def collect_errors():
    return []


def test_no_errors():
    errors = collect_errors()
    assert len(errors) == 0
