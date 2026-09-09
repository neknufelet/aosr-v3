"""樣本用的假測試：把一個呼叫的整數結果鎖死成 7。

壞在：``assert count_rows() == 7``。不是只有 ``len()`` 會鎖死數量，任何回整數的呼叫
直接對整數字面值都一樣：數字寫死在斷言裡，看不出那七筆是哪七筆；換掉其中一筆內容、
總數不變照樣綠，多一筆真的東西反而紅。

只給 assertions-not-pinned-to-counts 的樣本當道具用，不是真的測試。
"""


def count_rows():
    return 7


def test_rows():
    assert count_rows() == 7
