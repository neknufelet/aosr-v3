"""考卷樹底下一支叫 engine 的模組檔，而卡上又有 tests.engine. 那一籃。

這支檔裡類別底下的題，pytest 給的 classname 是 tests.engine.<類別名>——
收據那邊會算成 tests.engine. 那一籃，原始碼那邊算的是它住的 tests 那一層。
同一支考卷兩個答案，而 classname 講不出最後那一段是類別還是模組。
"""


class TestThing:
    def test_thing(self) -> None:
        assert isinstance("x", str)
