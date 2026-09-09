"""樣本用的假測試：鎖死的數量藏在 and 串起來的斷言中間。

壞在：``assert len(parts) == 3 and ...``。只比對斷言最外層是看不到的，要往 and 底下
（AST 的 BoolOp）走才找得到。這一份不是想像出來的壞法——立這張卡的時候，主線的
``tests/test_fixture_runner.py`` 自己就是這個形狀。

只給 assertions-not-pinned-to-counts 的樣本當道具用，不是真的測試。
"""


def split_report(line):
    return line.split()


def test_report_line():
    parts = split_report("scan_root=. files=1 hits=0")
    assert len(parts) == 3 and parts[1].startswith("files=")
