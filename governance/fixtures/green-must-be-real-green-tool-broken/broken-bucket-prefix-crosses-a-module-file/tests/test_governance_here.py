"""治理層的考卷：不載入引擎的套件，住 tests/ 根層是對的。

這一支是對照用的：它跟隔壁那支放錯位置的引擎考卷住同一層、classname 也落在同一個籃子
（tests.），差別只在「有沒有載入 aosr」。第三組要是寫成「tests/ 根層的考卷一律紅」，
這棵樣本就會多一筆違規——那個實作在這裡亮燈。
"""


def test_governance_helper() -> None:
    assert isinstance("x", str)
