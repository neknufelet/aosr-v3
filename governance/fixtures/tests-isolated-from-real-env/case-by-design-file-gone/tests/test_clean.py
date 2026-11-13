"""樣本用的乾淨考卷：不 spawn 版控工具、不寫真樹。這棵樹裡沒有卡上登記的 tests/conftest.py。

只給 tests-isolated-from-real-env 的樣本當道具用。
"""


def test_adds():
    assert 1 + 1 == 2
