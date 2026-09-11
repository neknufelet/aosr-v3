"""引擎考卷用關鍵字參數的字面字串放錯位置。

import_module(name="aosr.…") 與 __import__(name="aosr.…") 一樣是靜態看得見的字面值，
只讀位置參數的實作會漏掉這一種（找碴實測回 0）。
"""
import importlib


def test_keyword_import_of_engine() -> None:
    module = importlib.import_module(name="aosr.runtime")
    assert module is not None
