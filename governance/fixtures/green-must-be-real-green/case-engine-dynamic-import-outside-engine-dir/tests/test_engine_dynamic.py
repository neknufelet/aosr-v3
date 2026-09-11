"""引擎考卷用字面字串的動態載入放錯位置。"""
import importlib


def test_dynamic_import_of_engine() -> None:
    module = importlib.import_module("aosr.runtime")
    assert module is not None
