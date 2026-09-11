"""引擎考卷放錯位置：tests/ 根層直接載入 engine 的套件。

這一支不是真的考卷，是必紅樣本的道具：檢查只做 ast 解析，不會真的 import 它，
所以 tests/engine/ 底下沒有真的 aosr 這棵樹也無所謂。
"""
from aosr import runtime


def test_engine_uses_runtime() -> None:
    assert runtime is not None
