"""引擎考卷住進第三籃：卡上有 tests.tools. 這一籃，可是引擎那一籃是 tests.engine.。

這一支不是真的考卷，是必紅樣本的道具：檢查只做 ast 解析，不會真的 import 它。
舊實作（取籃子表裡最深的那幾籃當引擎的家）在這裡回 0——tests/tools 跟 tests/engine
一樣深，於是它也算成合法的家，引擎考卷搬過去就不紅了。
"""
from aosr import runtime


def test_engine_uses_runtime() -> None:
    assert runtime is not None
