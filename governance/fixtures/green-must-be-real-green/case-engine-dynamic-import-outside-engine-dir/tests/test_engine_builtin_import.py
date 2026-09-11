"""引擎考卷用內建的 __import__ 加上字面字串放錯位置。

__import__ 跟 importlib.import_module 一樣是靜態看得見的字面字串，也是最直覺的繞法——
不放進判準就等於留一條零成本的後門。
"""


def test_builtin_import_of_engine() -> None:
    module = __import__("aosr.runtime")
    assert module is not None
