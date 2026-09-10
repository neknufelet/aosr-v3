"""樣本道具：改名 import 進來的 print。

只給 style-guard 的樣本當道具用，不是真的程式。
"""
from builtins import print as write


def scan(names):
    """`write` 這個名字是 import 進來的 print，只是改了名。"""
    write("開始掃了")
    kept = [name for name in names if name]
    write("掃完，命中", len(kept))
    return kept
