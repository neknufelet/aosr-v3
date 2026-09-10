"""樣本道具：走 builtins 這個模組叫 print。

只給 style-guard 的樣本當道具用，不是真的程式。
"""
import builtins


def scan(names):
    """屬性寫法：`builtins.print` 跟 `print` 是同一個東西。"""
    builtins.print("開始掃了")
    kept = [name for name in names if name]
    builtins.print("掃完，命中", len(kept))
    return kept
