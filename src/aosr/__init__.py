"""新家的門面。

v3 裡只有這個套件，一塊一塊重新長（決策紙
``docs/decisions/engine-not-imported-new-home-grows-block-by-block.md``）。

這個檔刻意不 re-export 任何東西：規矩卡 ``layers-import-downward-only`` 把套件根的
``__init__.py`` 當成站在所有層之上的門面，它 import 得到每一層，而任何一層 import 它
就是往上引。門面裡放 re-export 等於給每一層開一條繞過分層的近路，所以這裡空著。
"""
from __future__ import annotations
