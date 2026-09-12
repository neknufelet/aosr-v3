"""新家第 3 層：材料。

這個檔刻意不 re-export 任何東西，理由跟套件根的門面一樣（規矩卡
``layers-import-downward-only`` 把資料夾的 ``__init__.py`` 當成站在所有層之上的門面）。
上一代那個門面把 ``response`` 與 ``source`` 的名字全部拉出來，於是「載入材料這一塊」
就等於「載入 JAX」——第 2 層想拿一個型別都得先付那個代價。新家不帶那個門面：要什麼
就從那一支拿（``from aosr.materials.response import FrequencyAxis``）。
"""
from __future__ import annotations
