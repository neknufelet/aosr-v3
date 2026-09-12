"""控制樣本用的假產生器：docstring 指到一份**存在**的設計說明。

``notes/design-note.md`` 在控制樣本這棵迷你樹裡找得到，所以這一個引用必須**不紅**——
它證明 blueprint/*.py 入掃描面之後，收窄判準（只解析 docstring 與註解、只認 repo 相對路徑）
沒有誤咬「指到存在檔案」的正常情形。

只給 refs-and-links-resolve 的樣本當道具用，不是真的程式。
"""

SPEC = "程式碼字串裡的路徑 blueprint 底下不管"


def spec_path() -> str:
    return SPEC
