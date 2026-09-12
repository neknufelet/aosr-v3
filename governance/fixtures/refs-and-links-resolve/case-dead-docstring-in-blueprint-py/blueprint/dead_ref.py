"""樣本用的假產生器：docstring 指到一張不存在的決策紙。

壞在：``docs/decisions/xxx.md`` 不存在。blueprint 底下的 .py 現在也入掃描面了，但只解析
docstring 與註解、而且只認「長得像這一棵 repo 相對路徑」的 token（前綴限 docs/、
governance/、blueprint/、src/、tests/，副檔名限 .md/.py/.json/.toml）。這一份證明那半在咬。

只給 refs-and-links-resolve 的樣本當道具用，不是真的程式。
"""

SPEC = "上一代的座標，這裡刻意不寫路徑形狀的 token"


def spec_path() -> str:
    # 程式碼字串或註解要是寫了路徑，blueprint 底下只看 docstring 與註解；
    # 這裡刻意不寫，讓這份樣本只犯「docstring 死引用」這一條。
    return SPEC
