"""樣本道具：把 print 指給另一個名字，再用那個名字說話。

只給 style-guard 的樣本當道具用，不是真的程式。
"""
p = print


def scan(names):
    """`p` 這個名字就是 print，只是換了拼法。"""
    p("開始掃了")
    kept = [name for name in names if name]
    p("掃完，命中", len(kept))
    return kept
