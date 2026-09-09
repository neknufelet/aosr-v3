"""樣本道具：一支拿 print 當紀錄用的假檢查程式。

只給 style-guard 的樣本當道具用，不是真的檢查。
"""
from pathlib import Path


def scan(root):
    print("開始掃了")
    hits = []
    for path in sorted(Path(root).iterdir()):
        print("看到", path.name)
        if path.suffix == ".py":
            hits.append(path.name)
    print("掃完，命中", len(hits))
    return hits
