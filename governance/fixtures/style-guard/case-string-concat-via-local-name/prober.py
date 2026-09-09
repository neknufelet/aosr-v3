"""樣本道具：拼好的字串先指給一個名字，再拿那個名字當路徑參數。

只給 style-guard 的樣本當道具用，不是真的程式。
"""
import os


def decision_exists(name):
    where = "docs" + "/" + name
    return os.path.exists(where)
