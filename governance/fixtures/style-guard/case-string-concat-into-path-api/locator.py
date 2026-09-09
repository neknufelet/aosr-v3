"""樣本道具：拼字串當路徑，交給 Path()。

只給 style-guard 的樣本當道具用，不是真的程式。
"""
from pathlib import Path


def receipt_path(name):
    return Path("governance" + "/" + name)
