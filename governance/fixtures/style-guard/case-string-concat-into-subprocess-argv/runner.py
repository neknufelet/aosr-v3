"""樣本道具：拼出來的路徑塞進子程序的 argv。

只給 style-guard 的樣本當道具用，不是真的程式。
"""
import subprocess


def show(name):
    proc = subprocess.run(["cat", "logs" + "/" + name], capture_output=True, text=True)
    return proc.stdout
