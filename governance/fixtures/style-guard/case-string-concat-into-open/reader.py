"""樣本道具：拼字串當路徑，直接餵給開檔。

開檔刻意寫在 with 的頭上，所以只犯第②條。
只給 style-guard 的樣本當道具用，不是真的程式。
"""


def read_log(name):
    with open("logs" + "/" + name, encoding="utf-8") as fh:
        return fh.read()
