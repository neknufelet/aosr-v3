"""樣本道具：開了檔不用 with。

只給 style-guard 的樣本當道具用，不是真的程式。
"""


def read_data():
    fh = open("data.txt", encoding="utf-8")
    body = fh.read()
    fh.close()
    return body
