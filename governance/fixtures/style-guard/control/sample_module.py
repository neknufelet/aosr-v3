"""控制樣本用的假模組：五條合規的寫法 ＋ 一條已知會咬的。

整份檔很短，所以第⑤條（一支檔的行數上限）也在這裡當「不誤咬」的證據。

只給 style-guard 的樣本當道具用，不是真的程式。
"""
import json
from pathlib import Path


def dump_report(root, name, rows):
    """合規：字串相加流進的是**檔案內容**，不是路徑。這一行是「不誤咬」的證據。"""
    body = json.dumps(rows, ensure_ascii=False) + "\n"
    (root / name).write_text(body, encoding="utf-8")


def read_first_line(root, name):
    """合規：開檔在 with 的頭上，而且路徑是用 / 接出來的，不是字串拼。"""
    with open(root / name, encoding="utf-8") as fh:
        return fh.readline()


def read_whole(root, name):
    """合規：根本沒開檔，Path 自己關。"""
    return Path(root, name).read_text(encoding="utf-8")


def summarise(rows):
    """合規：短函式，分支數也少。"""
    kept = [row for row in rows if row]
    return len(kept)


def load_and_log(root, name):
    """違規（已知會咬）：拿 print 當紀錄用，這個檔不在輸出層白名單上。"""
    rows = read_whole(root, name)
    print("debug: 讀到了", len(rows), "個字")
    return rows
