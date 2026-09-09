"""樣本道具：四條同時犯。

只給 style-guard 的樣本當道具用，不是真的程式。
"""


def load(name):
    """第②條（字串拼路徑流進開檔）＋第③條（開檔沒有 with）＋第①條（print 當紀錄）。"""
    fh = open("logs" + "/" + name, encoding="utf-8")
    body = fh.read()
    fh.close()
    print("debug: 讀完了", name)
    return body


def accumulate_everything_by_hand(total):
    """第④條：行數超過卡上登記的門檻。"""
    step_01 = total + 1
    total = step_01
    step_02 = total + 2
    total = step_02
    step_03 = total + 3
    total = step_03
    step_04 = total + 4
    total = step_04
    step_05 = total + 5
    total = step_05
    step_06 = total + 6
    total = step_06
    step_07 = total + 7
    total = step_07
    step_08 = total + 8
    total = step_08
    step_09 = total + 9
    total = step_09
    step_10 = total + 10
    total = step_10
    step_11 = total + 11
    total = step_11
    step_12 = total + 12
    total = step_12
    step_13 = total + 13
    total = step_13
    step_14 = total + 14
    total = step_14
    step_15 = total + 15
    total = step_15
    step_16 = total + 16
    total = step_16
    step_17 = total + 17
    total = step_17
    step_18 = total + 18
    total = step_18
    step_19 = total + 19
    total = step_19
    step_20 = total + 20
    total = step_20
    step_21 = total + 21
    total = step_21
    step_22 = total + 22
    total = step_22
    step_23 = total + 23
    total = step_23
    step_24 = total + 24
    total = step_24
    step_25 = total + 25
    total = step_25
    step_26 = total + 26
    total = step_26
    step_27 = total + 27
    total = step_27
    step_28 = total + 28
    total = step_28
    step_29 = total + 29
    total = step_29
    step_30 = total + 30
    total = step_30
    step_31 = total + 31
    total = step_31
    step_32 = total + 32
    total = step_32
    step_33 = total + 33
    total = step_33
    step_34 = total + 34
    total = step_34
    step_35 = total + 35
    total = step_35
    return total
