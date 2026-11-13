"""樣本道具：卡頂層 [[by_design]] 登記的 SHELL_SAYS_EMPTY 已經改名，這裡只剩 SHELL_SAYS_DONE。

改名的那一個同時也是協定碼、值落在離開碼白名單裡，所以它本身不紅；紅的是卡上那一筆資料
對不上程式。
"""
SHELL_SAYS_DONE = 0


def probe() -> int:
    return SHELL_SAYS_DONE
