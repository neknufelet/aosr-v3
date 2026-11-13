"""樣本道具：登記的協定碼還在，同一個名字在模組層又被賦值了一次。"""
SHELL_SAYS_EMPTY = 7

if True:
    SHELL_SAYS_EMPTY = 500


def probe(tries: int) -> int:
    return SHELL_SAYS_EMPTY if tries > 0 else 0
