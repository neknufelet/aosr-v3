"""樣本道具：登記的協定碼還在，同一支檔又多了一個沒登記的門檻常數。"""
SHELL_SAYS_EMPTY = 7
MAX_RETRIES = 5


def probe(tries: int) -> int:
    return SHELL_SAYS_EMPTY if tries > MAX_RETRIES else 0
