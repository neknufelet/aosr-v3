"""樣本產品程式：把契約門檻藏在函式簽章的預設值裡。"""


def judge(value: float, tolerance_rel: float = 2.0 ** -20) -> bool:
    return value <= tolerance_rel
