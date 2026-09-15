"""樣本產品程式：沒有契約常數。"""

PARIS_POINTS = 256


def judge(value: float, tolerance_rel: float) -> bool:
    return value <= tolerance_rel
