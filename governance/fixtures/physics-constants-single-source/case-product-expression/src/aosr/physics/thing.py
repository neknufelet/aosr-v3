"""樣本產品程式：ρc 用密度乘聲速的純數字運算式寫死。"""


def total(rho_c_pa_s_per_m: float) -> float:
    return rho_c_pa_s_per_m


def run() -> float:
    return total(rho_c_pa_s_per_m=1.2 * 343.0)
