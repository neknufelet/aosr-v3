"""樣本產品程式：呼叫時把密度寫死（產品真名 density_kg_m3）。"""


def total(density_kg_m3: float) -> float:
    return density_kg_m3


def run() -> float:
    return total(density_kg_m3=1.2)
