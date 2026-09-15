"""引擎考卷從唯一登記簿取精度契約，不從產品模組拿尺。"""
from pathlib import Path

from aosr.config.precision_contracts import load_precision_contracts


REGISTRY_PATH = Path(__file__).resolve().parents[2] / "blueprint" / "precision_contracts.toml"
CONTRACTS = load_precision_contracts(REGISTRY_PATH)

# 變異考卷的取樣間距，不是契約門檻：界線內那一點取「差÷界線 = 1−δ」、界線外那一點取
# 「差÷界線 = 1+δ」，δ 只在這裡寫一次，九支考卷從這裡讀，門檻值仍然只住登記簿。
MUTANT_MARGIN = 2.0**-10
"""界線兩側各推開多少（相對界線的比值），取 2^-10 讓夾擠收窄到界線附近。"""


def inside_factor(contract_fraction: float) -> float:
    """答案值的乘數：讓「差÷界線」**剛好**等於 ``contract_fraction``。

    產品判 ``|算出來的 − 答案| ≤ T·答案``（界線乘的是答案那一格），要讓
    ``|算出來的 − 答案| / (T·答案)`` 剛好是 ``f``（``f = (1∓δ)·T`` 除以 ``T``），
    答案得放 ``算出來的 / (1 − f·T)``——不是 ``·(1 + f·T)``，那樣會偏一個 ``T`` 量級的一階項
    （第四輪找碴點的）。傳入的 ``contract_fraction`` 就是 ``f·T``。
    """
    return 1.0 / (1.0 - contract_fraction)


def contract_value(name: str) -> float:
    """回傳一條已驗過 display/value 一致的正式門檻。"""
    return CONTRACTS[name].value
