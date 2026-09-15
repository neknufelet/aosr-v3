"""引擎考卷從唯一登記簿取精度契約，不從產品模組拿尺。"""
from pathlib import Path

from aosr.config.precision_contracts import load_precision_contracts


REGISTRY_PATH = Path(__file__).resolve().parents[2] / "blueprint" / "precision_contracts.toml"
CONTRACTS = load_precision_contracts(REGISTRY_PATH)


def contract_value(name: str) -> float:
    """回傳一條已驗過 display/value 一致的正式門檻。"""
    return CONTRACTS[name].value
