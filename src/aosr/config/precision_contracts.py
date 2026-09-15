"""精度契約登記簿的唯讀載入與欄位一致性驗證。"""
from __future__ import annotations

import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PrecisionContract(BaseModel):
    """一條具名精度契約；模型凍結且拒收未登記欄位。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    value: float = Field(gt=0.0, strict=True)
    display: str = Field(min_length=1)
    unit: Literal["relative", "absolute", "ulp"]
    decision_paper: str = Field(min_length=1)
    truth: str = Field(min_length=1)
    mutant_test: str = Field(min_length=1)

    @model_validator(mode="after")
    def display_matches_value(self) -> Self:
        """拒絕人看寫法與機器浮點值分離。"""
        if _parse_display(self.display) != self.value:
            raise ValueError("display 解析值不等於 value")
        return self


class _PrecisionContractRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: tuple[PrecisionContract, ...]


@dataclass(frozen=True)
class ComparisonTolerances:
    """路徑與總量裁判一起需要的三條登記值。"""

    reflection_ulp: float
    direct_energy_rel: float
    reflected_energy_rel_floor: float


def _parse_display(display: str) -> float:
    power = re.fullmatch(r"2\^-(\d+)", display)
    if power is not None:
        return math.ldexp(1.0, -int(power.group(1)))
    factors = display.split("·")
    if len(factors) > 2:
        raise ValueError("display 不是 2^-n 或十進位寫法")
    try:
        value = math.prod(float(factor) for factor in factors)
    except ValueError as exc:
        raise ValueError("display 不是 2^-n 或十進位寫法") from exc
    if not math.isfinite(value):
        raise ValueError("display 不是有限數")
    return value


def load_precision_contracts(path: str | Path) -> dict[str, PrecisionContract]:
    """從呼叫端必給的 TOML 路徑讀出以契約名為鍵的映射。"""
    with Path(path).open("rb") as config_file:
        loaded = tomllib.load(config_file)
    registry = _PrecisionContractRegistry.model_validate(loaded)
    contracts: dict[str, PrecisionContract] = {}
    for contract in registry.contract:
        if contract.name in contracts:
            raise ValueError(f"同名精度契約：{contract.name}")
        contracts[contract.name] = contract
    return contracts


def load_comparison_tolerances(path: str | Path) -> ComparisonTolerances:
    """從必給登記簿路徑取出路徑與總量裁判的三條界線。"""
    contracts = load_precision_contracts(path)
    return ComparisonTolerances(
        reflection_ulp=contracts["reflection_product_ulp"].value,
        direct_energy_rel=contracts["direct_energy_vs_legacy"].value,
        reflected_energy_rel_floor=contracts["reflected_energy_floor"].value,
    )
