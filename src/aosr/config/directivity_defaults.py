"""#505 指向性曲線與允許範圍的唯一登記簿；呼叫端必須給路徑。"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class AllowedRange(_Frozen):
    beta_min: float
    beta_max: float
    power_floor_min_db: float
    power_floor_max_db: float

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.beta_min > self.beta_max or self.power_floor_min_db > self.power_floor_max_db:
            raise ValueError("allowed_range 上下界順序錯誤")
        return self


class TwoParameterCurve(_Frozen):
    beta_limit: float
    beta_corner_hz: float = Field(gt=0.0)
    beta_exponent: float = Field(gt=0.0)
    power_floor_limit_db: float
    power_floor_corner_hz: float = Field(gt=0.0)
    power_floor_exponent: float = Field(gt=0.0)


class Provenance(_Frozen):
    status: str = Field(min_length=1)
    source_kind: str = Field(min_length=1)
    source: str = Field(min_length=1)
    conditions: str = Field(min_length=1)


class DirectivityDefaults(_Frozen):
    two_parameter: TwoParameterCurve
    allowed_range: AllowedRange
    provenance: Provenance

    @model_validator(mode="after")
    def limits_within_allowed_range(self) -> Self:
        curve, bounds = self.two_parameter, self.allowed_range
        if not bounds.beta_min <= curve.beta_limit <= bounds.beta_max:
            raise ValueError("beta_limit 超出 allowed_range")
        if not bounds.power_floor_min_db <= curve.power_floor_limit_db <= bounds.power_floor_max_db:
            raise ValueError("power_floor_limit_db 超出 allowed_range")
        return self


def load_directivity_defaults(path: str | Path) -> DirectivityDefaults:
    """依呼叫端給定的路徑讀 TOML；缺欄、多欄、非法數值皆拒收。"""
    with Path(path).open("rb") as file:
        return DirectivityDefaults.model_validate(tomllib.load(file))
