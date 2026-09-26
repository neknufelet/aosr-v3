"""#505 指向性曲線與允許範圍的唯一登記簿；呼叫端必須給路徑。"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, model_validator


class _Frozen(BaseModel):
    # 數值一律 Strict：TOML 的 true 在寬鬆模式會被當成 1，靜靜變成一個數字（品質登記簿踩過）。
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class AllowedRange(_Frozen):
    # β 小於 0 會讓側面比正前方大聲、功率下限高過 0 dB 會讓背後比正前方大聲，都不是這個模型。
    beta_min: StrictFloat = Field(ge=0.0)
    beta_max: StrictFloat
    power_floor_min_db: StrictFloat
    power_floor_max_db: StrictFloat = Field(le=0.0)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.beta_min > self.beta_max or self.power_floor_min_db > self.power_floor_max_db:
            raise ValueError("allowed_range 上下界順序錯誤")
        return self


class TwoParameterCurve(_Frozen):
    beta_limit: StrictFloat
    beta_corner_hz: StrictFloat = Field(gt=0.0)
    beta_exponent: StrictFloat = Field(gt=0.0)
    power_floor_limit_db: StrictFloat
    power_floor_corner_hz: StrictFloat = Field(gt=0.0)
    power_floor_exponent: StrictFloat = Field(gt=0.0)


class Provenance(_Frozen):
    # 受控字：今天只有工程基線、出自量測擬合；要升級得帶新決策紙，不在這裡自創狀態。
    status: Literal["baseline"]
    source_kind: Literal["measurement_fit"]
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
