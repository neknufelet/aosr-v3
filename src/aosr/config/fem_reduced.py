"""有限元素替身方程式（投影式降階）建模設定的唯讀載入（#561）。

路徑必給（``docs/decisions/config-loaders-keep-path-required.md``）；正式檔是
``config_path("fem_reduced.toml")``。這裡只有建模的選點與截斷設定，替身準不準的尺住精度契約登記簿。
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReducedModelSettings(BaseModel):
    """起點、殘差貪婪補點、完整解殘差上限與截斷比例；凍結、拒收未登記欄位。"""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    initial_impedance_count: int = Field(strict=True, ge=1)
    initial_frequency_count: int = Field(strict=True, ge=1)
    candidates_per_round: int = Field(strict=True, ge=1)
    batch: int = Field(strict=True, ge=1)
    stop_residual: float = Field(gt=0.0, lt=1.0)
    max_full_solves: int = Field(strict=True, ge=1)
    full_solve_max_relative_residual: float = Field(gt=0.0, lt=1.0)
    svd_relative_cutoff: float = Field(gt=0.0, lt=1.0)
    greedy_seed: int = Field(strict=True, ge=0)

    @model_validator(mode="after")
    def _batch_within_candidates(self) -> ReducedModelSettings:
        if self.batch > self.candidates_per_round:
            raise ValueError("batch 不能大於 candidates_per_round")
        return self


def load_reduced_model_settings(path: str | Path) -> ReducedModelSettings:
    """讀 ``[build]`` 一節；缺節或欄位不合就報錯。"""
    resolved = Path(path)
    with resolved.open("rb") as config_file:
        data = tomllib.load(config_file)
    try:
        section = data["build"]
    except KeyError as exc:
        raise ValueError(f"{resolved} 必須有 [build] 一節") from exc
    return ReducedModelSettings(**section)
