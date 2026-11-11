"""網頁替新切法建替身時的訓練／留出材料抽法與建模行程數的唯讀載入（#1098）。

路徑必給（``docs/decisions/config-loaders-keep-path-required.md``）；正式檔是 ``config_path("surrogate_training.toml")``。
建模的停止規則住 ``fem_reduced.toml``、替身准不准用的尺住精度契約登記簿，這裡都沒有。
"""
from __future__ import annotations

import math
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class RandomSets(_Frozen):
    seed: int = Field(strict=True, ge=0)
    random_sets: int = Field(strict=True, ge=0)


class Sampling(_Frozen):
    bare_probabilities: tuple[float, ...] = Field(min_length=1)
    rigid_every: int = Field(strict=True, ge=1)
    family_weights: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def _checked(self) -> Sampling:
        if any(not 0.0 <= value < 1.0 for value in self.bare_probabilities):
            raise ValueError("bare_probabilities 每一格要在 [0, 1) 之間（1 就是整組都原牆，不用抽）")
        if any(value <= 0.0 for value in self.family_weights.values()):
            raise ValueError("family_weights 每一格要大於 0")
        if not math.isclose(sum(self.family_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("family_weights 加起來要是 1")
        return self


class Compute(_Frozen):
    workers: int = Field(strict=True, ge=1)


class SurrogateTrainingSettings(_Frozen):
    training: RandomSets
    holdout: RandomSets
    sampling: Sampling
    compute: Compute

    @model_validator(mode="after")
    def _different_seeds(self) -> SurrogateTrainingSettings:
        if self.training.seed == self.holdout.seed:
            raise ValueError("訓練與留出的種子不能一樣（留出材料要跟訓練材料獨立）")
        return self


def load_surrogate_training_settings(path: str | Path) -> SurrogateTrainingSettings:
    """讀整份；缺節或欄位不合就報錯。"""
    with Path(path).open("rb") as config_file:
        return SurrogateTrainingSettings.model_validate(tomllib.load(config_file))
