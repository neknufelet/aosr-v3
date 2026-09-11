"""Validated MAT beta-continuation configuration loaded from TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path
from .paths import config_path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MatContinuationConfig(BaseModel):
    """Runtime beta-continuation constants for MAT patch geometry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    beta0: float = Field(gt=0.0)
    hold_steps: int = Field(ge=0)
    ramp_increment: float = Field(gt=0.0)
    ramp_every_steps: int = Field(gt=0)
    beta_cap: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _validate_cap(self) -> MatContinuationConfig:
        if self.beta_cap < self.beta0:
            raise ValueError("beta_cap must be >= beta0")
        return self


_CONFIG_PATH = config_path("mat_continuation.toml")


def load_mat_continuation(path: str | Path | None = None) -> MatContinuationConfig:
    """Load the ``[beta]`` continuation settings from the TOML SSOT."""

    resolved = _CONFIG_PATH if path is None else Path(path)
    with resolved.open("rb") as config_file:
        data = tomllib.load(config_file)
    try:
        section = data["beta"]
    except KeyError as exc:
        raise ValueError(f"{resolved} must define [beta]") from exc
    return MatContinuationConfig(**section)
