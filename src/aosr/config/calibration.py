"""Calibration runtime settings loaded from ``src/aosr/config/data/calibration.toml``（在新家，這一份住在 ``data`` 目錄底下）.

The low calibration edge is intentionally labelled provisional.  It is a
runtime decision owned by the calibration plan, not a Python constant hidden
in the observable functions.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from .paths import config_path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CalibrationBandConfig(_Frozen):
    """Calibration frequency band; ``f_lo_hz`` is provisional in v0."""

    f_lo_hz: float = Field(gt=0.0)
    f_hi_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _validate_order(self) -> CalibrationBandConfig:
        if self.f_hi_hz <= self.f_lo_hz:
            raise ValueError("f_hi_hz must exceed f_lo_hz")
        return self


class CalibrationRtConfig(_Frozen):
    """Schroeder RT validity settings."""

    valid_floor_hz: float = Field(gt=0.0)
    floor_ratio_min: float = Field(gt=1.0)


class CalibrationSmoothingConfig(_Frozen):
    """導入端 fractional-octave smoothing settings."""

    fraction: int = Field(ge=1)


class CalibrationConfig(_Frozen):
    """Validated CAL1 configuration."""

    band: CalibrationBandConfig
    rt: CalibrationRtConfig
    smoothing: CalibrationSmoothingConfig


_DEFAULT_CONFIG_PATH = config_path("calibration.toml")


def load_calibration(path: str | Path | None = None) -> CalibrationConfig:
    """Load and validate the calibration TOML, preserving the filename in errors."""

    resolved = _DEFAULT_CONFIG_PATH if path is None else Path(path)
    try:
        with resolved.open("rb") as config_file:
            data = tomllib.load(config_file)
        config = CalibrationConfig.model_validate(data)
    except (OSError, tomllib.TOMLDecodeError, ValidationError, TypeError, ValueError) as exc:
        raise ValueError(f"{resolved}: invalid calibration configuration: {exc}") from exc
    logger.debug("loaded calibration configuration from %s", resolved)
    return config
