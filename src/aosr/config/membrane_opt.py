"""Validated SSOT for the M15-P5-1d limp-membrane FEM lane."""

from __future__ import annotations

import tomllib
from pathlib import Path
from .paths import config_path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MembraneOptConfig(BaseModel):
    """Runtime constants for the additive 1d material optimisation lane."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    energy_frac: float = Field(gt=0.0, lt=1.0)
    barrier_t: float = Field(gt=0.0)
    barrier_relaxation: float = Field(gt=0.0)
    fmax_hz: float = Field(gt=0.0)
    elements_per_wavelength: int = Field(ge=1)
    re_z_floor_frac_rho_c: float = Field(gt=0.0, le=1.0e-3)
    re_z_floor_sharpness: float = Field(gt=0.0)
    w_decay: float = Field(ge=0.0)
    w_rt60: float = Field(ge=0.0)
    air_gap_min_m: float = Field(gt=0.0)
    air_gap_max_m: float = Field(gt=0.0)
    mass_min_kg_m2: float = Field(gt=0.0)
    mass_max_kg_m2: float = Field(gt=0.0)
    porous_min_m: float = Field(gt=0.0)
    porous_max_m: float = Field(gt=0.0)
    flow_resistivity_min: float = Field(gt=0.0)
    flow_resistivity_max: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> MembraneOptConfig:
        pairs = (
            ("air_gap", self.air_gap_min_m, self.air_gap_max_m),
            ("mass", self.mass_min_kg_m2, self.mass_max_kg_m2),
            ("porous", self.porous_min_m, self.porous_max_m),
            (
                "flow_resistivity",
                self.flow_resistivity_min,
                self.flow_resistivity_max,
            ),
        )
        for name, lower, upper in pairs:
            if lower >= upper:
                raise ValueError(
                    f"{name} lower bound must be below upper bound, got "
                    f"{lower} >= {upper}"
                )
        if self.w_decay != 0.0 or self.w_rt60 != 0.0:
            raise ValueError(
                "M15-P5-1d requires w_decay=0 and w_rt60=0; raw FEM flatness "
                "plus the same-band energy backstop are its only objectives"
            )
        return self


def load_membrane_opt(path: str | Path | None = None) -> MembraneOptConfig:
    """Load and validate the ``[membrane_1d]`` runtime configuration."""
    resolved = config_path("membrane_opt.toml") if path is None else Path(path)
    with resolved.open("rb") as config_file:
        data = tomllib.load(config_file)
    try:
        section = data["membrane_1d"]
    except KeyError as exc:
        raise ValueError(f"{resolved} must define [membrane_1d]") from exc
    return MembraneOptConfig(**section)
