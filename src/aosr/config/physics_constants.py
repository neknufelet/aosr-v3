"""Physics constants layer — frozen, optimiser-untouched (design §8).

Loaded from TOML via the stdlib ``tomllib`` (Python 3.11+). The characteristic
impedance of air ``rho_c = ρ₀·c`` is derived here and injected explicitly into
the adapter and solvers (design §7.4).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from .paths import config_path

from pydantic import BaseModel, ConfigDict, Field


class PhysicsConstants(BaseModel):
    """Frozen environment/physics constants."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    air_density_kg_m3: float = Field(gt=0.0, description="ρ₀, air density")
    sound_speed_m_s: float = Field(gt=0.0, description="c, speed of sound")
    ref_pressure_pa: float = Field(gt=0.0, description="reference SPL pressure (20 µPa)")
    air_viscosity_pa_s: float = Field(gt=0.0, description="µ, dynamic viscosity of air")

    @property
    def rho_c(self) -> float:
        """Characteristic impedance of air ρ₀·c, in Pa·s/m (≈ 411.6 at STP)."""
        return self.air_density_kg_m3 * self.sound_speed_m_s


def load_physics_constants(path: str | Path | None = None) -> PhysicsConstants:
    """Load and validate the ``physics_constants.toml`` in this package's ``data`` directory."""
    resolved = config_path("physics_constants.toml") if path is None else Path(path)
    with resolved.open("rb") as f:
        data = tomllib.load(f)
    return PhysicsConstants(**data)
