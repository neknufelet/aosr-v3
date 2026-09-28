"""Physics constants layer — frozen, optimiser-untouched (design §8).

Loaded from TOML via the stdlib ``tomllib`` (Python 3.11+). The characteristic
impedance of air ``rho_c = ρ₀·c`` is derived here and injected explicitly into
the adapter and solvers (design §7.4).
"""

from __future__ import annotations

import tomllib
from pathlib import Path

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


def load_physics_constants(path: str | Path) -> PhysicsConstants:
    """Load and validate the ``physics_constants.toml`` in this package's ``data`` directory."""
    with open(path, "rb") as f:  # noqa: PTH123  # expires=2026-12-08 reason=與上一代逐字相同的開檔寫法——`open(None)` 的 TypeError 訊息是行為契約，`Path(path).open(...)` 講的是另一句（找碴第二輪）
        data = tomllib.load(f)
    return PhysicsConstants(**data)
