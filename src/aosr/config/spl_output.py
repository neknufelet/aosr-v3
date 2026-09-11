"""Output-layer absolute-SPL reference (M15-P5-2 inc-4) — display-only.

Converts the bridge's RELATIVE per-receiver SPL (referenced to a free-field unit
monopole at 1 m) into ABSOLUTE dB SPL by adding a single reference offset

    ``L_ref = sensitivity_db + playback_level_db``     (additive dB)

at the CHORAS output layer ONLY. The optimiser/kernel are untouched: the octagon
ranking objective ``membrane_spectral_flatness`` is offset-invariant (a common dB
offset cancels inside ``std`` over bins), so this offset cannot change top-K
selection or any score — see ``design_20260621-m15-p5-2-inc4-absolute-spl-build-spec``.

Scope (design-harden GO-WITH-MUST-FIX):
  * Default ``0.0`` → byte-identical to the relative output (ships RELATIVE; real
    absolute numbers require a user-supplied measured sensitivity).
  * Single flat scalar across all bands; frequency-dependent sensitivity
    (Spinorama) and target-level calibration (S-line) are reserves.
  * Per-source sensitivities MUST be equal (flat-equal guard): output-layer
    additivity is exact only for a common scalar; per-source-different levels need
    a physics-layer entry (directivity increment), not this offset.
  * NOT part of optimisation identity — these never enter ``config_hash``.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

# SSOT defaults: 0.0 = unconfigured / relative passthrough (NOT a fabricated 88 dB).
DEFAULT_SENSITIVITY_DB: float = 0.0
DEFAULT_PLAYBACK_LEVEL_DB: float = 0.0
# Tolerance for the per-source flat-equal guard (dB).
FLAT_EQUAL_TOL_DB: float = 1.0e-6


class SplOutputConfig(BaseModel):
    """Output-layer absolute-SPL reference (display-only, not optimisation identity).

    ``allow_inf_nan=False`` rejects non-finite sensitivity/playback at construction
    (Pydantic's default would accept ``nan``/``inf`` and silently poison every SPL).
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    sensitivity_db: tuple[float, ...] = Field(
        default=(DEFAULT_SENSITIVITY_DB,),
        min_length=1,
        description="Per-source speaker sensitivity (dB SPL @1 m @ reference drive). "
        "inc-4 requires all entries equal (flat-equal guard); a single entry "
        "broadcasts to all sources.",
    )
    playback_level_db: float = Field(
        default=DEFAULT_PLAYBACK_LEVEL_DB,
        description="Global playback/drive level offset (dB) relative to reference drive.",
    )

    @property
    def l_ref_db(self) -> float:
        """Single scalar reference offset ``sensitivity_db + playback_level_db``.

        Enforces the flat-equal guard: per-source sensitivities that differ beyond
        ``FLAT_EQUAL_TOL_DB`` are rejected, because an output-layer additive offset
        is exact only for a COMMON scalar (the lanes sum sources coherently; a
        per-source-different amplitude does not factor out and must enter the
        physics layer = directivity increment).
        """
        first = self.sensitivity_db[0]
        for value in self.sensitivity_db[1:]:
            if abs(value - first) > FLAT_EQUAL_TOL_DB:
                raise ValueError(
                    f"per-source sensitivity_db differ ({self.sensitivity_db}); "
                    "output-layer absolute SPL (inc-4) requires a common scalar. "
                    "Per-source-different levels need a physics-layer entry "
                    "(directivity increment), not the display offset."
                )
        l_ref = first + self.playback_level_db
        if not math.isfinite(l_ref):
            raise ValueError(f"non-finite L_ref ({l_ref})")
        return l_ref

    @property
    def is_relative(self) -> bool:
        """``True`` when the offset is exactly 0 dB (output stays relative)."""
        return self.l_ref_db == 0.0
