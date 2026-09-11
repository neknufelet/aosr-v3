"""Canonical acoustic source reference shared by ALL lanes (audit 2026-06-22).

All lanes are summed in the energy-domain blend, so they MUST share one source
reference. Canonical = UNIT-AMPLITUDE monopole, |p(1 m)| = 1 (direct energy 1/r^2),
matching ISM (|Σp|^2), FEM-stitched, Ray-direct, RFZ. The handbook diffuse-field
density 4/R is referenced to UNIT SOURCE POWER; converting it to the unit-amplitude
monopole reference multiplies by 4π (W·ρ₀c = 4π for |p(1m)|=1, so the diffuse
squared pressure 4·W·ρ₀c/R = 16π/R). See physics-realign-audit-results.md (FIX-1/2/3).
"""

import math

# ``(-∇²-k²) p = S δ`` has the outgoing free-space solution
# ``p = S exp(-jkr)/(4πr)``.  The canonical ``exp(-jkr)/r`` reference therefore
# requires ``S=4π``.  Keep the numeric definition here and nowhere else.
CANONICAL_MONOPOLE_STRENGTH: float = 4.0 * math.pi

# Backward-compatible semantic alias used by the diffuse-field conversion:
# unit-power ``4/R`` -> unit-amplitude ``16π/R``.
DIFFUSE_MONOPOLE_4PI: float = CANONICAL_MONOPOLE_STRENGTH

# SCENE-1 D13 absolute-level naming anchor.  This is the power-equivalent
# monopole reference, not an ON-axis equalisation target: it documents the
# existing |p(1 m)|=1 reference and deliberately does not rescale any solver.
# With power-normalised directivity enabled, the on-axis relationship is
# L_axis(f) = 94 dB SPL + DI(f), where DI(f) = 20 log10 |D(theta=0,f)|.
REFERENCE_ANCHOR_KIND: str = "power-equivalent-monopole"
REFERENCE_SENSITIVITY_DB_SPL_1M_1W: float = 94.0
REFERENCE_SENSITIVITY_PRESSURE_PA: float = 1.0
REFERENCE_DRIVE_POWER_W: float = 1.0
HIGH_SPL_REFERENCE_DB_SPL_1M_1W: float = 114.0
HIGH_SPL_REFERENCE_PRESSURE_PA: float = 10.0
