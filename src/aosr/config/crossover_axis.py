"""Canonical frequency- and time-seam parameters for connected forward fields."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_T_C_CENTER_S = 0.08
DEFAULT_T_C_FADE_S = 0.02


@dataclass(frozen=True)
class CanonicalCrossover:
    """Canonical seam coordinates shared by connected-forward consumers."""

    seam_f_s: float
    t_c_center: float = DEFAULT_T_C_CENTER_S
    t_c_fade: float = DEFAULT_T_C_FADE_S


def build_canonical_crossover(
    seam_f_s: float,
    *,
    t_c_center: float = DEFAULT_T_C_CENTER_S,
    t_c_fade: float = DEFAULT_T_C_FADE_S,
) -> CanonicalCrossover:
    """Build the immutable canonical crossover contract."""
    return CanonicalCrossover(
        seam_f_s=float(seam_f_s),
        t_c_center=float(t_c_center),
        t_c_fade=float(t_c_fade),
    )
