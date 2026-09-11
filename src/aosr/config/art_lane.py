"""ART-lane structural constants (M15-P4.2 SSOT)."""

from __future__ import annotations

ART_N_PER_WALL_DEFAULT: int = 6
ART_FACE_GRID_N: int = 6
"""provisional coarse per-face ART grid resolution; box-ART precedent ART_N_PER_WALL_DEFAULT=6; re-pinned by CG-4 from the spatially-varying-alpha convergence curve"""
ART_NEUMANN_EPS_TAIL: float = 1e-4
ART_POWERITER_K: int = 50
ART_FORMFACTOR_EPS: float = 1e-12
ART_RECIPROCITY_TOL: float = 0.25
# Dense ART patch ceiling. Production polygon ART now uses the coarse per-face cell grid
# (`ART_FACE_GRID_N`), so this cap is retained as the RAM-bounded test-only oracle guard
# for FEM-triangle ART paths. At n_freq=68: rpf = P^2 * 68 * 4 bytes -> P=4498 (legacy
# production octagon boundary mesh oracle) = 5.5 GB, P=8192 = 18.3 GB. 8192 keeps oracle
# checks under ~18 GB on the 48 GB CPU-pinned host while covering the historical polygon
# FEM boundary-triangle count with ~1.8x headroom.
ART_P_CAP: int = 8192
# Static upper bound on the Neumann reflection-order scan length. The per-frequency
# effective depth is `tail_steps = ceil(ln(EPS_TAIL)/ln rho(R))` from the ACTUAL Perron
# root (spec §1.D); this cap only bounds the static `lax.scan` length. 256 covers the
# full production absorption sweep (down to alpha=0.05) at EPS_TAIL=1e-4 with margin.
# A per-frequency dynamic-length scan (cut compute to max(tail_steps)) is a P4.3 item.
ART_NEUMANN_K_MAX: int = 256

# ENGINE-4.4 (VAL-F6): T20 fit range and smooth-window softness for the WLS
# slope T60. The measured T20 samples the -5 -> -25 dB early decay; the
# dominant-Perron asymptotic rate systematically over-estimates it in rooms
# whose early decay is steeper than the slowest mode (CR2: +12..26%).
ART_WLS_T20_HI_DB: float = -5.0
ART_WLS_T20_LO_DB: float = -25.0
ART_WLS_WINDOW_SOFTNESS_DB: float = 1.0


def guard_art_patch_count(n_per_wall: int, *, context: str = "ART") -> None:
    """Fail loudly if an ART dispatch would exceed the dense patch cap."""
    if n_per_wall <= 0:
        raise ValueError(f"{context} n_per_wall must be positive; got {n_per_wall}")
    patch_count = 6 * n_per_wall * n_per_wall
    if patch_count > ART_P_CAP:
        raise ValueError(
            f"{context} patch count P={patch_count} exceeds ART_P_CAP={ART_P_CAP}; "
            f"lower n_per_wall (got {n_per_wall})"
        )


def guard_polygon_art_patch_count(
    n_tris: int,
    *,
    context: str = "polygon ART",
) -> None:
    """Fail loudly if a polygon ART boundary-triangle dispatch exceeds the cap."""
    if n_tris <= 0:
        raise ValueError(f"{context} triangle count must be positive; got {n_tris}")
    if n_tris > ART_P_CAP:
        raise ValueError(
            f"{context} patch count P={n_tris} exceeds ART_P_CAP={ART_P_CAP} "
            f"(RAM-bounded: rpf=(P,P,n_freq)f32 ~ {(n_tris * n_tris * 68 * 4) / 1e9:.1f} GB at "
            "n_freq=68); use a coarse per-face ART grid or lower polygon mesh target_nodes"
        )
