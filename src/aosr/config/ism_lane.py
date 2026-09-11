"""ISM-lane structural constants (PMO7 DI-2b SSOT)."""

from __future__ import annotations

ISM_FACE_GRID_N: int = 16
"""Per-face HF-ISM cell grid resolution pinned by DI-2b geometry-FD convergence.

N=16 with ISM_FACE_GRID_SAMPLES=11 gives per-DOF FD max_rel ~=0.027, meeting
DI-3 rel_tol=5e-2 with margin; the cost is negligible for the HF-ISM lane.
"""

ISM_FACE_GRID_SAMPLES: int = 11
"""Per-cell midpoint quadrature samples per UV axis for HF-ISM area averaging."""
