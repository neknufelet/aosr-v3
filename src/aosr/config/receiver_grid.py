"""Frozen defaults for the scoring/report receiver grid.

The grid samples room-wide spatial uniformity; it is separate from user-placed
listening receivers and receiver clusters. Changing any value here affects
ranking/report output and therefore requires the SCORING decision process.
"""

from __future__ import annotations

GRID_N: int = 4
"""Number of receiver-grid points along each horizontal room axis."""

GRID_MARGIN_M: float = 0.5
"""Clearance in metres between receiver-grid points and room walls."""

EAR_HEIGHT_M: float = 1.2
"""Seated ear height in metres used by the receiver grid."""
