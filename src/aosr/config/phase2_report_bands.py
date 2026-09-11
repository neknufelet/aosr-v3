"""Phase-2 report extraction bands and thresholds (UNWIRE-5 SSOT)."""

from __future__ import annotations

PHASE2_RFZ_THRESHOLD_DB: float = -6.0
PHASE2_RFZ_BAND_HZ: tuple[float, float] = (1000.0, 8000.0)
PHASE2_LATE_RT_BAND_HZ: tuple[float, float] = (500.0, 2000.0)
PHASE2_RFZ_WINDOW_S: float = 0.015
PHASE2_TARGET_SLOPE_DB_OCT: float = -0.7
