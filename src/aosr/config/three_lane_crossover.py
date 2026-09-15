"""第九段三路接合的 Eyring、Schroeder 與頻率邊界常數。

規格錨點是 ``docs/decisions/stage-nine-three-lane-stitch-and-report-with-interference.md``；這支只放
具名設定，算式住 ``aosr.physics.crossover``。既有 ``fem_lane.FEM_FMAX_CAP_HZ`` 是
上一代 250 Hz 凍結答案，刻意不在這裡改寫或沿用。
"""
from __future__ import annotations

from typing import Final

EYRING_COEFFICIENT_S_PER_M: Final[float] = 0.161
SCHROEDER_COEFFICIENT_SI: Final[float] = 2000.0
SCHROEDER_T60_BANDS_HZ: Final[tuple[float, float]] = (500.0, 1000.0)
CROSSOVER_LOWER_FLOOR_HZ: Final[float] = 150.0
