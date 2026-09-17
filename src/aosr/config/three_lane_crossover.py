"""第九段三路接合的 Eyring、Schroeder 與頻率邊界常數。

規格錨點是 ``docs/decisions/stage-nine-reflection-order-is-a-setting.md``；這支只放
具名設定，算式住 ``aosr.physics.crossover``。既有 ``fem_lane.FEM_FMAX_CAP_HZ`` 是
上一代 250 Hz 凍結答案，刻意不在這裡改寫或沿用。
"""
from __future__ import annotations

from typing import Final

EYRING_COEFFICIENT_S_PER_M: Final[float] = 0.161
SCHROEDER_COEFFICIENT_SI: Final[float] = 2000.0
SCHROEDER_T60_BANDS_HZ: Final[tuple[float, float]] = (500.0, 1000.0)
CROSSOVER_LOWER_FLOOR_HZ: Final[float] = 150.0
# 幾何路與晚期混響的交接階數 K：K 階以內的鏡面留在鏡像法，被散射掉的那一份與 K 階以上
# 全部交給晚期混響（決策紙第 5 條）。這個值是**相容性**的選擇——3 只為了跟上一代比對，
# 不是精度上的選擇。把它露出成呼叫端可調的設定、並放寬 ``physics.room_paths`` 那條
# 「四階以上報錯」的硬上限，是票 #341；在那之前這裡是唯一的一份，程式一路把它當參數帶
# 著走，不把 3 灑在各處。
REFLECTION_ORDER_K: Final[int] = 3
