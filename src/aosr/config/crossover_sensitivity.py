"""比較接法的設定；不被物理或模態入口載入。

依據：docs/decisions/crossover-sensitivity-is-a-ranking-reminder-not-a-rule-change.md 第 2 條。
頻率端點沿用正式設定，唯一新增數字是上一代的半個八度半寬。
"""
from typing import Final

LEGACY_HALF_WIDTH_OCTAVE: Final[float] = 0.5
