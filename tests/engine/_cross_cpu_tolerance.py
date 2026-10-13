"""跨處理器錄答案考卷的容差：逐格比 ``|算出 − 答案| ≤ max(絕對, 相對 × |答案|)``，複數比差的模。

依據是決策紙 ``docs/decisions/verification-split-stand-in-wiring-and-cross-cpu-tolerance.md`` 第 6 條與
「為什麼」那張表：已觀察到的最大差 × 100，往上取整到 1、2、5 開頭。這裡守的是「跟主線一樣」的回歸，
不是物理精度，所以不進精度契約登記簿；改值要改那張紙。鍵（頻率、帶中心、階數……）與判定照舊逐位，
不經過這裡。
"""
from __future__ import annotations

import cmath
import math
from collections.abc import Iterable
from dataclasses import dataclass

# ① 鏡像法、指向性與不解方程組的閉式公式（艾林殘響時間、交接、頻帶平均）：
# 雲端最大相對 2.6e-16（10-08 主線一跑的指向性路徑聲壓）、本機 0；
# 最近的物理契約 reflected_energy_floor 2^-23。
GEOMETRIC_REL = 5e-14
# ② 含解線性方程組的晚期家族：雲端最大 7.26e-16（#509）、本機 2.0e-15（OpenBLAS 換 Prescott 核心）；
# 最近的物理契約 late_energy_physical_property 2^-30。
LINEAR_SOLVE_REL = 2e-13
# ③ 評分層：沒有量得出差的雲端紀錄；輸入末位擾動量到絕對最大 2.13e-14、數值大於 5 的格相對最大 4.0e-15；
# 沒有物理契約，網頁顯示到小數 4 位。
SCORING_REL = 5e-13
SCORING_ABS = 5e-12

Number = float | complex | str


def _number(value: Number) -> float | complex:
    """字串只收 ``float.hex()`` 的形狀；十進位字串 fromhex 會照十六進位讀錯（'1.5' 讀成 1.3125）。"""
    if not isinstance(value, str):
        return value
    if "0x" not in value.lower() and value.lower().lstrip("+-") not in ("inf", "nan"):
        raise ValueError(f"不是 float.hex() 的字串：{value!r}")
    return float.fromhex(value)


@dataclass(frozen=True)
class Cell:
    """一格：位置、算出、答案，以及這一格的相對與絕對容差（絕對由答案或同列答案算，不由算出值算）。"""

    where: str
    actual: float | complex
    expected: float | complex
    rel: float
    abs_: float

    @property
    def tolerance(self) -> float:
        return max(self.abs_, self.rel * abs(self.expected))

    @property
    def ratio(self) -> float:
        """差 ÷ 容差；不超過 1 才算過。算出 NaN 或容差為零而有差，都是無限大。"""
        difference = abs(self.actual - self.expected)
        if math.isnan(difference):
            return math.inf
        if difference == 0:
            return 0.0
        return difference / self.tolerance if self.tolerance > 0 else math.inf


def cell(where: str, actual: Number, expected: Number, *, rel: float, abs_: float) -> Cell:
    """造一格；hex 字串先轉成數。答案不是有限數、或容差是負的或無限大，是考卷寫錯，直接報錯。"""
    answer = _number(expected)
    if not cmath.isfinite(answer):
        raise ValueError(f"{where}：答案不是有限數 {expected!r}")
    if not (math.isfinite(rel) and math.isfinite(abs_) and rel >= 0 and abs_ >= 0):
        raise ValueError(f"{where}：容差要是有限的非負數（相對 {rel!r}、絕對 {abs_!r}）")
    return Cell(where, _number(actual), answer, rel, abs_)


def assert_within(cells: Iterable[Cell]) -> None:
    """全部格都在容差內才過；不過時把每一格超出的位置、算出、答案、差、容差都列出來。"""
    outside = [c for c in cells if not c.ratio <= 1.0]
    if outside:
        raise AssertionError("超出跨處理器容差：\n" + "\n".join(
            f"{c.where}：算出 {c.actual!r}，答案 {c.expected!r}，差 {abs(c.actual - c.expected)!r}，"
            f"容差 {c.tolerance!r}" for c in outside))


def ratios(cells: Iterable[Cell]) -> dict[str, float]:
    """每一格的差 ÷ 容差，給量測與合併請求內文用。"""
    return {c.where: c.ratio for c in cells}
