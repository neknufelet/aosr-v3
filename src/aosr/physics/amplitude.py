"""反射乘積與每條路徑壓力的純物理算術，以及振幅精度契約的界線公式。

**這一支只 import 標準庫**（``math``、``cmath``、``dataclasses``），不碰 numpy、不碰 JAX。
它跟 :mod:`aosr.physics.room_paths` 住同一層（``physics``），可以拿下面那層 ``geometry``
的零件（規矩卡 ``layers-import-downward-only`` 只准 ``src/aosr`` 往下引）。它自己**不 print**
——命令列輸出全住在 ``room_paths.py``（``style-guard`` 的輸出層白名單只列那一支）——所以
這一支是純函式模組，回值、不解讀、不對人說話。

**公式照上一代，不准自己發明。** 反射係數 ``R = (Z·cosθ − ρc) / (Z·cosθ + ρc)``，跟
``blueprint/reference_amplitude_check.py`` 的 :func:`reflection_coefficient` 是**同一條公式**，
考卷會拿兩邊算出來的數字比（反射係數、入射 cos、反射乘積與路徑壓力逐格對，雙精度對雙精度
該逐位相同，見 ``tests/engine/test_amplitude.py`` 的逐位題）。入射角的 cos 照上一代「攤開直線」
的定義：對某面牆那一軸的 ``|receiver[axis] − image[axis]| / dist``（整條鏡像距離做分母），
不是重新對反彈點求角。反彈乘積是每一次反彈的 R 連乘（牆打幾次乘幾次 ``r ** count``）；
``r ** count`` 的前提是**整面牆同一個阻抗**（跟上一代逐次反彈取 cell 的做法在這個前提下
等價；分格材料是以後的事）；直達路徑（order 0）沒有反彈，乘積是 ``1+0j``。路徑壓力 =
(1/dist)·refl·exp(−i·2πf·τ)，``τ = dist / c``（聲速）。

**界線基底由呼叫端給。** 反射乘積用固定絕對界線；路徑壓力依決策紙保留隨相位成長的
公式。兩者的基底都從唯一登記簿傳入，本模組不持有門檻副本；直達反射乘積仍恰等於 1。

**材料是資料不是這一支的狀態。** :class:`Materials` 是凍結資料（``rho_c``、逐頻資料、六面牆
各自的 row-major 阻抗格網；整面牆是 1×1）；載入與格式收窄住在 ``room_paths.py`` 的
``load_room_input``，這一支只吃「已經收窄好的 :class:`Materials`」算物理量。
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Final

from aosr.geometry.shoebox import wall_count_signature

# 六面牆的規範順序（跟 v2 的 CANONICAL_WALL_ORDER、geometry 的 ``Wall.wall_names()`` 一致）：
# floor（z=0）、ceiling（z=Lz）、x0、xL、y0、yL。
CANONICAL_WALLS: Final[tuple[str, ...]] = ("floor", "ceiling", "x0", "xL", "y0", "yL")

# 一面牆一個頻帶的複數表面阻抗：輸入牆名與頻帶 index，回 Z（Pa·s/m）。
WallImpedance = Callable[[str, int], complex]
WallGrid = tuple[int, int, tuple[tuple[complex, ...], ...]]


@dataclass(frozen=True)
class Materials:
    """一份輸入檔 ``materials`` 節收窄後的結果：介質與六面牆的表面阻抗。

    ``frequencies_hz`` 是逐頻資料；``walls`` 只保留 1×1 整牆阻抗列，``wall_grids`` 是牆名 →
    ``(rows, cols, cells)``，其中 cells 依 row-major 攤平、每格一列頻帶阻抗。分格牆只准透過
    :meth:`impedance_at` 取值。
    """

    rho_c: float
    frequencies_hz: tuple[float, ...]
    walls: dict[str, tuple[complex, ...]]
    wall_grids: dict[str, WallGrid] | None = None

    def __post_init__(self) -> None:
        """舊形整牆清單正規化成 1×1；新形由載入器提供完整 row-major 格網。"""
        if self.wall_grids is None:
            object.__setattr__(
                self,
                "wall_grids",
                {wall: (1, 1, (values,)) for wall, values in self.walls.items()},
            )

    def impedance(self, wall: str, f_index: int) -> complex:
        """那面牆那一個頻帶的複數表面阻抗。牆名或頻帶 index 不對就當場炸。"""
        if self.grid(wall)[:2] != (1, 1):
            raise ValueError("這面牆有分格，用 impedance_at")
        try:
            row = self.walls[wall]
        except KeyError as exc:
            raise ValueError(f"未知牆名：{wall!r}") from exc
        if not 0 <= f_index < len(row):
            raise ValueError(f"頻帶 index {f_index} 超出範圍 [0, {len(row)})")
        return row[f_index]

    def grid(self, wall: str) -> WallGrid:
        """回一面牆的 ``(rows, cols, cells)``；cells 是 row-major 的頻帶列。"""
        if self.wall_grids is None:
            raise ValueError("材料格網尚未初始化")
        try:
            return self.wall_grids[wall]
        except KeyError as exc:
            raise ValueError(f"未知牆名：{wall!r}") from exc

    def impedance_at(self, wall: str, row: int, col: int, f_index: int) -> complex:
        """取一面牆指定格、指定頻帶的阻抗；格子以 row-major 攤平。"""
        rows, cols, cells = self.grid(wall)
        if not 0 <= row < rows or not 0 <= col < cols:
            raise ValueError(f"牆 {wall!r} 格子 ({row}, {col}) 超出 {rows}×{cols}")
        band = cells[row * cols + col]
        if not 0 <= f_index < len(band):
            raise ValueError(f"頻帶 index {f_index} 超出範圍 [0, {len(band)})")
        return band[f_index]

    def has_patches(self) -> bool:
        """任一牆不是 1×1 就是分格材料。"""
        return any(self.grid(wall)[:2] != (1, 1) for wall in CANONICAL_WALLS)


def reflection_coefficient(Z: complex, cos_theta: float, rho_c: float) -> complex:
    """``R = (Z·cosθ − ρc) / (Z·cosθ + ρc)``（跟 blueprint/reference_amplitude_check.py 同一條）。

    這是上一代材料轉接層的 ``z_to_r_from_cos``，考卷會拿 v3 這一支跟
    獨立檢查那一支的輸出比對；公式一字照抄，寫法自己寫。
    """
    z_cos = Z * cos_theta
    return (z_cos - rho_c) / (z_cos + rho_c)


def incidence_cos(
    axis: int,
    dist_m: float,
    receiver: tuple[float, float, float],
    image: tuple[float, float, float],
) -> float:
    """一面牆的入射角餘弦：那一軸的 ``|receiver − image| / dist``（攤開直線，整條距離當分母）。"""
    return abs(receiver[axis] - image[axis]) / dist_m


def reflection_product(
    materials: Materials,
    dist_m: float,
    receiver: tuple[float, float, float],
    image: tuple[float, float, float],
    counts: dict[str, int],
) -> tuple[complex, ...]:
    """一條路徑每個頻帶的反射乘積：對每一面牆，R 連乘 ``count`` 次（``r ** count``）。

    ``counts`` 是 :func:`aosr.geometry.shoebox.wall_count_signature` 的結果（每面牆打幾次）。
    ``r ** count`` 的前提是**整面牆同一個阻抗**（跟上一代逐次反彈取 cell 的做法在這個前提下
    等價；分格材料是以後的事）。直達路徑（全部 count 為 0）回 ``(1+0j, …)`` 恰好。入射 cos
    對同一面牆的每一次反彈都用「那一軸的 |receiver − image| / dist」——攤開直線的定義，
    不逐反彈求角。
    """
    out: list[complex] = []
    for f_index in range(len(materials.frequencies_hz)):
        product = complex(1.0, 0.0)
        for wall in CANONICAL_WALLS:
            count = counts.get(wall, 0)
            if count == 0:
                continue
            axis = _wall_axis(wall)
            cos = incidence_cos(axis, dist_m, receiver, image)
            r = reflection_coefficient(materials.impedance(wall, f_index), cos, materials.rho_c)
            product *= r if count == 1 else r ** count
        out.append(product)
    return tuple(out)


def reflection_product_by_bounce(
    materials: Materials,
    dist_m: float,
    receiver: tuple[float, float, float],
    image: tuple[float, float, float],
    bounces: tuple[tuple[str, int, int], ...],
) -> tuple[complex, ...]:
    """依逐次反彈的 ``(wall, row, col)`` 查 Z、算 R 並連乘。

    同一面牆可以在不同跳命中不同格；入射 cos 仍沿用攤開直線定義。直達路徑傳空
    ``bounces``，每頻帶恰回 ``1+0j``。
    """
    out: list[complex] = []
    for f_index in range(len(materials.frequencies_hz)):
        product = complex(1.0, 0.0)
        for wall, row, col in bounces:
            cos = incidence_cos(_wall_axis(wall), dist_m, receiver, image)
            coefficient = reflection_coefficient(
                materials.impedance_at(wall, row, col, f_index),
                cos,
                materials.rho_c,
            )
            product *= coefficient
        out.append(product)
    return tuple(out)


def path_pressure(
    materials: Materials,
    dist_m: float,
    sound_speed: float,
    refl_per_freq: tuple[complex, ...],
) -> tuple[complex, ...]:
    """一條路徑每個頻帶的壓力：``(1/dist)·refl·exp(−i·2πf·τ)``，``τ = dist / c``。

    雙精度（``complex``）。``refl_per_freq`` 是 :func:`reflection_product` 的結果，
    長度跟 ``frequencies_hz`` 一致。``dist_m`` 為 0（接收點正好落在鏡像聲源上）時丟
    ``ValueError`` 說接收點落在鏡像上——不准 ``ZeroDivisionError``、不准另塞小數補救。
    """
    if dist_m == 0.0:
        raise ValueError("接收點落在鏡像聲源上（dist_m 為 0），壓力沒定義")
    out: list[complex] = []
    for f_index, f in enumerate(materials.frequencies_hz):
        omega = 2.0 * math.pi * f
        tau = dist_m / sound_speed
        phase = cmath.exp(complex(0.0, -omega * tau))
        out.append((1.0 / dist_m) * refl_per_freq[f_index] * phase)
    return tuple(out)


def _wall_axis(wall: str) -> int:
    """牆名 → 那一軸（0=x、1=y、2=z）。只認六面牆。"""
    for name, axis in (
        ("floor", 2),
        ("ceiling", 2),
        ("x0", 0),
        ("xL", 0),
        ("y0", 1),
        ("yL", 1),
    ):
        if name == wall:
            return axis
    raise ValueError(f"未知牆名：{wall!r}")


def path_amplitude(
    materials: Materials,
    identity: tuple[int, int, int, int, int, int],
    dist_m: float,
    sound_speed: float,
    receiver: tuple[float, float, float],
    image: tuple[float, float, float],
) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    """一條路徑的完整振幅結果：回 ``(反射乘積、路徑壓力)`` 兩組（各六個頻帶）。

    反射乘積由 :func:`reflection_product`、壓力由 :func:`path_pressure` 依序推，
    直達路徑反射乘積恰 ``1+0j``。
    """
    counts = wall_count_signature(identity)
    refl = reflection_product(materials, dist_m, receiver, image, counts)
    pp = path_pressure(materials, dist_m, sound_speed, refl)
    return refl, pp


def reflection_tolerance(
    f: float,
    tau: float,
    abs_refl: float,
    base_tolerance_ulp: float,
) -> float:
    """反射乘積每分量的固定絕對差界線。

    決策紙選項 1：反射乘積每分量使用固定絕對界線，跟 f、τ、|refl| 無關。這三個參數
    保留是為了跟 :func:`pressure_tolerance` 同一張簽名、讓「界線不隨它們變」在呼叫點看得見。
    """
    del f, tau, abs_refl
    return base_tolerance_ulp


def pressure_tolerance(
    f: float,
    tau: float,
    abs_refl: float,
    base_tolerance_ulp: float,
) -> float:
    """路徑壓力的相對差界線；基底由呼叫端傳入。

    決策紙選項 1：每條路徑的壓力相對差不超過該式，``ω = 2πf``、``τ`` 到達時間。回傳的是
    **比值**；相對差以**複數差的模除以參考值的模**計：呼叫端拿來斷言 ``|Δp| ≤ 回傳值·|p|``
    （``Δp`` 是複數差，不是逐分量各比）。
    """
    omega_tau = 2.0 * math.pi * f * tau
    return base_tolerance_ulp * (omega_tau + 1.0) + base_tolerance_ulp / abs_refl
