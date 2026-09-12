"""獨立振幅重算：只用標準庫，把答案檔的反射乘積與路徑壓力雙精度重算並跟答案比。

**為什麼要有這一支。** 票 #194 的考卷（``tests/test_reference_amplitude.py``）要拿一支
「跟產生器完全無關」的程式，把 ``blueprint/reference_amplitude_{flat,varied}.json`` 裡每一條
路徑的反射乘積（``reflection_product``）與路徑壓力（``path_pressure``）重新算一遍，再跟答案檔
逐格比、判是否落在精度契約的界線內。這一支就是那支獨立程式：只 import 標準庫（``math``、
``cmath``、``dataclasses``），不 import numpy／jax／donor 任何東西，純函式不讀檔不印東西——
所以「答案檔的值」跟「驗它的算式」不共用任何一支會算錯就一起錯的程式（找碴那一層要獨立，
見 CLAUDE.md 七條習慣第 4 條）。

**完全不寫死任何房間尺寸、聲速、阻抗。** 房間幾何（三軸長度、聲速）、聲源接收點、每面牆
每個頻帶的表面阻抗、頻率全部由呼叫端餵進來（:func:`Inputs`），這一支一個物理數字都不自己定。
同一組尺寸只住在產生器檔頭那一份凍結常數，這裡要是也寫一份，會長成「改了產生器忘了改獨立
檢查」的第二份副本。

**入射角的 cos 用「攤開直線」的定義。** 對某面牆的每一次反彈，入射角餘弦是那一面牆那一軸的
``|receiver[axis] − full_image[axis]| / dist``（照 donor ``z_to_r_from_cos`` 入口餵進來的
``cos_theta``），不是重新對反彈點求角。反射係數 ``R = (Z·cosθ − ρc)/(Z·cosθ + ρc)``；每條路徑
的反射乘積是每一次反彈的 R 連乘（``r ** count``，牆打幾次乘幾次）；直達路徑（order 0）沒有
反彈、乘積是 ``1+0j``。路徑壓力 ``path_pressure = (1/dist)·refl·exp(−iωτ)``，``ω=2πf``、
``τ=dist/c``（照產生器重現的公式）。

**每面牆每個頻帶的阻抗由呼叫端饋。** :class:`WallImpedance` 給出一個 callable，輸入「牆名與
頻帶 index」回那面牆那個頻帶的複數表面阻抗 Z（Pa·s/m）。flat 那組六面牆都是同一個純實數
常數；varied 那組 ``y0`` 牆的 Z 隨頻帶「互異且虛部非零」、其餘五面維持常數——這個差別由考卷
從答案檔的 ``parameters.cases`` 讀出來餵，不寫死在這支。

**反彈展開照獨立幾何。** identity 六元組 → 每面牆反彈幾次的簽名（:func:`wall_count_signature`，
純從 ``(n, sign)`` 每軸推，跟 donor ``_axis_wall_counts`` 同一條拆法），這是**簽名**、不是逐次
求交。展開反彈（逐牆求交剥鏡像）住在 ``blueprint/reference_room_geometry.py``；這一支跟它同一個
演算法形狀，但也刻意獨立不 import 它——兩邊都要對得到同一份 identity 才算。這個模組只需要
「每面牆打幾次」就夠算反射乘積（入射角的 cos 只跟牆名與整條 dist 有關，跟反彈順序無關）。

**比對回差異清單，不回布林。** :func:`compare_cells` 對傳進來的「重算值」與「答案檔值」逐格比
（每格比**複數差的模**，雙精度對雙精度該逐位相同，一整格看成一筆差），回一串可讀的差異
字串（空清單＝完全一致）。
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Callable, Final

# 契約常數：反射乘積每分量絕對差 ≤ 2^-21（單精度四格）。這一格**故意住在這支的模組
# 常數**（考卷從這裡 import，不重抄），它錨在決策紙
# ``docs/decisions/precision-contract-amplitude-phase-scaled.md`` 選項 1。
# 這是「容差常數」不是「門檻卡管的門檻」：門檻數字只准住在規矩卡的登記簿（
# thresholds-live-only-in-registry），而這一格是物理契約的係數，錨在決策紙、由考卷
# import，不屬於那張卡的管轄。要動它得改決策紙，不是改門檻清單。
REFLECTION_CONTRACT_ULP: Final[float] = 2.0 ** -21

# 六面牆的規範順序（跟 v2 的 CANONICAL_WALL_ORDER 一致）：
# floor（z=0）、ceiling（z=Lz）、x0、xL、y0、yL。
_CANONICAL_WALLS: Final[tuple[tuple[str, tuple[int, str]], ...]] = (
    ("floor", (2, "zero")),
    ("ceiling", (2, "L")),
    ("x0", (0, "zero")),
    ("xL", (0, "L")),
    ("y0", (1, "zero")),
    ("yL", (1, "L")),
)


@dataclass(frozen=True)
class Inputs:
    """餵給獨立重算的全部物理輸入（全部由呼叫端饋，這一支不自己定）。"""

    lx: float
    """x 軸房間長度（m）。"""
    ly: float
    """y 軸房間長度（m）。"""
    lz: float
    """z 軸房間長度（m）。"""
    c: float
    """聲速（m/s）。"""
    rho_c: float
    """空氣特性阻抗 ρ·c（Pa·s/m）。"""
    src: tuple[float, float, float]
    """聲源座標（m）。"""
    recv: tuple[float, float, float]
    """接收點座標（m）。"""
    freqs_hz: tuple[float, ...]
    """頻率（Hz），六個頻帶。"""

    def length(self, axis: int) -> float:
        """軸 index（0=x、1=y、2=z）那一軸的房間長度。"""
        return (self.lx, self.ly, self.lz)[axis]


# 一面牆一個頻帶的複數表面阻抗。輸入是牆名與頻帶 index，回 Z（Pa·s/m）。
WallImpedance = Callable[[str, int], complex]


@dataclass(frozen=True)
class Recompute:
    """一條路徑的獨立重算結果（反射乘積與路徑壓力，每個頻帶一個複數）。"""

    index: int
    order: int
    identity: tuple[int, int, int, int, int, int]
    dist_m: float
    reflection_product: tuple[complex, ...]
    path_pressure: tuple[complex, ...]


def canonical_walls() -> tuple[str, ...]:
    """規範順序的六面牆名（跟 v2 的 ``CANONICAL_WALL_ORDER`` 一致）。"""
    return tuple(name for name, _ in _CANONICAL_WALLS)


def _axis_order(n: int, sign: int) -> int:
    """一軸的反射次數：``sign=+1`` 是 ``2*|n|``，``sign=-1`` 是 ``|2n-1|``。

    跟 donor 的 ``_axis_wall_counts`` 推出的一軸總反射次數一致。
    """
    if sign == 1:
        return 2 * abs(n)
    return abs(2 * n - 1)


def order_of(identity: tuple[int, int, int, int, int, int]) -> int:
    """從 identity 六元組獨立算出的階數（反射總次數），直達 0。

    六元組 ``(nx, sx, ny, sy, nz, sz)``：每軸一組 (反射次數, 正負號)。三軸反射次數相加
    就是這一條路徑的階數。
    """
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])
    return _axis_order(n[0], s[0]) + _axis_order(n[1], s[1]) + _axis_order(n[2], s[2])


def image_from_identity(
    lx: float,
    ly: float,
    lz: float,
    identity: tuple[int, int, int, int, int, int],
    src: tuple[float, float, float],
) -> tuple[float, float, float]:
    """從 identity 六元組直接算鏡像座標，任何階都適用：``image = 2*n*L + s*src`` 每軸。

    直達（全 0、全 +1）回聲源本尊。
    """
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])
    return (
        2.0 * n[0] * lx + s[0] * src[0],
        2.0 * n[1] * ly + s[1] * src[1],
        2.0 * n[2] * lz + s[2] * src[2],
    )


def _wall_axis_kind(wall: str) -> tuple[int, str]:
    """牆名 → (軸 index, kind)。未知牆名當場炸（這一支只認六面牆）。"""
    for name, (axis, kind) in _CANONICAL_WALLS:
        if name == wall:
            return (axis, kind)
    raise ValueError(f"未知牆名：{wall!r}")


def wall_count_signature(
    identity: tuple[int, int, int, int, int, int],
) -> dict[str, int]:
    """identity → 每面牆反彈幾次的簽名（多重集，**不看時間順序**）。

    純從 identity 六元組的「每軸 ``(n, sign)``」算（照 donor ``_axis_wall_counts``）：
    ``sign=+1`` 時該軸兩面牆各 ``|n|`` 次；``sign=-1`` 時總共 ``|2n-1|`` 次、依 ``n`` 正負拆給
    zero 面（floor/x0/y0）與 L 面（ceiling/xL/yL）。牆名鍵照 canonical 順序。
    """
    n = (identity[0], identity[2], identity[4])
    s = (identity[1], identity[3], identity[5])

    def _lo_hi(nx: int, sign: int) -> tuple[int, int]:
        """一軸的 (zero 面次數, L 面次數)，照 donor ``_axis_wall_counts``。"""
        if sign == 1:
            return abs(nx), abs(nx)
        m = 2 * nx - 1
        order = abs(m)
        if m > 0:
            return order // 2, (order + 1) // 2
        return (order + 1) // 2, order // 2

    z_lo, z_hi = _lo_hi(n[2], s[2])
    x_lo, x_hi = _lo_hi(n[0], s[0])
    y_lo, y_hi = _lo_hi(n[1], s[1])
    return {
        "floor": z_lo,
        "ceiling": z_hi,
        "x0": x_lo,
        "xL": x_hi,
        "y0": y_lo,
        "yL": y_hi,
    }


def incidence_cos(
    axis: int,
    dist_m: float,
    recv: tuple[float, float, float],
    image: tuple[float, float, float],
) -> float:
    """一面牆的入射角餘弦：那一軸的 ``|receiver − image| / dist``（攤開直線）。"""
    return abs(recv[axis] - image[axis]) / dist_m


def reflection_coefficient(Z: complex, cos_theta: float, rho_c: float) -> complex:
    """``R = (Z·cosθ − ρc) / (Z·cosθ + ρc)``（adapter ``z_to_r_from_cos``）。"""
    z_cos = Z * cos_theta
    return (z_cos - rho_c) / (z_cos + rho_c)


def recompute_path(
    inp: Inputs,
    impedance: WallImpedance,
    index: int,
    identity: tuple[int, int, int, int, int, int],
) -> Recompute:
    """獨立重算一條路徑的反射乘積與路徑壓力（雙精度，``cmath`` 相位）。

    反射乘積對每一跳的 R 連乘（``r ** count``）；直達（order 0）沒有反彈，乘積是 ``1+0j``。
    ``r ** count`` 的前提是**整面牆同一個阻抗**（跟上一代逐次反彈取 cell 的做法在這個前提下
    等價；分格材料是以後的事）。路徑壓力 ``(1/dist)·refl·exp(−iωτ)``，``ω=2πf``、``τ=dist/c``。
    """
    order = order_of(identity)
    image = image_from_identity(inp.lx, inp.ly, inp.lz, identity, inp.src)
    diff = (
        inp.recv[0] - image[0],
        inp.recv[1] - image[1],
        inp.recv[2] - image[2],
    )
    dist = math.sqrt(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
    counts = wall_count_signature(identity)

    refl_per_freq: list[complex] = []
    for f_idx in range(len(inp.freqs_hz)):
        product = complex(1.0, 0.0)
        for wall in canonical_walls():
            count = counts[wall]
            if count == 0:
                continue
            axis, _kind = _wall_axis_kind(wall)
            cos = incidence_cos(axis, dist, inp.recv, image)
            r = reflection_coefficient(impedance(wall, f_idx), cos, inp.rho_c)
            product *= r if count == 1 else r ** count
        refl_per_freq.append(product)

    pp_per_freq: list[complex] = []
    for f_idx in range(len(inp.freqs_hz)):
        omega = 2.0 * math.pi * inp.freqs_hz[f_idx]
        tau = dist / inp.c
        phase = cmath.exp(complex(0.0, -omega * tau))
        pp_per_freq.append((1.0 / dist) * refl_per_freq[f_idx] * phase)

    return Recompute(
        index=index,
        order=order,
        identity=identity,
        dist_m=dist,
        reflection_product=tuple(refl_per_freq),
        path_pressure=tuple(pp_per_freq),
    )


def compare_cells(
    computed: tuple[complex, ...],
    answer: tuple[complex, ...],
    identifier: str,
    kind: str,
) -> list[str]:
    """逐格比對兩組複數：``computed``（雙精度重算）對 ``answer``（答案檔），回差異清單。

    每一格比的是**複數差的模** ``|computed − answer|`` 是否為 0（不是逐分量各比——雙精度
    對雙精度該逐位相同，一整格看成一筆差）。``identifier`` 指明是哪條路徑、``kind`` 指明
    反射乘積還是路徑壓力。兩邊長度不同會額外報一筆。
    """
    diffs: list[str] = []
    for f_idx in range(min(len(computed), len(answer))):
        mine = computed[f_idx]
        ref = answer[f_idx]
        diff = abs(mine - ref)
        if diff != 0.0:
            diffs.append(
                f"{identifier} {kind}[{f_idx}]: 答案 {ref!r}，獨立重算 {mine!r}，"
                f"複數差模 {diff!r}"
            )
    if len(computed) != len(answer):
        diffs.append(f"{identifier} {kind} 頻帶數不同：{len(computed)} 對 {len(answer)}")
    return diffs


def reflection_tolerance(f: float, tau: float, abs_refl: float) -> float:
    """反射乘積每分量的契約界線：絕對差 ≤ ``REFLECTION_CONTRACT_ULP``。

    決策紙選項 1：反射乘積每分量絕對差不超過 2^-21，三個參數（f、τ、|refl|）在這個
    界線上都不進公式——反射乘積界線是固定的 2^-21，跟頻率、時延、大小無關。函式保留
    這三個參數是為了跟 :func:`pressure_tolerance` 同一張簽名、也讓「界線不隨它們變」這件
    事在呼叫點看得見。
    """
    del f, tau, abs_refl
    return REFLECTION_CONTRACT_ULP


def pressure_tolerance(f: float, tau: float, abs_refl: float) -> float:
    """路徑壓力的契約界線（相對差）：``2^-21·(ωτ + 1) + 2^-21/|refl|``。

    決策紙選項 1：每條路徑的壓力相對差不超過 ``2^-21·(ωτ+1) + 2^-21/|反射乘積|``，其中
    ``ω = 2πf``、``τ`` 是到達時間。``f``（Hz）、``tau``（s）、``abs_refl``（反射乘積的大小）
    三個參數一起決定界線，缺一個都算不出來。

    注意：界線是**相對**容差，回傳的是一個比值；相對差以**複數差的模除以參考值的模**計：
    呼叫端拿來斷言 ``|Δp| ≤ 回傳值·|p|``（``Δp`` 是複數差，不是逐分量各比）。
    """
    omega_tau = 2.0 * math.pi * f * tau
    return REFLECTION_CONTRACT_ULP * (omega_tau + 1.0) + REFLECTION_CONTRACT_ULP / abs_refl
