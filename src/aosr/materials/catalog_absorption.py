"""把型錄的無規入射頻帶吸音率換成細軸上的實數阻抗。

本模組與考卷共同落實決策紙 ``catalog-absorption-random-incidence-paris-inversion`` 的
「決定」七條：材料自帶頻帶軸，先在 log10 頻率上內插吸音率，再以 Paris 閉式反推硬側
實數阻抗，且由呼叫端注入 ρc。量測頻帶外平坦取最近端帶，並逐點標成延伸；性質考卷的
相對差界線從唯一精度契約登記簿取得。

這條新流程不呼叫 :meth:`aosr.materials.response.MaterialResponse.from_alpha`。後者是第二類
相容紀錄仍需保留的舊吸收率與夾值路徑。``impedance_on_axis`` 把內插後超過實數 Paris 模型
頂點的型錄值逐點夾到頂點並標出；單點反推
``normalized_impedance_from_random_incidence_absorption`` 遇到超過頂點仍報錯。55 度法只住在
考卷的對照組，不是產品備援。
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np


PARIS_GAUSS_LEGENDRE_POINTS: Final[int] = 256
"""複數阻抗 Paris 無規入射積分使用的固定 Gauss--Legendre 點數。"""

_PARIS_NODES, _PARIS_WEIGHTS = np.polynomial.legendre.leggauss(
    PARIS_GAUSS_LEGENDRE_POINTS
)
_PARIS_MU = (_PARIS_NODES + 1.0) / 2.0
_PARIS_MAPPED_WEIGHTS = _PARIS_WEIGHTS / 2.0


def random_incidence_absorption(zeta: float) -> float:
    """回傳實數正規化阻抗 ``zeta`` 的 Paris 無規入射吸音率。

    定義域是有限的 ``zeta >= 1``，也就是決策紙原記的 ``Z >= rho*c``；低於
    這條界線的軟側不是這個公開換算契約，而且閉式在極小 ``zeta`` 會因相消
    靜默失真，所以直接報錯。定義域內的括號嚴格遞增，且在下界等於
    ``3/2 - 2*log(2) > 0``，因此不再保留不可達的括號非正檢查。``log1p``
    則保留大阻抗時 ``log(1 + zeta)`` 的準確度。
    """
    if not math.isfinite(zeta) or zeta < 1.0:
        raise ValueError(f"zeta={zeta!r} must be finite and >= 1")
    inverse = 1.0 / zeta
    bracket = 1.0 + 1.0 / (1.0 + zeta) - 2.0 * inverse * math.log1p(zeta)
    return 8.0 * inverse * bracket


def complex_random_incidence_absorption(zeta: complex) -> float:
    """回傳複數正規化阻抗的 Paris 無規入射吸音率。

    以 256 點 Gauss--Legendre 計算
    ``2 integral_0^1 (1-|((zeta*mu-1)/(zeta*mu+1))|^2) mu dmu``。
    被動局部反應邊界要求有限且 ``Re(zeta) > 0``；不符就直接報錯。
    """
    value = complex(zeta)
    if (
        not math.isfinite(value.real)
        or not math.isfinite(value.imag)
        or value.real <= 0.0
    ):
        raise ValueError(f"zeta={value!r} must be finite with real part > 0")
    reflection = (value * _PARIS_MU - 1.0) / (value * _PARIS_MU + 1.0)
    angle_absorption = 1.0 - np.abs(reflection) ** 2
    return float(
        2.0 * np.sum(_PARIS_MAPPED_WEIGHTS * angle_absorption * _PARIS_MU)
    )


def _absorption_derivative(zeta: float) -> float:
    """Paris 閉式對 ``zeta`` 的解析導數，只供頂點二分使用。"""
    inverse = 1.0 / zeta
    logarithm = math.log1p(zeta)
    bracket = 1.0 + 1.0 / (1.0 + zeta) - 2.0 * inverse * logarithm
    bracket_derivative = -1.0 / (1.0 + zeta) ** 2 - 2.0 * (
        zeta / (1.0 + zeta) - logarithm
    ) * inverse**2
    return 8.0 * inverse * (bracket_derivative - bracket * inverse)


def _computed_peak() -> tuple[float, float]:
    """用導數夾號二分求唯一頂點，直到 binary64 區間無法再縮。"""
    lower = 1.0
    upper = 3.0
    if _absorption_derivative(lower) <= 0.0:
        raise RuntimeError("Paris peak lower bracket does not have positive derivative")
    if _absorption_derivative(upper) >= 0.0:
        raise RuntimeError("Paris peak upper bracket does not have negative derivative")

    while True:
        midpoint = lower + (upper - lower) / 2.0
        if midpoint == lower or midpoint == upper:
            break
        if _absorption_derivative(midpoint) > 0.0:
            lower = midpoint
        else:
            upper = midpoint

    candidates = (lower, upper)
    zeta = max(candidates, key=random_incidence_absorption)
    return zeta, random_incidence_absorption(zeta)


ZETA_AT_MAX_ABSORPTION, MAX_RANDOM_INCIDENCE_ABSORPTION = _computed_peak()
"""模組載入時從解析導數二分算出的 Paris 頂點與頂點吸音率。"""


def _validate_alpha(alpha: float, *, prefix: str = "") -> None:
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"{prefix}alpha={alpha!r} must be finite and > 0")
    if alpha > MAX_RANDOM_INCIDENCE_ABSORPTION:
        raise ValueError(
            f"{prefix}alpha={alpha!r} exceeds "
            f"maximum={MAX_RANDOM_INCIDENCE_ABSORPTION!r}"
        )


def normalized_impedance_from_random_incidence_absorption(alpha: float) -> float:
    """在 Paris 頂點以上的硬側分支反推正規化實數阻抗 ζ。

    這是純反推層，只接受實數 Paris 模型可達的吸音率；超過頂點仍報錯。型錄量測值的
    逐點夾值只由 :func:`impedance_on_axis` 負責，避免這支一般反推函式偷偷改輸入。

    下界是程式算出的頂點。硬側吸音率連續、嚴格遞減且在 ζ 趨近無限大時趨近零；
    因此從頂點開始反覆把上界加倍；若 binary64 的有限上界內能表示該解，就會得到吸音率
    不大於 ``alpha`` 的另一端，從而保證夾根。若加倍先溢位，代表這套有限數運算找不到有限
    阻抗解，函式會報錯而不回傳無限值。接著只用函式值保留夾根區間，直到 binary64 中點
    等於某一端、區間已無法再縮小；沒有另設差值容差。
    """
    _validate_alpha(alpha)
    if alpha == MAX_RANDOM_INCIDENCE_ABSORPTION:
        return ZETA_AT_MAX_ABSORPTION

    lower = ZETA_AT_MAX_ABSORPTION
    upper = 2.0 * lower
    while random_incidence_absorption(upper) > alpha:
        next_upper = 2.0 * upper
        if not math.isfinite(next_upper):
            raise ValueError(
                f"alpha={alpha!r}: doubling could not find a finite upper bound"
            )
        upper = next_upper

    while True:
        midpoint = lower + (upper - lower) / 2.0
        if midpoint == lower or midpoint == upper:
            break
        if random_incidence_absorption(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint

    return min(
        (lower, upper),
        key=lambda zeta: abs(random_incidence_absorption(zeta) - alpha),
    )


@dataclass(frozen=True)
class CatalogAbsorption:
    """一份材料型錄的頻帶中心與無規入射吸音率。"""

    material_id: str
    band_center_hz: tuple[float, ...]
    absorption: tuple[float, ...]

    def __post_init__(self) -> None:
        frequencies = tuple(float(value) for value in self.band_center_hz)
        absorption = tuple(float(value) for value in self.absorption)
        object.__setattr__(self, "band_center_hz", frequencies)
        object.__setattr__(self, "absorption", absorption)

        if len(frequencies) != len(absorption):
            raise ValueError("band_center_hz and absorption must have the same length")
        if not frequencies:
            raise ValueError("catalog must have at least one band")
        for index, frequency in enumerate(frequencies, start=1):
            if not math.isfinite(frequency) or frequency <= 0.0:
                raise ValueError(
                    f"band {index} frequency={frequency!r} must be finite and > 0"
                )
        if any(right <= left for left, right in zip(frequencies, frequencies[1:])):
            raise ValueError("band_center_hz must be strictly increasing")
        for index, alpha in enumerate(absorption, start=1):
            if not math.isfinite(alpha) or alpha <= 0.0:
                raise ValueError(
                    f"band {index} alpha={alpha!r} must be finite and > 0"
                )


@dataclass(frozen=True)
class CatalogImpedanceOnAxis:
    """型錄攤到細軸後的原始與實用吸音率、阻抗、延伸及夾值記號。

    六個逐點欄位都用不可變 ``tuple``；容器本身也凍結，避免呼叫端改掉其中一份而讓
    同一結果裡的頻率、吸音率、阻抗與旗標彼此錯位。``catalog_absorption`` 保留內插後的
    原始型錄值，``absorption`` 是逐點夾過、真正交給 Paris 反推的值。
    """

    frequencies_hz: tuple[float, ...]
    catalog_absorption: tuple[float, ...]
    absorption: tuple[float, ...]
    impedance_pa_s_per_m: tuple[float, ...]
    extrapolated: tuple[bool, ...]
    clamped: tuple[bool, ...]


def impedance_on_axis(
    catalog: CatalogAbsorption,
    frequencies_hz: Sequence[float],
    rho_c_pa_s_per_m: float,
) -> CatalogImpedanceOnAxis:
    """把頻帶 α 在 log10 頻率上內插，再逐點反推並乘上呼叫端的 ρc。

    低於最低中心與高於最高中心的點平坦取最近端帶，並標 ``extrapolated=True``；
    中心點和中心之間都是 ``False``。原始內插值可能超過 Paris 頂點（也可能大於 1）；
    內插完成後才逐點以嚴格大於判斷夾到頂點，等於頂點不標成夾值。
    """
    if not math.isfinite(rho_c_pa_s_per_m) or rho_c_pa_s_per_m <= 0.0:
        raise ValueError(
            f"rho_c={rho_c_pa_s_per_m!r} must be finite and > 0"
        )

    target = tuple(float(value) for value in frequencies_hz)
    for index, frequency in enumerate(target, start=1):
        if not math.isfinite(frequency) or frequency <= 0.0:
            raise ValueError(
                f"frequencies_hz point {index}={frequency!r} must be finite and > 0"
            )

    target_log = np.log10(np.asarray(target, dtype=np.float64))
    band_log = np.log10(np.asarray(catalog.band_center_hz, dtype=np.float64))
    catalog_alpha_array = np.interp(
        target_log,
        band_log,
        np.asarray(catalog.absorption, dtype=np.float64),
    )
    catalog_alpha = tuple(float(value) for value in catalog_alpha_array)
    clamped = tuple(
        value > MAX_RANDOM_INCIDENCE_ABSORPTION for value in catalog_alpha
    )
    alpha = tuple(
        MAX_RANDOM_INCIDENCE_ABSORPTION if is_clamped else value
        for value, is_clamped in zip(catalog_alpha, clamped)
    )
    impedance_values: list[float] = []
    for index, (value, is_clamped) in enumerate(zip(alpha, clamped), start=1):
        zeta = (
            ZETA_AT_MAX_ABSORPTION
            if is_clamped
            else normalized_impedance_from_random_incidence_absorption(value)
        )
        scaled = rho_c_pa_s_per_m * zeta
        if not math.isfinite(scaled):
            raise ValueError(
                f"impedance point {index} is not finite after scaling: "
                f"alpha={value!r}, rho_c={rho_c_pa_s_per_m!r}"
            )
        impedance_values.append(scaled)
    impedance = tuple(impedance_values)
    lowest = catalog.band_center_hz[0]
    highest = catalog.band_center_hz[-1]
    extrapolated = tuple(
        frequency < lowest or frequency > highest for frequency in target
    )
    return CatalogImpedanceOnAxis(
        frequencies_hz=target,
        catalog_absorption=catalog_alpha,
        absorption=alpha,
        impedance_pa_s_per_m=impedance,
        extrapolated=extrapolated,
        clamped=clamped,
    )
