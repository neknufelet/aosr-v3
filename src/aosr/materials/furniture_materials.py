"""家具頻帶資料的逐點阻抗；沿用型錄的 Paris（無規入射）反推，不自行讀物理常數。"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from aosr.config.furniture_materials import FurnitureMaterial
from aosr.materials.catalog_absorption import CatalogAbsorption, CatalogImpedanceOnAxis, impedance_on_axis


@dataclass(frozen=True)
class FurnitureImpedanceOnAxis(CatalogImpedanceOnAxis):
    """沿用原始 α、夾後 α、實數阻抗、延伸及夾值，另標預設未知延伸段。

    unknown_extrapolated 按預設資料判斷；即使複核上下界在未知頻帶有明給假設，
    仍標未知，不冒充量到。extrapolated 則只表示此次所選曲線是否在自己的軸外。
    未知只標落在未知八度帶裡的點：八度帶照 frequency_axis 的半開區間
    [中心÷√2, 中心×√2)，所以端帶中心以外、仍在端帶裡的點只標延伸、不標未知。
    """

    unknown_extrapolated: tuple[bool, ...]
    unknown_label: str = "未知（計算時用相鄰頻帶延伸代算）"


def furniture_impedance_on_axis(
    material: FurnitureMaterial,
    frequencies_hz: Sequence[float],
    rho_c_pa_s_per_m: float,
    *, curve: str = "default", kind: str | None = None,
) -> FurnitureImpedanceOnAxis:
    """預設、下界或上界用同一函式算；選界線須明給適用家具種類。

    先內插原始吸音率，再逐點夾到 Paris 頂點並反推；所有數值與兩個既有旗標
    直接來自 impedance_on_axis。未知旗標覆蓋預設軸兩端未知八度帶裡的點，不只中心點。
    ρc 由呼叫端從自己的物理條件帶進，不在此模組讀設定或複寫數字。
    """
    if curve not in ("default", "lower", "upper"):
        raise ValueError(f"未知吸音率曲線 {curve!r}")
    bounds = material.for_kind(kind) if kind is not None else None
    if curve == "default":
        series = material.default
    elif bounds is None:
        raise ValueError("上下界計算必須明給家具種類")
    else:
        series = bounds.lower if curve == "lower" else bounds.upper
    result = impedance_on_axis(
        CatalogAbsorption(material_id="furniture", band_center_hz=series.band_center_hz,
                          absorption=series.absorption), frequencies_hz, rho_c_pa_s_per_m,
    )
    low, high = material.default.band_center_hz[0], material.default.band_center_hz[-1]
    unknown_low = any(band < low for band in material.unknown_bands_hz)
    unknown_high = any(band > high for band in material.unknown_bands_hz)
    known_lower_edge, known_upper_edge = low / math.sqrt(2.0), high * math.sqrt(2.0)
    unknown = tuple((frequency < known_lower_edge and unknown_low)
                    or (frequency >= known_upper_edge and unknown_high)
                    for frequency in result.frequencies_hz)
    return FurnitureImpedanceOnAxis(
        frequencies_hz=result.frequencies_hz, catalog_absorption=result.catalog_absorption,
        absorption=result.absorption, impedance_pa_s_per_m=result.impedance_pa_s_per_m,
        extrapolated=result.extrapolated, clamped=result.clamped, unknown_extrapolated=unknown,
    )
