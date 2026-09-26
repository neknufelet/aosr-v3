"""三路報表的牆面材料換算小工具：阻抗表、散射表、無規入射吸收率、具名阻抗列（原樣從 three_lane_report 搬出，騰行數）。

:mod:`aosr.physics.three_lane_report` 用同名轉出，外面照舊 ``three_lane_report._wall_impedances`` 取用。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ
from aosr.geometry.shoebox import Wall
from aosr.materials.catalog_absorption import complex_random_incidence_absorption


def _wall_impedances(
    impedance_by_wall: Mapping[Wall, object],
) -> dict[Wall, float]:
    """只收 FEM 目前支援的六面頻率無關正有限實數阻抗。"""
    if set(impedance_by_wall) != set(Wall.all()):
        raise ValueError("impedance_by_wall 必須恰好包含 Wall.all() 的六面牆")
    result: dict[Wall, float] = {}
    for wall in Wall.all():
        value = impedance_by_wall[wall]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{wall.wall_name()} 的阻抗必須是頻率無關的正有限實數")
        impedance = float(value)
        if not math.isfinite(impedance) or impedance <= 0.0:
            raise ValueError(f"{wall.wall_name()} 的阻抗必須是頻率無關的正有限實數")
        result[wall] = impedance
    return result


def _scattering_by_name(
    scattering_by_wall: Mapping[Wall, float] | None,
) -> dict[str, float] | None:
    if scattering_by_wall is None:
        return None
    unknown = set(scattering_by_wall) - set(Wall.all())
    if unknown:
        raise ValueError("scattering_by_wall 只能使用 Wall.all() 的牆面")
    result: dict[str, float] = {}
    for wall, value in scattering_by_wall.items():
        scattering = float(value)
        if not math.isfinite(scattering) or not 0.0 <= scattering <= 1.0:
            raise ValueError(f"{wall.wall_name()} 的散射係數必須有限且落在 [0,1]")
        result[wall.wall_name()] = scattering
    return result


def _random_absorption_by_wall(
    wall_impedances: Mapping[Wall, float],
    rho_c_pa_s_per_m: float,
) -> dict[Wall, dict[float, float]]:
    if not math.isfinite(rho_c_pa_s_per_m) or rho_c_pa_s_per_m <= 0.0:
        raise ValueError("rho_c_pa_s_per_m 必須是有限正數")
    result: dict[Wall, dict[float, float]] = {}
    for wall in Wall.all():
        impedance = wall_impedances[wall]
        absorption = complex_random_incidence_absorption(
            complex(impedance / rho_c_pa_s_per_m)
        )
        result[wall] = {
            frequency_hz: absorption for frequency_hz in SCHROEDER_T60_BANDS_HZ
        }
    return result


def _named_impedance_rows(
    wall_impedances: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
) -> dict[str, tuple[complex, ...]]:
    return {
        wall.wall_name(): tuple(
            complex(wall_impedances[wall]) for _frequency in frequencies_hz
        )
        for wall in Wall.all()
    }
