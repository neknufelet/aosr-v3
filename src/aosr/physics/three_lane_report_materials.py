"""三路報表的牆面材料換算小工具：阻抗表、散射表（原樣從 three_lane_report 搬出，騰行數）。

:mod:`aosr.physics.three_lane_report` 用同名轉出，外面照舊 ``three_lane_report._wall_impedances`` 取用。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from aosr.geometry.shoebox import Wall


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
