"""兩聲道聆聽軸與早期反射方向分區；角度界線只讀品質登記簿。"""
from __future__ import annotations

import math
from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from aosr.config.quality_targets import EntryStatus, QualityPurpose, SettingEntry
from aosr.scoring.contract_base import FrozenModel
from aosr.scoring.placement import CoordinateM


class DirectionZone(StrEnum):
    FRONT = "front"
    LATERAL = "lateral"
    REAR = "rear"
    VERTICAL = "vertical"


class ZoneLimits(FrozenModel):
    """方向分區的三條登記簿角度界線。"""

    vertical_min_abs_elevation_deg: float = Field(gt=0.0, lt=90.0)
    front_max_abs_azimuth_deg: float = Field(gt=0.0)
    rear_min_abs_azimuth_deg: float = Field(lt=180.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if not 0.0 < self.front_max_abs_azimuth_deg < self.rear_min_abs_azimuth_deg < 180.0:
            raise ValueError("前向與後向界線必須滿足 0 < 前 < 後 < 180")
        return self


def zone_limits(purpose: QualityPurpose) -> tuple[ZoneLimits, dict[str, EntryStatus]]:
    """從指定用途讀三條角度與其狀態，供評估器傳遞基線標記。"""
    keys = (
        "direction_zones.vertical_min_abs_elevation_deg",
        "direction_zones.front_max_abs_azimuth_deg",
        "direction_zones.rear_min_abs_azimuth_deg",
    )
    values: dict[str, float] = {}
    statuses: dict[str, EntryStatus] = {}
    for key in keys:
        entry = purpose.entry(key)
        if not isinstance(entry, SettingEntry) or entry.unit != "deg" or not isinstance(entry.value, float):
            raise ValueError(f"{key} 必須是 deg 浮點設定")
        values[key.removeprefix("direction_zones.")] = entry.value
        statuses[key] = entry.status
    return ZoneLimits.model_validate(values), statuses


class ListeningAxisUndefined(ValueError):
    """聲道數或水平幾何使聆聽軸無法定義。"""

    def __init__(self) -> None:
        super().__init__("聆聽軸定不出來")


def listening_axis(
    speakers: tuple[CoordinateM, ...], main_receiver: CoordinateM,
) -> tuple[float, float]:
    """由主位選定喇叭連線的水平垂直平分線朝向。"""
    if len(speakers) != 2:
        raise ListeningAxisUndefined()
    left, right = speakers
    dx, dy = right[0] - left[0], right[1] - left[1]
    span = math.hypot(dx, dy)
    if not math.isfinite(span) or span == 0.0:
        raise ListeningAxisUndefined()
    normal = (-dy / span, dx / span)
    midpoint = ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)
    to_speakers = (midpoint[0] - main_receiver[0], midpoint[1] - main_receiver[1])
    side = normal[0] * to_speakers[0] + normal[1] * to_speakers[1]
    if not math.isfinite(side) or side == 0.0:
        raise ListeningAxisUndefined()
    sign = math.copysign(1.0, side)
    return normal[0] * sign, normal[1] * sign


def listening_angles(
    direction_vector: CoordinateM, axis: tuple[float, float],
) -> tuple[float, float]:
    """回傳相對水平角與仰角；逆時針為正，水平角範圍 (-180, 180]。"""
    x, y, z = direction_vector
    horizontal = math.hypot(x, y)
    axis_length = math.hypot(*axis)
    if not all(math.isfinite(value) for value in (*direction_vector, *axis)) or axis_length == 0.0 or math.hypot(horizontal, z) == 0.0:
        raise ValueError("方向或聆聽軸必須是非零有限向量")
    azimuth = math.degrees(math.atan2(axis[0] * y - axis[1] * x, axis[0] * x + axis[1] * y))
    if azimuth == -180.0:
        azimuth = 180.0
    elevation = math.degrees(math.atan2(z, horizontal))
    return azimuth, elevation


def classify(azimuth: float, elevation: float, limits: ZoneLimits) -> DirectionZone:
    """先以仰角判垂直，再按相對聆聽軸水平角判三區。"""
    if not math.isfinite(azimuth) or not math.isfinite(elevation):
        raise ValueError("角度必須是有限數")
    if abs(elevation) >= limits.vertical_min_abs_elevation_deg:
        return DirectionZone.VERTICAL
    if abs(azimuth) < limits.front_max_abs_azimuth_deg:
        return DirectionZone.FRONT
    if abs(azimuth) <= limits.rear_min_abs_azimuth_deg:
        return DirectionZone.LATERAL
    return DirectionZone.REAR
