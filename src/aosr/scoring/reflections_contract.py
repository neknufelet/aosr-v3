"""反射與顫動回音的一候選完整凍結輸出；評估器在 ``reflections``，代價與警戒在 ``reflections_cost``。"""
from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from aosr.geometry.shoebox import WALL_SEQUENCE_ORDER
from aosr.physics.room_paths import NUMERICALLY_GUARDED_ORDER_K
from aosr.scoring.contract_base import (
    FrequencyRange,
    FrozenModel,
    InputProvenance,
    MetricState,
    ReasonCode,
)
from aosr.scoring.direction_zones import DirectionZone, ZoneLimits, classify


REFLECTIONS_AND_ECHO_EVALUATOR_VERSION: Final[str] = "aosr.scoring.reflections.v2"
CONFIRMED_NO_REFLECTION: Final[frozenset[ReasonCode]] = frozenset({
    ReasonCode.NO_REFLECTION_IN_ZONE_POINT, ReasonCode.ZERO_REFLECTION_ENERGY,
})

class ReflectionSource(StrEnum):
    PATH_TABLE = "path_table"
    WINDOW_EXTENSION = "window_extension"


class MetricCell(FrozenModel):
    """一個可缺值的診斷量；值、狀態與原因碼同進同出。"""

    value: float | None
    state: MetricState
    reason_codes: tuple[ReasonCode, ...]

    @model_validator(mode="after")
    def _coherent(self) -> Self:
        if (self.value is not None) != (self.state is MetricState.MEASURED):
            raise ValueError("值／狀態／原因碼不一致")
        if (self.value is None) != bool(self.reason_codes):
            raise ValueError("值／狀態／原因碼不一致")
        return self


class ReflectionPath(FrozenModel):
    """單條反射原始路徑與方向視圖；索引先主表後補算表。"""

    source: ReflectionSource
    source_index: Annotated[int, Field(ge=0)]
    order: Annotated[int, Field(ge=1)]
    wall_sequence: tuple[str, ...] = Field(min_length=1, description=WALL_SEQUENCE_ORDER)
    relative_direct_delay_s: Annotated[float, Field(ge=0.0)]
    room_azimuth_deg: float
    room_elevation_deg: float
    listening_azimuth_deg: float
    listening_elevation_deg: float
    zone: DirectionZone
    within_window: bool
    broadband_level_db: float | None
    broadband_state: MetricState
    broadband_reason_codes: tuple[ReasonCode, ...]

    @model_validator(mode="after")
    def _diagnostic_is_coherent(self) -> Self:
        MetricCell(value=self.broadband_level_db, state=self.broadband_state,
                     reason_codes=self.broadband_reason_codes)
        if len(self.wall_sequence) != self.order:
            raise ValueError("牆序列長度必須等於反射階數")
        return self


class ZonePoint(FrozenModel):
    """一個細頻率點在一區的最強窗內路徑與能量合計。"""

    frequency_hz: Annotated[float, Field(gt=0.0)]
    strongest_level_db: float | None
    strongest_delay_s: Annotated[float, Field(ge=0.0)] | None
    strongest_path_index: Annotated[int, Field(ge=0)] | None
    strongest_state: MetricState
    strongest_reason_codes: tuple[ReasonCode, ...]
    total_energy_db: MetricCell

    @model_validator(mode="after")
    def _strongest_is_coherent(self) -> Self:
        MetricCell(value=self.strongest_level_db, state=self.strongest_state,
                     reason_codes=self.strongest_reason_codes)
        has_path = self.strongest_delay_s is not None and self.strongest_path_index is not None
        if (self.strongest_delay_s is None) != (self.strongest_path_index is None):
            raise ValueError("最強反射的延遲與路徑索引必須同進同出")
        if self.strongest_level_db is not None and not has_path:
            raise ValueError("最強反射有聲級時必須有延遲與路徑索引")
        if has_path and self.strongest_level_db is None and ReasonCode.ZERO_REFLECTION_ENERGY not in self.strongest_reason_codes:
            raise ValueError("零能量路徑必須帶零能量原因碼")
        if not has_path and ReasonCode.ZERO_REFLECTION_ENERGY in self.strongest_reason_codes:
            raise ValueError("零能量原因必須指到路徑")
        if not has_path and self.total_energy_db.state is MetricState.MEASURED:
            raise ValueError("沒有窗內反射時總能量不可已量")
        if ReasonCode.ZERO_REFLECTION_ENERGY in self.strongest_reason_codes and self.total_energy_db.state is MetricState.MEASURED:
            raise ValueError("零能量反射不可寫成已量總能量")
        if self.strongest_state is MetricState.MEASURED and ReasonCode.NO_REFLECTION_IN_ZONE_POINT in self.total_energy_db.reason_codes:
            raise ValueError("已量最強反射的總能量不可是沒有反射")
        return self


class ZoneResult(FrozenModel):
    """一區在全細頻率軸的原始判定與診斷。"""

    zone: DirectionZone
    points: tuple[ZonePoint, ...]

    @model_validator(mode="after")
    def _frequencies_increase(self) -> Self:
        frequencies = tuple(point.frequency_hz for point in self.points)
        if any(left >= right for left, right in zip(frequencies, frequencies[1:])):
            raise ValueError("頻率必須遞增")
        return self


class ReflectionChannel(FrozenModel):
    """一支喇叭在一接收點的全部反射與四區原始結果。"""

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    speaker_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)
    is_primary: bool
    provenance: InputProvenance
    reflections: tuple[ReflectionPath, ...]
    zones: tuple[ZoneResult, ...]
    total_window_energy_db: tuple[MetricCell, ...]
    report_order_k: Annotated[int, Field(ge=0)]
    computed_order_k: Annotated[int, Field(ge=0)]
    coverage: Literal["complete", "not_provable", "missing"]
    validation: Literal["validated", "unvalidated", "missing"]
    state: MetricState
    reason_codes: tuple[ReasonCode, ...]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.state is MetricState.MEASURED) == bool(self.reason_codes):
            raise ValueError("聲道狀態與原因碼不一致")
        if self.provenance.speaker_id != self.speaker_id or self.provenance.receiver_id != self.receiver_id:
            raise ValueError("出身與聲道身分不一致")
        if self.computed_order_k < self.report_order_k:
            raise ValueError("補算階數不得小於主報表階數")
        if self.validation == "validated" and self.computed_order_k > NUMERICALLY_GUARDED_ORDER_K:
            raise ValueError("驗證階數不得超過數值保證階數")
        if self.coverage != "complete" and self.state is MetricState.MEASURED:
            raise ValueError("覆蓋未完成的聲道不可已量")
        zones = [item.zone for item in self.zones]
        if len(zones) != len(set(zones)):
            raise ValueError("聲道四區結果不可重複")
        if set(zones) != set(DirectionZone):
            raise ValueError("聲道必須保留四區結果")
        axes = [tuple(point.frequency_hz for point in zone.points) for zone in self.zones]
        if len(set(axes)) != 1:
            raise ValueError("同一聲道四區頻率軸必須一致")
        if len(self.total_window_energy_db) != len(axes[0]):
            raise ValueError("逐點長度與頻率軸必須一致")
        order = [(0 if path.source is ReflectionSource.PATH_TABLE else 1, path.source_index)
                 for path in self.reflections]
        if order != sorted(set(order)):
            raise ValueError("反射路徑必須先主表後補算且來源索引遞增不重複")
        for path in self.reflections:
            if path.source is ReflectionSource.PATH_TABLE and path.order > self.report_order_k:
                raise ValueError("主表反射階數不得超過主報表階數")
            if path.source is ReflectionSource.WINDOW_EXTENSION and not self.report_order_k < path.order <= self.computed_order_k:
                raise ValueError("補算反射階數必須高於主報表且不超過補算階數")
        for zone in self.zones:
            for point in zone.points:
                index = point.strongest_path_index
                if index is not None and (index >= len(self.reflections) or
                                          self.reflections[index].zone is not zone.zone or
                                          not self.reflections[index].within_window):
                    raise ValueError("最強反射路徑索引必須指向同區窗內路徑")
                if index is not None and point.strongest_delay_s != self.reflections[index].relative_direct_delay_s:
                    raise ValueError("最強反射延遲必須等於所指路徑")
        if not any(path.within_window for path in self.reflections) and any(
            cell.state is MetricState.MEASURED for cell in self.total_window_energy_db
        ):
            raise ValueError("沒有反射路徑時窗內總能量不可已量")
        return self


class WallPairBandRisk(FrozenModel):
    """一對平行牆在一個 1/3 八度子帶的三個獨立量值。

    全反射記成算不出，但它是持續度無限長、最該掛警戒的情況；代價那一刀要把
    FULL_REFLECTION 當成掛警戒，不能當不掛。
    """

    frequency_hz: Annotated[float, Field(gt=0.0, description="子帶的精確中心頻率")]
    nominal_center_hz: Annotated[int, Field(gt=0)]
    lower_hz: Annotated[float, Field(gt=0.0)]
    upper_hz: Annotated[float, Field(gt=0.0)]
    round_trip_loss_db: MetricCell
    decay_duration_s: MetricCell
    room_t20_s: MetricCell

    @model_validator(mode="after")
    def _physical_values(self) -> Self:
        if not self.lower_hz < self.frequency_hz < self.upper_hz:
            raise ValueError("子帶下界、精確中心、上界必須依序遞增")
        if self.round_trip_loss_db.value is not None and self.round_trip_loss_db.value <= 0.0:
            raise ValueError("全反射損耗不得寫成有限已量值")
        if self.decay_duration_s.value is not None and self.decay_duration_s.value <= 0.0:
            raise ValueError("衰減持續度必須為正")
        if self.room_t20_s.value is not None and self.room_t20_s.value <= 0.0:
            raise ValueError("本房 t20_s 必須為正")
        for cause in (ReasonCode.ZERO_RETENTION, ReasonCode.FULL_REFLECTION):
            if cause in self.round_trip_loss_db.reason_codes and cause not in self.decay_duration_s.reason_codes:
                raise ValueError("來回損耗不可計算時持續度須保留同一原因")
        return self


class WallPairRisk(FrozenModel):
    """一對平行牆的來回延遲與逐 1/3 八度子帶顫動資料。"""

    walls: tuple[str, str]
    round_trip_delay_s: MetricCell
    bands: tuple[WallPairBandRisk, ...]

    @model_validator(mode="after")
    def _bands_increase(self) -> Self:
        frequencies = tuple(band.frequency_hz for band in self.bands)
        if self.walls[0] == self.walls[1]:
            raise ValueError("兩面牆必須相異")
        if any(left >= right for left, right in zip(frequencies, frequencies[1:])):
            raise ValueError("牆對頻率必須遞增")
        if self.round_trip_delay_s.value is not None and self.round_trip_delay_s.value <= 0.0:
            raise ValueError("來回延遲必須為正")
        return self


class ReflectionsAndEchoPayload(FrozenModel):
    """每候選一份：主位計分，周圍點與逐牆資料完整保留。"""

    category: Literal["reflections_and_echo"]
    window_upper_ms: Annotated[float, Field(gt=0.0)]
    window_upper_s: Annotated[float, Field(gt=0.0)]
    frequency_range_hz: FrequencyRange
    zone_limits: ZoneLimits
    listening_axis_xy: tuple[float, float]
    listening_axis_rule: str = Field(min_length=1)
    primary_receiver_id: str = Field(min_length=1)
    includes_speaker_directivity: Literal[False]
    channels: tuple[ReflectionChannel, ...] = Field(min_length=1)
    wall_pairs: tuple[WallPairRisk, ...]
    flutter_alert_band_centers_hz: tuple[int, ...] = Field(min_length=1)

    def _check_path_views(self) -> None:
        for channel in self.channels:
            for point in channel.zones[0].points:
                if not self.frequency_range_hz[0] <= point.frequency_hz <= self.frequency_range_hz[1]:
                    raise ValueError("逐點頻率必須落在頻率範圍內")
            for path in channel.reflections:
                if classify(path.listening_azimuth_deg, path.listening_elevation_deg, self.zone_limits) is not path.zone:
                    raise ValueError("反射路徑分區與角度不一致")
                if path.within_window != (path.relative_direct_delay_s <= self.window_upper_s):
                    raise ValueError("反射路徑窗內旗標與延遲不一致")

    def _check_wall_pairs(self) -> None:
        expected_pairs = {("x0", "xL"), ("y0", "yL"), ("floor", "ceiling")}
        if len({item.walls for item in self.wall_pairs}) != len(self.wall_pairs):
            raise ValueError("平行牆對不可重複")
        if {item.walls for item in self.wall_pairs} != expected_pairs:
            raise ValueError("必須保留三對平行牆")
        band_axes = {tuple((band.nominal_center_hz, band.lower_hz,
                            band.frequency_hz, band.upper_hz) for band in pair.bands)
                     for pair in self.wall_pairs}
        if len(band_axes) != 1:
            raise ValueError("三對平行牆的子帶清單必須一致")
        names = [band.nominal_center_hz for band in self.wall_pairs[0].bands]
        if len(names) != len(set(names)):
            raise ValueError("牆對子帶標稱帶名不可重複")
        if len(self.flutter_alert_band_centers_hz) != len(set(self.flutter_alert_band_centers_hz)):
            raise ValueError("顫動警戒標稱帶名不可重複")
        if not set(self.flutter_alert_band_centers_hz) <= set(names):
            raise ValueError("顫動警戒標稱帶名須存在於牆對子帶")

    @model_validator(mode="after")
    def _identities_and_axes(self) -> Self:
        if self.window_upper_s != self.window_upper_ms / 1000.0:
            raise ValueError("時間窗 ms 與 s 不一致")
        if self.frequency_range_hz[0] >= self.frequency_range_hz[1]:
            raise ValueError("頻率範圍必須遞增")
        if not math.isclose(math.hypot(*self.listening_axis_xy), 1.0):
            raise ValueError("聆聽軸必須是水平單位向量")
        identities = [(item.role, item.receiver_id) for item in self.channels]
        if len(identities) != len(set(identities)):
            raise ValueError("角色與接收點不可重複")
        if identities != sorted(identities):
            raise ValueError("聲道必須按角色、接收點排序")
        roles = {item.role for item in self.channels}
        if len(roles) != 2:
            raise ValueError("目前只支援兩支喇叭")
        role_speakers = {(item.role, item.speaker_id) for item in self.channels}
        if len(role_speakers) != len(roles):
            raise ValueError("每個角色必須固定對應一支喇叭")
        if len({item.speaker_id for item in self.channels}) != len(roles):
            raise ValueError("每個角色必須固定對應一支不同喇叭")
        receiver_roles: dict[str, set[str]] = {}
        for item in self.channels:
            receiver_roles.setdefault(item.receiver_id, set()).add(item.role)
        if any(found != roles for found in receiver_roles.values()):
            raise ValueError("每個接收點都必須保留每個角色")
        if self.primary_receiver_id not in receiver_roles:
            raise ValueError("主位接收點不存在")
        primary_roles = {item.role for item in self.channels if item.is_primary and item.receiver_id == self.primary_receiver_id}
        if roles != primary_roles:
            raise ValueError("主位每個角色都必須有一支且身分一致")
        if any(item.is_primary != (item.receiver_id == self.primary_receiver_id) for item in self.channels):
            raise ValueError("主位旗標必須對應主位接收點")
        self._check_wall_pairs()
        if not any(channel.state is MetricState.MEASURED for channel in self.channels):
            raise ValueError("全部聲道不可估")
        measured_axes = {tuple(point.frequency_hz for point in channel.zones[0].points)
                         for channel in self.channels if channel.state is MetricState.MEASURED}
        if len(measured_axes) != 1:
            raise ValueError("各聲道逐點頻率必須一致")
        self._check_path_views()
        if any(channel.coverage != "complete" for channel in self.channels if channel.is_primary):
            raise ValueError("主位時間窗未蓋滿時整類不可估")
        if any(channel.state is not MetricState.MEASURED for channel in self.channels if channel.is_primary):
            raise ValueError("主位每支聲道都必須是已量")
        return self
