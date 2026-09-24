"""由報表八度帶的精確界三等分；標稱帶名只供顯示。

彙整的對象跟報表八度 T20／T30 一樣，是 ``late_decay`` 逐頻點**已經擬合好**的 T20／T30（沒有換成
另一種衰減曲線合成再擬合）；同一批逐頻結果分別彙整成原八度報表與這一份 1/3 八度，不是把八度平均值
拆成三份。差別只在權重：報表八度是逐點平均，這裡按八度寬度加權（平行牆對也用同一支
:func:`subband_weighted_mean`）；在正式細軸上兩者的差只來自兩端格子被帶界切掉，大小看曲線在帶內怎麼變、也看曲線在
哪裡歸零：平的曲線兩者相等；拿幾條合成 T20 量過，三個子帶值的平均對八度值的相對差從十萬分之五
到萬分之九都有（例如 T20 跟頻率成正比時約萬分之六）。這是量過幾次的例子、沒有考卷守，不是上界；
單一子帶對八度值本來就可以差很多，這裡講的是三個子帶值的平均。
"""

from __future__ import annotations

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config import frequency_axis
from aosr.physics.late_decay import LateDecayBand
from aosr.physics.report_io import ReportInput, scene_fingerprint
from aosr.physics.three_lane_report import ThreeLaneReport


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ThirdOctaveBand(BaseModel):
    """一個嵌在既有報表八度帶中的對數三等分子帶。"""

    model_config = FROZEN

    nominal_center_hz: int = Field(gt=0)
    octave_center_hz: float = Field(gt=0.0)
    lower_hz: float = Field(gt=0.0)
    upper_hz: float = Field(gt=0.0)
    center_hz: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _ordered_edges(self) -> Self:
        if not self.lower_hz < self.center_hz < self.upper_hz:
            raise ValueError("子帶下界、中心、上界必須依序遞增")
        return self


class ThirdOctaveDecayRow(BaseModel):
    """一個子帶的既有逐頻擬合加權結果與各自不可估原因。"""

    model_config = FROZEN

    band: ThirdOctaveBand
    point_count: int = Field(ge=0)
    t20_s: float | None = Field(default=None, gt=0.0)
    t20_unavailable_reason: str | None = Field(default=None, min_length=1)
    t30_s: float | None = Field(default=None, gt=0.0)
    t30_unavailable_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _value_matches_reason(self) -> Self:
        for value, reason in (
            (self.t20_s, self.t20_unavailable_reason),
            (self.t30_s, self.t30_unavailable_reason),
        ):
            if (value is None) == (reason is None):
                raise ValueError("衰減時間有值則不能有原因，無值則必須有原因")
        return self


class ThirdOctaveDecay(BaseModel):
    """與報表場景綁定、依頻率排序的 1/3 八度帶衰減。"""

    model_config = FROZEN

    scene_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    rows: tuple[ThirdOctaveDecayRow, ...]

    @model_validator(mode="after")
    def _rows_ascend(self) -> Self:
        """列照下界嚴格遞增：評估器按標稱名找帶，少一帶或順序反了要在這裡就擋下。"""
        lowers = tuple(row.band.lower_hz for row in self.rows)
        if any(left >= right for left, right in zip(lowers, lowers[1:])):
            raise ValueError("rows 必須照子帶下界嚴格遞增")
        return self


def third_octave_bands() -> tuple[ThirdOctaveBand, ...]:
    """依所屬八度帶與子帶序號對表標稱名，依精確八度界算帶界。"""
    root_two = math.sqrt(2.0)
    rows: list[ThirdOctaveBand] = []
    for octave_index, center in enumerate(
        frequency_axis.GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
    ):
        edges = (
            center / root_two,
            center * 2 ** (-1.0 / 6.0),
            center * 2 ** (1.0 / 6.0),
            center * root_two,
        )
        centers = (center * 2 ** (-1.0 / 3.0), center, center * 2 ** (1.0 / 3.0))
        for index, nominal in enumerate(
            frequency_axis.GEOMETRIC_REPORT_THIRD_OCTAVE_NOMINAL_HZ[octave_index]
        ):
            rows.append(ThirdOctaveBand(
                nominal_center_hz=nominal,
                octave_center_hz=center,
                lower_hz=edges[index],
                upper_hz=edges[index + 1],
                center_hz=centers[index],
            ))
    return tuple(rows)


def subband_weighted_mean(
    frequencies_hz: tuple[float, ...],
    values: tuple[float, ...],
    band: ThirdOctaveBand,
) -> float:
    """一個子帶的八度寬度加權平均：先照半開帶界（下界含、上界不含）挑點，再用共用八度格子算權重。

    整房殘響與平行牆對都走這一支，兩邊的帶界與權重才是同一套（老闆 2026-09-24「牆對與整房殘響共用
    帶界及權重」）。``octave_cells_in_range`` 本身兩端都含，所以挑點一定要在這裡先做，剛好落在界上的
    點才不會被兩個子帶各算一次。
    """
    if len(frequencies_hz) != len(values):
        raise ValueError("逐頻點頻率與值的數量不同")
    selected = tuple(
        (frequency, value) for frequency, value in zip(frequencies_hz, values, strict=True)
        if band.lower_hz <= frequency < band.upper_hz
    )
    if not selected:
        raise ValueError(f"{band.nominal_center_hz} Hz 子帶內沒有逐頻點")
    weights = frequency_axis.octave_cells_in_range(
        tuple(frequency for frequency, _value in selected), (band.lower_hz, band.upper_hz)
    )
    numerator = sum(weight * value for weight, (_f, value) in zip(weights, selected, strict=True))
    return numerator / sum(weights)


def _weighted_decay(
    points: tuple[LateDecayBand, ...],
    band: ThirdOctaveBand,
    *,
    t30: bool,
) -> float:
    """子帶的 T20 或 T30；逐頻點缺值而母帶沒有原因就拒。"""
    values: list[float] = []
    for point in points:
        value = point.t30_s if t30 else point.t20_s
        if value is None:
            raise ValueError("逐頻晚期衰減缺值，但所屬八度帶沒有不可估原因")
        values.append(value)
    return subband_weighted_mean(tuple(point.frequency_hz for point in points), tuple(values), band)


def _decay_row(
    band: ThirdOctaveBand,
    report: ThreeLaneReport,
) -> ThirdOctaveDecayRow:
    """優先承接母帶原因，其次才從子帶既有逐頻點計算。"""
    parent = next(
        (row for row in report.bands if row.center_frequency_hz == band.octave_center_hz),
        None,
    )
    if parent is None:
        raise ValueError(f"報表缺少 {band.octave_center_hz:g} Hz 八度帶")
    points = tuple(
        point for point in report.late_decay.bands
        if band.lower_hz <= point.frequency_hz < band.upper_hz
    )
    empty_reason = "子帶內沒有晚期衰減細軸點" if not points else None
    t20_reason = parent.t20_unavailable_reason or empty_reason
    t30_reason = parent.t30_unavailable_reason or empty_reason
    return ThirdOctaveDecayRow(
        band=band,
        point_count=len(points),
        t20_s=None if t20_reason else _weighted_decay(points, band, t30=False),
        t20_unavailable_reason=t20_reason,
        t30_s=None if t30_reason else _weighted_decay(points, band, t30=True),
        t30_unavailable_reason=t30_reason,
    )


def build_third_octave_decay(
    report: ThreeLaneReport, inputs: ReportInput,
) -> ThirdOctaveDecay:
    """只讀既有逐頻 T20／T30，驗反射階數與低頻軸後綁場景指紋。"""
    if not isinstance(report, ThreeLaneReport):
        raise ValueError(f"report 不是 ThreeLaneReport：{type(report).__name__}")
    if report.reflection_order_k != inputs.reflection_order_k:
        raise ValueError("inputs 的反射階數跟 report 不同")
    if report.low_frequency_axis is not inputs.low_frequency_axis:
        raise ValueError("inputs 的低頻軸跟 report 不同")
    return ThirdOctaveDecay(
        scene_fingerprint=scene_fingerprint(inputs),
        rows=tuple(_decay_row(band, report) for band in third_octave_bands()),
    )
