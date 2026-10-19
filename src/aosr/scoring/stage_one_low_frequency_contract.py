"""第一階段低音量測的凍結契約（票 #558）：輸入曲線、每支喇叭的兩個分項與列出不計分的量。

照 ``contract.py`` 開頭「新的類別 payload 住自己的模組」的做法另開一支；這一份不進七類候選包
（category payload，七類評估的數字那一塊），也不走排名層，是第一階段另外交給搜尋的篩選量。
原因碼用本模組自己的列舉，不往 ``contract_base`` 的共用列舉加——那份彙整照宣告順序排。
"""
from __future__ import annotations

import math
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.scoring.contract import Feature
from aosr.scoring.contract_base import FrequencyRange
from aosr.scoring.contract_base import FrozenModel as _FrozenModel


STAGE_ONE_LOW_FREQUENCY_EVALUATOR_VERSION: Final = "aosr.scoring.stage_one_low_frequency.v1"
# 機器讀的假設標示（assumption label，固定字面值）與給人看的那一句；後一句照老闆 2026-10-19 原話逐字
# （#558 第 ② 塊第 3 題之二，三階段紙第一階段那條收錄），不改寫、不縮短。
IDEAL_SOURCE_ASSUMPTION: Final = "ideal_source_no_loudspeaker_low_frequency_rolloff"
IDEAL_SOURCE_ASSUMPTION_TEXT: Final = (
    "未加入實際喇叭的低頻衰減，排名只適用於本次理想聲源基準；換播放系統後可能不同"
)
IdealSourceAssumption = Literal["ideal_source_no_loudspeaker_low_frequency_rolloff"]
IdealSourceAssumptionText = Literal[
    "未加入實際喇叭的低頻衰減，排名只適用於本次理想聲源基準；換播放系統後可能不同"
]
# 吃哪一欄：報表逐點表的有限元素能量（FEM energy，有限元素那一路的能量），不吃三路接合的總能量
# ——總能量在交接帶混進幾何路，聲源用解析指向性時還帶指向性，跟「理想聲源、沒有額外聲源修正」衝突。
EnergyColumn = Literal["fem_energy"]
_CurveInputConfig = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=True)


class StageOneReasonCode(StrEnum):
    """一支喇叭第一階段低音不可估的原因；不填零、不內插、不外插。"""

    # 主位與周圍點在計分範圍裡的頻率軸不是逐點相同（不內插去對齊）。
    FREQUENCY_AXIS_MISMATCH = "frequency_axis_mismatch"
    # 計分範圍裡的能量有非有限或非正的值（取不了 dB）。
    NON_POSITIVE_ENERGY = "non_positive_energy"
    # 計分範圍裡的點少到定不出一條直線。
    INSUFFICIENT_POINTS = "insufficient_points"
    # 計分範圍裡（上下界當虛擬點）有缺段：少交的頻率不補成假資料。
    SCORING_RANGE_GAP = "scoring_range_gap"


class StageOneSeatCurve(BaseModel):
    """一支喇叭在一個座位的低頻能量曲線；``fem_energy`` 的 None 是那個頻點沒有有限元素值。

    能量准收非有限與非正值，那是評估器要回「不可估」的情形，不在建構時就把證據擋掉；
    頻率軸必須有限、為正、嚴格遞增，而且與能量等長。
    """

    model_config = _CurveInputConfig

    receiver_id: str = Field(min_length=1)
    # 主位是 None；周圍點寫相對主位的方向（direction，例如 front、up），照座位組的寫法。
    direction_relative_to_primary: str | None
    report_id: str = Field(min_length=1)
    frequencies_hz: tuple[float, ...] = Field(min_length=1)
    fem_energy: tuple[float | None, ...]

    @model_validator(mode="after")
    def _axis_is_finite_and_ascending(self) -> Self:
        if len(self.fem_energy) != len(self.frequencies_hz):
            raise ValueError("fem_energy 必須與 frequencies_hz 等長")
        axis = self.frequencies_hz
        if not all(math.isfinite(value) and value > 0.0 for value in axis):
            raise ValueError("frequencies_hz 必須是有限的正頻率")
        if any(upper <= lower for lower, upper in zip(axis, axis[1:], strict=False)):
            raise ValueError("frequencies_hz 必須嚴格遞增")
        if self.direction_relative_to_primary is not None and not self.direction_relative_to_primary.strip():
            raise ValueError("周圍點的方向不可為空白")
        return self


class StageOneSpeakerCurves(BaseModel):
    """一支喇叭在主位與周圍點的曲線；周圍點的代號與方向都不可重複，也不可跟主位同代號。"""

    model_config = _CurveInputConfig

    speaker_id: str = Field(min_length=1)
    primary: StageOneSeatCurve
    surrounding: tuple[StageOneSeatCurve, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _seats_are_unambiguous(self) -> Self:
        if self.primary.direction_relative_to_primary is not None:
            raise ValueError("主位不帶相對主位方向")
        if any(seat.direction_relative_to_primary is None for seat in self.surrounding):
            raise ValueError("周圍點必須帶相對主位方向")
        ids = [self.primary.receiver_id, *(seat.receiver_id for seat in self.surrounding)]
        directions = [seat.direction_relative_to_primary for seat in self.surrounding]
        if len(ids) != len(set(ids)) or len(directions) != len(set(directions)):
            raise ValueError("座位代號與周圍方向都不可重複")
        return self

    @property
    def seat_layout(self) -> tuple[str, tuple[tuple[str, str | None], ...]]:
        """主位代號與周圍點（代號、方向）的排序清單；同一份輸入的每支喇叭要一樣。"""
        return self.primary.receiver_id, tuple(sorted(
            (seat.receiver_id, seat.direction_relative_to_primary) for seat in self.surrounding
        ))


class StageOneLowFrequencyInput(BaseModel):
    """一個候選每支喇叭 × 主位＋周圍點的低頻能量；每支喇叭用同一組座位。"""

    model_config = _CurveInputConfig

    candidate_id: str = Field(min_length=1)
    speakers: tuple[StageOneSpeakerCurves, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _speakers_share_one_seat_group(self) -> Self:
        speaker_ids = [item.speaker_id for item in self.speakers]
        if len(speaker_ids) != len(set(speaker_ids)):
            raise ValueError("speaker_id 不可重複")
        if len({item.seat_layout for item in self.speakers}) != 1:
            raise ValueError("每支喇叭的主位與周圍點（代號、方向）必須相同")
        return self


class SeatDirectionDifference(_FrozenModel):
    """一個周圍點對主位的逐頻 dB 差與它的加權均方根；兩邊同一個聲源與音量基準、不各自調平。"""

    receiver_id: str = Field(min_length=1)
    direction_relative_to_primary: str = Field(min_length=1)
    report_id: str = Field(min_length=1)
    difference_rms_db: Annotated[float, Field(ge=0.0)]
    # (頻率 Hz, 這一點減主位的 dB)，只含計分範圍裡的軸點。
    difference_curve_db: tuple[tuple[float, float], ...] = Field(min_length=1)


class WorstSeatDirection(_FrozenModel):
    """曲線差均方根最大的那個周圍點；只提醒，不淘汰、不改可估狀態。"""

    receiver_id: str = Field(min_length=1)
    direction_relative_to_primary: str = Field(min_length=1)
    difference_rms_db: Annotated[float, Field(ge=0.0)]


class StageOneSpeakerPayload(_FrozenModel):
    """一支喇叭的兩個計分項（principal，進分數的主項）與列出不計分（reported，只印）的量。"""

    primary_report_id: str = Field(min_length=1)
    # 計分：主位曲線扣自己的最小平方直線後的加權均方根（dB），與六點曲線差均方根的等權平均（dB）。
    own_line_ripple_rms_db: Annotated[float, Field(ge=0.0)]
    seat_robustness_rms_db: Annotated[float, Field(ge=0.0)]
    directions: tuple[SeatDirectionDifference, ...] = Field(min_length=1)
    worst_direction: WorstSeatDirection
    # 列出不計分：自己那條直線（dB 對 log2 f）的斜率與截距、主位殘差的峰谷清單與摘要索引。
    own_line_slope_db_per_octave: float
    own_line_intercept_db: float
    features: tuple[Feature, ...]
    strongest_peak_index: Annotated[int, Field(ge=0)] | None
    deepest_dip_index: Annotated[int, Field(ge=0)] | None
    # 資料在計分範圍裡真正蓋到的頻率（首末兩個軸點）；不外插到範圍上下界。
    data_range_hz: FrequencyRange

    @model_validator(mode="after")
    def _worst_is_one_of_the_directions(self) -> Self:
        named = {(item.receiver_id, item.difference_rms_db) for item in self.directions}
        if (self.worst_direction.receiver_id, self.worst_direction.difference_rms_db) not in named:
            raise ValueError("最差方向必須是 directions 裡的一筆")
        return self


class StageOneSpeakerEvaluation(_FrozenModel):
    """一支喇叭的結果：measured（已量）帶 payload、不帶原因；unavailable（不可估）反過來。"""

    speaker_id: str = Field(min_length=1)
    state: Literal["measured", "unavailable"]
    payload: StageOneSpeakerPayload | None
    reason_codes: tuple[StageOneReasonCode, ...]

    @model_validator(mode="after")
    def _state_matches_payload(self) -> Self:
        measured = self.state == "measured"
        if measured != (self.payload is not None) or measured == bool(self.reason_codes):
            raise ValueError("measured 必帶 payload 且無原因；unavailable 無 payload 且必帶原因")
        return self


class StageOneLowFrequencyEvaluation(_FrozenModel):
    """一個候選的第一階段低音量測：每支喇叭各一組，左右兩支不在這一層合成一個數。

    假設標示兩格是固定字面值，每份結果必帶；計分範圍照呼叫端給的抄，不在這裡寫預設。
    原始能量不另存，只記每條曲線的報表代號（report id，指回存原始曲線的那份報表）。
    """

    evaluator_version: Literal["aosr.scoring.stage_one_low_frequency.v1"]
    candidate_id: str = Field(min_length=1)
    energy_column: EnergyColumn
    scoring_range_hz: FrequencyRange
    primary_receiver_id: str = Field(min_length=1)
    assumption: IdealSourceAssumption
    assumption_text: IdealSourceAssumptionText
    speakers: tuple[StageOneSpeakerEvaluation, ...] = Field(min_length=1)
