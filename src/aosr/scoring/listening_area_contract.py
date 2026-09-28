"""聆聽區穩定度的凍結契約：從 ``contract.py`` 原樣搬出（#518 前置——那一支貼著 1000 行上限）。

逐字搬移、內容不改；``contract.py`` 用原名再匯出，既有呼叫端不必跟著改匯入來源。契約版本照舊由
``contract.CONTRACT_SCHEMA_VERSION`` 代表。
"""
from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from aosr.scoring.contract_base import FrozenModel as _FrozenModel
from aosr.scoring.contract_base import Flag, InputProvenance, ReasonCode


class DeviationEndpoint(_FrozenModel):
    """一筆偏差端點的接收點身分、相對主位方向與彙總重要性。"""

    receiver_id: str = Field(min_length=1)
    direction_relative_to_primary: str | None
    importance: Annotated[float, Field(ge=0.0)]


class WorstDeviation(_FrozenModel):
    """不加權的原始最差偏差與形成這筆比較的兩端。"""

    value: Annotated[float, Field(ge=0.0)]
    receiver: DeviationEndpoint
    reference: DeviationEndpoint


class DeviationAggregate(_FrozenModel):
    """一種比較集合的加權平均偏差與不加權最差偏差。"""

    weighted_mean_deviation: Annotated[float, Field(ge=0.0)]
    worst_deviation: WorstDeviation


class StabilityComparison(_FrozenModel):
    """同一品質輸出的兩組獨立結果；沒有周圍配對只讓第二組明確留空。"""

    primary_to_surrounding: DeviationAggregate
    surrounding_to_surrounding: DeviationAggregate | None
    surrounding_to_surrounding_reason: ReasonCode | None

    @model_validator(mode="after")
    def _peer_result_and_reason_agree(self) -> Self:
        if self.surrounding_to_surrounding is None:
            if self.surrounding_to_surrounding_reason != ReasonCode.NO_SURROUNDING_PAIRS:
                raise ValueError("周圍彼此留空時必須標 no_surrounding_pairs")
        elif self.surrounding_to_surrounding_reason is not None:
            raise ValueError("周圍彼此有結果時不准帶留空原因")
        return self


class PeakDipOccurrence(_FrozenModel):
    """主位的一個峰或谷在多少點有對應；第一版不列只在周圍彼此共有的峰谷。"""

    kind: Literal["peak", "dip"]
    primary_center_frequency_hz: Annotated[float, Field(gt=0.0)]
    receiver_ids: tuple[str, ...] = Field(min_length=1)
    occurrence_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def _count_matches_named_receivers(self) -> Self:
        if self.occurrence_count != len(self.receiver_ids):
            raise ValueError("occurrence_count 必須等於 receiver_ids 的實際數量")
        return self


class TargetDeviationPositionSpread(_FrozenModel):
    """音色對目標偏差曲線的位置間逐頻率最大最小差，只供診斷、不算代價。"""

    frequency_hz: Annotated[float, Field(gt=0.0)]
    spread_db: Annotated[float, Field(ge=0.0)]


class ReceiverPointProvenance(_FrozenModel):
    """真正疊入聆聽區結果的一份單點音色評估出身。"""

    receiver_id: str = Field(min_length=1)
    report_id: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    settings_fingerprint: str = Field(min_length=1)


class ReceiverPointFrequencySupport(_FrozenModel):
    """一個接收點的上游音色實際頻率支撐完整序列（#489）。"""

    receiver_id: str = Field(min_length=1)
    timbre_frequencies_hz: tuple[Annotated[float, Field(gt=0.0)], ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _frequencies_ascend(self) -> Self:
        if any(left >= right for left, right in zip(
            self.timbre_frequencies_hz, self.timbre_frequencies_hz[1:]
        )):
            raise ValueError("接收點音色頻率支撐必須嚴格遞增")
        return self


class ListeningAreaFrequencySupport(_FrozenModel):
    """聆聽區整體音量軸與所有已量接收點的上游音色軸（#489）。"""

    overall_level_frequencies_hz: tuple[Annotated[float, Field(gt=0.0)], ...] = Field(
        min_length=1
    )
    points: tuple[ReceiverPointFrequencySupport, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def _frequencies_and_receivers_are_valid(self) -> Self:
        axis = self.overall_level_frequencies_hz
        if any(left >= right for left, right in zip(axis, axis[1:])):
            raise ValueError("整體音量頻率支撐必須嚴格遞增")
        ids = [point.receiver_id for point in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("接收點頻率支撐的 receiver_id 不可重複")
        return self


class ListeningAreaStabilityPayload(_FrozenModel):
    """聆聽區穩定性的身分、逐點出身、四種量法與診斷曲線。

    診斷曲線來自音色評估已扣平均的「對目標偏差曲線」，所以對整體音量差盲；這不是
    漏量，因為 ``overall_level_stability`` 另行量整體音量差。音色的「對目標偏差均方根」
    不另算位置散布：直線目標下它與傾斜加起伏是同一份資訊，排名層也只拿它作對照，
    在這裡再算會把同一件事算兩次。
    """

    category: Literal["listening_area_stability"]
    candidate_id: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    receiver_set_fingerprint: str = Field(min_length=1)
    timbre_settings_fingerprint: str = Field(min_length=1)
    settings_fingerprint: str = Field(min_length=1)
    point_provenance: tuple[ReceiverPointProvenance, ...] = Field(min_length=2)
    frequency_support: ListeningAreaFrequencySupport
    tilt_stability: StabilityComparison
    ripple_rms_stability: StabilityComparison
    overall_level_stability: StabilityComparison
    peak_dip_consistency: StabilityComparison
    peak_dip_occurrences: tuple[PeakDipOccurrence, ...]
    target_deviation_position_spread_curve_db: tuple[TargetDeviationPositionSpread, ...]
    target_deviation_common_frequency_count: Annotated[int, Field(ge=0)]
    target_deviation_discarded_frequency_value_count: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _support_matches_point_provenance(self) -> Self:
        if tuple(point.receiver_id for point in self.frequency_support.points) != tuple(
            point.receiver_id for point in self.point_provenance
        ):
            raise ValueError("frequency_support 的接收點與 point_provenance 順序必須一致")
        return self


class ListeningAreaChannel(_FrozenModel):
    """一支聆聽區聲道的角色、喇叭、完整單支結果與出身。"""

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    speaker_id: str = Field(min_length=1)
    payload: ListeningAreaStabilityPayload
    provenance: InputProvenance
    flags: tuple[Flag, ...]


class ListeningAreaChannelsPayload(_FrozenModel):
    """候選聆聽區的逐聲道彙總；保留單支結果及共同比較身分。"""

    category: Literal["listening_area_stability_channels"]
    channel_group_fingerprint: str = Field(min_length=64, max_length=64)
    receiver_set_fingerprint: str = Field(min_length=1)
    primary_receiver_id: str = Field(min_length=1)
    listening_area_evaluator_version: str = Field(min_length=1)
    listening_area_settings_fingerprint: str = Field(min_length=1)
    channels: tuple[ListeningAreaChannel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _channel_identities_are_unique(self) -> Self:
        roles = [item.role for item in self.channels]
        speakers = [item.speaker_id for item in self.channels]
        if len(roles) != len(set(roles)) or len(speakers) != len(set(speakers)):
            raise ValueError("聆聽區聲道的角色與 speaker_id 都不可重複")
        return self
