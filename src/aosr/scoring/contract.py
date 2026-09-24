"""這是評估器→排名層的凍結契約；任何變更都必須走合併請求。

第一層評估器產出本模組的模型，第三層排名只轉送場景身分與分類評估、不自行補造擺位。契約明分
measured（已量未算代價）、costed（已算類代價）與 unavailable（不可估）三種狀態。
"""
from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import Field, model_validator

from aosr.config.quality_targets import Unit
from aosr.scoring.placement import Placement, merge_placements
# 共用底座搬去 contract_base（#351：新的類別 payload 住自己的模組、不准回頭拿這一支）；
# 這裡用原名明示再匯出，既有呼叫端（含考卷）不必跟著改匯入來源。
from aosr.scoring.contract_base import FROZEN as FROZEN
from aosr.scoring.contract_base import Flag as Flag
from aosr.scoring.contract_base import FrequencyRange as FrequencyRange
from aosr.scoring.contract_base import FrozenModel as _FrozenModel
from aosr.scoring.contract_base import MetricState as MetricState
from aosr.scoring.contract_base import ReasonCode as ReasonCode


CONTRACT_SCHEMA_VERSION: Final[str] = "aosr.scoring.contract.v3"
CostDirection = Literal["below_range", "within_range", "above_range"]
SceneFingerprint = Annotated[
    str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
]


class QualityCategory(StrEnum):
    """完整七類聆聽品質的受控名稱。"""

    TIMBRE_BALANCE = "timbre_balance"
    LISTENING_AREA_STABILITY = "listening_area_stability"
    LOW_FREQUENCY_DECAY = "low_frequency_decay"
    REFLECTIONS_AND_ECHO = "reflections_and_echo"
    REVERBERATION = "reverberation"
    CHANNEL_MATCHING = "channel_matching"
    SPATIAL_IMPRESSION = "spatial_impression"


class EvaluationState(StrEnum):
    """評估結果是否可估、是否已換算類代價的明確狀態。"""

    MEASURED = "measured"
    COSTED = "costed"
    UNAVAILABLE = "unavailable"


class SchroederPosition(StrEnum):
    """整個頻帶相對 Schroeder 交界的位置。"""

    BELOW = "below"
    ABOVE = "above"
    CROSSING = "crossing"


class ModelValidationStatus(StrEnum):
    """輸入報表宣告的模型驗證狀態；與頻帶相對交界的位置互不推導。"""

    VALIDATED = "validated"
    EXPERIMENTAL = "experimental"
    UNSUPPORTED = "unsupported"
    UNCHECKED = "unchecked"


class RawQuantity(_FrozenModel):
    """一個帶受控單位的原始量，供排名表保留可追查數字。"""

    name: str = Field(min_length=1)
    value: float
    unit: Unit


class InputProvenance(_FrozenModel):
    """評估器吃到哪份報表與哪個聲源／接收點；排名層只轉不造。"""

    report_id: str = Field(min_length=1)
    engine_commit: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)


class Feature(_FrozenModel):
    """音色殘差曲線上的一個峰或谷；寬度未知（邊界不完整）才是 None，否則必須為正。"""

    kind: Literal["peak", "dip"]
    center_frequency_hz: Annotated[float, Field(gt=0.0)]
    depth_db: float
    width_octave: Annotated[float, Field(gt=0.0)] | None
    flags: tuple[Flag, ...]


ModelValidationFrequencyRange = FrequencyRange | tuple[()]


class TimbrePayload(_FrozenModel):
    """音色評估器五格量法的完整拍板輸出。"""

    category: Literal["timbre_balance"]
    tilt_db_per_octave: float
    tilt_fit_range_hz: FrequencyRange
    tilt_dependency_range_hz: FrequencyRange
    target_tilt_db_per_octave: float
    target_deviation_rms_db: Annotated[float, Field(ge=0.0)]
    deviation_curve: tuple[tuple[float, float], ...]
    residual_rms_db: Annotated[float, Field(ge=0.0)]
    ripple_range_hz: FrequencyRange
    ripple_dependency_range_hz: FrequencyRange
    features: tuple[Feature, ...]
    strongest_peak_index: Annotated[int, Field(ge=0)] | None
    deepest_dip_index: Annotated[int, Field(ge=0)] | None
    data_range_hz: FrequencyRange
    coverage_range_hz: FrequencyRange
    model_validation_status: ModelValidationStatus
    model_validation_frequency_range_hz: ModelValidationFrequencyRange

    @model_validator(mode="after")
    def _ranges_ascend(self) -> Self:
        """有值的頻率範圍都要遞增；顛倒的範圍讓「覆蓋不到」無法判定。"""
        for name in (
            "tilt_fit_range_hz",
            "tilt_dependency_range_hz",
            "ripple_range_hz",
            "ripple_dependency_range_hz",
            "data_range_hz",
            "coverage_range_hz",
        ):
            lower, upper = getattr(self, name)
            if lower >= upper:
                raise ValueError(f"{name} 必須遞增")
        if self.model_validation_frequency_range_hz:
            lower, upper = self.model_validation_frequency_range_hz
            if lower >= upper:
                raise ValueError("model_validation_frequency_range_hz 必須遞增")
        unchecked = self.model_validation_status is ModelValidationStatus.UNCHECKED
        if unchecked != (not self.model_validation_frequency_range_hz):
            # 報表那一側的約定：沒查表＝狀態 unchecked 而且範圍是空的，兩格同進同出。
            raise ValueError("能力範圍是空的若且唯若狀態是 unchecked")
        return self

    @model_validator(mode="after")
    def _summary_indices_point_at_features(self) -> Self:
        """摘要索引必須指到清單裡真的存在、而且種類相符的特徵；否則摘要是捏造的。"""
        for name, kind in (("strongest_peak_index", "peak"), ("deepest_dip_index", "dip")):
            index = getattr(self, name)
            if index is None:
                continue
            if index >= len(self.features):
                raise ValueError(f"{name} 超出 features 範圍")
            if self.features[index].kind != kind:
                raise ValueError(f"{name} 指到的特徵不是 {kind}")
        return self


class TimbreChannel(_FrozenModel):
    """主位一個聲道的角色、真實喇叭代號與完整單支音色結果。"""

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    speaker_id: str = Field(min_length=1)
    payload: TimbrePayload
    provenance: InputProvenance
    flags: tuple[Flag, ...]


class TimbreChannelsPayload(_FrozenModel):
    """候選主位的逐聲道音色；保留每支原始輸出與共同比較身分。"""

    category: Literal["timbre_balance_channels"]
    channel_group_fingerprint: str = Field(min_length=64, max_length=64)
    primary_receiver_id: str = Field(min_length=1)
    timbre_evaluator_version: str = Field(min_length=1)
    timbre_settings_fingerprint: str = Field(min_length=1)
    channels: tuple[TimbreChannel, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _channel_identities_are_unique(self) -> Self:
        roles = [item.role for item in self.channels]
        speakers = [item.speaker_id for item in self.channels]
        if len(roles) != len(set(roles)) or len(speakers) != len(set(speakers)):
            raise ValueError("音色聲道的角色與 speaker_id 都不可重複")
        return self


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
    tilt_stability: StabilityComparison
    ripple_rms_stability: StabilityComparison
    overall_level_stability: StabilityComparison
    peak_dip_consistency: StabilityComparison
    peak_dip_occurrences: tuple[PeakDipOccurrence, ...]
    target_deviation_position_spread_curve_db: tuple[TargetDeviationPositionSpread, ...]
    target_deviation_common_frequency_count: Annotated[int, Field(ge=0)]
    target_deviation_discarded_frequency_value_count: Annotated[int, Field(ge=0)]


class LowFrequencyDecayPayload(_FrozenModel):
    """低頻時間表現尚未定欄位；只保留可辨識類別。"""

    category: Literal["low_frequency_decay"]


class ReflectionsAndEchoPayload(_FrozenModel):
    """反射與回音尚未定欄位；只保留可辨識類別。"""

    category: Literal["reflections_and_echo"]


class ReverberationMetric(_FrozenModel):
    """一個殘響量值自己的值、狀態與原因；各指標不共用或吞併狀態。"""

    value: float | None
    unit: Literal["s", "1"]
    state: MetricState
    reason_codes: tuple[ReasonCode, ...]
    reason: str | None

    @model_validator(mode="after")
    def _value_state_and_reason_agree(self) -> Self:
        if self.state == MetricState.MEASURED:
            if self.value is None or self.reason_codes or self.reason is not None:
                raise ValueError("measured 必須有值且不准帶不可估原因")
        elif self.value is not None or not self.reason_codes or not self.reason:
            raise ValueError("unavailable／not_computable 必須無值並帶機器原因與人話")
        return self


class ReverberationBand(_FrozenModel):
    """一個八度帶的 T20、診斷 T30／T20，以及兩個彼此獨立的可信度標記。"""

    center_frequency_hz: Annotated[float, Field(gt=0.0)]
    band_range_hz: FrequencyRange
    schroeder_position: SchroederPosition
    model_validation_status: ModelValidationStatus
    t20: ReverberationMetric
    t30: ReverberationMetric
    fitting_difference: ReverberationMetric

    @model_validator(mode="after")
    def _range_and_metrics_are_consistent(self) -> Self:
        lower, upper = self.band_range_hz
        if not lower < self.center_frequency_hz < upper:
            raise ValueError("band_range_hz 必須遞增並夾住中心頻率")
        if self.t20.unit != "s" or self.t30.unit != "s":
            raise ValueError("T20 與 T30 的單位必須是秒")
        if self.fitting_difference.unit != "1":
            raise ValueError("擬合差異 T30/T20 必須是無因次")
        for name in ("t20", "t30", "fitting_difference"):
            metric = getattr(self, name)
            if metric.state == MetricState.MEASURED and metric.value <= 0.0:
                raise ValueError(f"{name} 量到的值必須為正")
        return self


class AdjacentBandChange(_FrozenModel):
    """由低中心頻率往高中心頻率的 T20 帶正負號對數比；絕對大小由讀者按需取得。"""

    lower_center_frequency_hz: Annotated[float, Field(gt=0.0)]
    upper_center_frequency_hz: Annotated[float, Field(gt=0.0)]
    signed_log_ratio: float | None
    state: MetricState
    reason_codes: tuple[ReasonCode, ...]
    reason: str | None

    @model_validator(mode="after")
    def _value_state_and_direction_agree(self) -> Self:
        if self.lower_center_frequency_hz >= self.upper_center_frequency_hz:
            raise ValueError("相鄰帶必須由低中心頻率指向高中心頻率")
        if self.state == MetricState.MEASURED:
            if self.signed_log_ratio is None:
                raise ValueError("measured 的相鄰帶變化必須有對數比")
            if self.reason_codes or self.reason is not None:
                raise ValueError("measured 的相鄰帶變化不准帶原因")
        elif (
            self.signed_log_ratio is not None
            or not self.reason_codes
            or not self.reason
        ):
            raise ValueError("不可計算的相鄰帶變化必須無值並帶機器原因與人話")
        return self


class ReverberationPayload(_FrozenModel):
    """逐帶殘響長短、擬合診斷，以及每一對相鄰帶的相對突變。"""

    category: Literal["reverberation"]
    bands: tuple[ReverberationBand, ...] = Field(min_length=1)
    adjacent_band_changes: tuple[AdjacentBandChange, ...]
    logarithm_base: Annotated[float, Field(gt=1.0)]

    @model_validator(mode="after")
    def _logarithm_and_adjacency_are_consistent(self) -> Self:
        if len(self.adjacent_band_changes) != len(self.bands) - 1:
            raise ValueError("adjacent_band_changes 必須逐一對應相鄰頻帶")
        for lower, upper, change in zip(
            self.bands[:-1], self.bands[1:], self.adjacent_band_changes, strict=True
        ):
            if (
                change.lower_center_frequency_hz != lower.center_frequency_hz
                or change.upper_center_frequency_hz != upper.center_frequency_hz
            ):
                raise ValueError("相鄰帶變化的兩端必須對應 bands 裡的相鄰項目")
        return self


class ChannelIdentity(_FrozenModel):
    """聲道組裡一支喇叭的可擴充角色與實際喇叭身分。"""

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    speaker_id: str = Field(min_length=1)


class ChannelComparisonPair(_FrozenModel):
    """有方向的聲道比較對；所有差值固定是左欄減右欄。"""

    left_role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    right_role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")

    @model_validator(mode="after")
    def _roles_are_distinct(self) -> Self:
        if self.left_role == self.right_role:
            raise ValueError("比較對的兩個角色不可相同")
        return self


class ChannelSourceEvaluation(_FrozenModel):
    """一個接收點的一支聲道原始音色評估；只收尚未算代價的原始結果。"""

    schema_version: str
    candidate_id: str = Field(min_length=1)
    role: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    state: Literal[EvaluationState.MEASURED, EvaluationState.UNAVAILABLE]
    payload: TimbrePayload | None
    raw_quantities: tuple[RawQuantity, ...]
    flags: tuple[Flag, ...]
    reason_codes: tuple[ReasonCode, ...]
    evaluator_version: str = Field(min_length=1)
    settings_fingerprint: str = Field(min_length=1)
    provenance: InputProvenance

    @model_validator(mode="after")
    def _raw_timbre_state_is_complete(self) -> Self:
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("聲道原始音色結果版本不符")
        if self.provenance.speaker_id != self.speaker_id:
            raise ValueError("聲道原始音色結果的 speaker_id 不一致")
        if self.state == EvaluationState.MEASURED:
            if self.payload is None or not self.raw_quantities or self.reason_codes:
                raise ValueError("measured 聲道原始結果必須完整且不帶不可估原因")
        elif self.payload is not None or self.raw_quantities or not self.reason_codes:
            raise ValueError("unavailable 聲道原始結果必須無量值並帶原因")
        return self


class ChannelPointSources(_FrozenModel):
    """一個接收點輸入的全部聲道原始結果；未列入比較的聲道也保留。"""

    receiver_id: str = Field(min_length=1)
    channels: tuple[ChannelSourceEvaluation, ...] = Field(min_length=1)


class ChannelFrequencyDifference(_FrozenModel):
    """同一頻率平滑後的左欄減右欄音色差。"""

    frequency_hz: Annotated[float, Field(gt=0.0)]
    left_minus_right_db: float


class ChannelFeatureDifference(_FrozenModel):
    """只出現在比較對一邊的峰或谷。"""

    present_in_role: str = Field(min_length=1)
    feature: Feature


class ChannelPointMatch(_FrozenModel):
    """一個接收點的一個聲道比較對；不可估時不捏造任何差值。"""

    receiver_id: str = Field(min_length=1)
    importance: Annotated[float, Field(ge=0.0)]
    left_role: str = Field(min_length=1)
    right_role: str = Field(min_length=1)
    state: Literal[MetricState.MEASURED, MetricState.UNAVAILABLE]
    reason_codes: tuple[ReasonCode, ...]
    reason: str | None
    tilt_difference_db_per_octave: float | None
    ripple_rms_difference_db: float | None
    broadband_level_difference_db: float | None
    direct_time_difference_ms: float | None
    frequency_difference_curve_db: tuple[ChannelFrequencyDifference, ...]
    unmatched_features: tuple[ChannelFeatureDifference, ...]

    @model_validator(mode="after")
    def _state_matches_values(self) -> Self:
        values = (
            self.tilt_difference_db_per_octave,
            self.ripple_rms_difference_db,
            self.broadband_level_difference_db,
            self.direct_time_difference_ms,
        )
        if self.state == MetricState.MEASURED:
            if any(value is None for value in values):
                raise ValueError("measured 聲道逐點結果必須有四個量值")
            if self.reason_codes or self.reason is not None:
                raise ValueError("measured 聲道逐點結果不准帶不可估原因")
        elif any(value is not None for value in values) or not self.reason_codes or not self.reason:
            raise ValueError("unavailable 聲道逐點結果必須無量值並帶原因")
        return self


class ChannelMetricAggregate(_FrozenModel):
    """逐點絕對差沿聆聽區權重彙總，另保留未加權最差點。"""

    weighted_mean_absolute_difference: Annotated[float, Field(ge=0.0)]
    worst_absolute_difference: Annotated[float, Field(ge=0.0)]
    worst_receiver_id: str = Field(min_length=1)


class ChannelComparisonAggregate(_FrozenModel):
    """一個明列比較對的四種彙總；沒有可估點時四種都明確留空。"""

    left_role: str = Field(min_length=1)
    right_role: str = Field(min_length=1)
    assessed_receiver_ids: tuple[str, ...]
    unavailable_receiver_ids: tuple[str, ...]
    tilt_difference: ChannelMetricAggregate | None
    ripple_rms_difference: ChannelMetricAggregate | None
    broadband_level_difference: ChannelMetricAggregate | None
    direct_time_difference: ChannelMetricAggregate | None

    @model_validator(mode="after")
    def _availability_matches_aggregates(self) -> Self:
        aggregates = (
            self.tilt_difference,
            self.ripple_rms_difference,
            self.broadband_level_difference,
            self.direct_time_difference,
        )
        if self.assessed_receiver_ids and any(item is None for item in aggregates):
            raise ValueError("有可估點時四種聲道彙總都必須存在")
        if not self.assessed_receiver_ids and any(item is not None for item in aggregates):
            raise ValueError("沒有可估點時不准捏造聲道彙總")
        return self


class ChannelBroadbandSupport(_FrozenModel):
    """寬頻音量實際納入的最低頻點、最高頻點與點數。"""

    lowest_frequency_hz: Annotated[float, Field(gt=0.0)]
    highest_frequency_hz: Annotated[float, Field(gt=0.0)]
    frequency_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def _bounds_match_count(self) -> Self:
        if self.lowest_frequency_hz > self.highest_frequency_hz:
            raise ValueError("寬頻支撐的最低頻率不可高於最高頻率")
        same_endpoint = self.lowest_frequency_hz == self.highest_frequency_hz
        if same_endpoint != (self.frequency_count == 1):
            raise ValueError("寬頻支撐的端點必須與頻率點數一致")
        return self


class ChannelMatchingPayload(_FrozenModel):
    """聲道匹配的五個比較身分、原始單聲道結果、逐點差、彙總與診斷。"""

    category: Literal["channel_matching"]
    candidate_id: str = Field(min_length=1)
    receiver_set_fingerprint: str = Field(min_length=1)
    timbre_settings_fingerprint: str = Field(min_length=1)
    listening_area_settings_fingerprint: str = Field(min_length=1)
    channel_group_fingerprint: str = Field(min_length=1)
    channels: tuple[ChannelIdentity, ...] = Field(min_length=2)
    comparisons: tuple[ChannelComparisonPair, ...] = Field(min_length=1)
    point_sources: tuple[ChannelPointSources, ...]
    point_results: tuple[ChannelPointMatch, ...] = Field(min_length=1)
    aggregates: tuple[ChannelComparisonAggregate, ...] = Field(min_length=1)
    broadband_support: ChannelBroadbandSupport
    direct_time_cost_enabled: bool

    @model_validator(mode="after")
    def _structure_is_unambiguous(self) -> Self:
        roles = [channel.role for channel in self.channels]
        speakers = [channel.speaker_id for channel in self.channels]
        if len(roles) != len(set(roles)) or len(speakers) != len(set(speakers)):
            raise ValueError("聲道角色與 speaker_id 都不可重複")
        pairs = tuple(
            (item.left_role, item.right_role) for item in self.comparisons
        )
        if len(pairs) != len(set(pairs)):
            raise ValueError("聲道比較對不可重複")
        if any(left not in roles or right not in roles for left, right in pairs):
            raise ValueError("聲道比較對必須引用聲道組內角色")
        self._rows_are_unambiguous(pairs)
        unavailable = next(
            (item for item in self.point_results if item.state is not MetricState.MEASURED),
            None,
        )
        if unavailable is not None:
            # 票 #403：該量的點有一個不可估，整類就該是不可估；已量的輸出裡不准夾著不可估的列。
            raise ValueError(
                f"接收點 {unavailable.receiver_id} 不可估：聲道匹配整類應回不可估，不是帶著缺點的已量"
            )
        return self

    def _rows_are_unambiguous(self, pairs: tuple[tuple[str, str], ...]) -> None:
        pair_set = set(pairs)
        aggregate_pairs = [
            (item.left_role, item.right_role) for item in self.aggregates
        ]
        unexpected = next(
            (pair for pair in aggregate_pairs if pair not in pair_set), None
        )
        if unexpected is not None:
            raise ValueError(f"聲道比較對 {unexpected[0]}/{unexpected[1]} 未明列")
        for pair in pairs:
            if aggregate_pairs.count(pair) != 1:
                raise ValueError(
                    f"聲道比較對 {pair[0]}/{pair[1]} 必須剛好一條彙總列"
                )

        result_keys = [
            (item.receiver_id, item.left_role, item.right_role)
            for item in self.point_results
        ]
        seen: set[tuple[str, str, str]] = set()
        for receiver, left, right in result_keys:
            if (receiver, left, right) in seen:
                raise ValueError(
                    f"接收點 {receiver} 的聲道比較對 {left}/{right} 必須剛好一條逐點結果"
                )
            seen.add((receiver, left, right))
            if (left, right) not in pair_set:
                raise ValueError(f"聲道比較對 {left}/{right} 未明列（接收點 {receiver}）")
        receivers = {receiver for receiver, _, _ in result_keys}
        missing = next(
            (
                (receiver, left, right)
                for receiver in sorted(receivers)
                for left, right in pairs
                if (receiver, left, right) not in seen
            ),
            None,
        )
        if missing is not None:
            raise ValueError(
                f"接收點 {missing[0]} 的聲道比較對 {missing[1]}/{missing[2]} 缺逐點結果"
            )
        self._aggregate_membership_is_unambiguous(pairs, result_keys)

    def _aggregate_membership_is_unambiguous(
        self,
        pairs: tuple[tuple[str, str], ...],
        result_keys: list[tuple[str, str, str]],
    ) -> None:
        receivers_by_pair = {
            pair: {
                receiver
                for receiver, left, right in result_keys
                if (left, right) == pair
            }
            for pair in pairs
        }
        measured_by_pair = {
            pair: {
                item.receiver_id
                for item in self.point_results
                if (item.left_role, item.right_role) == pair
                and item.state is MetricState.MEASURED
            }
            for pair in pairs
        }
        for aggregate in self.aggregates:
            pair = (aggregate.left_role, aggregate.right_role)
            assessed = aggregate.assessed_receiver_ids
            unavailable = aggregate.unavailable_receiver_ids
            for name, receiver_ids in (
                ("assessed_receiver_ids", assessed),
                ("unavailable_receiver_ids", unavailable),
            ):
                repeated = next(
                    (item for item in receiver_ids if receiver_ids.count(item) > 1),
                    None,
                )
                if repeated is not None:
                    raise ValueError(
                        f"聲道比較對 {pair[0]}/{pair[1]} 的 {name} 重複接收點 {repeated}"
                    )
            overlap = set(assessed) & set(unavailable)
            if overlap:
                receiver = sorted(overlap)[0]
                raise ValueError(
                    f"聲道比較對 {pair[0]}/{pair[1]} 的接收點 {receiver} 同時列為已評與不可估"
                )
            listed = set(assessed) | set(unavailable)
            expected = receivers_by_pair[pair]
            if expected - listed:
                receiver = sorted(expected - listed)[0]
                raise ValueError(
                    f"聲道比較對 {pair[0]}/{pair[1]} 的接收點 {receiver} 漏掉逐點結果"
                )
            if listed - expected:
                receiver = sorted(listed - expected)[0]
                raise ValueError(
                    f"聲道比較對 {pair[0]}/{pair[1]} 的接收點 {receiver} 沒有逐點結果"
                )
            misfiled = set(assessed) ^ measured_by_pair[pair]
            if misfiled:
                # 已評清單＝那一對逐點結果是已量的點；互換兩張清單會讓不可估的點冒充已評。
                receiver = sorted(misfiled)[0]
                raise ValueError(
                    f"聲道比較對 {pair[0]}/{pair[1]} 的接收點 {receiver} 已評／不可估的歸類跟逐點結果不符"
                )


class SpatialImpressionPayload(_FrozenModel):
    """空間感尚未定欄位；只保留可辨識類別。"""

    category: Literal["spatial_impression"]


CategoryPayload = Annotated[
    TimbrePayload
    | TimbreChannelsPayload
    | ListeningAreaStabilityPayload
    | LowFrequencyDecayPayload
    | ReflectionsAndEchoPayload
    | ReverberationPayload
    | ChannelMatchingPayload
    | SpatialImpressionPayload,
    Field(discriminator="category"),
]

_PAYLOAD_OUTER_CATEGORIES: Final[dict[str, QualityCategory]] = {
    "timbre_balance": QualityCategory.TIMBRE_BALANCE,
    "timbre_balance_channels": QualityCategory.TIMBRE_BALANCE,
    "listening_area_stability": QualityCategory.LISTENING_AREA_STABILITY,
    "low_frequency_decay": QualityCategory.LOW_FREQUENCY_DECAY,
    "reflections_and_echo": QualityCategory.REFLECTIONS_AND_ECHO,
    "reverberation": QualityCategory.REVERBERATION,
    "channel_matching": QualityCategory.CHANNEL_MATCHING,
    "spatial_impression": QualityCategory.SPATIAL_IMPRESSION,
}


def in_declared_order[E: StrEnum](items: Iterable[E]) -> tuple[E, ...]:
    """去重後照列舉宣告的順序排：彙總類的原因碼與旗標不准跟著輸入的順序變（票 #417）。"""
    found = set(items)
    return tuple(member for member in type(next(iter(found))) if member in found) if found else ()


class UnassessedBand(_FrozenModel):
    """第二層沒有代價的頻帶；保留頻帶身分與第一層給的原因碼。"""

    center_frequency_hz: Annotated[float, Field(gt=0.0)]
    state: Literal["unassessed"] = "unassessed"
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1)


class CategoryCost(_FrozenModel):
    """第二層算出的類代價、逐項診斷、未評估頻帶與所用代價設定指紋。"""

    value: float
    components: dict[str, float]
    component_directions: dict[str, CostDirection] = Field(default_factory=dict)
    unassessed_bands: tuple[UnassessedBand, ...] = ()
    cost_settings_fingerprint: str = Field(min_length=1)


class CategoryEvaluation(_FrozenModel):
    """一個候選的一類評估輸出；另帶這一類實際用到的擺位。

    九條跨層不變條件在這裡守 1、2、3、4、6，第 9 條在候選包。

    第 5（類代價有限）、7（標記與原因是受控列舉）、8（任何數值不得 NaN／無限）三條由
    欄位層守：``FROZEN`` 的 ``allow_inf_nan=False`` 與 ``tuple[Flag, ...]``／``tuple[ReasonCode, ...]``
    的列舉型別在建構時就拒收；不另寫一支永遠跑不到的 validator（找碴席實測那三支是死碼）。
    """

    schema_version: str
    candidate_id: str = Field(min_length=1)
    scene_fingerprint: SceneFingerprint
    placement: Placement
    category: QualityCategory
    state: EvaluationState
    payload: CategoryPayload | None
    raw_quantities: tuple[RawQuantity, ...]
    category_cost: CategoryCost | None
    flags: tuple[Flag, ...]
    reason_codes: tuple[ReasonCode, ...]
    evaluator_version: str
    settings_fingerprint: str
    provenance: InputProvenance

    @model_validator(mode="after")
    def _schema_version_matches(self) -> Self:
        """不變條件 1：輸出版本必須等於這份凍結契約版本。"""
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("schema_version 不等於 CONTRACT_SCHEMA_VERSION")
        return self

    @model_validator(mode="after")
    def _payload_matches_category(self) -> Self:
        """不變條件 2：可估輸出有具名量，payload（類別資料）與外層類別相同。"""
        if self.state != EvaluationState.UNAVAILABLE and self.payload is None:
            raise ValueError("可估狀態必須帶 payload")
        if self.state == EvaluationState.UNAVAILABLE and self.payload is not None:
            raise ValueError("unavailable 不准帶 payload（不填零也不捏造數值）")
        if (
            self.payload is not None
            and _PAYLOAD_OUTER_CATEGORIES[self.payload.category] is not self.category
        ):
            raise ValueError("payload.category 必須對應外層 category")
        return self

    @model_validator(mode="after")
    def _estimable_state_matches_cost(self) -> Self:
        """不變條件 3：measured（已量）無代價，costed（已算代價）必有代價；兩者都是可估，
        所以不准帶不可估原因、而且至少要有一個原始量（「已量」卻沒有量是自相矛盾）。"""
        if self.state == EvaluationState.MEASURED and self.category_cost is not None:
            raise ValueError("measured 的 category_cost 必須是空的")
        if self.state == EvaluationState.COSTED and self.category_cost is None:
            raise ValueError("costed 的 category_cost 不可為空")
        if self.state != EvaluationState.UNAVAILABLE:
            if not self.placement.speaker_positions_m:
                raise ValueError("可估狀態的 placement.speaker_positions_m 至少要有一筆")
            if not self.placement.receiver_positions_m:
                raise ValueError("可估狀態的 placement.receiver_positions_m 至少要有一筆")
            if self.reason_codes:
                raise ValueError("可估狀態的 reason_codes 必須是空的")
            if not self.raw_quantities:
                raise ValueError("可估狀態的 raw_quantities 至少要有一個")
        return self

    @model_validator(mode="after")
    def _unavailable_has_reason(self) -> Self:
        """不變條件 4：unavailable（不可估）沒有代價，而且至少帶一個原因。"""
        if self.state != EvaluationState.UNAVAILABLE:
            return self
        if self.category_cost is not None:
            raise ValueError("unavailable 的 category_cost 必須是空的")
        if not self.reason_codes:
            raise ValueError("unavailable 的 reason_codes 至少要有一個")
        return self


    @model_validator(mode="after")
    def _comparison_identity_is_nonempty(self) -> Self:
        """不變條件 6：評估器版本與設定指紋都不可為空白。"""
        for name, value in (
            ("evaluator_version", self.evaluator_version),
            ("settings_fingerprint", self.settings_fingerprint),
        ):
            if not value.strip():
                raise ValueError(f"{name} 不可為空")
        return self


class CandidateEvaluation(_FrozenModel):
    """同候選、同場景的一包分類評估；同代號同座標，每類最多一條。"""

    schema_version: str
    candidate_id: str = Field(min_length=1)
    scene_fingerprint: SceneFingerprint
    evaluations: tuple[CategoryEvaluation, ...]

    @model_validator(mode="after")
    def _schema_version_matches(self) -> Self:
        """候選包也是輸出物件，版本必須等於同一份凍結契約。"""
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("schema_version 不等於 CONTRACT_SCHEMA_VERSION")
        return self

    @model_validator(mode="after")
    def _identity_matches_envelope(self) -> Self:
        """不變條件 9：每條候選身分與場景指紋都等於外層，排名層不補造。"""
        for item in self.evaluations:
            if item.candidate_id != self.candidate_id:
                raise ValueError("candidate_id 與候選包外層不一致")
            if item.scene_fingerprint != self.scene_fingerprint:
                raise ValueError("scene_fingerprint 與候選包外層不一致")
        merge_placements(item.placement for item in self.evaluations)
        return self

    @model_validator(mode="after")
    def _estimable_timbre_uses_channel_aggregate(self) -> Self:
        """候選裡的可估音色必須完整列出主位各聲道，不接受任選一支的單點結果。"""
        for item in self.evaluations:
            if (
                item.category is QualityCategory.TIMBRE_BALANCE
                and item.state is not EvaluationState.UNAVAILABLE
                and not isinstance(item.payload, TimbreChannelsPayload)
            ):
                raise ValueError("候選包的可估音色必須先走聲道彙總")
        return self

    @model_validator(mode="after")
    def _categories_are_unique(self) -> Self:
        """同一候選的一類最多一條，否則類代價沒有唯一答案。"""
        seen: set[QualityCategory] = set()
        for item in self.evaluations:
            if item.category in seen:
                raise ValueError(f"候選包重複 category：{item.category.value}")
            seen.add(item.category)
        return self
