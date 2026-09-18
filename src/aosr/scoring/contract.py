"""這是評估器→排名層的凍結契約；任何變更都必須走合併請求。

第一層評估器產出本模組的模型，第三層排名只轉送出身、不自行補造。契約明分
measured（已量未算代價）、costed（已算類代價）與 unavailable（不可估）三種狀態。
"""
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


CONTRACT_SCHEMA_VERSION: Final[str] = "aosr.scoring.contract.v1"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


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


class Flag(StrEnum):
    """跨評估器傳遞、但不直接等於不可估原因的受控標記。"""

    CROSSOVER_BAND = "crossover_band"
    UNVALIDATED = "unvalidated"
    NO_DIRECTIVITY = "no_directivity"
    DATA_COVERAGE_SHORT = "data_coverage_short"
    FEATURE_TOO_NARROW = "feature_too_narrow"
    FEATURE_BOUNDARY_INCOMPLETE = "feature_boundary_incomplete"
    BASELINE_SETTINGS = "baseline_settings"


class ReasonCode(StrEnum):
    """不可估時由評估器填入的受控原因代碼。"""

    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    MISSING_POINTS = "missing_points"
    NON_POSITIVE_ENERGY = "non_positive_energy"
    SOLVER_UNAVAILABLE = "solver_unavailable"
    EVALUATOR_NOT_IMPLEMENTED = "evaluator_not_implemented"


class _FrozenModel(BaseModel):
    """共用凍結、拒收多餘欄位與非有限數的模型底座。"""

    model_config = FROZEN


class RawQuantity(_FrozenModel):
    """一個帶受控單位的原始量，供排名表保留可追查數字。"""

    name: str = Field(min_length=1)
    value: float
    unit: Literal["dB", "dB/oct", "Hz", "oct", "1"]


class Feature(_FrozenModel):
    """音色殘差曲線上的一個峰或谷；寬度未知（邊界不完整）才是 None，否則必須為正。"""

    kind: Literal["peak", "dip"]
    center_frequency_hz: Annotated[float, Field(gt=0.0)]
    depth_db: float
    width_octave: Annotated[float, Field(gt=0.0)] | None
    flags: tuple[Flag, ...]


FrequencyRange = tuple[Annotated[float, Field(gt=0.0)], Annotated[float, Field(gt=0.0)]]


class TimbrePayload(_FrozenModel):
    """音色評估器五格量法的完整拍板輸出。"""

    category: Literal["timbre_balance"]
    tilt_db_per_octave: float
    tilt_fit_range_hz: FrequencyRange
    target_tilt_db_per_octave: float
    target_deviation_rms_db: Annotated[float, Field(ge=0.0)]
    deviation_curve: tuple[tuple[float, float], ...]
    residual_rms_db: Annotated[float, Field(ge=0.0)]
    ripple_range_hz: FrequencyRange
    features: tuple[Feature, ...]
    strongest_peak_index: Annotated[int, Field(ge=0)] | None
    deepest_dip_index: Annotated[int, Field(ge=0)] | None
    data_range_hz: FrequencyRange
    coverage_range_hz: FrequencyRange

    @model_validator(mode="after")
    def _ranges_ascend(self) -> Self:
        """四個頻率範圍都要下端小於上端；顛倒的範圍讓「覆蓋不到」無法判定。"""
        for name in ("tilt_fit_range_hz", "ripple_range_hz", "data_range_hz", "coverage_range_hz"):
            lower, upper = getattr(self, name)
            if lower >= upper:
                raise ValueError(f"{name} 必須遞增")
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


class ListeningAreaStabilityPayload(_FrozenModel):
    """聆聽區穩定性尚未定欄位；只保留可辨識類別。"""

    category: Literal["listening_area_stability"]


class LowFrequencyDecayPayload(_FrozenModel):
    """低頻時間表現尚未定欄位；只保留可辨識類別。"""

    category: Literal["low_frequency_decay"]


class ReflectionsAndEchoPayload(_FrozenModel):
    """反射與回音尚未定欄位；只保留可辨識類別。"""

    category: Literal["reflections_and_echo"]


class ReverberationPayload(_FrozenModel):
    """殘響尚未定欄位；只保留可辨識類別。"""

    category: Literal["reverberation"]


class ChannelMatchingPayload(_FrozenModel):
    """聲道匹配尚未定欄位；只保留可辨識類別。"""

    category: Literal["channel_matching"]


class SpatialImpressionPayload(_FrozenModel):
    """空間感尚未定欄位；只保留可辨識類別。"""

    category: Literal["spatial_impression"]


CategoryPayload = Annotated[
    TimbrePayload
    | ListeningAreaStabilityPayload
    | LowFrequencyDecayPayload
    | ReflectionsAndEchoPayload
    | ReverberationPayload
    | ChannelMatchingPayload
    | SpatialImpressionPayload,
    Field(discriminator="category"),
]


class InputProvenance(_FrozenModel):
    """評估器吃到哪份報表與哪個聲源／接收點；排名層只轉不造。"""

    report_id: str = Field(min_length=1)
    engine_commit: str = Field(min_length=1)
    speaker_id: str = Field(min_length=1)
    receiver_id: str = Field(min_length=1)


class CategoryCost(_FrozenModel):
    """第二層算出的類代價、主要分項與所用代價設定指紋。"""

    value: float
    components: dict[str, float]
    cost_settings_fingerprint: str = Field(min_length=1)


class CategoryEvaluation(_FrozenModel):
    """一個候選的一類評估輸出；九條跨層不變條件在這裡守 1、2、3、4、6，第 9 條在候選包。

    第 5（類代價有限）、7（標記與原因是受控列舉）、8（任何數值不得 NaN／無限）三條由
    欄位層守：``FROZEN`` 的 ``allow_inf_nan=False`` 與 ``tuple[Flag, ...]``／``tuple[ReasonCode, ...]``
    的列舉型別在建構時就拒收；不另寫一支永遠跑不到的 validator（找碴席實測那三支是死碼）。
    """

    schema_version: str
    candidate_id: str = Field(min_length=1)
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
        if self.payload is not None and self.payload.category != self.category.value:
            raise ValueError("payload.category 必須等於外層 category")
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
    """同候選、同輸入報表出身的一包分類評估；每類最多一條。"""

    schema_version: str
    candidate_id: str = Field(min_length=1)
    provenance: InputProvenance
    evaluations: tuple[CategoryEvaluation, ...]

    @model_validator(mode="after")
    def _schema_version_matches(self) -> Self:
        """候選包也是輸出物件，版本必須等於同一份凍結契約。"""
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("schema_version 不等於 CONTRACT_SCHEMA_VERSION")
        return self

    @model_validator(mode="after")
    def _identity_matches_envelope(self) -> Self:
        """不變條件 9：每條候選身分與輸入出身都等於外層，排名層不補造。"""
        for item in self.evaluations:
            if item.candidate_id != self.candidate_id:
                raise ValueError("candidate_id 與候選包外層不一致")
            if item.provenance != self.provenance:
                raise ValueError("provenance 與候選包外層不一致")
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
