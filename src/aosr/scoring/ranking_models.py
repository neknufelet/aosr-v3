"""排名輸出的資料模型與參考基準字串（票 #357 原文從 ``ranking.py`` 搬出，騰行數；#505 第三刀）。

``ranking.py`` 貼著單檔行數上限；這一支只放模型，不放排名邏輯。``ranking.py`` 用原名明示再匯出，
既有呼叫端（含考卷）不必跟著改匯入來源。
"""
from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from aosr.config.quality_targets import (
    EntryStatus,
    QualificationValue,
    Unit,
)
from aosr.physics.report_facts import NO_BASIS_COUNT, facts
from aosr.scoring.category_registry import EliminationReason, NotEvaluatedReason
from aosr.scoring.contract import (
    CategoryEvaluation,
    CostDirection,
    Flag,
    QualityCategory,
    ReasonCode,
    UnassessedBand,
)
from aosr.scoring.cost_shapes import ComponentRole
from aosr.scoring import recommendation as rec
from aosr.scoring.review_alert import ReviewAlert


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

# 代價與權重那幾格的參考基準；每個數字旁另有評估器版本、設定指紋與聲源模型指紋（見 ComparisonIdentity）。
COST_REFERENCE: Final[str] = (
    "無因次：超出或偏離的量除以登記簿的較差參考（worse_reference）；"
    "各項的 1 不代表同等的品質損失，不是品質判決的絕對尺度"
)
WEIGHT_REFERENCE: Final[str] = "沒有基準（只是登記簿的權重，第一版等權是試驗基線、不是中立）"
RAW_REFERENCE: Final[str] = "評估器輸出的原始量，單位見同一列的 raw_unit；排名層只轉不改"


class CandidateStatus(StrEnum):
    """候選的五種狀態；``illegal`` 只留給候選產生層，這一刀沒有規則會產生它。"""

    RANKABLE = "rankable"
    ELIMINATED = "eliminated"
    NOT_EVALUATED = "not_evaluated"
    NOT_COMPARABLE = "not_comparable"
    ILLEGAL = "illegal"


class ExternalAcceptance(StrEnum):
    """外部底線（可施工、預算、噪音、失真）的驗收狀態；排名層只讀不算。"""

    NOT_CHECKED = "not_checked"
    PASSED = "passed"
    FAILED = "failed"


class EligibilityApplication(StrEnum):
    """一條資料資格規則對一類有沒有套上。"""

    APPLIED = "applied"
    NOT_APPLICABLE = "not_applicable"


class _FrozenModel(BaseModel):
    """凍結、拒收多餘欄位與非有限數的輸出模型底座。"""

    model_config = FROZEN


class RankingContext(_FrozenModel):
    """呼叫端給的表頭資料；日期與引擎版本由呼叫端給，這一支不讀時鐘。"""

    purpose: str = Field(min_length=1)
    receiver_set_fingerprint: str = Field(min_length=1)
    channel_group_fingerprint: str = Field(min_length=1)
    run_date: date
    engine_version: str = Field(min_length=1)


class ExternalFloors(_FrozenModel):
    """專案宣告的外部底線結果；排名層只讀。沒列到的候選記「未檢查」。"""

    declared_by: str = Field(min_length=1)
    verdicts: dict[str, Literal[ExternalAcceptance.PASSED, ExternalAcceptance.FAILED]]


class ComparisonIdentity(_FrozenModel):
    """一類評估的比較身分：評估器、聲源模型指紋、兩份設定，以及實際評估支撐。"""

    category: QualityCategory
    evaluator_version: str
    source_model_fingerprint: str | None
    settings_fingerprint: str
    cost_settings_fingerprint: str
    assessed_support: str = ""


class ComponentLine(_FrozenModel):
    """類內一個分項：原始值與代價並列，角色說明它進不進類代價。"""

    name: str
    role: ComponentRole
    raw_value: float | None = Field(
        json_schema_extra=facts("原始量", "見 raw_unit", RAW_REFERENCE)
    )
    raw_unit: Unit
    cost: float = Field(json_schema_extra=facts("代價", "1", COST_REFERENCE))
    weight: float | None = Field(json_schema_extra=facts("權重", "1", WEIGHT_REFERENCE))


class CategoryLine(_FrozenModel):
    """一個候選一類的類代價、權重、分項與完整評估（原始量、標記、原因都在 evaluation 裡）。"""

    identity: ComparisonIdentity
    category_cost: float = Field(json_schema_extra=facts("代價", "1", COST_REFERENCE))
    category_weight: float = Field(
        json_schema_extra=facts("權重", "1", WEIGHT_REFERENCE)
    )
    weighted_cost: float = Field(json_schema_extra=facts("代價", "1", COST_REFERENCE))
    components: tuple[ComponentLine, ...]
    component_directions: dict[str, CostDirection]
    unassessed_bands: tuple[UnassessedBand, ...]
    evaluation: CategoryEvaluation


class UncoveredCategory(_FrozenModel):
    """選評類沒蓋到：沒送來，或評估器回不可估（原因原樣帶著）。"""

    category: QualityCategory
    reason_codes: tuple[ReasonCode, ...]


class MissingCategory(_FrozenModel):
    """讓候選未評估的一類：缺的類、排名層的原因與評估器給的原因。"""

    category: QualityCategory
    reason: NotEvaluatedReason
    evaluator_reason_codes: tuple[ReasonCode, ...]


class EligibilityRuleUse(_FrozenModel):
    """一條資料資格規則讀進來的值，以及它對每一類套上沒有。"""

    key: str
    value: QualificationValue
    per_category: dict[QualityCategory, EligibilityApplication]


class CalibrationSource(_FrozenModel):
    """判定整份結果校準狀態時看過的一條：登記簿條目，或評估器自報的基線設定標記。"""

    kind: Literal["registry", "evaluator_flag"]
    key: str
    status: EntryStatus


class RankingHeader(_FrozenModel):
    """第一塊：用途、各份指紋、蓋到哪幾類、日期、引擎版本與校準狀態。"""

    purpose: str
    registry_fingerprint: str
    main_table_identity: tuple[ComparisonIdentity, ...]
    receiver_set_fingerprint: str
    channel_group_fingerprint: str
    mandatory_covered: tuple[QualityCategory, ...]
    optional_covered: tuple[QualityCategory, ...]
    optional_uncovered: tuple[QualityCategory, ...]
    run_date: date
    engine_version: str
    calibration: EntryStatus
    calibration_note: str
    calibration_sources: tuple[CalibrationSource, ...]
    eligibility_rules: tuple[EligibilityRuleUse, ...]
    external_floors_declared_by: str | None
    overall_acceptance: ExternalAcceptance
    acceptance_note: str | None


class RankableRow(_FrozenModel):
    """第二塊的一列：名次、J、逐類代價、警戒、標記；外部驗收、複核、推薦三個狀態分開記（#449）。"""

    status: Literal[CandidateStatus.RANKABLE]
    rank: int = Field(ge=1, json_schema_extra=facts("名次", "1", NO_BASIS_COUNT))
    candidate_id: str
    scene_fingerprint: str
    total_cost: float = Field(json_schema_extra=facts("代價", "1", COST_REFERENCE))
    categories: tuple[CategoryLine, ...]
    review_alerts: tuple[ReviewAlert, ...]
    uncovered: tuple[UncoveredCategory, ...]
    flags: tuple[Flag, ...]
    external_acceptance: ExternalAcceptance
    review_status: rec.ReviewStatus
    recommendation_status: rec.RecommendationStatus
    not_final_reasons: tuple[rec.NotFinalReason, ...] = Field(min_length=1)


class EliminatedRow(_FrozenModel):
    """第三塊的一列：全部違反的底線原因，同時缺的類也一併列出；沒有名次也沒有 J。"""

    status: Literal[CandidateStatus.ELIMINATED]
    candidate_id: str
    scene_fingerprint: str
    reasons: tuple[EliminationReason, ...] = Field(min_length=1)
    missing: tuple[MissingCategory, ...]
    evaluations: tuple[CategoryEvaluation, ...]
    external_acceptance: ExternalAcceptance
    # 被外部底線淘汰的候選同時踩了峰谷警戒也要查得出來，不因為出局就丟掉（#445 找碴）。
    review_alerts: tuple[ReviewAlert, ...]


class NotEvaluatedRow(_FrozenModel):
    """第四塊的一列：缺的類與原因代碼；評估器送來的東西原樣保留。"""

    status: Literal[CandidateStatus.NOT_EVALUATED]
    candidate_id: str
    scene_fingerprint: str
    missing: tuple[MissingCategory, ...] = Field(min_length=1)
    evaluations: tuple[CategoryEvaluation, ...]
    external_acceptance: ExternalAcceptance


class NotComparableRow(_FrozenModel):
    """第五塊的一列：只列代號與指紋，不列分數——這是比較規則，不是品質判決。"""

    status: Literal[CandidateStatus.NOT_COMPARABLE]
    candidate_id: str
    identity: tuple[ComparisonIdentity, ...]


class CandidateFilterCounts(_FrozenModel):
    """候選產生層的過濾計數（不合法的數量與原因）；第一刀沒有規則，結構先留著。"""

    illegal_count: int = Field(
        ge=0, json_schema_extra=facts("計數", "1", NO_BASIS_COUNT)
    )
    illegal_reasons: dict[str, int]


class NotComparableBlock(_FrozenModel):
    """第五塊：不可同表的候選，加上候選產生層的過濾計數。"""

    rows: tuple[NotComparableRow, ...]
    candidate_filter: CandidateFilterCounts


class RankingResult(_FrozenModel):
    """五塊排名資料；格式是機器讀的資料，前台另題。"""

    schema_version: Literal["aosr.scoring.ranking.v1"]  # 等於 RANKING_SCHEMA_VERSION
    header: RankingHeader
    rankable: tuple[RankableRow, ...]
    eliminated: tuple[EliminatedRow, ...]
    not_evaluated: tuple[NotEvaluatedRow, ...]
    not_comparable: NotComparableBlock

    def status_of(self, candidate_id: str) -> CandidateStatus:
        """回傳一個候選落在哪一塊；不在任何一塊就報錯，不猜。"""
        zones: tuple[tuple[CandidateStatus, tuple[str, ...]], ...] = (
            (
                CandidateStatus.RANKABLE,
                tuple(row.candidate_id for row in self.rankable),
            ),
            (
                CandidateStatus.ELIMINATED,
                tuple(row.candidate_id for row in self.eliminated),
            ),
            (
                CandidateStatus.NOT_EVALUATED,
                tuple(row.candidate_id for row in self.not_evaluated),
            ),
            (
                CandidateStatus.NOT_COMPARABLE,
                tuple(row.candidate_id for row in self.not_comparable.rows),
            ),
        )
        for status, ids in zones:
            if candidate_id in ids:
                return status
        raise KeyError(f"排名結果裡沒有這個候選：{candidate_id}")
