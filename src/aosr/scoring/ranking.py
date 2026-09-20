"""排名層最小版：把凍結評估契約換成類代價、總代價 J、五種候選狀態與五塊排名資料（票 #357）。

規則出處是票 #354 的四格拍板（第 1、1.5、2、3 格）與 #345 第 5 格；門檻、目標、較差參考與
權重一律從品質登記簿（``src/aosr/config/data/quality_targets.toml``）讀，這一支不寫任何數字。

**第二層（代價）。** 三型代價公式只住 :func:`_shape_cost` 一處。音色在直線目標下：傾斜
（``in_range_best``）與殘差均方根（``less_is_better``）是主要分項，按類內權重加權和成類代價；
峰谷（``beyond_threshold_only``）是底線保護，只擋不加；對目標偏差（``less_is_better``）只當
對照印出來——直線目標時它等於傾斜加起伏同一份資訊，加進去就算兩次。

**第三層（狀態）。** 每個候選對全部底線都檢查、全部列出原因，不碰到第一條就停：
資料資格、底線保護超標、比較相容、外部底線。狀態的先後是「淘汰」→「未評估」→
「不可同表比較」→「可排名」：淘汰是真的踩線，就算同時缺類也照樣列在淘汰區、缺的類一併列出；
「不可同表比較」只在過了底線、資料也齊的候選之間分，所以它不是品質判決。
「不合法」這一格只留列舉與過濾計數的欄位，候選產生層（#354 第 1.5 格）另票，這一刀沒有規則會產生它。

**資料資格那三條**（不可估帶數上限、關鍵帶、最低覆蓋）是按頻帶評的類才有的證據；契約裡
今天沒有任何一類的 payload 帶頻帶清單，所以照樣從登記簿讀進來、在表頭對每一類記成
``not_applicable``，不是整體「未檢查」，也不另收契約外的輸入。等殘響（#348）的 payload 帶了
頻帶清單，把那一類放進 :data:`_BAND_EVIDENCE_CATEGORIES` 並在 :func:`_floor_violations` 接上三條判準即可。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from enum import StrEnum
from typing import Final, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from aosr.config.quality_targets import (
    EntryStatus,
    QualificationEntry,
    QualificationValue,
    QualityPurpose,
    QualityTargets,
)
from aosr.physics.report_facts import NO_BASIS_COUNT, facts
from aosr.scoring.contract import (
    CandidateEvaluation,
    CategoryCost,
    CategoryEvaluation,
    EvaluationState,
    Feature,
    Flag,
    InputProvenance,
    ListeningAreaStabilityPayload,
    QualityCategory,
    ReasonCode,
    TimbrePayload,
)
from aosr.scoring.cost_shapes import (
    ComponentRole,
    scalar as _scalar,
    shape_cost as _shape_cost,
    target as _target,
    weight_table as _weight_table,
)
from aosr.scoring.listening_area_cost import (
    _LISTENING_AREA_ROLES,
    _LISTENING_AREA_TARGET_KEYS,
    _LISTENING_AREA_WEIGHTS_KEY,
    _listening_area_principal_weights,
    cost_listening_area_evaluation as cost_listening_area_evaluation,
)


RANKING_SCHEMA_VERSION: Final[str] = "aosr.scoring.ranking.v1"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

# 代價與權重那幾格的參考基準；每個數字旁另有評估器版本與設定指紋（見 ComparisonIdentity）。
COST_REFERENCE: Final[str] = (
    "無因次：超出或偏離的量除以登記簿的較差參考（worse_reference）；"
    "各項的 1 不代表同等的品質損失，不是品質判決的絕對尺度"
)
WEIGHT_REFERENCE: Final[str] = "沒有基準（只是登記簿的權重，第一版等權是試驗基線、不是中立）"
RAW_REFERENCE: Final[str] = "評估器輸出的原始量，單位見同一列的 raw_unit；排名層只轉不改"

_MANDATORY_KEY: Final[str] = "ranking.mandatory_categories"
_OPTIONAL_KEY: Final[str] = "ranking.optional_categories"
_CATEGORY_WEIGHTS_KEY: Final[str] = "ranking.category_weights"
_ELIGIBILITY_KEYS: Final[tuple[str, ...]] = (
    "ranking.eligibility.max_unavailable_bands",
    "ranking.eligibility.critical_bands",
    "ranking.eligibility.min_valid_bands",
)
# 哪幾類的 payload 帶得出頻帶證據；今天一類都沒有（見檔頭）。
_BAND_EVIDENCE_CATEGORIES: Final[frozenset[QualityCategory]] = frozenset()

_TIMBRE_WEIGHTS_KEY: Final[str] = "timbre_balance.within_category_weights"
_TILT_KEY: Final[str] = "timbre_balance.target_tilt_db_per_octave"
_RESIDUAL_KEY: Final[str] = "timbre_balance.residual_rms_db"
_DEVIATION_KEY: Final[str] = "timbre_balance.target_deviation_rms_db"
_PEAK_KEY: Final[str] = "timbre_balance.peak_depth_db"
_DIP_KEY: Final[str] = "timbre_balance.dip_depth_db"

BASELINE_NOTE: Final[str] = "未校準，不是品質判決"
CALIBRATED_NOTE: Final[str] = "參與排名的登記簿條目與評估設定全部已校準"
NOT_CHECKED_NOTE: Final[str] = "外部底線未宣告或未檢查：整體驗收未檢查，不是整體合格"


class CandidateStatus(StrEnum):
    """候選的五種狀態；``illegal`` 只留給候選產生層，這一刀沒有規則會產生它。"""

    RANKABLE = "rankable"
    ELIMINATED = "eliminated"
    NOT_EVALUATED = "not_evaluated"
    NOT_COMPARABLE = "not_comparable"
    ILLEGAL = "illegal"


class EliminationReason(StrEnum):
    """踩到硬底線的受控原因代碼；一個候選踩幾條就列幾條。"""

    TIMBRE_PEAK_BEYOND_LIMIT = "timbre_peak_beyond_limit"
    TIMBRE_DIP_BEYOND_LIMIT = "timbre_dip_beyond_limit"
    LISTENING_AREA_TILT_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_tilt_primary_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_TILT_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_tilt_surrounding_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_RIPPLE_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_ripple_primary_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_RIPPLE_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_ripple_surrounding_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_LEVEL_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_level_primary_to_surrounding_worst_beyond_limit"
    )
    LISTENING_AREA_LEVEL_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT = (
        "listening_area_level_surrounding_to_surrounding_worst_beyond_limit"
    )
    EXTERNAL_FLOOR_FAILED = "external_floor_failed"


class NotEvaluatedReason(StrEnum):
    """該算的沒算到的受控原因代碼；不進排名、也不算淘汰。"""

    MANDATORY_CATEGORY_MISSING = "mandatory_category_missing"
    MANDATORY_CATEGORY_UNAVAILABLE = "mandatory_category_unavailable"
    COST_NOT_COMPUTED = "cost_not_computed"


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
    """一類評估的比較身分：哪支評估器、哪份評估設定、哪份代價設定算的。"""

    category: QualityCategory
    evaluator_version: str
    settings_fingerprint: str
    cost_settings_fingerprint: str


class ComponentLine(_FrozenModel):
    """類內一個分項：原始值與代價並列，角色說明它進不進類代價。"""

    name: str
    role: ComponentRole
    raw_value: float | None = Field(json_schema_extra=facts("原始量", "見 raw_unit", RAW_REFERENCE))
    raw_unit: str
    cost: float = Field(json_schema_extra=facts("代價", "1", COST_REFERENCE))
    weight: float | None = Field(json_schema_extra=facts("權重", "1", WEIGHT_REFERENCE))


class CategoryLine(_FrozenModel):
    """一個候選一類的類代價、權重、分項與完整評估（原始量、標記、原因都在 evaluation 裡）。"""

    identity: ComparisonIdentity
    category_cost: float = Field(json_schema_extra=facts("代價", "1", COST_REFERENCE))
    category_weight: float = Field(json_schema_extra=facts("權重", "1", WEIGHT_REFERENCE))
    weighted_cost: float = Field(json_schema_extra=facts("代價", "1", COST_REFERENCE))
    components: tuple[ComponentLine, ...]
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
    """第二塊的一列：名次、J、逐類代價與分項原始值、標記、外部驗收狀態。"""

    status: Literal[CandidateStatus.RANKABLE]
    rank: int = Field(ge=1, json_schema_extra=facts("名次", "1", NO_BASIS_COUNT))
    candidate_id: str
    provenance: InputProvenance
    total_cost: float = Field(json_schema_extra=facts("代價", "1", COST_REFERENCE))
    categories: tuple[CategoryLine, ...]
    uncovered: tuple[UncoveredCategory, ...]
    flags: tuple[Flag, ...]
    external_acceptance: ExternalAcceptance


class EliminatedRow(_FrozenModel):
    """第三塊的一列：全部違反的底線原因，同時缺的類也一併列出；沒有名次也沒有 J。"""

    status: Literal[CandidateStatus.ELIMINATED]
    candidate_id: str
    provenance: InputProvenance
    reasons: tuple[EliminationReason, ...] = Field(min_length=1)
    missing: tuple[MissingCategory, ...]
    evaluations: tuple[CategoryEvaluation, ...]
    external_acceptance: ExternalAcceptance


class NotEvaluatedRow(_FrozenModel):
    """第四塊的一列：缺的類與原因代碼；評估器送來的東西原樣保留。"""

    status: Literal[CandidateStatus.NOT_EVALUATED]
    candidate_id: str
    provenance: InputProvenance
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

    illegal_count: int = Field(ge=0, json_schema_extra=facts("計數", "1", NO_BASIS_COUNT))
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
            (CandidateStatus.RANKABLE, tuple(row.candidate_id for row in self.rankable)),
            (CandidateStatus.ELIMINATED, tuple(row.candidate_id for row in self.eliminated)),
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


# ── 登記簿讀取 ──────────────────────────────────────────────────────────────


def _qualification(purpose: QualityPurpose, key: str) -> QualificationEntry:
    """讀一條資格規則。"""
    entry = purpose.entry(key)
    if not isinstance(entry, QualificationEntry):
        raise TypeError(f"{key} 不是資格規則")
    return entry


def _category_list(purpose: QualityPurpose, key: str) -> frozenset[QualityCategory]:
    """讀必評或選評清單；名稱必須是受控類名。"""
    value = _qualification(purpose, key).value
    if not isinstance(value, tuple) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{key} 必須是類名清單")
    return frozenset(QualityCategory(str(item)) for item in value)


# ── 第二層：代價 ────────────────────────────────────────────────────────────


def _counted_depths(features: tuple[Feature, ...], kind: str) -> tuple[float, ...]:
    """峰或谷之中進底線保護的深度；太窄的保留在清單上、但不進代價。"""
    return tuple(
        feature.depth_db
        for feature in features
        if feature.kind == kind and Flag.FEATURE_TOO_NARROW not in feature.flags
    )


def _protection_costs(payload: TimbrePayload, purpose: QualityPurpose) -> dict[str, float]:
    """峰、谷各自超出界線的代價；大於零就是踩到底線保護。"""
    costs: dict[str, float] = {}
    for kind, key in (("peak", _PEAK_KEY), ("dip", _DIP_KEY)):
        target = _target(purpose, key)
        if target.cost_shape != "beyond_threshold_only":
            raise ValueError(f"{key} 是底線保護，cost_shape 必須是 beyond_threshold_only")
        costs[kind] = _shape_cost(target, _counted_depths(payload.features, kind))
    return costs


def _timbre_components(payload: TimbrePayload, purpose: QualityPurpose) -> dict[str, float]:
    """音色四樣輸出各自的代價（#345 第 5 格的配法）。"""
    protection = _protection_costs(payload, purpose)
    return {
        "tilt": _shape_cost(_target(purpose, _TILT_KEY), (payload.tilt_db_per_octave,)),
        "residual_rms": _shape_cost(_target(purpose, _RESIDUAL_KEY), (payload.residual_rms_db,)),
        "target_deviation": _shape_cost(
            _target(purpose, _DEVIATION_KEY), (payload.target_deviation_rms_db,)
        ),
        "peaks_dips": protection["peak"] + protection["dip"],
    }


# 音色分項的角色：主要分項進類代價、對照只印、保護只擋（#345 第 5 格，直線目標）。
_TIMBRE_ROLES: Final[dict[str, ComponentRole]] = {
    "tilt": "principal",
    "residual_rms": "principal",
    "target_deviation": "reference",
    "peaks_dips": "protection",
}
_TIMBRE_TARGET_KEYS: Final[dict[str, tuple[str, ...]]] = {
    "tilt": (_TILT_KEY,),
    "residual_rms": (_RESIDUAL_KEY,),
    "target_deviation": (_DEVIATION_KEY,),
    "peaks_dips": (_PEAK_KEY, _DIP_KEY),
}


def _principal_weights(purpose: QualityPurpose) -> dict[str, float]:
    """類內權重表的名稱必須剛好是主要分項；多一項或少一項都報錯，不靜靜忽略。"""
    weights = {item.name: item.value for item in _weight_table(purpose, _TIMBRE_WEIGHTS_KEY).item}
    principal = {name for name, role in _TIMBRE_ROLES.items() if role == "principal"}
    if set(weights) != principal:
        raise ValueError(f"{_TIMBRE_WEIGHTS_KEY} 的名稱必須剛好是 {sorted(principal)}")
    return weights


def cost_timbre_evaluation(
    evaluation: CategoryEvaluation,
    purpose: QualityPurpose,
    cost_settings_fingerprint: str,
) -> CategoryEvaluation:
    """把一條已量的音色輸出升成新的 ``costed`` 物件；輸入不動。

    類代價只加主要分項（傾斜、殘差均方根）；``target_deviation`` 是對照、``peaks_dips`` 是
    底線保護，兩者照算、放進 ``components`` 帶出去，但不進 ``value``。直線目標之外的目標曲線
    要另拍配法，所以評估器用的目標傾斜必須等於登記簿那一條，不等就報錯。
    """
    if evaluation.state is not EvaluationState.MEASURED:
        raise ValueError("音色代價只接 measured 評估")
    payload = evaluation.payload
    if not isinstance(payload, TimbrePayload):
        raise TypeError("音色代價必須收到 TimbrePayload")
    if payload.target_tilt_db_per_octave != _scalar(_target(purpose, _TILT_KEY)):
        raise ValueError("評估器用的目標傾斜與登記簿不同，代價沒有唯一答案")
    components = _timbre_components(payload, purpose)
    weights = _principal_weights(purpose)
    value = sum(components[name] * weight for name, weight in weights.items())
    category_cost = CategoryCost(
        value=value,
        components=components,
        cost_settings_fingerprint=cost_settings_fingerprint,
    )
    # 走一次完整驗證（不用 model_copy）：costed 的不變條件要在建構時就被契約檢查一次。
    document = evaluation.model_dump(mode="python")
    document.update(state=EvaluationState.COSTED, category_cost=category_cost)
    return CategoryEvaluation.model_validate(document)


Coster = Callable[[CategoryEvaluation, QualityPurpose, str], CategoryEvaluation]
# 排名層會算代價的類；不在這裡的類送 measured 來就是「可估但沒類代價」。
_COSTERS: Final[dict[QualityCategory, Coster]] = {
    QualityCategory.TIMBRE_BALANCE: cost_timbre_evaluation,
    QualityCategory.LISTENING_AREA_STABILITY: cost_listening_area_evaluation,
}


def _timbre_raw(payload: TimbrePayload, name: str) -> float | None:
    """分項的原始值；峰谷那一項是進保護的特徵裡最大的 |深度|，沒有就是 None。"""
    if name == "tilt":
        return payload.tilt_db_per_octave
    if name == "residual_rms":
        return payload.residual_rms_db
    if name == "target_deviation":
        return payload.target_deviation_rms_db
    depths = _counted_depths(payload.features, "peak") + _counted_depths(payload.features, "dip")
    return max((abs(depth) for depth in depths), default=None)


def _shared_unit(purpose: QualityPurpose, keys: tuple[str, ...]) -> str:
    """一個分項對到好幾條登記簿條目（峰與谷）時，單位必須一樣；不一樣就報錯，不靜靜取第一條。"""
    units = {_target(purpose, key).unit for key in keys}
    if len(units) != 1:
        raise ValueError(f"{list(keys)} 的單位不一致：{sorted(units)}")
    return units.pop()


def _component_lines(evaluation: CategoryEvaluation, purpose: QualityPurpose) -> tuple[ComponentLine, ...]:
    """類內分項逐條列出；音色照角色表，別類的已算代價原樣標「reported」。"""
    cost = evaluation.category_cost
    if cost is None:
        raise ValueError("只有 costed 評估有分項")
    payload = evaluation.payload
    if isinstance(payload, ListeningAreaStabilityPayload):
        weights = _listening_area_principal_weights(purpose)
        lines: list[ComponentLine] = []
        for name, component_cost in cost.components.items():
            target_name, separator, _ = name.partition(".")
            role = _LISTENING_AREA_ROLES[target_name]
            if separator and role == "principal":
                role = "reported"
            lines.append(
                ComponentLine(
                    name=name,
                    role=role,
                    raw_value=None,
                    raw_unit=_target(
                        purpose, _LISTENING_AREA_TARGET_KEYS[target_name]
                    ).unit,
                    cost=component_cost,
                    weight=weights.get(name),
                )
            )
        return tuple(lines)
    if not isinstance(payload, TimbrePayload):
        return tuple(
            ComponentLine(name=name, role="reported", raw_value=None, raw_unit="1", cost=value, weight=None)
            for name, value in sorted(cost.components.items())
        )
    weights = _principal_weights(purpose)
    return tuple(
        ComponentLine(
            name=name,
            role=role,
            raw_value=_timbre_raw(payload, name),
            raw_unit=_shared_unit(purpose, _TIMBRE_TARGET_KEYS[name]),
            cost=cost.components[name],
            weight=weights.get(name),
        )
        for name, role in _TIMBRE_ROLES.items()
    )


# ── 第三層：逐候選判定 ──────────────────────────────────────────────────────


class _Rules(NamedTuple):
    """這一跑從登記簿讀進來的全部規則。"""

    purpose: QualityPurpose
    registry_fingerprint: str
    mandatory: frozenset[QualityCategory]
    optional: frozenset[QualityCategory]
    category_weights: dict[str, float]


class _Assessment(NamedTuple):
    """一個候選走完全部底線之後的判定；排成哪一塊由呼叫端決定。"""

    candidate: CandidateEvaluation
    evaluations: tuple[CategoryEvaluation, ...]
    lines: tuple[CategoryLine, ...]
    uncovered: tuple[UncoveredCategory, ...]
    missing: tuple[MissingCategory, ...]
    eliminations: tuple[EliminationReason, ...]
    external: ExternalAcceptance


def _read_rules(registry: QualityTargets, purpose_name: str) -> _Rules:
    """讀必評／選評清單與類權重；兩張清單不准重疊。"""
    purpose = registry.purpose(purpose_name)
    mandatory = _category_list(purpose, _MANDATORY_KEY)
    optional = _category_list(purpose, _OPTIONAL_KEY)
    if mandatory & optional:
        raise ValueError("同一類不能同時是必評與選評")
    weights = {item.name: item.value for item in _weight_table(purpose, _CATEGORY_WEIGHTS_KEY).item}
    return _Rules(purpose, registry.fingerprint, mandatory, optional, weights)


def _settle(evaluation: CategoryEvaluation, rules: _Rules) -> CategoryEvaluation:
    """已量而且排名層會算的類就換成 costed；其餘原樣。"""
    coster = _COSTERS.get(evaluation.category)
    if evaluation.state is EvaluationState.MEASURED and coster is not None:
        return coster(evaluation, rules.purpose, rules.registry_fingerprint)
    return evaluation


def _category_line(evaluation: CategoryEvaluation, rules: _Rules) -> CategoryLine:
    """一條 costed 評估換成排名表的一類；類權重缺了就報錯，不補 1。"""
    cost = evaluation.category_cost
    if cost is None:
        raise ValueError("只有 costed 評估能進排名表")
    weight = rules.category_weights[evaluation.category.value]
    return CategoryLine(
        identity=ComparisonIdentity(
            category=evaluation.category,
            evaluator_version=evaluation.evaluator_version,
            settings_fingerprint=evaluation.settings_fingerprint,
            cost_settings_fingerprint=cost.cost_settings_fingerprint,
        ),
        category_cost=cost.value,
        category_weight=weight,
        weighted_cost=cost.value * weight,
        components=_component_lines(evaluation, rules.purpose),
        evaluation=evaluation,
    )


def _classify(
    evaluations: tuple[CategoryEvaluation, ...], rules: _Rules
) -> tuple[list[CategoryLine], list[UncoveredCategory], list[MissingCategory]]:
    """逐類分成：進表的、選評沒蓋到的、讓候選未評估的。缺的類不補任何數。"""
    lines: list[CategoryLine] = []
    uncovered: list[UncoveredCategory] = []
    missing: list[MissingCategory] = []
    present = {evaluation.category for evaluation in evaluations}
    for evaluation in evaluations:
        category = evaluation.category
        if evaluation.state is EvaluationState.COSTED:
            lines.append(_category_line(evaluation, rules))
        elif evaluation.state is EvaluationState.MEASURED:
            missing.append(MissingCategory(
                category=category,
                reason=NotEvaluatedReason.COST_NOT_COMPUTED,
                evaluator_reason_codes=evaluation.reason_codes,
            ))
        elif category in rules.mandatory:
            missing.append(MissingCategory(
                category=category,
                reason=NotEvaluatedReason.MANDATORY_CATEGORY_UNAVAILABLE,
                evaluator_reason_codes=evaluation.reason_codes,
            ))
        else:
            uncovered.append(UncoveredCategory(category=category, reason_codes=evaluation.reason_codes))
    for category in sorted(rules.mandatory - present):
        missing.append(MissingCategory(
            category=category,
            reason=NotEvaluatedReason.MANDATORY_CATEGORY_MISSING,
            evaluator_reason_codes=(),
        ))
    for category in sorted(rules.optional - present):
        uncovered.append(UncoveredCategory(category=category, reason_codes=()))
    return lines, uncovered, missing


def _floor_violations(
    evaluations: tuple[CategoryEvaluation, ...], rules: _Rules, external: ExternalAcceptance
) -> tuple[EliminationReason, ...]:
    """全部底線都檢查、全部列出；比較相容不在這裡（它不是淘汰，是分表）。"""
    reasons: list[EliminationReason] = []
    for evaluation in evaluations:
        payload = evaluation.payload
        if evaluation.state is not EvaluationState.COSTED:
            continue
        if isinstance(payload, TimbrePayload):
            protection = _protection_costs(payload, rules.purpose)
            if protection["peak"] > 0.0:
                reasons.append(EliminationReason.TIMBRE_PEAK_BEYOND_LIMIT)
            if protection["dip"] > 0.0:
                reasons.append(EliminationReason.TIMBRE_DIP_BEYOND_LIMIT)
        elif isinstance(payload, ListeningAreaStabilityPayload):
            cost = evaluation.category_cost
            if cost is None:
                raise ValueError("costed 聆聽區評估缺 category_cost")
            protections = (
                (
                    "tilt_worst_deviation.primary_to_surrounding",
                    EliminationReason.LISTENING_AREA_TILT_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT,
                ),
                (
                    "tilt_worst_deviation.surrounding_to_surrounding",
                    EliminationReason.LISTENING_AREA_TILT_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT,
                ),
                (
                    "ripple_rms_worst_deviation.primary_to_surrounding",
                    EliminationReason.LISTENING_AREA_RIPPLE_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT,
                ),
                (
                    "ripple_rms_worst_deviation.surrounding_to_surrounding",
                    EliminationReason.LISTENING_AREA_RIPPLE_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT,
                ),
                (
                    "overall_level_worst_deviation.primary_to_surrounding",
                    EliminationReason.LISTENING_AREA_LEVEL_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT,
                ),
                (
                    "overall_level_worst_deviation.surrounding_to_surrounding",
                    EliminationReason.LISTENING_AREA_LEVEL_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT,
                ),
            )
            reasons.extend(
                reason
                for name, reason in protections
                if cost.components.get(name, 0.0) > 0.0
            )
    if external is ExternalAcceptance.FAILED:
        reasons.append(EliminationReason.EXTERNAL_FLOOR_FAILED)
    return tuple(reasons)


def _assess(candidate: CandidateEvaluation, rules: _Rules, external: ExternalAcceptance) -> _Assessment:
    """一個候選走完第二層與全部底線；登記簿沒指定必評或選評的類報錯，不靜靜丟掉。"""
    unknown = {item.category for item in candidate.evaluations} - rules.mandatory - rules.optional
    if unknown:
        names = sorted(category.value for category in unknown)
        raise ValueError(f"候選 {candidate.candidate_id} 帶了登記簿沒指定必評或選評的類：{names}")
    evaluations = tuple(_settle(item, rules) for item in candidate.evaluations)
    lines, uncovered, missing = _classify(evaluations, rules)
    return _Assessment(
        candidate=candidate,
        evaluations=evaluations,
        lines=tuple(lines),
        uncovered=tuple(uncovered),
        missing=tuple(missing),
        eliminations=_floor_violations(evaluations, rules, external),
        external=external,
    )


def _external_verdict(floors: ExternalFloors | None, candidate_id: str) -> ExternalAcceptance:
    """外部底線沒宣告、或宣告了沒列到這個候選，都是「未檢查」。"""
    if floors is None:
        return ExternalAcceptance.NOT_CHECKED
    return ExternalAcceptance(floors.verdicts.get(candidate_id, ExternalAcceptance.NOT_CHECKED))


# ── 第三層：分表、排序與表頭 ────────────────────────────────────────────────


def _identity(assessment: _Assessment) -> tuple[ComparisonIdentity, ...]:
    """同表條件：已評估類集合、每類的評估器版本與兩份指紋全部相同。"""
    return tuple(sorted((line.identity for line in assessment.lines), key=lambda item: item.category.value))


def _split_tables(
    contenders: Sequence[_Assessment],
) -> tuple[tuple[ComparisonIdentity, ...], list[_Assessment], list[_Assessment]]:
    """人數最多那一組當主表；同人數時取身分字典序最小的一組，只為可重現、不是品質判決。"""
    groups: dict[tuple[ComparisonIdentity, ...], list[_Assessment]] = {}
    for assessment in contenders:
        groups.setdefault(_identity(assessment), []).append(assessment)
    if not groups:
        return (), [], []
    main_key = min(
        groups,
        key=lambda key: (-len(groups[key]), [item.model_dump_json() for item in key]),
    )
    others = [item for key, members in groups.items() if key != main_key for item in members]
    return main_key, groups[main_key], others


def _total_cost(assessment: _Assessment) -> float:
    """J＝Σ 類代價 × 類權重；只加真的算到的類，缺類不補值。"""
    return sum(line.weighted_cost for line in assessment.lines)


def _row_flags(assessment: _Assessment) -> tuple[Flag, ...]:
    """類層標記與峰谷特徵標記的聯集，照列舉順序；原件仍在每一類的 evaluation 裡。"""
    found: set[Flag] = set()
    for evaluation in assessment.evaluations:
        found.update(evaluation.flags)
        if isinstance(evaluation.payload, TimbrePayload):
            for feature in evaluation.payload.features:
                found.update(feature.flags)
    return tuple(flag for flag in Flag if flag in found)


def _registry_sources(categories: set[QualityCategory], rules: _Rules) -> list[CalibrationSource]:
    """真的被排名層讀來判狀態或算代價的登記簿條目；沒套上的資格規則不算參與。"""
    purpose = rules.purpose
    sources = [
        CalibrationSource(kind="registry", key=key, status=_qualification(purpose, key).status)
        for key in (_MANDATORY_KEY, _OPTIONAL_KEY)
    ]
    weight_items = {item.name: item for item in _weight_table(purpose, _CATEGORY_WEIGHTS_KEY).item}
    for category in sorted(categories):
        name = f"{_CATEGORY_WEIGHTS_KEY}.{category.value}"
        sources.append(CalibrationSource(kind="registry", key=name, status=weight_items[category.value].status))
        if category is QualityCategory.TIMBRE_BALANCE:
            for keys in _TIMBRE_TARGET_KEYS.values():
                sources.extend(
                    CalibrationSource(
                        kind="registry", key=key, status=_target(purpose, key).status
                    )
                    for key in keys
                )
            sources.extend(
                CalibrationSource(
                    kind="registry",
                    key=f"{_TIMBRE_WEIGHTS_KEY}.{item.name}",
                    status=item.status,
                )
                for item in _weight_table(purpose, _TIMBRE_WEIGHTS_KEY).item
            )
        elif category is QualityCategory.LISTENING_AREA_STABILITY:
            sources.extend(
                CalibrationSource(
                    kind="registry", key=key, status=_target(purpose, key).status
                )
                for key in _LISTENING_AREA_TARGET_KEYS.values()
            )
            sources.extend(
                CalibrationSource(
                    kind="registry",
                    key=f"{_LISTENING_AREA_WEIGHTS_KEY}.{item.name}",
                    status=item.status,
                )
                for item in _weight_table(purpose, _LISTENING_AREA_WEIGHTS_KEY).item
            )
    return sources


def _calibration_sources(assessments: Sequence[_Assessment], rules: _Rules) -> tuple[CalibrationSource, ...]:
    """參與排名的登記簿條目，加上這一批裡評估器自報「用了基線設定」的每一條評估。

    旗標不只看進得了表的類：掛在不可估、或沒算到代價的那一條上一樣算（找碴席實測那種情形
    原本會回 calibrated）。寧可多標未校準，不讓同一批裡有人明說用了基線、表頭卻寫已校準。
    """
    categories = {line.identity.category for item in assessments for line in item.lines}
    sources = _registry_sources(categories, rules)
    for assessment in assessments:
        for evaluation in assessment.evaluations:
            if Flag.BASELINE_SETTINGS in evaluation.flags:
                key = f"{assessment.candidate.candidate_id}/{evaluation.category.value}"
                sources.append(CalibrationSource(kind="evaluator_flag", key=key, status="baseline"))
    return tuple(sources)


def _eligibility_rules(assessments: Sequence[_Assessment], rules: _Rules) -> tuple[EligibilityRuleUse, ...]:
    """三條資料資格規則照樣讀進來，逐類記套上或未套用（沒有頻帶證據的類是未套用）。"""
    seen = sorted({item.category for assessment in assessments for item in assessment.evaluations})
    per_category = {
        category: EligibilityApplication.APPLIED
        if category in _BAND_EVIDENCE_CATEGORIES
        else EligibilityApplication.NOT_APPLICABLE
        for category in seen
    }
    return tuple(
        EligibilityRuleUse(
            key=key, value=_qualification(rules.purpose, key).value, per_category=per_category
        )
        for key in _ELIGIBILITY_KEYS
    )


def _overall_acceptance(floors: ExternalFloors | None, ranked: Sequence[_Assessment]) -> ExternalAcceptance:
    """整體驗收：沒宣告、榜上沒有候選、或榜上有任何一個未檢查，就是「未檢查」。"""
    if floors is None or not ranked or any(item.external is ExternalAcceptance.NOT_CHECKED for item in ranked):
        return ExternalAcceptance.NOT_CHECKED
    return ExternalAcceptance.PASSED


def _header(
    context: RankingContext,
    rules: _Rules,
    main_key: tuple[ComparisonIdentity, ...],
    assessments: Sequence[_Assessment],
    ranked: Sequence[_Assessment],
    floors: ExternalFloors | None,
) -> RankingHeader:
    """第一塊；任何一條參與的來源是 baseline，整份結果就是 baseline。"""
    covered = {identity.category for identity in main_key}
    sources = _calibration_sources(assessments, rules)
    calibration: EntryStatus = (
        "baseline" if any(source.status == "baseline" for source in sources) else "calibrated"
    )
    acceptance = _overall_acceptance(floors, ranked)
    return RankingHeader(
        purpose=context.purpose,
        registry_fingerprint=rules.registry_fingerprint,
        main_table_identity=main_key,
        receiver_set_fingerprint=context.receiver_set_fingerprint,
        channel_group_fingerprint=context.channel_group_fingerprint,
        mandatory_covered=tuple(sorted(rules.mandatory & covered)),
        optional_covered=tuple(sorted(rules.optional & covered)),
        optional_uncovered=tuple(sorted(rules.optional - covered)),
        run_date=context.run_date,
        engine_version=context.engine_version,
        calibration=calibration,
        calibration_note=BASELINE_NOTE if calibration == "baseline" else CALIBRATED_NOTE,
        calibration_sources=sources,
        eligibility_rules=_eligibility_rules(assessments, rules),
        external_floors_declared_by=None if floors is None else floors.declared_by,
        overall_acceptance=acceptance,
        acceptance_note=NOT_CHECKED_NOTE if acceptance is ExternalAcceptance.NOT_CHECKED else None,
    )


def _rankable_rows(ranked: Sequence[_Assessment]) -> tuple[RankableRow, ...]:
    """J 由小到大排；同分照候選代號排，只為可重現。"""
    ordered = sorted(ranked, key=lambda item: (_total_cost(item), item.candidate.candidate_id))
    return tuple(
        RankableRow(
            status=CandidateStatus.RANKABLE,
            rank=index,
            candidate_id=item.candidate.candidate_id,
            provenance=item.candidate.provenance,
            total_cost=_total_cost(item),
            categories=tuple(sorted(item.lines, key=lambda line: line.identity.category.value)),
            uncovered=item.uncovered,
            flags=_row_flags(item),
            external_acceptance=item.external,
        )
        for index, item in enumerate(ordered, start=1)
    )


def _require_unique_ids(candidates: Sequence[CandidateEvaluation]) -> None:
    """同一批裡候選代號重複就報錯；兩列同名的排名表讀不出誰是誰。"""
    seen: set[str] = set()
    for candidate in candidates:
        if candidate.candidate_id in seen:
            raise ValueError(f"候選代號重複：{candidate.candidate_id}")
        seen.add(candidate.candidate_id)


def _external_names_are_known(floors: ExternalFloors | None, candidates: Sequence[CandidateEvaluation]) -> None:
    """外部底線列到不在這一批的候選代號就報錯；拼錯的代號會讓真的候選靜靜變成未檢查。"""
    if floors is None:
        return
    unknown = set(floors.verdicts) - {candidate.candidate_id for candidate in candidates}
    if unknown:
        raise ValueError(f"外部底線列到不在這一批的候選：{sorted(unknown)}")


def rank_candidates(
    candidates: Sequence[CandidateEvaluation],
    registry: QualityTargets,
    context: RankingContext,
    external_floors: ExternalFloors | None = None,
) -> RankingResult:
    """吃一批候選與登記簿，回五塊排名資料；狀態先後見檔頭。"""
    _require_unique_ids(candidates)
    _external_names_are_known(external_floors, candidates)
    rules = _read_rules(registry, context.purpose)
    assessments = [
        _assess(candidate, rules, _external_verdict(external_floors, candidate.candidate_id))
        for candidate in candidates
    ]
    eliminated = [item for item in assessments if item.eliminations]
    not_evaluated = [item for item in assessments if not item.eliminations and item.missing]
    contenders = [item for item in assessments if not item.eliminations and not item.missing]
    main_key, ranked, others = _split_tables(contenders)
    return RankingResult(
        schema_version="aosr.scoring.ranking.v1",
        header=_header(context, rules, main_key, assessments, ranked, external_floors),
        rankable=_rankable_rows(ranked),
        eliminated=tuple(
            EliminatedRow(
                status=CandidateStatus.ELIMINATED,
                candidate_id=item.candidate.candidate_id,
                provenance=item.candidate.provenance,
                reasons=item.eliminations,
                missing=item.missing,
                evaluations=item.evaluations,
                external_acceptance=item.external,
            )
            for item in eliminated
        ),
        not_evaluated=tuple(
            NotEvaluatedRow(
                status=CandidateStatus.NOT_EVALUATED,
                candidate_id=item.candidate.candidate_id,
                provenance=item.candidate.provenance,
                missing=item.missing,
                evaluations=item.evaluations,
                external_acceptance=item.external,
            )
            for item in not_evaluated
        ),
        not_comparable=NotComparableBlock(
            rows=tuple(
                NotComparableRow(
                    status=CandidateStatus.NOT_COMPARABLE,
                    candidate_id=item.candidate.candidate_id,
                    identity=_identity(item),
                )
                for item in others
            ),
            candidate_filter=CandidateFilterCounts(illegal_count=0, illegal_reasons={}),
        ),
    )
