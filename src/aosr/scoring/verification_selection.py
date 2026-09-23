"""#435 搜尋榜單的驗證名單；驗完前五名仍不合格後的遞補留給下一刀。"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt

from aosr.config.quality_targets import (
    QualityPurpose,
    QualityTargets,
    SettingEntry,
    TargetEntry,
    Unit,
)
from aosr.scoring.category_registry import EliminationReason
from aosr.scoring.contract import (
    CategoryEvaluation,
    ChannelMatchingPayload,
    ListeningAreaStabilityPayload,
)
from aosr.scoring.ranking import EliminatedRow, RankableRow, RankingResult


_FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


@dataclass(frozen=True)
class _LineSpec:
    target_key: str
    reason: EliminationReason
    unit: Unit
    metric: str
    group: str | None = None


_LINES = (
    _LineSpec(
        "listening_area_stability.tilt_worst_deviation",
        EliminationReason.LISTENING_AREA_TILT_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT,
        "dB/oct",
        "tilt_stability",
        "primary_to_surrounding",
    ),
    _LineSpec(
        "listening_area_stability.tilt_worst_deviation",
        EliminationReason.LISTENING_AREA_TILT_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT,
        "dB/oct",
        "tilt_stability",
        "surrounding_to_surrounding",
    ),
    _LineSpec(
        "listening_area_stability.ripple_rms_worst_deviation",
        EliminationReason.LISTENING_AREA_RIPPLE_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT,
        "dB",
        "ripple_rms_stability",
        "primary_to_surrounding",
    ),
    _LineSpec(
        "listening_area_stability.ripple_rms_worst_deviation",
        EliminationReason.LISTENING_AREA_RIPPLE_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT,
        "dB",
        "ripple_rms_stability",
        "surrounding_to_surrounding",
    ),
    _LineSpec(
        "listening_area_stability.overall_level_worst_deviation",
        EliminationReason.LISTENING_AREA_LEVEL_PRIMARY_TO_SURROUNDING_WORST_BEYOND_LIMIT,
        "dB",
        "overall_level_stability",
        "primary_to_surrounding",
    ),
    _LineSpec(
        "listening_area_stability.overall_level_worst_deviation",
        EliminationReason.LISTENING_AREA_LEVEL_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT,
        "dB",
        "overall_level_stability",
        "surrounding_to_surrounding",
    ),
    _LineSpec(
        "channel_matching.tilt_difference_worst",
        EliminationReason.CHANNEL_MATCHING_TILT_WORST_BEYOND_LIMIT,
        "dB/oct",
        "tilt_difference",
    ),
    _LineSpec(
        "channel_matching.ripple_rms_difference_worst",
        EliminationReason.CHANNEL_MATCHING_RIPPLE_WORST_BEYOND_LIMIT,
        "dB",
        "ripple_rms_difference",
    ),
    _LineSpec(
        "channel_matching.broadband_level_difference_worst",
        EliminationReason.CHANNEL_MATCHING_LEVEL_WORST_BEYOND_LIMIT,
        "dB",
        "broadband_level_difference",
    ),
    _LineSpec(
        "channel_matching.direct_time_difference_worst",
        EliminationReason.CHANNEL_MATCHING_DIRECT_TIME_WORST_BEYOND_LIMIT,
        "ms",
        "direct_time_difference",
    ),
)


class RegistryNumber(BaseModel):
    """這一輪讀到的登記簿原值、單位與出處。"""

    model_config = _FROZEN

    key: str
    value: StrictInt | StrictFloat
    unit: Unit
    status: str
    source_kind: str
    source: str


class VerificationLine(BaseModel):
    """一條底線與對應觀測換軸差，範圍由原值直接相乘。"""

    model_config = _FROZEN

    key: str
    elimination_reason: EliminationReason
    limit: RegistryNumber
    observed_axis_shift: RegistryNumber
    margin: float


class VerificationRules(BaseModel):
    """本輪固定的挑選規則與尚待決定的抽查及預算。"""

    model_config = _FROZEN

    purpose: str
    registry_fingerprint: str
    top_n: RegistryNumber
    multiplier: RegistryNumber
    # 直達聲時間差那條線開不開也會影響選人（關著就不讀那條），一起記下值與出處（找碴席抓到）。
    direct_time_cost_enabled: RegistryNumber
    lines: tuple[VerificationLine, ...]
    spot_check_count: int | None
    spot_check_note: str | None
    budget_candidates: int | None
    budget_note: str | None
    spot_check_seed: int


class SelectionReason(BaseModel):
    """前 N、近線或抽查；近線同時記 q、L、m 與無因次距離。"""

    model_config = _FROZEN

    kind: Literal["top_n", "near_line", "spot_check"]
    line_key: str | None = None
    q: float | None = None
    limit: float | None = None
    margin: float | None = None
    distance: float | None = None


class VerificationCandidate(BaseModel):
    """完整待驗名單的一列，預算只改執行狀態。"""

    model_config = _FROZEN

    candidate_id: str
    search_rank: int | None
    reasons: tuple[SelectionReason, ...] = Field(min_length=1)
    # 受控字串（跟 #449 的狀態同一種寫法）：scheduled＝排入驗證；not_verified_over_budget＝超出預算、尚未驗證。
    status: Literal["scheduled", "not_verified_over_budget"]


class NotAddedCandidate(BaseModel):
    """可比較榜單與淘汰表中未被追加的候選及原因。"""

    model_config = _FROZEN

    candidate_id: str
    reasons: tuple[str, ...] = Field(min_length=1)


class VerificationSelection(BaseModel):
    """本輪完整名單、未追加原因及不參與挑選的兩區計數。"""

    model_config = _FROZEN

    rules: VerificationRules
    candidates: tuple[VerificationCandidate, ...]
    not_added: tuple[NotAddedCandidate, ...]
    not_comparable_count: int = Field(ge=0)
    not_evaluated_count: int = Field(ge=0)


def _number(entry: SettingEntry | TargetEntry, unit: Unit) -> RegistryNumber:
    if entry.unit != unit:
        raise ValueError(f"{entry.key} 預期單位 {unit}，登記簿是 {entry.unit}")
    if not isinstance(entry.value, int | float) or isinstance(entry.value, bool):
        raise TypeError(f"{entry.key} 預期單一數值")
    return RegistryNumber(
        key=entry.key,
        value=entry.value,
        unit=unit,
        status=entry.status,
        source_kind=entry.source_kind,
        source=entry.source,
    )


def _setting(purpose: QualityPurpose, key: str, unit: Unit) -> RegistryNumber:
    entry = purpose.entry(key)
    if not isinstance(entry, SettingEntry):
        raise TypeError(f"{key} 預期量法設定")
    return _number(entry, unit)


def _target(purpose: QualityPurpose, key: str, unit: Unit) -> RegistryNumber:
    entry = purpose.entry(key)
    if (
        not isinstance(entry, TargetEntry)
        or entry.cost_shape != "beyond_threshold_only"
    ):
        raise TypeError(f"{key} 預期淘汰底線")
    return _number(entry, unit)


def _rules(
    purpose: QualityPurpose,
    registry_fingerprint: str,
    spot_check_count: int | None,
    budget_candidates: int | None,
    spot_check_seed: int,
) -> VerificationRules:
    top_n = _setting(purpose, "verification.top_n", "1")
    multiplier = _setting(purpose, "verification.range_multiplier", "1")
    if not isinstance(top_n.value, int) or top_n.value <= 0 or multiplier.value < 0:
        raise ValueError("前 N 須為正整數，倍數須非負")
    direct_time = _setting(purpose, "channel_matching.direct_time_cost_enabled", "1")
    lines = tuple(_line(purpose, spec, multiplier.value) for spec in _LINES)
    if any(line.observed_axis_shift.value < 0 for line in lines):
        raise ValueError("觀測換軸差須非負")
    return VerificationRules(
        purpose=purpose.name,
        registry_fingerprint=registry_fingerprint,
        top_n=top_n,
        multiplier=multiplier,
        direct_time_cost_enabled=direct_time,
        lines=lines,
        spot_check_count=spot_check_count,
        spot_check_note="抽查數量待定，這一輪沒有抽查"
        if spot_check_count is None
        else None,
        budget_candidates=budget_candidates,
        budget_note="預算待定，沒有截斷" if budget_candidates is None else None,
        spot_check_seed=spot_check_seed,
    )


def _line(
    purpose: QualityPurpose, spec: _LineSpec, multiplier: float
) -> VerificationLine:
    key = f"{spec.target_key}.{spec.group}" if spec.group else spec.target_key
    observed = _setting(purpose, f"verification.observed_axis_shift.{key}", spec.unit)
    return VerificationLine(
        key=key,
        elimination_reason=spec.reason,
        limit=_target(purpose, spec.target_key, spec.unit),
        observed_axis_shift=observed,
        margin=observed.value * multiplier,
    )


def _raw_value(evaluation: CategoryEvaluation, spec: _LineSpec) -> float | None:
    payload = evaluation.payload
    if isinstance(payload, ListeningAreaStabilityPayload) and spec.group is not None:
        comparison = getattr(payload, spec.metric)
        aggregate = getattr(comparison, spec.group)
        return None if aggregate is None else aggregate.worst_deviation.value
    if isinstance(payload, ChannelMatchingPayload) and spec.group is None:
        if (
            spec.metric == "direct_time_difference"
            and not payload.direct_time_cost_enabled
        ):
            return None
        values = (
            metric.worst_absolute_difference
            for comparison in payload.aggregates
            if (metric := getattr(comparison, spec.metric)) is not None
        )
        return max(values, default=None)
    return None


def _measurements(
    evaluations: tuple[CategoryEvaluation, ...],
    rules: VerificationRules,
) -> dict[EliminationReason, tuple[float, VerificationLine]]:
    measured = {}
    for spec, line in zip(_LINES, rules.lines, strict=True):
        values = tuple(
            value
            for evaluation in evaluations
            if (value := _raw_value(evaluation, spec)) is not None
        )
        if values:
            measured[spec.reason] = (max(values), line)
    return measured


def _near_reasons(
    measured: dict[EliminationReason, tuple[float, VerificationLine]],
) -> tuple[SelectionReason, ...]:
    reasons = []
    for q, line in measured.values():
        limit, margin = line.limit.value, line.margin
        # 範圍是 0 的線（直達聲時間差：只看距離與聲速，換軸不會變）換軸不可能翻盤，不算近線；
        # 否則剛好等於線的候選會被追加、還排在最前面擠掉真正要驗的人（找碴席抓到）。
        if margin > 0.0 and limit - margin <= q <= limit + margin:
            distance = abs(q - limit) / margin
            reasons.append(
                SelectionReason(
                    kind="near_line",
                    line_key=line.key,
                    q=q,
                    limit=limit,
                    margin=margin,
                    distance=distance,
                )
            )
    return tuple(reasons)


def _certain_failures(
    reasons: tuple[EliminationReason, ...],
    measured: dict[EliminationReason, tuple[float, VerificationLine]],
) -> tuple[str, ...]:
    certain = []
    for reason in reasons:
        if reason is EliminationReason.EXTERNAL_FLOOR_FAILED:
            certain.append(reason.value)
        elif reason not in measured:
            certain.append(f"{reason.value}：沒有可核對的原始最差值")
        else:
            q, line = measured[reason]
            if q > line.limit.value + line.margin:
                certain.append(
                    f"{reason.value}：q={q} > L+m={line.limit.value + line.margin}"
                )
    return tuple(certain)


def _review_others(
    ranking: RankingResult,
    rules: VerificationRules,
    top_ids: set[str],
) -> tuple[
    dict[str, tuple[int | None, tuple[SelectionReason, ...]]],
    list[NotAddedCandidate],
    list[tuple[float, str]],
    list[str],
]:
    selected: dict[str, tuple[int | None, tuple[SelectionReason, ...]]] = {}
    outside: list[NotAddedCandidate] = []
    near_order: list[tuple[float, str]] = []
    spot_pool: list[str] = []
    rows: tuple[RankableRow | EliminatedRow, ...] = (
        *ranking.rankable,
        *ranking.eliminated,
    )
    for row in rows:
        if row.candidate_id in top_ids:
            continue
        if isinstance(row, RankableRow):
            evaluations = tuple(line.evaluation for line in row.categories)
            failures: tuple[EliminationReason, ...] = ()
            search_rank: int | None = row.rank
        else:
            evaluations = row.evaluations
            failures = row.reasons
            search_rank = None
        measured = _measurements(evaluations, rules)
        blocked = _certain_failures(failures, measured)
        near = _near_reasons(measured)
        if blocked:
            outside.append(
                NotAddedCandidate(candidate_id=row.candidate_id, reasons=blocked)
            )
        elif near:
            selected[row.candidate_id] = (search_rank, near)
            distances = (
                reason.distance for reason in near if reason.distance is not None
            )
            near_order.append((min(distances), row.candidate_id))
        else:
            spot_pool.append(row.candidate_id)
    return selected, outside, near_order, spot_pool


def _draw_spots(
    ranking: RankingResult,
    rules: VerificationRules,
    spot_pool: list[str],
    selected: dict[str, tuple[int | None, tuple[SelectionReason, ...]]],
    outside: list[NotAddedCandidate],
) -> list[str]:
    count = rules.spot_check_count
    if count is None:
        drawn: list[str] = []
    else:
        if count > len(spot_pool):
            raise ValueError("抽查數量大於範圍外且未排除的候選數")
        drawn = random.Random(rules.spot_check_seed).sample(sorted(spot_pool), count)
    ranks = {row.candidate_id: row.rank for row in ranking.rankable}
    for candidate_id in drawn:
        selected[candidate_id] = (
            ranks.get(candidate_id),
            (SelectionReason(kind="spot_check"),),
        )
    outside_note = (
        "範圍外，抽查數量待定，本輪沒有抽查" if count is None else "範圍外，本輪未抽中"
    )
    outside.extend(
        NotAddedCandidate(candidate_id=candidate_id, reasons=(outside_note,))
        for candidate_id in spot_pool
        if candidate_id not in drawn
    )
    return drawn


def select_for_verification(
    ranking: RankingResult,
    registry: QualityTargets,
    spot_check_count: int | None,
    budget_candidates: int | None,
    spot_check_seed: int,
) -> VerificationSelection:
    """固定本輪規則，先列全名單，再依候選數預算標記執行狀態。

    規則要跟排名用同一份登記簿（整份指紋相同）：拿新規則去挑舊排名的人，淘汰線與換軸差會對不上（施工席提的疑慮）。
    """
    if registry.fingerprint != ranking.header.registry_fingerprint:
        raise ValueError("驗證規則的登記簿跟排名用的不是同一份（指紋不同）")
    purpose = registry.purpose(ranking.header.purpose)
    if spot_check_count is not None and spot_check_count < 0:
        raise ValueError("抽查數量不可為負")
    if budget_candidates is not None and budget_candidates < 0:
        raise ValueError("候選預算不可為負")
    rules = _rules(
        purpose, registry.fingerprint, spot_check_count, budget_candidates, spot_check_seed
    )
    top = sorted(ranking.rankable, key=lambda row: (row.rank, row.candidate_id))[
        : int(rules.top_n.value)
    ]
    top_ids = {row.candidate_id for row in top}
    selected: dict[str, tuple[int | None, tuple[SelectionReason, ...]]] = {
        row.candidate_id: (
            row.rank,
            (
                SelectionReason(kind="top_n"),
                *_near_reasons(
                    _measurements(
                        tuple(line.evaluation for line in row.categories), rules
                    )
                ),
            ),
        )
        for row in top
    }
    added, outside, near_order, spot_pool = _review_others(ranking, rules, top_ids)
    selected.update(added)
    drawn = _draw_spots(ranking, rules, spot_pool, selected, outside)
    ordered = [row.candidate_id for row in top]
    ordered += [candidate_id for _, candidate_id in sorted(near_order)]
    ordered += drawn
    candidates = tuple(
        VerificationCandidate(
            candidate_id=candidate_id,
            search_rank=selected[candidate_id][0],
            reasons=selected[candidate_id][1],
            status="scheduled"
            if budget_candidates is None or index < budget_candidates
            else "not_verified_over_budget",
        )
        for index, candidate_id in enumerate(ordered)
    )
    return VerificationSelection(
        rules=rules,
        candidates=candidates,
        not_added=tuple(sorted(outside, key=lambda item: item.candidate_id)),
        not_comparable_count=len(ranking.not_comparable.rows),
        not_evaluated_count=len(ranking.not_evaluated),
    )
