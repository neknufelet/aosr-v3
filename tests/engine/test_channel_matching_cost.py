"""聲道匹配第二層代價考卷（票 #350 第一段）。"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Final

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import (
    QualityTargets,
    SettingEntry,
    TargetEntry,
    load_quality_targets,
)
from aosr.scoring.channel_matching_cost import cost_channel_matching_evaluation
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    ChannelComparisonPair,
    ChannelIdentity,
    ChannelMatchingPayload,
    ChannelMetricAggregate,
    EvaluationState,
    Feature,
    InputProvenance,
    QualityCategory,
    RawQuantity,
)
from aosr.scoring.ranking import (
    CandidateStatus,
    EliminationReason,
    RankingContext,
    rank_candidates,
)


_PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
_SOURCE: Final[str] = "測試基線，未查證；正式值等 #358"


def _registry(
    *,
    direct_time_cost_enabled: bool = False,
    zero_channel_weights: bool = False,
) -> QualityTargets:
    document = load_quality_targets(config_path("quality_targets.toml")).model_dump(
        mode="json", by_alias=True
    )
    settings = {
        item["key"]: item for item in document["purpose"][0]["setting"]
    }
    settings["channel_matching.direct_time_cost_enabled"]["value"] = int(
        direct_time_cost_enabled
    )
    if zero_channel_weights:
        table = next(
            item
            for item in document["purpose"][0]["weight"]
            if item["key"] == "channel_matching.within_category_weights"
        )
        for item in table["item"]:
            item["value"] = 0.0
    return QualityTargets.model_validate(document)


def _registry_with_direct_time_unit(tmp_path: Path, changed: str) -> QualityTargets:
    text = config_path("quality_targets.toml").read_text(encoding="utf-8")
    switch = (
        'key = "channel_matching.direct_time_cost_enabled"\n'
        'value = 0\nunit = "1"'
    )
    assert switch in text
    text = text.replace(switch, switch.replace("value = 0", "value = 1"), 1)
    key = "channel_matching.direct_time_difference_weighted_mean"
    marker = f'key = "{key}"'
    head, found, tail = text.partition(marker)
    assert found
    block, next_table, rest = tail.partition("[[purpose.")
    assert 'unit = "ms"' in block
    path = tmp_path / "quality_targets.toml"
    path.write_text(
        head
        + found
        + block.replace('unit = "ms"', f'unit = "{changed}"', 1)
        + next_table
        + rest,
        encoding="utf-8",
    )
    return load_quality_targets(path)


def _feature() -> Feature:
    return Feature(
        kind="dip",
        center_frequency_hz=100.0,
        depth_db=-8.0,
        width_octave=0.25,
        flags=(),
    )


def _point_document() -> dict[str, object]:
    return {
        "receiver_id": "main",
        "importance": 1.0,
        "left_role": "left",
        "right_role": "right",
        "state": "measured",
        "reason_codes": [],
        "reason": None,
        "tilt_difference_db_per_octave": 2.0,
        "ripple_rms_difference_db": 3.0,
        "broadband_level_difference_db": 4.0,
        "direct_time_difference_ms": 5.0,
        "frequency_difference_curve_db": [
            {"frequency_hz": 100.0, "left_minus_right_db": 1.0}
        ],
        "unmatched_features": [
            {"present_in_role": "left", "feature": _feature()}
        ],
    }


def _aggregate_document() -> dict[str, object]:
    values = {
        "tilt_difference": 2.0,
        "ripple_rms_difference": 3.0,
        "broadband_level_difference": 4.0,
        "direct_time_difference": 5.0,
    }
    return {
        "left_role": "left",
        "right_role": "right",
        "assessed_receiver_ids": ["main"],
        "unavailable_receiver_ids": [],
        **{
            name: {
                "weighted_mean_absolute_difference": value,
                "worst_absolute_difference": value,
                "worst_receiver_id": "main",
            }
            for name, value in values.items()
        },
    }


def _payload(direct_time_cost_enabled: bool) -> ChannelMatchingPayload:
    return ChannelMatchingPayload.model_validate(
        {
            "category": "channel_matching",
            "candidate_id": "candidate-a",
            "receiver_set_fingerprint": "receiver-set-a",
            "timbre_settings_fingerprint": "timbre-settings-a",
            "listening_area_settings_fingerprint": "listening-settings-a",
            "channel_group_fingerprint": "channel-group-a",
            "channels": [
                {"role": "left", "speaker_id": "speaker-left"},
                {"role": "right", "speaker_id": "speaker-right"},
            ],
            "comparisons": [{"left_role": "left", "right_role": "right"}],
            "point_sources": [],
            "point_results": [_point_document()],
            "aggregates": [_aggregate_document()],
            "direct_time_cost_enabled": direct_time_cost_enabled,
        }
    )


def _measured(*, direct_time_cost_enabled: bool = False) -> CategoryEvaluation:
    return CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id="candidate-a",
        category=QualityCategory.CHANNEL_MATCHING,
        state=EvaluationState.MEASURED,
        payload=_payload(direct_time_cost_enabled),
        raw_quantities=(RawQuantity(name="assessed_point_count", value=1.0, unit="1"),),
        category_cost=None,
        flags=(),
        reason_codes=(),
        evaluator_version="channel-matching-fixture-v1",
        settings_fingerprint="channel-settings-a",
        provenance=InputProvenance(
            report_id="channel-report",
            engine_commit="engine-fixture",
            speaker_id="channel-group-a",
            receiver_id="receiver-set-a",
        ),
    )


def _cost(
    evaluation: CategoryEvaluation, registry: QualityTargets
) -> CategoryEvaluation:
    return cost_channel_matching_evaluation(
        evaluation,
        registry.purpose(_PURPOSE),
        registry.fingerprint,
    )


def _with_metric_summary(
    evaluation: CategoryEvaluation,
    *,
    metric: str,
    mean: float,
    worst: float,
    receiver_id: str = "main",
) -> CategoryEvaluation:
    payload = evaluation.payload
    assert isinstance(payload, ChannelMatchingPayload)
    aggregate = payload.aggregates[0]
    summary = getattr(aggregate, metric)
    assert summary is not None
    changed_summary = summary.model_copy(
        update={
            "weighted_mean_absolute_difference": mean,
            "worst_absolute_difference": worst,
            "worst_receiver_id": receiver_id,
        }
    )
    changed_aggregate = aggregate.model_copy(
        update={metric: changed_summary}
    )
    return evaluation.model_copy(
        update={
            "payload": payload.model_copy(
                update={"aggregates": (changed_aggregate,)}
            )
        }
    )


def _safe_measured(*, direct_time_cost_enabled: bool) -> CategoryEvaluation:
    evaluation = _measured(direct_time_cost_enabled=direct_time_cost_enabled)
    for metric, mean, worst in (
        ("tilt_difference", 1.0, 1.5),
        ("ripple_rms_difference", 2.0, 3.0),
        ("broadband_level_difference", 3.0, 4.0),
        ("direct_time_difference", 0.5, 1.0),
    ):
        evaluation = _with_metric_summary(
            evaluation,
            metric=metric,
            mean=mean,
            worst=worst,
        )
    return evaluation


def _channel_only_registry(*, direct_time_cost_enabled: bool = False) -> QualityTargets:
    document = _registry(
        direct_time_cost_enabled=direct_time_cost_enabled
    ).model_dump(mode="json", by_alias=True)
    for row in document["purpose"][0]["qualification"]:
        if row["key"] == "ranking.mandatory_categories":
            row["value"] = ["channel_matching"]
        elif row["key"] == "ranking.optional_categories":
            row["value"] = [
                name for name in row["value"] if name != "channel_matching"
            ]
    return QualityTargets.model_validate(document)


def _candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=evaluation.candidate_id,
        provenance=evaluation.provenance,
        evaluations=(evaluation,),
    )


def _context() -> RankingContext:
    return RankingContext(
        purpose=_PURPOSE,
        receiver_set_fingerprint="receiver-set-a",
        channel_group_fingerprint="channel-group-a",
        run_date=date(2026, 9, 20),
        engine_version="engine-fixture",
    )


def test_direct_time_is_absent_from_cost_by_default_and_enters_only_when_switch_is_on() -> None:
    """時間量值永遠保留；只有登記簿與 payload 都開啟時才可進類代價。"""
    off_registry = _registry(direct_time_cost_enabled=False)
    on_registry = _registry(direct_time_cost_enabled=True)

    off = _cost(_measured(direct_time_cost_enabled=False), off_registry)
    on = _cost(_measured(direct_time_cost_enabled=True), on_registry)

    assert off.category_cost is not None
    assert on.category_cost is not None
    assert "direct_time_difference.weighted_mean" not in off.category_cost.components
    assert "direct_time_difference.worst" not in off.category_cost.components
    assert on.category_cost.components["direct_time_difference.weighted_mean"] > 0.0
    assert on.category_cost.components["direct_time_difference.worst"] > 0.0
    assert on.category_cost.value > off.category_cost.value
    assert isinstance(off.payload, ChannelMatchingPayload)
    assert off.payload.point_results[0].direct_time_difference_ms == pytest.approx(5.0)


def test_direct_time_rejects_registry_unit_mismatch(tmp_path: Path) -> None:
    """若聲道時間代價沒聲明 ms，秒單位的門檻會被當成毫秒直接套用。"""
    key = "channel_matching.direct_time_difference_weighted_mean"
    registry = _registry_with_direct_time_unit(tmp_path, "s")

    with pytest.raises(
        ValueError,
        match=f"{key} 單位應為 ms，登記簿寫 s",
    ):
        _cost(_measured(direct_time_cost_enabled=True), registry)


def test_diagnostic_curve_and_unmatched_features_never_change_cost() -> None:
    """逐頻率曲線與單邊峰谷只能診斷；改它們不得改任何代價分項。"""
    registry = _registry()
    measured = _measured()
    assert isinstance(measured.payload, ChannelMatchingPayload)
    point = measured.payload.point_results[0]
    changed_point = point.model_copy(
        update={
            "frequency_difference_curve_db": (
                point.frequency_difference_curve_db[0].model_copy(
                    update={"left_minus_right_db": 99.0}
                ),
            ),
            "unmatched_features": (),
        }
    )
    changed_payload = measured.payload.model_copy(
        update={"point_results": (changed_point,)}
    )
    changed = measured.model_copy(update={"payload": changed_payload})

    baseline = _cost(measured, registry)
    diagnostic_changed = _cost(changed, registry)

    assert baseline.category_cost is not None
    assert diagnostic_changed.category_cost is not None
    assert diagnostic_changed.category_cost == baseline.category_cost


@pytest.mark.parametrize(
    "component",
    ("tilt_difference", "ripple_rms_difference", "broadband_level_difference"),
)
def test_weighted_mean_uses_dead_band_while_worst_is_a_separate_protection(
    component: str,
) -> None:
    """三種可動代價各自讀登記簿容許帶，最差值另列保護而不重複加進主代價。"""
    registry = _registry()
    purpose = registry.purpose(_PURPOSE)
    target = purpose.entry(f"channel_matching.{component}_weighted_mean")
    protection = purpose.entry(f"channel_matching.{component}_worst")

    costed = _cost(_measured(), registry)

    assert isinstance(target, TargetEntry)
    assert isinstance(protection, TargetEntry)
    assert target.cost_shape == "in_range_best"
    assert protection.cost_shape == "beyond_threshold_only"
    assert costed.category_cost is not None
    assert f"{component}.weighted_mean" in costed.category_cost.components
    assert f"{component}.worst" in costed.category_cost.components


@pytest.mark.parametrize(
    ("metric", "mean", "threshold", "reason", "direct_time_cost_enabled"),
    (
        (
            "tilt_difference",
            1.0,
            1.5,
            EliminationReason.CHANNEL_MATCHING_TILT_WORST_BEYOND_LIMIT,
            False,
        ),
        (
            "ripple_rms_difference",
            2.0,
            3.0,
            EliminationReason.CHANNEL_MATCHING_RIPPLE_WORST_BEYOND_LIMIT,
            False,
        ),
        (
            "broadband_level_difference",
            3.0,
            4.0,
            EliminationReason.CHANNEL_MATCHING_LEVEL_WORST_BEYOND_LIMIT,
            False,
        ),
        (
            "direct_time_difference",
            0.5,
            1.0,
            EliminationReason.CHANNEL_MATCHING_DIRECT_TIME_WORST_BEYOND_LIMIT,
            True,
        ),
    ),
)
def test_worst_breach_eliminates_candidate_without_changing_principal_cost(
    metric: str,
    mean: float,
    threshold: float,
    reason: EliminationReason,
    direct_time_cost_enabled: bool,
) -> None:
    """最差超標要真正在排名層淘汰；同一加權平均下主代價不得被保護值污染。"""
    safe = _safe_measured(direct_time_cost_enabled=direct_time_cost_enabled)
    breached = _with_metric_summary(
        safe,
        metric=metric,
        mean=mean,
        worst=threshold + 0.5,
    )
    registry = _registry(direct_time_cost_enabled=direct_time_cost_enabled)

    safe_costed = _cost(safe, registry)
    breached_costed = _cost(breached, registry)
    result = rank_candidates(
        [_candidate(breached)],
        _channel_only_registry(
            direct_time_cost_enabled=direct_time_cost_enabled
        ),
        _context(),
    )

    assert safe_costed.category_cost is not None
    assert breached_costed.category_cost is not None
    assert breached_costed.category_cost.value == safe_costed.category_cost.value
    assert result.status_of("candidate-a") is CandidateStatus.ELIMINATED
    (row,) = result.eliminated
    assert row.reasons == (reason,)


def test_worst_components_name_both_comparison_pair_and_receiver() -> None:
    """多個比較對各自保留最差接收點，不能只剩一個跨比較對 max。"""
    measured = _measured()
    payload = measured.payload
    assert isinstance(payload, ChannelMatchingPayload)
    first_point = payload.point_results[0]
    first_aggregate = payload.aggregates[0]

    def at_front(
        summary: ChannelMetricAggregate | None,
    ) -> ChannelMetricAggregate:
        assert summary is not None
        return summary.model_copy(update={"worst_receiver_id": "front"})

    second_metrics = {
        "tilt_difference": at_front(first_aggregate.tilt_difference),
        "ripple_rms_difference": at_front(first_aggregate.ripple_rms_difference),
        "broadband_level_difference": at_front(
            first_aggregate.broadband_level_difference
        ),
        "direct_time_difference": at_front(
            first_aggregate.direct_time_difference
        ),
    }
    second_aggregate = first_aggregate.model_copy(
        update={
            "left_role": "right",
            "right_role": "center",
            "assessed_receiver_ids": ("front",),
            **second_metrics,
        }
    )
    expanded = payload.model_copy(
        update={
            "channels": (
                *payload.channels,
                ChannelIdentity(role="center", speaker_id="speaker-center"),
            ),
            "comparisons": (
                *payload.comparisons,
                ChannelComparisonPair(left_role="right", right_role="center"),
            ),
            "point_results": (
                first_point,
                first_point.model_copy(
                    update={
                        "receiver_id": "front",
                        "left_role": "right",
                        "right_role": "center",
                    }
                ),
            ),
            "aggregates": (first_aggregate, second_aggregate),
        }
    )

    costed = _cost(measured.model_copy(update={"payload": expanded}), _registry())

    assert costed.category_cost is not None
    assert "tilt_difference.worst.left-right.main" in costed.category_cost.components
    assert "tilt_difference.worst.right-center.front" in costed.category_cost.components


def test_zero_sum_of_participating_weights_is_an_explicit_error() -> None:
    """有可估主分項但參與權重和為零時，整類不得靜默維持 measured。"""
    with pytest.raises(ValueError, match="參與主代價的權重和必須大於零"):
        _cost(_measured(), _registry(zero_channel_weights=True))


def test_channel_matching_registry_entries_are_all_provisional_and_keep_switch_off() -> None:
    """這一段只能落 baseline 佔位數字，也不得偷把位置用的時間代價開關打開。"""
    purpose = _registry().purpose(_PURPOSE)
    entries = tuple(
        entry for entry in purpose.entries if entry.key.startswith("channel_matching.")
    )
    records = tuple(
        record
        for entry in entries
        for record in (entry.item if hasattr(entry, "item") else (entry,))
    )
    switch = purpose.entry("channel_matching.direct_time_cost_enabled")

    assert {record.status for record in records} == {"baseline"}
    assert {record.source for record in records} == {_SOURCE}
    # 開關是量法設定；先縮型別再讀值（union 直接取 .value 型別警衛會擋）。
    assert isinstance(switch, SettingEntry)
    assert switch.value == 0
