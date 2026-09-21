"""殘響第二層代價與第三層排名資格考卷（票 #348 第二段）。"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Final

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, SettingEntry, load_quality_targets
from aosr.scoring import ranking
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    InputProvenance,
    QualityCategory,
    ReasonCode,
    ReverberationPayload,
)
from aosr.scoring.reverberation_cost import cost_reverberation_evaluation
from aosr.scoring.ranking import CandidateStatus, RankingContext
from tests.engine._placement import POINT_PLACEMENT


_PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
_SOURCE: Final[str] = "測試基線，未查證；正式值等 #358"
_SCENE_FINGERPRINT: Final[str] = "a" * 64
_CENTERS: Final[tuple[float, ...]] = (
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
)
_CONTEXT: Final[RankingContext] = RankingContext(
    purpose=_PURPOSE,
    receiver_set_fingerprint="receivers-fixture",
    channel_group_fingerprint="channels-fixture",
    run_date=date(2026, 9, 20),
    engine_version="engine-fixture",
)


def _registry(
    *,
    max_unavailable: int = 1,
    critical_bands: tuple[float, ...] = (125.0,),
    min_valid: int = 2,
    nominal_t20: float = 1.0,
) -> QualityTargets:
    """把正式登記簿的新條目釘成手算尺，避免考卷跟產品暫定值一起漂。"""
    document = load_quality_targets(config_path("quality_targets.toml")).model_dump(
        mode="json", by_alias=True
    )
    purpose = document["purpose"][0]
    settings = {item["key"]: item for item in purpose["setting"]}
    targets = {item["key"]: item for item in purpose["target"]}
    weights = {item["key"]: item for item in purpose["weight"]}
    qualifications = {item["key"]: item for item in purpose["qualification"]}
    settings["reverberation.target_band_centers_hz"]["value"] = list(_CENTERS)
    settings["reverberation.target_t20_nominal_s_by_band"]["value"] = [
        nominal_t20
    ] * len(_CENTERS)
    settings["reverberation.target_t20_tolerance_s_by_band"]["value"] = [0.1] * len(_CENTERS)
    settings["reverberation.adjacent_t20_logarithm_base"]["value"] = 2.0
    targets["reverberation.t20_interval_excess_s"]["worse_reference"] = 0.5
    jump = targets["reverberation.adjacent_t20_log_ratio_excess"]
    jump["value"] = 0.5
    jump["worse_reference"] = 1.0
    for item in weights["reverberation.within_category_weights"]["item"]:
        item["value"] = 1.0
    qualifications["ranking.mandatory_categories"]["value"] = ["reverberation"]
    qualifications["ranking.optional_categories"]["value"] = [
        name
        for name in qualifications["ranking.optional_categories"]["value"]
        if name != "reverberation"
    ]
    qualifications["ranking.eligibility.max_unavailable_bands"]["value"] = max_unavailable
    qualifications["ranking.eligibility.critical_bands"]["value"] = list(critical_bands)
    qualifications["ranking.eligibility.min_valid_bands"]["value"] = min_valid
    return QualityTargets.model_validate(document)


def _registry_with_unit(
    tmp_path: Path, *, key: str, expected: str, changed: str
) -> QualityTargets:
    text = config_path("quality_targets.toml").read_text(encoding="utf-8")
    marker = f'key = "{key}"'
    head, found, tail = text.partition(marker)
    assert found
    block, next_table, rest = tail.partition("[[purpose.")
    old = f'unit = "{expected}"'
    assert old in block
    path = tmp_path / "quality_targets.toml"
    path.write_text(
        head + found + block.replace(old, f'unit = "{changed}"', 1) + next_table + rest,
        encoding="utf-8",
    )
    return load_quality_targets(path)


def _metric(
    value: float | None,
    *,
    unit: str = "s",
    reason: str = "衰減範圍不足",
) -> dict[str, object]:
    if value is not None:
        return {
            "value": value,
            "unit": unit,
            "state": "measured",
            "reason_codes": [],
            "reason": None,
        }
    return {
        "value": None,
        "unit": unit,
        "state": "unavailable",
        "reason_codes": ["insufficient_decay_range"],
        "reason": reason,
    }


def _bands(
    centers: tuple[float, ...],
    values: tuple[float | None, ...],
    t30_scale: float,
) -> list[dict[str, object]]:
    bands: list[dict[str, object]] = []
    for center, t20 in zip(centers, values, strict=True):
        t30 = None if t20 is None else t20 * t30_scale
        bands.append(
            {
                "center_frequency_hz": center,
                "band_range_hz": [center / math.sqrt(2.0), center * math.sqrt(2.0)],
                "schroeder_position": "crossing" if center == 125.0 else "above",
                "model_validation_status": "experimental",
                "t20": _metric(t20),
                "t30": _metric(t30),
                "fitting_difference": _metric(
                    None if t20 is None else t30_scale,
                    unit="1",
                    reason="T20 不可估，無法計算 T30/T20",
                ),
            }
        )
    return bands


def _changes(
    centers: tuple[float, ...], values: tuple[float | None, ...]
) -> list[dict[str, object]]:
    changes: list[dict[str, object]] = []
    for lower_center, upper_center, lower, upper in zip(
        centers[:-1], centers[1:], values[:-1], values[1:], strict=True
    ):
        signed_log_ratio = None
        if lower is not None and upper is not None:
            signed_log_ratio = math.log(upper / lower, 2.0)
        changes.append(
            {
                "lower_center_frequency_hz": lower_center,
                "upper_center_frequency_hz": upper_center,
                "signed_log_ratio": signed_log_ratio,
                "state": "measured" if signed_log_ratio is not None else "not_computable",
                "reason_codes": []
                if signed_log_ratio is not None
                else ["insufficient_decay_range"],
                "reason": None
                if signed_log_ratio is not None
                else "相鄰帶任一端 T20 不可估",
            }
        )
    return changes


def _reverberation(
    values: tuple[float | None, ...],
    *,
    candidate_id: str = "candidate-a",
    t30_scale: float = 1.1,
    flags: tuple[str, ...] = (),
    unavailable_change: int | None = None,
    centers: tuple[float, ...] | None = None,
) -> CategoryEvaluation:
    centers = _CENTERS[: len(values)] if centers is None else centers
    changes = _changes(centers, values)
    if unavailable_change is not None:
        original = changes[unavailable_change]
        changes[unavailable_change] = {
            **original,
            "signed_log_ratio": None,
            "state": "not_computable",
            "reason_codes": ["insufficient_decay_range"],
            "reason": "這一對沒有可用的對數比",
        }
    provenance = InputProvenance(
        report_id=f"reverberation-report-{candidate_id}",
        engine_commit="fixture-engine",
        speaker_id="left",
        receiver_id="main-seat",
    )
    return CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "scene_fingerprint": _SCENE_FINGERPRINT,
            "placement": POINT_PLACEMENT,
            "category": "reverberation",
            "state": "measured",
            "payload": {
                "category": "reverberation",
                "bands": _bands(centers, values, t30_scale),
                "adjacent_band_changes": changes,
                "logarithm_base": 2.0,
            },
            "raw_quantities": [{"name": "band_count", "value": len(values), "unit": "1"}],
            "category_cost": None,
            "flags": list(flags),
            "reason_codes": [],
            "evaluator_version": "reverberation-fixture-v1",
            "settings_fingerprint": "reverberation-settings-a",
            "provenance": provenance,
        }
    )


def _rank(
    evaluation: CategoryEvaluation,
    registry: QualityTargets | None = None,
) -> ranking.RankingResult:
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=evaluation.candidate_id,
        scene_fingerprint=evaluation.scene_fingerprint,
        evaluations=(evaluation,),
    )
    return ranking.rank_candidates((candidate,), registry or _registry(), _CONTEXT)


def _rank_together(
    evaluations: tuple[CategoryEvaluation, ...], registry: QualityTargets
) -> ranking.RankingResult:
    candidates = tuple(
        CandidateEvaluation(
            schema_version=CONTRACT_SCHEMA_VERSION,
            candidate_id=evaluation.candidate_id,
            scene_fingerprint=evaluation.scene_fingerprint,
            evaluations=(evaluation,),
        )
        for evaluation in evaluations
    )
    return ranking.rank_candidates(candidates, registry, _CONTEXT)


def _costed(result: ranking.RankingResult) -> CategoryEvaluation:
    if result.rankable:
        return result.rankable[0].categories[0].evaluation
    return result.not_evaluated[0].evaluations[0]


def test_t20_cost_is_zero_inside_each_interval_and_grows_only_outside() -> None:
    inside = _costed(_rank(_reverberation((0.9, 1.0, 1.1, 1.0))))
    outside = _costed(_rank(_reverberation((1.3, 1.0, 1.1, 1.0))))

    assert inside.category_cost is not None
    assert outside.category_cost is not None
    assert inside.category_cost.components["t20_target_interval.125Hz"] == 0.0
    assert outside.category_cost.components["t20_target_interval.125Hz"] == pytest.approx(0.4)
    assert outside.category_cost.value > inside.category_cost.value


def test_reverberation_nominal_t20_rejects_registry_unit_mismatch(
    tmp_path: Path,
) -> None:
    """若殘響代價沒聲明 s，T20 名義值被錯標成 dB 仍會形成目標區間。"""
    key = "reverberation.target_t20_nominal_s_by_band"
    registry = _registry_with_unit(tmp_path, key=key, expected="s", changed="dB")

    with pytest.raises(
        ValueError,
        match=f"{key} 單位應為 s，登記簿寫 dB",
    ):
        cost_reverberation_evaluation(
            _reverberation((0.5,) * len(_CENTERS)),
            registry.purpose(_PURPOSE),
            registry.fingerprint,
        )


def test_unavailable_band_is_omitted_instead_of_averaged_as_zero() -> None:
    result = _rank(
        _reverberation((1.6, None)),
        _registry(max_unavailable=1, critical_bands=(125.0,), min_valid=1),
    )
    costed = _costed(result)

    assert costed.category_cost is not None
    assert costed.category_cost.components["t20_target_interval"] == pytest.approx(1.0)
    assert "t20_target_interval.250Hz" not in costed.category_cost.components
    assert costed.payload is not None
    assert isinstance(costed.payload, ReverberationPayload)
    assert costed.payload.bands[1].t20.reason_codes == ("insufficient_decay_range",)


def test_two_unavailable_bands_are_named_with_reasons_without_changing_cost() -> None:
    result = _rank(
        _reverberation((1.6, None, 1.0, None, 1.0, 1.0)),
        _registry(max_unavailable=2, critical_bands=(125.0,), min_valid=4),
    )
    costed = _costed(result)

    assert costed.category_cost is not None
    assert costed.category_cost.components["t20_target_interval"] == pytest.approx(0.25)
    assert [
        (band.center_frequency_hz, band.reason_codes)
        for band in costed.category_cost.unassessed_bands
    ] == [
        (250.0, ("insufficient_decay_range",)),
        (1000.0, ("insufficient_decay_range",)),
    ]
    assert result.rankable[0].categories[0].unassessed_bands == (
        costed.category_cost.unassessed_bands
    )


def test_equal_interval_excess_keeps_above_and_below_direction() -> None:
    registry = _registry(min_valid=1)
    above = _costed(_rank(_reverberation((1.3,)), registry))
    below = _costed(_rank(_reverberation((0.7,)), registry))

    assert above.category_cost is not None
    assert below.category_cost is not None
    component = "t20_target_interval.125Hz"
    assert above.category_cost.components[component] == pytest.approx(0.4)
    assert below.category_cost.components[component] == pytest.approx(0.4)
    assert above.category_cost.component_directions[component] == "above_range"
    assert below.category_cost.component_directions[component] == "below_range"


def test_t30_and_fitting_difference_do_not_change_cost() -> None:
    first = _costed(_rank(_reverberation((1.0, 1.0, 1.0, 1.0), t30_scale=1.1)))
    changed = _costed(_rank(_reverberation((1.0, 1.0, 1.0, 1.0), t30_scale=8.0)))

    assert first.category_cost is not None
    assert changed.category_cost is not None
    assert changed.category_cost == first.category_cost


def test_adjacent_jump_cost_starts_after_the_registered_threshold() -> None:
    boundary = _costed(_rank(_reverberation((1.0, math.sqrt(2.0)))))
    beyond = _costed(_rank(_reverberation((1.0, 2.0))))

    assert boundary.category_cost is not None
    assert beyond.category_cost is not None
    assert boundary.category_cost.components["adjacent_t20_jump.125-250Hz"] == pytest.approx(0.0)
    assert beyond.category_cost.components["adjacent_t20_jump.125-250Hz"] == pytest.approx(0.5)


def test_downward_adjacent_jump_uses_absolute_excess_over_registered_threshold() -> None:
    downward = _costed(_rank(_reverberation((1.0, 0.5))))

    assert downward.category_cost is not None
    assert downward.category_cost.components["adjacent_t20_jump.125-250Hz"] == pytest.approx(0.5)


def _assert_eligibility_failure(
    values: tuple[float | None, ...],
    *,
    max_unavailable: int = 1,
    critical_bands: tuple[float, ...] = (125.0,),
    min_valid: int = 2,
    expected_reason: str,
) -> None:
    evaluation = _reverberation(values)
    registry = _registry(
        max_unavailable=max_unavailable,
        critical_bands=critical_bands,
        min_valid=min_valid,
    )
    result = _rank(evaluation, registry)

    assert result.status_of("candidate-a") is CandidateStatus.NOT_EVALUATED
    assert result.eliminated == ()
    assert [gap.reason.value for gap in result.not_evaluated[0].missing] == [
        expected_reason
    ]
    cost = result.not_evaluated[0].evaluations[0].category_cost
    assert cost is not None
    assert cost.value == 0.0


def test_too_many_unavailable_bands_blocks_ranking_without_adding_cost() -> None:
    _assert_eligibility_failure(
        (1.0, None, None),
        min_valid=1,
        expected_reason="reverberation_too_many_unavailable_bands",
    )


def test_unavailable_critical_band_blocks_ranking_without_adding_cost() -> None:
    _assert_eligibility_failure(
        (1.0, None, 1.0, 1.0, 1.0, 1.0),
        critical_bands=(250.0,),
        expected_reason="reverberation_critical_band_unavailable",
    )


def test_too_few_valid_bands_blocks_ranking_without_adding_cost() -> None:
    _assert_eligibility_failure(
        (1.0, None, None, None, None, None),
        max_unavailable=5,
        expected_reason="reverberation_insufficient_valid_bands",
    )


def test_crossover_and_unvalidated_flags_reach_the_ranking_row() -> None:
    result = _rank(
        _reverberation(
            (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
            flags=("crossover_band", "unvalidated"),
        )
    )

    assert result.status_of("candidate-a") is CandidateStatus.RANKABLE
    assert {flag.value for flag in result.rankable[0].flags} == {
        "crossover_band",
        "unvalidated",
    }


def test_missing_worst_band_changes_cost_but_separates_comparison_tables() -> None:
    registry = _registry(min_valid=4, nominal_t20=0.5)
    purpose = registry.purpose(_PURPOSE)
    full = _reverberation(
        (0.5, 0.5, 0.5, 0.5, 0.5, 5.0), candidate_id="full"
    )
    missing = _reverberation(
        (0.5, 0.5, 0.5, 0.5, 0.5, None), candidate_id="missing-worst"
    )
    full_costed = cost_reverberation_evaluation(full, purpose, registry.fingerprint)
    missing_costed = cost_reverberation_evaluation(
        missing, purpose, registry.fingerprint
    )

    result = _rank_together((full, missing), registry)

    assert result.status_of("full") is not result.status_of("missing-worst")
    assert {
        result.status_of("full"),
        result.status_of("missing-worst"),
    } == {CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE}
    assert full_costed.category_cost is not None
    assert missing_costed.category_cost is not None
    assert full_costed.category_cost.value > 0.0
    assert missing_costed.category_cost.value == 0.0


def test_same_missing_band_stays_in_one_table_and_sorts_by_cost() -> None:
    registry = _registry(min_valid=4, nominal_t20=0.5)
    lower = _reverberation(
        (0.5, 0.5, 0.5, 0.5, 0.5, None), candidate_id="lower"
    )
    higher = _reverberation(
        (0.2, 0.5, 0.5, 0.5, 0.5, None), candidate_id="higher"
    )

    result = _rank_together((higher, lower), registry)

    assert [row.candidate_id for row in result.rankable] == ["lower", "higher"]
    assert not result.not_comparable.rows


def test_different_adjacent_pair_support_separates_comparison_tables() -> None:
    registry = _registry(min_valid=4, nominal_t20=0.5)
    complete = _reverberation((0.5,) * 6, candidate_id="complete-pairs")
    missing_pair = _reverberation(
        (0.5,) * 6, candidate_id="missing-pair", unavailable_change=2
    )

    result = _rank_together((complete, missing_pair), registry)

    assert {
        result.status_of("complete-pairs"),
        result.status_of("missing-pair"),
    } == {CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE}


def test_band_support_alone_separates_tables_when_pair_support_is_equal() -> None:
    """最高帶不可估、跟「最高帶可估但最後一對算不出」的配對支撐一樣；只靠頻帶那一半也要分表。"""
    registry = _registry(min_valid=4, nominal_t20=0.5)
    last_pair = len(_CENTERS) - 2
    all_bands = _reverberation(
        (0.5,) * 6, candidate_id="all-bands", unavailable_change=last_pair
    )
    top_band_missing = _reverberation(
        (0.5,) * 5 + (None,), candidate_id="top-band-missing"
    )

    result = _rank_together((all_bands, top_band_missing), registry)

    assert {
        result.status_of("all-bands"),
        result.status_of("top-band-missing"),
    } == {CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE}


def test_absent_band_row_counts_unavailable_and_is_reported_unassessed() -> None:
    registry = _registry(max_unavailable=0, min_valid=5)

    result = _rank(_reverberation((1.0,) * 5), registry)

    assert result.status_of("candidate-a") is CandidateStatus.NOT_EVALUATED
    assert {
        gap.reason.value for gap in result.not_evaluated[0].missing
    } == {"reverberation_too_many_unavailable_bands"}
    cost = result.not_evaluated[0].evaluations[0].category_cost
    assert cost is not None
    assert [
        (band.center_frequency_hz, band.reason_codes)
        for band in cost.unassessed_bands
    ] == [(4000.0, (ReasonCode.BAND_ROW_MISSING,))]


def test_absent_row_and_unavailable_band_are_listed_together_in_frequency_order() -> None:
    """整列沒送來（4000 Hz）與不可估（125 Hz）同時出現：兩種都列、照頻率排、原因各自說。"""
    registry = _registry(min_valid=4)
    evaluation = _reverberation((None, 0.5, 0.5, 0.5, 0.5))

    costed = cost_reverberation_evaluation(
        evaluation, registry.purpose(_PURPOSE), registry.fingerprint
    )

    assert costed.category_cost is not None
    listed = [
        (band.center_frequency_hz, band.reason_codes)
        for band in costed.category_cost.unassessed_bands
    ]
    assert [center for center, _ in listed] == [_CENTERS[0], _CENTERS[-1]]
    assert listed[-1][1] == (ReasonCode.BAND_ROW_MISSING,)
    assert ReasonCode.BAND_ROW_MISSING not in listed[0][1]


def test_duplicate_band_center_is_rejected_by_the_contract() -> None:
    """同一個中心頻率送兩列會讓好的那一帶在平均裡算兩次；契約靠「相鄰帶必須由低指向高」拒收。"""
    with pytest.raises(ValueError, match="由低中心頻率指向高中心頻率"):
        _reverberation((0.5, 0.5, 5.0), centers=(125.0, 125.0, 250.0))


@pytest.mark.parametrize(
    "category",
    (
        QualityCategory.TIMBRE_BALANCE,
        QualityCategory.LISTENING_AREA_STABILITY,
    ),
)
def test_categories_without_band_support_keep_empty_comparison_support(
    category: QualityCategory,
) -> None:
    """沒有提供評估支撐接點的類這一格是空的。聲道匹配從票 #403 起有自己的支撐，不在這裡。"""
    evaluation = _reverberation((1.0,) * 6)

    assert CATEGORY_REGISTRY[category].comparison_support(evaluation) == ""


_FORMAL_INTERVAL_KEYS = frozenset(
    {
        "reverberation.target_t20_nominal_s_by_band",
        "reverberation.target_t20_tolerance_s_by_band",
    }
)


def test_formal_reverberation_interval_is_the_decided_one() -> None:
    """票 #430 拍的區間：125 Hz 是 0.3–0.7 秒（低頻上限往上翹），250–4000 Hz 是 0.3–0.6 秒。

    直接對區間的上下限，不對 nominal／tolerance 各自的值——那兩格只是區間的另一種寫法。
    """
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE)
    columns: list[tuple[float, ...]] = []
    for key in ("target_band_centers_hz", "target_t20_nominal_s_by_band", "target_t20_tolerance_s_by_band"):
        entry = purpose.entry(f"reverberation.{key}")
        assert isinstance(entry, SettingEntry)
        assert isinstance(entry.value, tuple)
        columns.append(entry.value)
    intervals = {
        center: (nominal - width, nominal + width)
        for center, nominal, width in zip(*columns, strict=True)
    }

    assert intervals[125.0] == pytest.approx((0.3, 0.7))
    for center, interval in intervals.items():
        if center != 125.0:
            assert interval == pytest.approx((0.3, 0.6)), center
    for key in _FORMAL_INTERVAL_KEYS:
        formal = purpose.entry(key)
        assert isinstance(formal, SettingEntry)
        assert formal.status == "calibrated"


def test_formal_reverberation_registry_entries_are_all_provisional() -> None:
    """區間那兩條在票 #430 升成正式數字；其餘殘響條目仍然全是暫定。"""
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE)
    entries = tuple(
        entry
        for entry in purpose.entries
        if entry.key.startswith("reverberation.") and entry.key not in _FORMAL_INTERVAL_KEYS
    )
    records = tuple(
        record
        for entry in entries
        for record in (entry.item if hasattr(entry, "item") else (entry,))
    )

    assert entries
    assert {record.status for record in records} == {"baseline"}
    assert {record.source for record in records} == {_SOURCE}
