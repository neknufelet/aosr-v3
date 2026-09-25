"""#489：音色與聆聽區按真正讀到的頻率分表。"""
from __future__ import annotations

import json
import math
from datetime import date

import pytest
from pydantic import ValidationError

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring.channel_matching import ChannelComparison, ChannelDefinition, ChannelGroup
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CandidateEvaluation, CategoryEvaluation, EvaluationState,
    InputProvenance, ListeningAreaFrequencySupport, ListeningAreaStabilityPayload,
    ModelValidationStatus, QualityCategory, TimbreChannelsPayload, TimbrePayload,
)
from aosr.scoring.listening_area import (
    LISTENING_AREA_EVALUATOR_VERSION, ReceiverPointResult, evaluate_listening_area,
)
from aosr.scoring import listening_area_cost
from aosr.scoring.listening_area_cost import cost_listening_area_evaluation
from aosr.scoring.ranking import CandidateStatus, RankingContext, RankingResult, rank_candidates
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet
from aosr.scoring.timbre import TimbreInput, evaluate_timbre
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.scoring.timbre_cost import comparison_support as timbre_support
from aosr.scoring.timbre_cost import cost_timbre_evaluation


_SCENE = "a" * 64
_PURPOSE = "dedicated_two_channel_listening_room"
_TARGETS = config_path("quality_targets.toml")
_AXIS = GEOMETRIC_LANE_FREQUENCIES_HZ


def _changed_middle(axis: tuple[float, ...]) -> tuple[float, ...]:
    middle = min(range(1, len(axis) - 1), key=lambda index: abs(axis[index] - 1000.0))
    return (*axis[:middle], (axis[middle - 1] + axis[middle]) / 2.0, *axis[middle + 1:])


def _curved(axis: tuple[float, ...], slope_db_per_octave: float,
            peak_db: float) -> tuple[float, ...]:
    """有斜率、在 250 Hz 有一個峰（或谷）的能量曲線；控制組要有不是 0 的分數才看得出分數變了沒。"""
    return tuple(
        1e7 * 10.0 ** ((slope_db_per_octave * math.log2(f / 1000.0)
                        + peak_db * math.exp(-(math.log2(f / 250.0) / 0.15) ** 2)) / 10.0)
        for f in axis
    )


def _timbre(candidate: str, receiver: str, speaker: str,
            axis: tuple[float, ...], energy: tuple[float, ...] | None = None) -> CategoryEvaluation:
    position = (1.0, 2.0, 1.2) if receiver == "main" else (1.1, 2.0, 1.2)
    data = TimbreInput(
        candidate_id=candidate, scene_fingerprint=_SCENE, speaker_id=speaker,
        receiver_id=receiver, source_position_m=(0.2, 0.3, 1.1),
        receiver_position_m=position, frequencies_hz=axis,
        total_energy=(1e7,) * len(axis) if energy is None else energy,
        source_reference="fixture-source",
        report_flags=(), model_validation_status=ModelValidationStatus.VALIDATED,
        model_validation_frequency_range_hz=(20.0, 8000.0),
        provenance=InputProvenance(
            report_id=f"report-{candidate}-{speaker}-{receiver}",
            engine_commit="fixture-engine", speaker_id=speaker, receiver_id=receiver,
        ),
    )
    result = evaluate_timbre(data, purpose=_PURPOSE, quality_targets_path=_TARGETS)
    assert result.state is EvaluationState.MEASURED
    return result


def _group() -> ChannelGroup:
    return ChannelGroup(
        channels=(ChannelDefinition(role="left", speaker_id="speaker-left"),),
        comparisons=(), feature_match_tolerance_hz=10.0,
    )


def _channels(candidate: str, axis: tuple[float, ...],
              energy: tuple[float, ...] | None = None) -> CategoryEvaluation:
    single = _timbre(candidate, "main", "speaker-left", axis, energy)
    result = evaluate_timbre_channels(
        _group(), "main", {"left": single},
        candidate_id=candidate, scene_fingerprint=_SCENE,
        timbre_settings_fingerprint=single.settings_fingerprint,
    )
    assert result.state is EvaluationState.MEASURED
    return result


def _receivers(*, reversed_order: bool = False) -> ReceiverSet:
    points = (
        ReceiverPoint(receiver_id="main", position_m=(1.0, 2.0, 1.2),
                      role=ReceiverRole.PRIMARY, importance=1.0),
        ReceiverPoint(receiver_id="front", position_m=(1.1, 2.0, 1.2),
                      role=ReceiverRole.SURROUNDING, importance=1.0,
                      direction_relative_to_primary="front"),
    )
    return ReceiverSet(points=points[::-1] if reversed_order else points)


def _listening(candidate: str, main_axis: tuple[float, ...],
               front_axis: tuple[float, ...],
               level_axis: tuple[float, ...] = _AXIS, *,
               shapes: tuple[tuple[float, float], tuple[float, float]] | None = None,
               reversed_order: bool = False) -> CategoryEvaluation:
    receivers = _receivers(reversed_order=reversed_order)

    def energy(axis: tuple[float, ...], index: int) -> tuple[float, ...] | None:
        return None if shapes is None else _curved(axis, *shapes[index])

    points = tuple(
        ReceiverPointResult(
            receiver_id=receiver, receiver_set_fingerprint=receivers.fingerprint,
            timbre_evaluation=_timbre(candidate, receiver, "speaker-left", axis, energy(axis, index)),
            frequencies_hz=level_axis,
            total_energy=energy(level_axis, index) or (1e7,) * len(level_axis),
        )
        for index, (receiver, axis) in enumerate((("main", main_axis), ("front", front_axis)))
    )
    fingerprint = points[0].timbre_evaluation.settings_fingerprint
    result = evaluate_listening_area(
        receivers, points, candidate_id=candidate, speaker_id="speaker-left",
        timbre_settings_fingerprint=fingerprint,
        scene_fingerprint=_SCENE, feature_match_tolerance_hz=10.0,
        broadband_range_hz=(20.0, 8000.0),
    )
    assert result.state is EvaluationState.MEASURED
    return result


def _registry(*required: QualityCategory) -> QualityTargets:
    document = load_quality_targets(_TARGETS).model_dump(mode="json", by_alias=True)
    for row in document["purpose"][0]["qualification"]:
        if row["key"] == "ranking.mandatory_categories":
            row["value"] = [category.value for category in required]
        elif row["key"] == "ranking.optional_categories":
            row["value"] = [
                name for name in row["value"] if name not in {item.value for item in required}
            ]
    return QualityTargets.model_validate(document)


def _rank(*candidates: CandidateEvaluation,
          required: tuple[QualityCategory, ...]) -> RankingResult:
    context = RankingContext(
        purpose=_PURPOSE, receiver_set_fingerprint=_receivers().fingerprint,
        channel_group_fingerprint=_group().fingerprint, run_date=date(2026, 9, 25),
        engine_version="fixture-engine",
    )
    return rank_candidates(candidates, _registry(*required), context)


def _candidate(candidate: str, *evaluations: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION, candidate_id=candidate,
        scene_fingerprint=_SCENE, evaluations=evaluations,
    )


def test_timbre_only_mixed_axes_are_not_comparable() -> None:
    """規則 3：只帶音色也須阻擋混軸；同一條物理曲線不能默默同表。"""
    first = _channels("first", _AXIS)
    second = _channels("second", _changed_middle(_AXIS))
    result = _rank(_candidate("first", first), _candidate("second", second),
                   required=(QualityCategory.TIMBRE_BALANCE,))
    assert {result.status_of(name) for name in ("first", "second")} == {
        CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE,
    }


# 控制組的答案是改動前的主線（這一刀之前）用同一組曲線實跑出來的數字（工作區 489-timbre-la-support/），
# 不是再呼叫一次被測的代價函式：代價函式自己變了，兩邊會一起變，那樣證不了「分數不變」。
_MAIN_TIMBRE_COST = {"steep": 0.2354061690312267, "gentle": 0.07770689312940236}
_MAIN_LISTENING_COST = {"spread": 0.6429367842635172, "close": 0.0}
_MAIN_LISTENING_RAW = {
    "spread": (1.4200514507692488, 0.32676162144836407, 2.3230997649550176, 4.0),
    "close": (0.22249940506240576, 0.1557517330157993, 0.36243734849381326, 0.0),
}
_SHAPES = {"steep": (-0.8, 6.0), "gentle": (-0.2, 2.0)}
_LISTENING_SHAPES = {"spread": ((-0.8, 6.0), (0.4, -4.0)), "close": ((-0.3, 2.0), (-0.1, 1.0))}


def test_timbre_same_axis_keeps_table_and_direct_cost() -> None:
    """老闆：「保留同軸同表、分數不變的控制組」——同一條軸兩條不同曲線，同表，類代價等於改動前主線的答案。"""
    evaluations = {name: _channels(name, _AXIS, _curved(_AXIS, *shape)) for name, shape in _SHAPES.items()}
    result = _rank(*(_candidate(name, item) for name, item in evaluations.items()),
                   required=(QualityCategory.TIMBRE_BALANCE,))
    assert {result.status_of(name) for name in evaluations} == {CandidateStatus.RANKABLE}
    costs = {row.candidate_id: row.categories[0].category_cost for row in result.rankable}
    assert costs == pytest.approx(_MAIN_TIMBRE_COST, rel=1e-12)
    assert [row.candidate_id for row in result.rankable] == ["gentle", "steep"]


def test_timbre_interior_point_changes_full_support() -> None:
    """規則 1：頭尾與點數相同，只換中間點仍要分表。"""
    first, second = _channels("first", _AXIS), _channels("second", _changed_middle(_AXIS))
    assert json.loads(timbre_support(first))["channels"][0]["frequencies_hz"] == [
        frequency for frequency in _AXIS if 20.0 <= frequency <= 8000.0
    ]
    assert timbre_support(first) != timbre_support(second)


def test_timbre_outside_coverage_is_not_support_or_score() -> None:
    """規則 1：覆蓋範圍外的頻點不進支撐，也不改原始量。"""
    first_axis = (10.0, *_AXIS, 16000.0)
    second_axis = (11.0, *_AXIS, 17000.0)
    first, second = _channels("first", first_axis), _channels("second", second_axis)
    assert timbre_support(first) == timbre_support(second)
    assert first.raw_quantities == second.raw_quantities
    result = _rank(_candidate("first", first), _candidate("second", second),
                   required=(QualityCategory.TIMBRE_BALANCE,))
    assert all(result.status_of(name) is CandidateStatus.RANKABLE
               for name in ("first", "second"))


def test_surrounding_timbre_axis_reaches_listening_support() -> None:
    """規則 2：主位相同，周圍點的上游音色頻軸改變仍須分表。"""
    first = _listening("first", _AXIS, _AXIS)
    second = _listening("second", _AXIS, _changed_middle(_AXIS))
    assert timbre_support(_channels("first", _AXIS)) == timbre_support(
        _channels("second", _AXIS)
    )
    assert isinstance(first.payload, ListeningAreaStabilityPayload)
    assert isinstance(second.payload, ListeningAreaStabilityPayload)
    first_points = first.payload.frequency_support.points
    second_points = second.payload.frequency_support.points
    assert first_points[0].timbre_frequencies_hz == second_points[0].timbre_frequencies_hz
    assert first_points[1].timbre_frequencies_hz == tuple(
        frequency for frequency in _AXIS if 20.0 <= frequency <= 8000.0
    )
    assert second_points[1].timbre_frequencies_hz == tuple(
        frequency for frequency in _changed_middle(_AXIS) if 20.0 <= frequency <= 8000.0
    )
    assert listening_area_cost.comparison_support(first) != listening_area_cost.comparison_support(second)
    result = _rank(_candidate("first", _channels("first", _AXIS), first),
                   _candidate("second", _channels("second", _AXIS), second),
                   required=(QualityCategory.TIMBRE_BALANCE,
                             QualityCategory.LISTENING_AREA_STABILITY))
    assert {result.status_of(name) for name in ("first", "second")} == {
        CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE,
    }


def test_overall_level_axis_reaches_listening_support() -> None:
    """規則 2：整體音量實際使用的頻率串改變也須分表。"""
    first = _listening("first", _AXIS, _AXIS)
    second = _listening("second", _AXIS, _AXIS, _changed_middle(_AXIS))
    assert listening_area_cost.comparison_support(first) != listening_area_cost.comparison_support(second)
    result = _rank(_candidate("first", _channels("first", _AXIS), first),
                   _candidate("second", _channels("second", _AXIS), second),
                   required=(QualityCategory.TIMBRE_BALANCE,
                             QualityCategory.LISTENING_AREA_STABILITY))
    assert {result.status_of(name) for name in ("first", "second")} == {
        CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE,
    }


def test_listening_same_axis_keeps_table_raw_quantities_and_cost() -> None:
    """老闆：「保留同軸同表、分數不變的控制組」——聆聽區同軸同表，原始量與類代價等於改動前主線的答案。"""
    evaluations = {name: _listening(name, _AXIS, _AXIS, shapes=shapes)
                   for name, shapes in _LISTENING_SHAPES.items()}
    for name, item in evaluations.items():
        by_name = {quantity.name: quantity.value for quantity in item.raw_quantities}
        expected = _MAIN_LISTENING_RAW[name]
        for index, metric in enumerate(("tilt", "ripple_rms", "overall_level", "peak_dip_consistency")):
            for kind in ("weighted_mean_deviation", "worst_deviation"):
                key = f"{metric}.primary_to_surrounding.{kind}"
                assert by_name[key] == pytest.approx(expected[index], rel=1e-12)
    result = _rank(*(_candidate(name, _channels(name, _AXIS, _curved(_AXIS, *shapes[0])), item)
                     for name, item in evaluations.items()
                     for shapes in (_LISTENING_SHAPES[name],)),
                   required=(QualityCategory.TIMBRE_BALANCE,
                             QualityCategory.LISTENING_AREA_STABILITY))
    assert {result.status_of(name) for name in evaluations} == {CandidateStatus.RANKABLE}
    costs = {}
    for row in result.rankable:
        lines = [category for category in row.categories
                 if category.identity.category is QualityCategory.LISTENING_AREA_STABILITY]
        assert lines
        costs[row.candidate_id] = lines[0].category_cost
    assert costs == pytest.approx(_MAIN_LISTENING_COST, rel=1e-12, abs=1e-15)


def test_listening_support_ignores_receiver_list_order() -> None:
    """接收點清單可以重排（ReceiverSet 的說明）：同一組點倒過來排，聆聽區支撐相同、同表。"""
    shapes = _LISTENING_SHAPES["close"]
    first = _listening("first", _AXIS, _changed_middle(_AXIS), shapes=shapes)
    second = _listening("second", _AXIS, _changed_middle(_AXIS), shapes=shapes, reversed_order=True)
    assert isinstance(first.payload, ListeningAreaStabilityPayload)
    assert isinstance(second.payload, ListeningAreaStabilityPayload)
    assert [p.receiver_id for p in first.payload.frequency_support.points] != [
        p.receiver_id for p in second.payload.frequency_support.points]
    assert listening_area_cost.comparison_support(first) == listening_area_cost.comparison_support(second)


def test_timbre_dependencies_fit_in_coverage() -> None:
    """規則 1：傾斜與起伏依賴範圍須落在覆蓋範圍內，支撐才不漏讀點。"""
    evaluation = _timbre("first", "main", "speaker-left", _AXIS)
    assert isinstance(evaluation.payload, TimbrePayload)
    payload = evaluation.payload
    assert "frequency_support_hz" not in TimbrePayload.model_fields
    for lower, upper in (payload.tilt_dependency_range_hz,
                         payload.ripple_dependency_range_hz):
        assert payload.coverage_range_hz[0] <= lower < upper <= payload.coverage_range_hz[1]
    assert payload.frequency_support_hz == tuple(
        frequency for frequency in _AXIS
        if payload.coverage_range_hz[0] <= frequency <= payload.coverage_range_hz[1]
    )


@pytest.mark.parametrize("bad", (
    "empty_level", "unsorted_level", "empty_timbre", "unsorted_timbre",
    "duplicate_receiver", "wrong_receiver", "wrong_order", "missing_section",
))
def test_listening_frequency_support_rejects_bad_contract(bad: str) -> None:
    """規則 2：支撐非空遞增、點不重複且與出身順序一致；缺節也拒收。"""
    evaluation = _listening("first", _AXIS, _AXIS)
    assert isinstance(evaluation.payload, ListeningAreaStabilityPayload)
    document = evaluation.payload.model_dump(mode="python")
    support = document["frequency_support"]
    if bad == "empty_level":
        support["overall_level_frequencies_hz"] = ()
    elif bad == "unsorted_level":
        support["overall_level_frequencies_hz"] = (200.0, 100.0)
    elif bad == "empty_timbre":
        support["points"][0]["timbre_frequencies_hz"] = ()
    elif bad == "unsorted_timbre":
        support["points"][0]["timbre_frequencies_hz"] = (200.0, 100.0)
    elif bad == "duplicate_receiver":
        support["points"][1]["receiver_id"] = "main"
    elif bad == "wrong_receiver":
        support["points"][1]["receiver_id"] = "other"
    elif bad == "wrong_order":
        support["points"] = list(reversed(support["points"]))
    else:
        del document["frequency_support"]
    with pytest.raises(ValidationError):
        ListeningAreaStabilityPayload.model_validate(document)


def test_frequency_support_contract_and_evaluator_version() -> None:
    """規則 2：完整兩層凍結支撐經評估器產出，且聆聽區版本升至 v5。"""
    evaluation = _listening("first", _AXIS, _AXIS)
    assert isinstance(evaluation.payload, ListeningAreaStabilityPayload)
    support = evaluation.payload.frequency_support
    expected = tuple(frequency for frequency in _AXIS if 20.0 <= frequency <= 8000.0)
    assert support.overall_level_frequencies_hz == expected
    assert support.points
    assert tuple(point.receiver_id for point in support.points) == ("main", "front")
    assert all(point.timbre_frequencies_hz == expected for point in support.points)
    document = support.model_dump(mode="json")
    document["points"] = sorted(document["points"], key=lambda point: str(point["receiver_id"]))
    assert listening_area_cost.comparison_support(evaluation) == json.dumps(
        document, sort_keys=True, separators=(",", ":")
    )
    assert LISTENING_AREA_EVALUATOR_VERSION == "aosr.scoring.listening_area.v5"
    assert evaluation.evaluator_version == LISTENING_AREA_EVALUATOR_VERSION


def test_listening_comparison_support_is_empty_for_other_payload() -> None:
    """規則 2：非聆聽區 payload 不得捏造聆聽區比較支撐。"""
    assert listening_area_cost.comparison_support(_channels("first", _AXIS)) == ""


def _two_channels(candidate: str, left_axis: tuple[float, ...],
                  right_axis: tuple[float, ...]) -> CategoryEvaluation:
    group = ChannelGroup(
        channels=(ChannelDefinition(role="left", speaker_id="speaker-left"),
                  ChannelDefinition(role="right", speaker_id="speaker-right")),
        comparisons=(ChannelComparison(left_role="left", right_role="right"),),
        feature_match_tolerance_hz=10.0,
    )
    left = _timbre(candidate, "main", "speaker-left", left_axis)
    right = _timbre(candidate, "main", "speaker-right", right_axis)
    result = evaluate_timbre_channels(
        group, "main", {"left": left, "right": right},
        candidate_id=candidate, scene_fingerprint=_SCENE,
        timbre_settings_fingerprint=left.settings_fingerprint,
    )
    assert result.state is EvaluationState.MEASURED
    return result


def test_timbre_support_records_each_channel_own_frequencies() -> None:
    """規則 2：每一支聲道記自己實際讀到的頻率；右聲道換軸時只有右聲道那一格跟著變。"""
    same = _two_channels("same", _AXIS, _AXIS)
    mixed = _two_channels("mixed", _AXIS, _changed_middle(_AXIS))
    assert isinstance(mixed.payload, TimbreChannelsPayload)
    low, high = mixed.payload.channels[0].payload.coverage_range_hz

    def by_role(evaluation: CategoryEvaluation) -> dict[str, list[float]]:
        document = json.loads(timbre_support(evaluation))
        return {channel["role"]: channel["frequencies_hz"] for channel in document["channels"]}

    # 另一條路：直接拿輸入軸濾覆蓋範圍，不讀被測的支撐屬性。
    expected_left = [f for f in _AXIS if low <= f <= high]
    expected_right = [f for f in _changed_middle(_AXIS) if low <= f <= high]
    assert expected_left != expected_right
    assert by_role(mixed) == {"left": expected_left, "right": expected_right}
    assert by_role(same) == {"left": expected_left, "right": expected_left}


def test_listening_support_rejects_duplicate_receivers_by_itself() -> None:
    """規則 2：聆聽區支撐自己就不准同一個接收點出現兩次（不靠跟出身比對才擋）。"""
    points = ({"receiver_id": "main", "timbre_frequencies_hz": (100.0, 200.0)},
              {"receiver_id": "front", "timbre_frequencies_hz": (100.0, 200.0)})
    good = {"overall_level_frequencies_hz": (100.0, 200.0), "points": points}
    assert ListeningAreaFrequencySupport.model_validate(good).points
    duplicated = {**good, "points": (points[0], {**points[1], "receiver_id": "main"})}
    with pytest.raises(ValidationError):
        ListeningAreaFrequencySupport.model_validate(duplicated)
