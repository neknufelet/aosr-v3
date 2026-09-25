"""#480 反射的擺位與比較支撐：拒用的周圍點不拖垮主位、只換一個中間頻點就不同表。"""
from __future__ import annotations

from dataclasses import replace

from aosr.scoring.contract import EvaluationState, QualityCategory, ReasonCode
from aosr.scoring.ranking import CandidateStatus, rank_candidates
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from aosr.scoring.reflections_cost import comparison_support
from tests.engine import test_ranking_recost as recost
from tests.engine import test_reflections as fixtures
from tests.engine import test_reflections_ranking as ranking_fixtures


def _channel_reasons(evaluation_payload: object) -> dict[tuple[str, str], tuple[ReasonCode, ...]]:
    assert isinstance(evaluation_payload, ReflectionsAndEchoPayload)
    return {(channel.role, channel.receiver_id): channel.reason_codes
            for channel in evaluation_payload.channels}


def test_rejected_surrounding_point_does_not_veto_primary_placement() -> None:
    """周圍點先因篩查對不上被拒用，它帶的別支喇叭座標不准再拿來否決主位（舊規則會整類不可估）。"""
    left, right = fixtures._pair()
    around_left = fixtures._record("left", 1.3, "around")
    around_right = fixtures._record("right", 2.5, "around")
    around_left = replace(around_left, report=around_left.report.model_copy(update={
        "scene": around_left.report.scene.model_copy(update={"source_m": right.report.scene.source_m})
    }))
    result = fixtures._evaluate((left, right, around_left, around_right))
    assert result.state is EvaluationState.MEASURED
    reasons = _channel_reasons(result.payload)
    assert reasons["left", "around"] == (ReasonCode.REFLECTION_SCREEN_OR_WINDOW_MISMATCH,)
    assert reasons["right", "around"] == ()
    assert dict(result.placement.speaker_positions_m) == {
        "left": left.report.scene.source_m.as_tuple(), "right": right.report.scene.source_m.as_tuple()}


def test_surrounding_point_with_stale_speaker_is_rejected_locally() -> None:
    """周圍點自己的資料自洽、只有喇叭座標是舊的：只有擺位抓得到，只記那一支，主位照常可估。"""
    left, right = fixtures._pair()
    stale_left = fixtures._record("left", 1.35, "s1", receiver_y=2.2)
    s1_right = fixtures._record("right", 2.5, "s1", receiver_y=2.2)
    result = fixtures._evaluate((left, right, stale_left, s1_right))
    assert result.state is EvaluationState.MEASURED
    reasons = _channel_reasons(result.payload)
    assert reasons["left", "s1"] == (ReasonCode.PLACEMENT_MISMATCH,)
    assert reasons["right", "s1"] == ()
    assert dict(result.placement.speaker_positions_m)["left"] == left.report.scene.source_m.as_tuple()
    assert "s1" in dict(result.placement.receiver_positions_m)


def test_surrounding_point_whose_channels_disagree_on_receiver_is_rejected() -> None:
    """同一個周圍點的兩支接收點座標不同：那一點兩支都記擺位不符，這一點不進有效擺位。"""
    left, right = fixtures._pair()
    s1_left = fixtures._record("left", 1.3, "s1", receiver_y=2.2)
    s1_right = fixtures._record("right", 2.5, "s1", receiver_y=2.25)
    result = fixtures._evaluate((left, right, s1_left, s1_right))
    assert result.state is EvaluationState.MEASURED
    reasons = _channel_reasons(result.payload)
    assert reasons["left", "s1"] == (ReasonCode.PLACEMENT_MISMATCH,)
    assert reasons["right", "s1"] == (ReasonCode.PLACEMENT_MISMATCH,)
    assert set(dict(result.placement.receiver_positions_m)) == {"main"}


def test_primary_placement_conflict_still_makes_category_unavailable() -> None:
    """主位兩支對主位接收點的座標不同：分數要用的資料自己衝突，整類不可估（跟以前一樣）。"""
    left = fixtures._record("left", 1.3)
    right = fixtures._record("right", 2.5, receiver_y=1.95)
    result = fixtures._evaluate((left, right))
    assert result.state is EvaluationState.UNAVAILABLE
    assert result.reason_codes == (ReasonCode.PLACEMENT_MISMATCH,)


def test_interior_frequency_change_splits_comparison_tables() -> None:
    """頭、尾、點數都一樣，只換一個中間頻點：沒量到同一批頻率，不准同表。"""
    axis_a = (250.0, 300.0, 500.0, 1000.0, 4000.0, 8000.0, 9000.0)
    axis_b = (250.0, 300.0, 500.0, 2000.0, 4000.0, 8000.0, 9000.0)
    first = fixtures._evaluate((fixtures._record("left", 1.3, axis=axis_a),
                                fixtures._record("right", 2.5, axis=axis_a)))
    second = fixtures._evaluate((fixtures._record("left", 1.3, axis=axis_b),
                                 fixtures._record("right", 2.5, axis=axis_b)))
    assert comparison_support(first) != comparison_support(second)
    first = first.model_copy(update={"candidate_id": "axis-a"})
    second = second.model_copy(update={"candidate_id": "axis-b"})
    registry = recost._registry_for(QualityCategory.REFLECTIONS_AND_ECHO)
    result = rank_candidates((ranking_fixtures._candidate(first), ranking_fixtures._candidate(second)),
                             registry, recost._CONTEXT)
    assert {result.status_of("axis-a"), result.status_of("axis-b")} == {
        CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE}
