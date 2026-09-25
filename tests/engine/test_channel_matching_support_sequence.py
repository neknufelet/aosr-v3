"""票 #481：聲道匹配的寬頻頻率支撐列整串頻率，只換一個中間頻點也分得出來。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from aosr.scoring.channel_matching_cost import comparison_support
from aosr.scoring.contract import (
    CategoryEvaluation,
    ChannelBroadbandSupport,
    EvaluationState,
    ReasonCode,
)
from aosr.scoring.ranking import CandidateStatus, rank_candidates
from tests.engine import test_channel_matching as fixtures

# 頭、尾、點數都一樣，只有中間那一點不同（1000 對 1500 Hz）。
_AXIS_A = (100.0, 200.0, 1000.0, 4000.0, 6000.0)
_AXIS_B = (100.0, 200.0, 1500.0, 4000.0, 6000.0)


def _evaluated(candidate_id: str, axis_main: tuple[float, ...],
               axis_front: tuple[float, ...]) -> CategoryEvaluation:
    receivers, group = fixtures._receivers(), fixtures._group()
    points = tuple(
        fixtures._point(
            receivers, group, receiver_id,
            left_frequencies_hz=axis, right_frequencies_hz=axis,
            left_energy=(1.0,) * len(axis), right_energy=(1.0,) * len(axis),
            candidate_id=candidate_id,
        )
        for receiver_id, axis in (("main", axis_main), ("front", axis_front))
    )
    return fixtures._evaluate(receivers, group, points, candidate_id=candidate_id)


def _statuses(*evaluations: CategoryEvaluation) -> dict[str, CandidateStatus]:
    receivers, group = fixtures._receivers(), fixtures._group()
    result = rank_candidates(
        [fixtures._candidate(item) for item in evaluations],
        fixtures._channel_only_registry(),
        fixtures._ranking_context(receivers, group),
    )
    return {item.candidate_id: result.status_of(item.candidate_id) for item in evaluations}


def test_one_interior_frequency_change_separates_comparison_tables() -> None:
    """兩個候選頭、尾、點數一樣、只換一個中間頻點：比較支撐不同，不准放進同一張表。"""
    first = _evaluated("axis-a", _AXIS_A, _AXIS_A)
    second = _evaluated("axis-b", _AXIS_B, _AXIS_B)
    assert first.state is second.state is EvaluationState.MEASURED
    assert comparison_support(first) != comparison_support(second)
    assert set(_statuses(first, second).values()) == {
        CandidateStatus.RANKABLE, CandidateStatus.NOT_COMPARABLE,
    }


def test_same_axis_is_the_control_and_stays_in_one_table() -> None:
    """控制組：兩個候選同一條軸，比較支撐逐字相同、同一張表。"""
    first = _evaluated("same-a", _AXIS_A, _AXIS_A)
    second = _evaluated("same-b", _AXIS_A, _AXIS_A)
    assert comparison_support(first) == comparison_support(second)
    assert set(_statuses(first, second).values()) == {CandidateStatus.RANKABLE}


def test_interior_frequency_mismatch_inside_one_candidate_is_unavailable() -> None:
    """同一候選：主位與周圍點頭、尾、點數一樣但中間一點不同，整類不可估（#403「每一點都要一樣」）。"""
    mixed = _evaluated("mixed", _AXIS_A, _AXIS_B)
    assert mixed.state is EvaluationState.UNAVAILABLE
    assert mixed.reason_codes == (ReasonCode.FREQUENCY_AXIS_MISMATCH,)


@pytest.mark.parametrize("change", ("not_increasing", "lowest", "highest", "count"))
def test_support_summary_must_match_the_sequence(change: str) -> None:
    """頭、尾、點數只是給人讀的摘要，必須跟整串頻率對得上。"""
    good = {"frequencies_hz": (100.0, 200.0, 400.0), "lowest_frequency_hz": 100.0,
            "highest_frequency_hz": 400.0, "frequency_count": 3}
    assert ChannelBroadbandSupport.model_validate(good).frequencies_hz == (100.0, 200.0, 400.0)
    broken = dict(good)
    if change == "not_increasing":
        broken["frequencies_hz"] = (100.0, 400.0, 200.0)
        broken["highest_frequency_hz"] = 200.0
    elif change == "lowest":
        broken["lowest_frequency_hz"] = 50.0
    elif change == "highest":
        broken["highest_frequency_hz"] = 800.0
    else:
        broken["frequency_count"] = 4
    with pytest.raises(ValidationError):
        ChannelBroadbandSupport.model_validate(broken)
