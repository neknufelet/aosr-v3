"""同一組距離與聲速換頻率軸，直達聲時間差逐位不變。"""

from __future__ import annotations

from aosr.config.frequency_axis import (
    GEOMETRIC_LANE_FREQUENCIES_HZ,
    VERIFICATION_REPORT_FREQUENCIES_HZ,
)
from aosr.scoring.contract import ChannelMatchingPayload
from tests.engine.test_channel_matching import _evaluate, _group, _point, _receivers


def test_direct_time_difference_is_identical_across_frequency_axes() -> None:
    receivers = _receivers()
    group = _group()
    search_axis = GEOMETRIC_LANE_FREQUENCIES_HZ
    verification_axis = VERIFICATION_REPORT_FREQUENCIES_HZ
    assert search_axis != verification_axis

    def evaluated(axis: tuple[float, ...]) -> ChannelMatchingPayload:
        points = tuple(
            _point(
                receivers,
                group,
                receiver_id,
                left_distance_m=2.0,
                right_distance_m=2.2,
                left_frequencies_hz=axis,
                right_frequencies_hz=axis,
                left_energy=(2.0,) * len(axis),
                right_energy=(1.0,) * len(axis),
            )
            for receiver_id in ("main", "front")
        )
        payload = _evaluate(receivers, group, points).payload
        assert isinstance(payload, ChannelMatchingPayload)
        return payload

    search = evaluated(search_axis)
    verification = evaluated(verification_axis)
    assert tuple(
        point.direct_time_difference_ms for point in search.point_results
    ) == tuple(point.direct_time_difference_ms for point in verification.point_results)
    assert tuple(
        aggregate.direct_time_difference for aggregate in search.aggregates
    ) == tuple(
        aggregate.direct_time_difference for aggregate in verification.aggregates
    )
