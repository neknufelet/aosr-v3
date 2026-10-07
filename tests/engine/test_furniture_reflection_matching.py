"""近似沒有仍列單側／雙側皆無；不冒充資料缺失。"""
from __future__ import annotations

import pytest

from aosr.scoring.contract import ReasonCode, MetricState
from aosr.scoring.channel_matching_reflections_contract import ReflectionAsymmetry, ReflectionAsymmetryPoint, ReflectionAsymmetryState
from aosr.scoring.reflections_contract import APPROXIMATE_NO_REFLECTION
from tests.engine import test_reflections as fixtures
from tests.engine import test_channel_matching_reflections as matching
from tests.engine.test_furniture_reflections import furniture_records, _zero_records


@pytest.mark.parametrize("zero", [False, True])
def test_approximate_one_sided_and_both_absent_are_available(zero: bool) -> None:
    room = {"Lx": 6.0, "Ly": 5.0, "Lz": 3.0}
    records = (fixtures._record("left", 0.6, room=room, receiver_y=1.2),
               fixtures._record("right", 1.8, room=room, receiver_y=1.2))
    if zero:
        records = (records[0], _zero_records((records[1],))[0])
    upstream = fixtures._evaluate(furniture_records(records))
    diagnosis = matching._diagnosis(upstream)
    assert diagnosis.state is MetricState.MEASURED
    assert diagnosis.one_sided
    assert all(point.state is not ReflectionAsymmetryState.UNAVAILABLE for point in diagnosis.points)
    assert all(set(row.reason_codes) <= APPROXIMATE_NO_REFLECTION for row in diagnosis.one_sided)
    absent = tuple(point for point in diagnosis.points if point.state is ReflectionAsymmetryState.BOTH_ABSENT)
    assert absent
    assert all(point.left_minus_right_db is None and point.reason_codes
               and set(point.reason_codes) <= APPROXIMATE_NO_REFLECTION for point in absent)
    if zero:
        assert any(ReasonCode.APPROXIMATE_ZERO_REFLECTION_ENERGY in row.reason_codes
                   for row in diagnosis.one_sided)


@pytest.mark.parametrize("reason", ["approximate_no_reflection_in_zone_point", "approximate_zero_reflection_energy"])
def test_approximate_absence_alone_cannot_be_an_unavailable_cell(reason: str) -> None:
    with pytest.raises(ValueError, match="真正缺失"):
        ReflectionAsymmetryPoint.model_validate({
            "left_role": "left", "right_role": "right", "receiver_id": "main", "zone": "front",
            "frequency_hz": 500.0, "state": "unavailable", "left_minus_right_db": None,
            "left": None, "right": None, "reason_codes": (reason,),
        })


@pytest.mark.parametrize("scope", ["within_cell", "across_cells"])
def test_reflection_asymmetry_rejects_mixed_absence_modes(scope: str) -> None:
    diagnosis = matching._diagnosis(fixtures._evaluate(furniture_records()))
    document = diagnosis.model_dump(mode="python")
    absent = [point for point in document["points"] if point["state"] is ReflectionAsymmetryState.BOTH_ABSENT]
    first, second = absent[:2]
    if scope == "within_cell":
        first["reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT, ReasonCode.APPROXIMATE_ZERO_REFLECTION_ENERGY)
    else:
        first["reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
        second["reason_codes"] = (ReasonCode.APPROXIMATE_NO_REFLECTION_IN_ZONE_POINT,)
    with pytest.raises(ValueError, match="左右差診斷不准混用近似沒有與確認沒有"):
        ReflectionAsymmetry.model_validate(document)


def test_reflection_asymmetry_does_not_require_furniture_flag_for_approximate_codes() -> None:
    diagnosis = matching._diagnosis(fixtures._evaluate(furniture_records()))
    document = diagnosis.model_dump(mode="python")
    document["source_flags"] = ()
    accepted = ReflectionAsymmetry.model_validate(document)
    assert accepted.points == diagnosis.points
    assert accepted.source_flags == ()
