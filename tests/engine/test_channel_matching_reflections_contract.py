"""反射左右差契約守衛：四態、清單、鍵及來源必須自洽。"""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from aosr.scoring.channel_matching_reflections_contract import (
    ReflectionAsymmetry, ReflectionAsymmetryPoint, ReflectionAsymmetryState,
)
from aosr.scoring.contract import MetricState, ReasonCode
from tests.engine.test_channel_matching_reflections import _diagnosis, _one_sided


def _section() -> ReflectionAsymmetry:
    return _diagnosis(_one_sided())


def _reject(document: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ReflectionAsymmetry.model_validate(document)


@pytest.mark.parametrize(("state", "damage"), (
    (ReflectionAsymmetryState.MEASURED, "wrong_difference"),
    (ReflectionAsymmetryState.MEASURED, "no_left"),
    (ReflectionAsymmetryState.MEASURED, "no_right"),
    (ReflectionAsymmetryState.MEASURED, "extra_reason"),
    (ReflectionAsymmetryState.ONE_SIDED, "zero_difference"),
    (ReflectionAsymmetryState.ONE_SIDED, "both_sides"),
    (ReflectionAsymmetryState.ONE_SIDED, "no_sides"),
    (ReflectionAsymmetryState.ONE_SIDED, "no_reason"),
    (ReflectionAsymmetryState.ONE_SIDED, "missing_reason"),
    (ReflectionAsymmetryState.BOTH_ABSENT, "zero_difference"),
    (ReflectionAsymmetryState.BOTH_ABSENT, "left_side"),
    (ReflectionAsymmetryState.BOTH_ABSENT, "right_side"),
    (ReflectionAsymmetryState.BOTH_ABSENT, "no_reason"),
    (ReflectionAsymmetryState.BOTH_ABSENT, "missing_reason"),
    (ReflectionAsymmetryState.UNAVAILABLE, "zero_difference"),
    (ReflectionAsymmetryState.UNAVAILABLE, "left_side"),
    (ReflectionAsymmetryState.UNAVAILABLE, "right_side"),
    (ReflectionAsymmetryState.UNAVAILABLE, "no_reason"),
    (ReflectionAsymmetryState.UNAVAILABLE, "absence_only"),
))
def test_each_cell_state_rejects_wrong_value_or_evidence(
    state: ReflectionAsymmetryState, damage: str,
) -> None:
    """四態的差值、兩側來源與原因碼逐格互斥。"""
    document = _section().model_dump(mode="python")
    measured = next(p for p in document["points"] if p["state"] is ReflectionAsymmetryState.MEASURED)
    cell = next((p for p in document["points"] if p["state"] is state), None)
    if cell is None:
        assert state is ReflectionAsymmetryState.UNAVAILABLE
        cell = deepcopy(measured)
        cell.update(state=state, left=None, right=None,
                    left_minus_right_db=None, reason_codes=(ReasonCode.INSUFFICIENT_COVERAGE,))
    assert ReflectionAsymmetryPoint.model_validate(cell)
    broken = deepcopy(cell)
    if damage == "wrong_difference":
        broken["left_minus_right_db"] += 1.0
    elif damage in {"zero_difference", "no_left", "no_right"}:
        if damage == "zero_difference":
            broken["left_minus_right_db"] = 0.0
        else:
            broken["left" if damage == "no_left" else "right"] = None
    elif damage == "extra_reason":
        broken["reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
    elif damage == "both_sides":
        missing = "left" if broken["left"] is None else "right"
        broken[missing] = measured[missing]
    elif damage == "no_sides":
        broken["left"] = broken["right"] = None
    elif damage in {"left_side", "right_side"}:
        side = "left" if damage == "left_side" else "right"
        broken[side] = measured[side]
    elif damage == "no_reason":
        broken["reason_codes"] = ()
    elif damage == "absence_only":
        broken["reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    else:
        broken["reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
    with pytest.raises(ValidationError):
        ReflectionAsymmetryPoint.model_validate(broken)


@pytest.mark.parametrize(("state", "damage"), (
    (MetricState.MEASURED, "reason"),
    (MetricState.MEASURED, "version"),
    (MetricState.MEASURED, "window"),
    (MetricState.MEASURED, "primary"),
    (MetricState.MEASURED, "range"),
    (MetricState.UNAVAILABLE, "no_reason"),
    (MetricState.UNAVAILABLE, "points"),
    (MetricState.NOT_COMPUTABLE, "state"),
))
def test_section_state_rejects_bad_identity_or_cells(state: MetricState, damage: str) -> None:
    """整節已量與不可估的必要欄位及允許狀態。"""
    measured = _section().model_dump(mode="python")
    baseline = deepcopy(measured)
    if state is not MetricState.MEASURED:
        baseline.update(state=MetricState.UNAVAILABLE,
                        reason_codes=(ReasonCode.INSUFFICIENT_COVERAGE,),
                        points=(), one_sided=())
    assert ReflectionAsymmetry.model_validate(baseline)
    broken = deepcopy(baseline)
    if damage == "reason":
        broken["reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
    elif damage in {"version", "window", "primary", "range"}:
        field = {"version": "reflections_evaluator_version", "window": "window_upper_ms",
                 "primary": "primary_receiver_id", "range": "frequency_range_hz"}[damage]
        broken[field] = None
    elif damage == "no_reason":
        broken["reason_codes"] = ()
    elif damage == "points":
        broken["points"] = tuple(point for point in measured["points"]
                                 if point["state"] is ReflectionAsymmetryState.MEASURED)
        assert broken["points"]
    else:
        broken["state"] = MetricState.NOT_COMPUTABLE
    with pytest.raises(ValidationError):
        ReflectionAsymmetry.model_validate(broken)


@pytest.mark.parametrize("change", ("extra", "missing", "content"))
def test_one_sided_list_must_equal_cells_exactly(change: str) -> None:
    """單側清單多、少、內容不一致都拒收。"""
    document = _section().model_dump(mode="python")
    rows = list(document["one_sided"])
    assert rows
    if change == "extra":
        rows.append(deepcopy(rows[0]))
    elif change == "missing":
        rows.pop()
    else:
        rows[0] = {**rows[0], "absent_role": rows[0]["present"]["role"]}
    document["one_sided"] = rows
    _reject(document)


@pytest.mark.parametrize("change", ("duplicate", "order"))
def test_point_keys_cannot_repeat_or_change_order(change: str) -> None:
    """逐格鍵唯一，且依比較對、接收點、區、頻率排序。"""
    document = _section().model_dump(mode="python")
    points = list(document["points"])
    if change == "duplicate":
        points.insert(0, deepcopy(points[0]))
    else:
        points[0], points[1] = points[1], points[0]
    document["points"] = points
    _reject(document)


@pytest.mark.parametrize("change", ("role", "right_role", "speaker", "walls"))
def test_side_rejects_wrong_role_provenance_or_wall_count(change: str) -> None:
    """來源的角色、喇叭與牆序列必須各自對齊（左右兩邊的角色都要對）。"""
    document = _section().model_dump(mode="python")
    cell = next(p for p in document["points"] if p["state"] is ReflectionAsymmetryState.MEASURED)
    side = cell["left"]
    assert side is not None
    if change == "role":
        side["role"] = cell["right_role"]
    elif change == "right_role":
        assert cell["right"] is not None
        cell["right"]["role"] = cell["left_role"]
    elif change == "speaker":
        side["provenance"]["speaker_id"] = "wrong"
    else:
        side["wall_sequence"] = ()
    _reject(document)


def test_measured_section_rejects_empty_points() -> None:
    """已量整節不能沒有逐格資料。"""
    document = _section().model_dump(mode="python")
    document["points"] = ()
    document["one_sided"] = ()
    _reject(document)


def test_unavailable_section_rejects_points() -> None:
    """不可估整節不得保留逐格值。"""
    document = _section().model_dump(mode="python")
    document["state"] = MetricState.UNAVAILABLE
    document["reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
    _reject(document)


def test_nonfinite_difference_is_rejected() -> None:
    """反射差不可填無限大。"""
    document = _section().model_dump(mode="python")
    cell = next(p for p in document["points"] if p["state"] is ReflectionAsymmetryState.MEASURED)
    cell["left_minus_right_db"] = float("inf")
    _reject(document)


def test_duplicate_comparison_declaration_is_rejected() -> None:
    """比較對的排序依宣告，宣告本身不能重複讓鍵的次序失真。"""
    document = _section().model_dump(mode="python")
    document["comparison_order"] = (*document["comparison_order"], document["comparison_order"][0])
    _reject(document)


def test_channel_matching_payload_requires_reflection_section() -> None:
    """聲道匹配 payload 必須明列診斷節，未交時也要帶不可估節。"""
    from aosr.scoring.contract import ChannelMatchingPayload
    from tests.engine import test_channel_matching as matching

    receivers = matching._receivers()
    group = matching._group()
    evaluation = matching._evaluate(receivers, group, matching._channel_points(receivers, group))
    assert isinstance(evaluation.payload, ChannelMatchingPayload)
    document = evaluation.payload.model_dump(mode="python")
    del document["reflection_asymmetry"]
    with pytest.raises(ValidationError):
        ChannelMatchingPayload.model_validate(document)


def test_point_rejects_same_role_on_both_columns() -> None:
    """比較對左右欄不可是同一個角色。"""
    document = _section().model_dump(mode="python")
    cell = next(p for p in document["points"] if p["state"] is ReflectionAsymmetryState.BOTH_ABSENT)
    cell["right_role"] = cell["left_role"]
    with pytest.raises(ValidationError):
        ReflectionAsymmetryPoint.model_validate(cell)


def test_confirmed_absence_reasons_are_a_closed_pair() -> None:
    """雙側都沒有可以同時帶兩種確認沒有的碼；只帶確認沒有的碼卻記不可估，拒收。"""
    both = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT, ReasonCode.ZERO_REFLECTION_ENERGY)
    document = _section().model_dump(mode="python")
    cell = next(p for p in document["points"] if p["state"] is ReflectionAsymmetryState.BOTH_ABSENT)
    cell["reason_codes"] = both
    assert ReflectionAsymmetryPoint.model_validate(cell).reason_codes == both
    cell["state"] = ReflectionAsymmetryState.UNAVAILABLE
    with pytest.raises(ValidationError):
        ReflectionAsymmetryPoint.model_validate(cell)


@pytest.mark.parametrize("change", ("range_order", "undeclared_pair", "outside_range"))
def test_section_rejects_bad_range_or_undeclared_pair(change: str) -> None:
    """頻率範圍要遞增；逐格的比較對要宣告過、頻率要落在範圍內。"""
    document = _section().model_dump(mode="python")
    low, high = document["frequency_range_hz"]
    if change == "range_order":
        document["frequency_range_hz"] = (high, low)
    elif change == "undeclared_pair":
        document["comparison_order"] = (("right", "left"),)
    else:
        document["frequency_range_hz"] = (low, document["points"][-1]["frequency_hz"] * 0.5)
    _reject(document)
