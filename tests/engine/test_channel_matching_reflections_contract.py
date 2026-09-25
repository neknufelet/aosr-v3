"""反射左右差契約守衛：四態、清單、鍵及來源必須自洽。"""
from __future__ import annotations

from copy import deepcopy

import pytest
from pydantic import ValidationError

from aosr.scoring.channel_matching_reflections_contract import (
    ReflectionAsymmetry, ReflectionAsymmetryState,
)
from aosr.scoring.contract import MetricState, ReasonCode
from tests.engine.test_channel_matching_reflections import _diagnosis, _one_sided


def _document() -> dict:
    return _diagnosis(_one_sided()).model_dump(mode="python")


def _reject(document: dict) -> None:
    with pytest.raises(ValidationError):
        ReflectionAsymmetry.model_validate(document)


@pytest.mark.parametrize("state", tuple(ReflectionAsymmetryState))
def test_each_cell_state_rejects_wrong_value_or_evidence(state: ReflectionAsymmetryState) -> None:
    """四態的差值、兩邊來源與原因碼綁死，不能把缺資料冒充已量。"""
    document = _document()
    cell = next((p for p in document["points"] if p["state"] is state), None)
    if cell is None:
        if state is ReflectionAsymmetryState.UNAVAILABLE:
            cell = deepcopy(document["points"][0])
            cell.update(state=state, left=None, right=None,
                        left_minus_right_db=None, reason_codes=(ReasonCode.INSUFFICIENT_COVERAGE,))
        else:
            pytest.fail(f"fixture missing {state}")
    if state is ReflectionAsymmetryState.MEASURED:
        cell["left_minus_right_db"] = cell["left_minus_right_db"] + 1.0
    elif state is ReflectionAsymmetryState.UNAVAILABLE:
        cell["reason_codes"] = (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)
    else:
        cell["left_minus_right_db"] = 0.0
    from aosr.scoring.channel_matching_reflections_contract import ReflectionAsymmetryPoint
    with pytest.raises(ValidationError):
        ReflectionAsymmetryPoint.model_validate(cell)


@pytest.mark.parametrize("change", ("extra", "missing", "content"))
def test_one_sided_list_must_equal_cells_exactly(change: str) -> None:
    """單側清單多、少、內容不一致都拒收。"""
    document = _document()
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
    document = _document()
    points = list(document["points"])
    if change == "duplicate":
        points.insert(0, deepcopy(points[0]))
    else:
        points[0], points[1] = points[1], points[0]
    document["points"] = points
    _reject(document)


@pytest.mark.parametrize("change", ("role", "speaker", "walls"))
def test_side_rejects_wrong_role_provenance_or_wall_count(change: str) -> None:
    """來源的角色、喇叭與牆序列必須各自對齊。"""
    document = _document()
    cell = next(p for p in document["points"] if p["state"] is ReflectionAsymmetryState.MEASURED)
    side = cell["left"]
    assert side is not None
    if change == "role":
        side["role"] = cell["right_role"]
    elif change == "speaker":
        side["provenance"]["speaker_id"] = "wrong"
    else:
        side["wall_sequence"] = ()
    _reject(document)


def test_measured_section_rejects_empty_points() -> None:
    """已量整節不能沒有逐格資料。"""
    document = _document()
    document["points"] = ()
    document["one_sided"] = ()
    _reject(document)


def test_unavailable_section_rejects_points() -> None:
    """不可估整節不得保留逐格值。"""
    document = _document()
    document["state"] = MetricState.UNAVAILABLE
    document["reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
    _reject(document)


def test_nonfinite_difference_is_rejected() -> None:
    """反射差不可填無限大。"""
    document = _document()
    cell = next(p for p in document["points"] if p["state"] is ReflectionAsymmetryState.MEASURED)
    cell["left_minus_right_db"] = float("inf")
    _reject(document)


def test_duplicate_comparison_declaration_is_rejected() -> None:
    """比較對的排序依宣告，宣告本身不能重複讓鍵的次序失真。"""
    document = _document()
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
