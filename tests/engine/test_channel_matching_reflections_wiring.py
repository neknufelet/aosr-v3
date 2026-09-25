"""反射診斷走聲道匹配入口的整合考卷。"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.channel_matching import (
    CHANNEL_MATCHING_EVALUATOR_VERSION, ChannelComparison, ChannelDefinition,
    ChannelGroup, ChannelPointInput, evaluate_channel_matching,
)
from aosr.scoring.channel_matching_cost import cost_channel_matching_evaluation
from aosr.scoring.channel_matching_reflections import reflection_asymmetry
from aosr.scoring.channel_matching_reflections_contract import (
    ReflectionAsymmetry, ReflectionAsymmetryState,
)
from aosr.scoring.contract import (
    CategoryEvaluation, ChannelComparisonPair, ChannelIdentity, ChannelMatchingPayload,
    EvaluationState, MetricState, ReasonCode,
)
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet
from aosr.scoring.reflections_cost import cost_reflections_evaluation
from tests.engine import test_channel_matching as matching
from tests.engine import test_reflections as fixtures

_Inputs = tuple[ReceiverSet, ChannelGroup, tuple[ChannelPointInput, ...]]

def _section_of(evaluation: CategoryEvaluation) -> ReflectionAsymmetry:
    assert isinstance(evaluation.payload, ChannelMatchingPayload)
    return evaluation.payload.reflection_asymmetry


def _integrated_inputs(upstream: CategoryEvaluation) -> _Inputs:
    """用反射報表的實際擺位給聲道匹配的音色樣本，完整走兩個入口。"""
    receivers_doc = matching._receivers().model_dump(mode="python")
    positions = dict(upstream.placement.receiver_positions_m)
    for point in receivers_doc["points"]:
        point["position_m"] = positions[point["receiver_id"]]
    receivers = ReceiverSet.model_validate(receivers_doc)
    group = ChannelGroup(
        channels=(ChannelDefinition(role="left", speaker_id="left"),
                  ChannelDefinition(role="right", speaker_id="right")),
        comparisons=matching._group().comparisons, feature_match_tolerance_hz=10.0,
    )
    speakers = dict(upstream.placement.speaker_positions_m)
    points = []
    for receiver in ("main", "front"):
        document = matching._point(receivers, group, receiver).model_dump(mode="python")
        for response in document["responses"]:
            role = response["role"]
            timbre = response["timbre_evaluation"]
            timbre["scene_fingerprint"] = upstream.scene_fingerprint
            timbre["provenance"]["speaker_id"] = role
            timbre["placement"] = {
                "speaker_positions_m": [(role, speakers[role])],
                "receiver_positions_m": [(receiver, positions[receiver])],
            }
        points.append(ChannelPointInput.model_validate(document))
    return receivers, group, tuple(points)


def _integrated_upstream() -> CategoryEvaluation:
    room = {"Lx": 6.0, "Ly": 5.0, "Lz": 3.0}
    records = tuple(fixtures._record(role, y, receiver=receiver, room=room,
                                     receiver_y=1.2, receiver_x=x)
                    for role, y in (("left", 0.6), ("right", 1.8))
                    for receiver, x in (("main", 3.2), ("front", 4.4)))
    return fixtures._evaluate(records)


def _matching_with(reflections: CategoryEvaluation | None, inputs: _Inputs,
                   scene_fingerprint: str) -> CategoryEvaluation:
    receivers, group, points = inputs
    return evaluate_channel_matching(
        receivers, points, candidate_id="candidate-a",
        scene_fingerprint=scene_fingerprint,
        timbre_settings_fingerprint=matching._TIMBRE_SETTINGS,
        listening_area_settings_fingerprint=matching._LISTENING_SETTINGS,
        channel_group=group, purpose=matching._PURPOSE,
        quality_targets_path=matching._TARGETS, sound_speed_m_s=343.0,
        reflections=reflections,
    )


def _integrated_evaluate(upstream: CategoryEvaluation, inputs: _Inputs) -> CategoryEvaluation:
    return _matching_with(upstream, inputs, upstream.scene_fingerprint)


def test_missing_point_does_not_make_whole_channel_matching_unavailable() -> None:
    """主對話約定（見決策紙）：單側改為缺資料只讓那格 unavailable，聲道匹配整類仍已量。"""
    upstream = _integrated_upstream()
    inputs = _integrated_inputs(upstream)
    baseline = _integrated_evaluate(upstream, inputs)
    assert baseline.state is EvaluationState.MEASURED
    section = _section_of(baseline)
    one = next(row for row in section.one_sided if row.receiver_id == "front")
    document = upstream.model_dump(mode="python")
    for channel in document["payload"]["channels"]:
        if (channel["role"], channel["receiver_id"]) != (one.absent_role, "front"):
            continue
        for zone in channel["zones"]:
            if zone["zone"] == one.zone:
                for point in zone["points"]:
                    if point["frequency_hz"] == one.frequency_hz:
                        point["strongest_reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
                        point["strongest_state"] = MetricState.UNAVAILABLE
    changed = CategoryEvaluation.model_validate(document)
    result = _integrated_evaluate(changed, inputs)
    assert result.state is EvaluationState.MEASURED
    after = _section_of(result)
    assert after.state is MetricState.MEASURED
    assert tuple(p for p in after.points if p.receiver_id == "main") == tuple(
        p for p in section.points if p.receiver_id == "main")
    assert next(p for p in after.points if (p.receiver_id, p.zone, p.frequency_hz)
                == ("front", one.zone, one.frequency_hz)).state is ReflectionAsymmetryState.UNAVAILABLE
    assert not any((row.receiver_id, row.zone, row.frequency_hz)
                   == ("front", one.zone, one.frequency_hz) for row in after.one_sided)


def test_reflections_only_change_fingerprint_not_matching_cost_or_flags() -> None:
    """老闆決定（見決策紙）：只做診斷，不進代價。主對話約定（見決策紙）：有交沒交反射的指紋不同。"""
    upstream = _integrated_upstream()
    inputs = _integrated_inputs(upstream)
    missing = _matching_with(None, inputs, upstream.scene_fingerprint)
    present = _matching_with(upstream, inputs, upstream.scene_fingerprint)
    assert CHANNEL_MATCHING_EVALUATOR_VERSION == "aosr.scoring.channel_matching.v6"
    assert missing.settings_fingerprint != present.settings_fingerprint
    assert missing.state == present.state is EvaluationState.MEASURED
    assert missing.raw_quantities == present.raw_quantities
    assert missing.flags == present.flags
    assert _section_of(missing).state is MetricState.UNAVAILABLE
    assert _section_of(present).state is MetricState.MEASURED
    assert _section_of(present).source_flags == upstream.flags
    registry = load_quality_targets(matching._TARGETS)
    purpose = registry.purpose(matching._PURPOSE)
    a = cost_channel_matching_evaluation(missing, purpose, registry.fingerprint)
    b = cost_channel_matching_evaluation(present, purpose, registry.fingerprint)
    assert a.category_cost == b.category_cost
    changed = upstream.model_dump(mode="python")
    changed["settings_fingerprint"] = "different-reflection-settings"
    alternate = CategoryEvaluation.model_validate(changed)
    other = _matching_with(alternate, inputs, upstream.scene_fingerprint)
    assert other.settings_fingerprint != present.settings_fingerprint


def test_missing_whole_reflection_channel_keeps_matching_measured() -> None:
    """老闆決定（見決策紙）：不因單側存在而把整類判不可估。主對話約定（見決策紙）：整支缺失也不外溢。"""
    upstream = _integrated_upstream()
    inputs = _integrated_inputs(upstream)
    baseline = _integrated_evaluate(upstream, inputs)
    original = _section_of(baseline)
    row = next(item for item in original.one_sided if item.receiver_id == "front")
    document = upstream.model_dump(mode="python")
    channel = next(ch for ch in document["payload"]["channels"]
                   if (ch["role"], ch["receiver_id"]) == (row.absent_role, "front"))
    channel.update(state=MetricState.UNAVAILABLE,
                   reason_codes=(ReasonCode.REFLECTION_WINDOW_INCOMPLETE,),
                   coverage="not_provable")
    for zone in channel["zones"]:
        zone["points"] = ()
    channel["total_window_energy_db"] = ()
    result = _integrated_evaluate(CategoryEvaluation.model_validate(document), inputs)
    assert result.state is EvaluationState.MEASURED
    section = _section_of(result)
    assert section.state is MetricState.MEASURED
    assert tuple(p for p in section.points if p.receiver_id == "main") == tuple(
        p for p in original.points if p.receiver_id == "main")
    front = [p for p in section.points if p.receiver_id == "front"]
    assert front
    assert all(p.state is ReflectionAsymmetryState.UNAVAILABLE for p in front)
    assert all(row.receiver_id != "front" for row in section.one_sided)


def test_changed_reflections_window_registry_changes_matching_fingerprint(tmp_path: Path) -> None:
    """反射量法時間窗改變時，聲道匹配的設定指紋也跟著變。"""
    upstream = _integrated_upstream()
    receivers, group, points = _integrated_inputs(upstream)
    original = _integrated_evaluate(upstream, (receivers, group, points))
    source = config_path("quality_targets.toml").read_text()
    key = "reflections_and_echo.window_upper_ms"
    pattern = rf'(key = "{re.escape(key)}"\nvalue = )([^\n]+)'
    changed, count = re.subn(pattern, lambda match: match.group(1) + str(float(match.group(2)) + 1.0), source)
    assert count == 1 and changed != source
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed)
    room = {"Lx": 6.0, "Ly": 5.0, "Lz": 3.0}
    records = tuple(fixtures._record(role, y, receiver=receiver, room=room,
                                     receiver_y=1.2, receiver_x=x, window_s=0.016)
                    for role, y in (("left", 0.6), ("right", 1.8))
                    for receiver, x in (("main", 3.2), ("front", 4.4)))
    changed_upstream = fixtures._evaluate(records, path)
    assert upstream.settings_fingerprint != changed_upstream.settings_fingerprint
    changed_result = evaluate_channel_matching(
        receivers, points, candidate_id="candidate-a",
        scene_fingerprint=upstream.scene_fingerprint,
        timbre_settings_fingerprint=matching._TIMBRE_SETTINGS,
        listening_area_settings_fingerprint=matching._LISTENING_SETTINGS,
        channel_group=group, purpose=matching._PURPOSE,
        quality_targets_path=matching._TARGETS, sound_speed_m_s=343.0,
        reflections=changed_upstream,
    )
    assert changed_result.settings_fingerprint != original.settings_fingerprint




@pytest.mark.parametrize(("change", "reason"), (
    ("candidate", ReasonCode.CANDIDATE_ID_MISMATCH),
    ("scene", ReasonCode.SCENE_FINGERPRINT_MISMATCH),
    ("placement", ReasonCode.PLACEMENT_MISMATCH),
))
def test_wiring_uses_matching_identity_not_reflections_identity(
    change: str, reason: ReasonCode,
) -> None:
    """入口把聲道匹配自己的身分交給診斷。"""
    upstream = _integrated_upstream()
    inputs = _integrated_inputs(upstream)
    document = upstream.model_dump(mode="python")
    if change == "candidate":
        document["candidate_id"] = "another-candidate"
    elif change == "scene":
        document["scene_fingerprint"] = "b" * 64
    else:
        speakers = list(document["placement"]["speaker_positions_m"])
        name, coordinate = speakers[0]
        speakers[0] = (name, (coordinate[0] + 0.1, *coordinate[1:]))
        document["placement"]["speaker_positions_m"] = speakers
    changed = CategoryEvaluation.model_validate(document)
    result = _matching_with(changed, inputs, upstream.scene_fingerprint)
    assert result.state is EvaluationState.MEASURED
    section = _section_of(result)
    assert section.state is MetricState.UNAVAILABLE
    assert section.reason_codes == (reason,)


def test_other_seat_does_not_require_reflection_channel() -> None:
    """其他座位不在該量接收點集合。"""
    upstream = _integrated_upstream()
    receivers, group, points = _integrated_inputs(upstream)
    document = receivers.model_dump(mode="python")
    other = dict(document["points"][1])
    other.update(receiver_id="other", position_m=(4.0, 2.0, 1.2),
                 role=ReceiverRole.OTHER_SEAT, direction_relative_to_primary=None)
    document["points"] = (*document["points"], other)
    expanded = ReceiverSet.model_validate(document)
    revised = []
    for point in points:
        row = point.model_dump(mode="python")
        row["receiver_set_fingerprint"] = expanded.fingerprint
        revised.append(ChannelPointInput.model_validate(row))
    result = _integrated_evaluate(upstream, (expanded, group, tuple(revised)))
    assert result.state is EvaluationState.MEASURED
    assert _section_of(result).state is MetricState.MEASURED


def test_reflections_parameter_is_required_keyword_only() -> None:
    """反射輸入必須由呼叫端明交。"""
    parameter = inspect.signature(evaluate_channel_matching).parameters["reflections"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_wrong_category_is_rejected_before_matching_unavailability() -> None:
    """即使聲道匹配身分錯，也在入口拒收錯類。"""
    upstream = _integrated_upstream()
    inputs = _integrated_inputs(upstream)
    wrong = matching._evaluate(matching._receivers(), matching._group(),
                               matching._channel_points(matching._receivers(), matching._group()))
    with pytest.raises(ValueError, match="必須交反射類評估"):
        _matching_with(wrong, inputs, "b" * 64)


def test_costed_reflections_are_rejected_at_matching_entry() -> None:
    """已算代價的上游不可再充當原始診斷。"""
    upstream = _integrated_upstream()
    registry = load_quality_targets(config_path("quality_targets.toml"))
    costed = cost_reflections_evaluation(
        upstream, registry.purpose(matching._PURPOSE), registry.fingerprint)
    assert costed.state is EvaluationState.COSTED
    with pytest.raises(ValueError, match="已算代價"):
        _integrated_evaluate(costed, _integrated_inputs(upstream))
    with pytest.raises(ValueError, match="已算代價"):
        reflection_asymmetry(
            costed, candidate_id=upstream.candidate_id,
            scene_fingerprint=upstream.scene_fingerprint,
            channels=(ChannelIdentity(role="left", speaker_id="left"),
                      ChannelIdentity(role="right", speaker_id="right")),
            comparisons=(ChannelComparisonPair(left_role="left", right_role="right"),),
            receiver_ids=("main", "front"), primary_receiver_id="main",
            placement=upstream.placement)


def test_reflection_evaluator_version_changes_matching_fingerprint() -> None:
    """設定指紋也記反射評估器版本。"""
    upstream = _integrated_upstream()
    inputs = _integrated_inputs(upstream)
    original = _integrated_evaluate(upstream, inputs)
    document = upstream.model_dump(mode="python")
    document["evaluator_version"] = "older-reflections"
    changed = CategoryEvaluation.model_validate(document)
    result = _integrated_evaluate(changed, inputs)
    assert result.settings_fingerprint != original.settings_fingerprint


def test_unavailable_channel_with_absence_only_reason_stays_measured() -> None:
    """不可估聲道只寫確認沒有時，逐格補缺值碼。"""
    upstream = _integrated_upstream()
    document = upstream.model_dump(mode="python")
    changed = False
    for channel in document["payload"]["channels"]:
        if channel["receiver_id"] == "front" and channel["role"] == "right":
            channel.update(state=MetricState.UNAVAILABLE,
                           reason_codes=(ReasonCode.NO_REFLECTION_IN_ZONE_POINT,))
            for zone in channel["zones"]:
                zone["points"] = ()
            channel["total_window_energy_db"] = ()
            changed = True
    assert changed
    result = _integrated_evaluate(CategoryEvaluation.model_validate(document),
                                  _integrated_inputs(upstream))
    assert result.state is EvaluationState.MEASURED
    section = _section_of(result)
    assert section.state is MetricState.MEASURED
    front = [point for point in section.points if point.receiver_id == "front"]
    assert front
    assert all(point.state is ReflectionAsymmetryState.UNAVAILABLE and
               point.reason_codes == (ReasonCode.CHANNEL_RESULT_UNAVAILABLE,
                                      ReasonCode.NO_REFLECTION_IN_ZONE_POINT)
               for point in front)


def test_no_measured_channel_frequency_points_makes_section_unavailable() -> None:
    """全部已量聲道的頻點皆空時，整節須報覆蓋不足。"""
    upstream = _integrated_upstream()
    document = upstream.model_dump(mode="python")
    changed = False
    for channel in document["payload"]["channels"]:
        if channel["state"] is MetricState.MEASURED:
            for zone in channel["zones"]:
                changed |= bool(zone["points"])
                zone["points"] = ()
            channel["total_window_energy_db"] = ()
    assert changed
    result = _integrated_evaluate(CategoryEvaluation.model_validate(document),
                                  _integrated_inputs(upstream))
    assert result.state is EvaluationState.MEASURED
    section = _section_of(result)
    assert section.state is MetricState.UNAVAILABLE
    assert section.reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)
