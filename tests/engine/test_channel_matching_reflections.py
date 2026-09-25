"""反射左右差診斷：老闆第 3 格的原始 L−R、交換標籤與缺值語意。"""
from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.scoring.channel_matching import (
    CHANNEL_MATCHING_EVALUATOR_VERSION, ChannelComparison, ChannelDefinition, ChannelGroup,
    ChannelPointInput, evaluate_channel_matching,
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
from aosr.scoring.direction_zones import DirectionZone
from aosr.scoring.placement import Placement
from aosr.scoring.receiver_set import ReceiverSet
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from aosr.scoring.reflections_cost import cost_reflections_evaluation
from tests.engine import test_channel_matching as matching
from tests.engine import test_reflections as fixtures


_PAIR = (ChannelComparisonPair(left_role="left", right_role="right"),)
_CHANNELS = (ChannelIdentity(role="left", speaker_id="left"),
             ChannelIdentity(role="right", speaker_id="right"))
_Inputs = tuple[ReceiverSet, ChannelGroup, tuple[ChannelPointInput, ...]]


def _section_of(evaluation: CategoryEvaluation) -> ReflectionAsymmetry:
    assert isinstance(evaluation.payload, ChannelMatchingPayload)
    return evaluation.payload.reflection_asymmetry


def _diagnosis(upstream: CategoryEvaluation, *, channels: tuple[ChannelIdentity, ...] = _CHANNELS,
               comparisons: tuple[ChannelComparisonPair, ...] = _PAIR,
               receiver_ids: tuple[str, ...] = ("main",)) -> ReflectionAsymmetry:
    return reflection_asymmetry(
        upstream, candidate_id=upstream.candidate_id,
        scene_fingerprint=upstream.scene_fingerprint,
        channels=channels, comparisons=comparisons, receiver_ids=receiver_ids,
        primary_receiver_id="main", placement=upstream.placement,
    )


def _swapped(records: tuple[ReflectionInput, ...]) -> CategoryEvaluation:
    group = ChannelGroup(
        channels=(ChannelDefinition(role="left", speaker_id="right"),
                  ChannelDefinition(role="right", speaker_id="left")),
        comparisons=(ChannelComparison(left_role="left", right_role="right"),),
        feature_match_tolerance_hz=0.0,
    )
    return evaluate_reflections(
        group, tuple(replace(item, role="right" if item.role == "left" else "left") for item in records),
        primary_receiver_id="main", candidate_id="candidate-a",
        purpose="dedicated_two_channel_listening_room",
        quality_targets_path=config_path("quality_targets.toml"),
    )


def _one_sided() -> CategoryEvaluation:
    room = {"Lx": 6.0, "Ly": 5.0, "Lz": 3.0}
    return fixtures._evaluate((
        fixtures._record("left", 0.6, room=room, receiver_y=1.2, receiver_x=3.2),
        fixtures._record("right", 1.8, room=room, receiver_y=1.2, receiver_x=3.2),
    ))


def _changed_level(upstream: CategoryEvaluation, role: str, zone: DirectionZone,
                   frequency: float, level: float) -> CategoryEvaluation:
    document = upstream.model_dump(mode="python")
    payload = document["payload"]
    changed = False
    for channel in payload["channels"]:
        if channel["role"] != role:
            continue
        for zone_row in channel["zones"]:
            if zone_row["zone"] != zone:
                continue
            for point in zone_row["points"]:
                if point["frequency_hz"] == frequency:
                    assert point["strongest_level_db"] is not None
                    point["strongest_level_db"] = level
                    changed = True
    assert changed, "找不到要改的那一點，這題會變成沒在測"
    return CategoryEvaluation.model_validate(document)


def test_measured_cells_keep_signed_left_minus_right_from_raw_payload() -> None:
    """原話：最強單條反射聲級做 L−R，保留正負號。"""
    upstream = fixtures._evaluate(fixtures._pair())
    assert isinstance(upstream.payload, ReflectionsAndEchoPayload)
    actual = _diagnosis(upstream)
    assert actual.state is MetricState.MEASURED
    expected_keys = {
        (pair.left_role, pair.right_role, channel.receiver_id, zone.zone, point.frequency_hz)
        for pair in _PAIR for channel in upstream.payload.channels if channel.role == pair.left_role
        for zone in channel.zones for point in zone.points
    }
    assert {(p.left_role, p.right_role, p.receiver_id, p.zone, p.frequency_hz) for p in actual.points} == expected_keys
    by_role = {channel.role: channel for channel in upstream.payload.channels}
    for cell in actual.points:
        left = next(p for z in by_role[cell.left_role].zones if z.zone is cell.zone
                    for p in z.points if p.frequency_hz == cell.frequency_hz)
        right = next(p for z in by_role[cell.right_role].zones if z.zone is cell.zone
                     for p in z.points if p.frequency_hz == cell.frequency_hz)
        if left.strongest_level_db is not None and right.strongest_level_db is not None:
            assert cell.left_minus_right_db == left.strongest_level_db - right.strongest_level_db
        else:
            assert cell.left_minus_right_db is None


def test_below_threshold_levels_still_keep_difference() -> None:
    """原話：兩邊都低於門檻時，差異仍須保留。"""
    upstream = fixtures._evaluate(fixtures._pair())
    section = _diagnosis(upstream)
    cell = next(p for p in section.points if p.state is ReflectionAsymmetryState.MEASURED)
    registry = load_quality_targets(config_path("quality_targets.toml"))
    entry = registry.purpose("dedicated_two_channel_listening_room").entry(
        f"reflections_and_echo.zone_threshold_db.{cell.zone.value}")
    assert isinstance(entry, SettingEntry) and isinstance(entry.value, float)
    threshold = entry.value
    changed = _changed_level(upstream, cell.left_role, cell.zone, cell.frequency_hz, threshold - 2)
    changed = _changed_level(changed, cell.right_role, cell.zone, cell.frequency_hz, threshold - 4)
    found = next(p for p in _diagnosis(changed).points if p.zone is cell.zone and p.frequency_hz == cell.frequency_hz)
    assert found.state is ReflectionAsymmetryState.MEASURED
    assert found.left_minus_right_db == (threshold - 2) - (threshold - 4)
    assert found.left_minus_right_db != 0.0


def test_swapping_only_labels_negates_each_measured_difference() -> None:
    """原話：交換左右後差值反號；純換標籤，實體報表不動。"""
    original = _diagnosis(fixtures._evaluate(fixtures._pair()))
    changed = _diagnosis(_swapped(fixtures._pair()), channels=(
        ChannelIdentity(role="left", speaker_id="right"),
        ChannelIdentity(role="right", speaker_id="left"),
    ))
    measured = [p for p in original.points if p.state is ReflectionAsymmetryState.MEASURED]
    assert measured
    for old, new in zip(original.points, changed.points, strict=True):
        if old.state is ReflectionAsymmetryState.MEASURED:
            assert new.state is ReflectionAsymmetryState.MEASURED
            assert old.left_minus_right_db is not None
            assert new.left_minus_right_db == -old.left_minus_right_db


def test_one_sided_and_both_absent_remain_distinct_with_source() -> None:
    """原話：單側有反射另列延遲及來源；雙側均無反射要分開記。"""
    upstream = _one_sided()
    section = _diagnosis(upstream)
    assert section.state is MetricState.MEASURED
    assert section.one_sided
    assert any(p.state is ReflectionAsymmetryState.BOTH_ABSENT for p in section.points)
    for cell in section.points:
        if cell.state is ReflectionAsymmetryState.ONE_SIDED:
            row = next(row for row in section.one_sided if (row.receiver_id, row.zone, row.frequency_hz)
                       == (cell.receiver_id, cell.zone, cell.frequency_hz))
            assert row.present.delay_s >= 0.0
            assert row.present.provenance.report_id
            assert row.present.wall_sequence
            assert cell.left_minus_right_db is None
        elif cell.state is ReflectionAsymmetryState.BOTH_ABSENT:
            assert cell.left_minus_right_db is None
            assert not any((row.receiver_id, row.zone, row.frequency_hz)
                           == (cell.receiver_id, cell.zone, cell.frequency_hz) for row in section.one_sided)


def test_swapping_one_sided_moves_present_role_but_keeps_path_evidence() -> None:
    """原話：交換左右後有反射側別交換，聲級不變、差值仍空。"""
    room = {"Lx": 6.0, "Ly": 5.0, "Lz": 3.0}
    records = (fixtures._record("left", 0.6, room=room, receiver_y=1.2),
               fixtures._record("right", 1.8, room=room, receiver_y=1.2))
    original = _diagnosis(fixtures._evaluate(records))
    swapped = _diagnosis(_swapped(records), channels=(
        ChannelIdentity(role="left", speaker_id="right"),
        ChannelIdentity(role="right", speaker_id="left"),
    ))
    assert original.one_sided
    for old in original.one_sided:
        new = next(row for row in swapped.one_sided if (row.receiver_id, row.zone, row.frequency_hz)
                   == (old.receiver_id, old.zone, old.frequency_hz))
        assert new.present.role != old.present.role
        assert new.absent_role == old.present.role
        assert new.present.speaker_id == old.present.speaker_id
        assert new.present.level_db == old.present.level_db
        assert new.present.delay_s == old.present.delay_s
        assert (new.present.source, new.present.source_index, new.present.wall_sequence) == (
            old.present.source, old.present.source_index, old.present.wall_sequence)
        assert next(p for p in swapped.points if (p.receiver_id, p.zone, p.frequency_hz)
                    == (new.receiver_id, new.zone, new.frequency_hz)).left_minus_right_db is None


def test_unknown_point_reason_is_unavailable_and_leaves_one_sided_list() -> None:
    """原話：確認沒有改成資料缺失，改判不可估、不能留在單側清單。"""
    upstream = _one_sided()
    original = _diagnosis(upstream)
    one = original.one_sided[0]
    document = upstream.model_dump(mode="python")
    for channel in document["payload"]["channels"]:
        if channel["role"] != one.absent_role:
            continue
        for zone in channel["zones"]:
            if zone["zone"] != one.zone:
                continue
            for point in zone["points"]:
                if point["frequency_hz"] == one.frequency_hz:
                    point["strongest_reason_codes"] = [ReasonCode.INSUFFICIENT_COVERAGE]
                    point["strongest_state"] = MetricState.UNAVAILABLE
    changed = CategoryEvaluation.model_validate(document)
    section = _diagnosis(changed)
    cell = next(p for p in section.points if (p.zone, p.frequency_hz) == (one.zone, one.frequency_hz))
    assert cell.state is ReflectionAsymmetryState.UNAVAILABLE
    assert cell.left_minus_right_db is None
    assert ReasonCode.INSUFFICIENT_COVERAGE in cell.reason_codes
    assert not any((row.zone, row.frequency_hz) == (one.zone, one.frequency_hz) for row in section.one_sided)
    assert section.state is MetricState.MEASURED


def test_threshold_only_changes_cost_not_raw_diagnosis(tmp_path: Path) -> None:
    """原話：只改超標門檻，原始診斷完全不變。"""
    original_text = config_path("quality_targets.toml").read_text()
    changed_text = original_text
    for zone in DirectionZone:
        key = f"reflections_and_echo.zone_threshold_db.{zone.value}"
        pattern = rf'(key = "{re.escape(key)}"\nvalue = )([^\n]+)'
        changed_text, count = re.subn(pattern, lambda match: match.group(1) + str(float(match.group(2)) - 20.0), changed_text)
        assert count == 1
    assert changed_text != original_text
    path = tmp_path / "quality_targets.toml"
    path.write_text(changed_text)
    before = fixtures._evaluate(fixtures._pair())
    after = fixtures._evaluate(fixtures._pair(), path)
    assert _diagnosis(before) == _diagnosis(after)
    registry_a = load_quality_targets(config_path("quality_targets.toml"))
    registry_b = load_quality_targets(path)
    cost_a = cost_reflections_evaluation(before, registry_a.purpose("dedicated_two_channel_listening_room"), registry_a.fingerprint)
    cost_b = cost_reflections_evaluation(after, registry_b.purpose("dedicated_two_channel_listening_room"), registry_b.fingerprint)
    assert cost_a.category_cost != cost_b.category_cost


def test_reverse_comparison_uses_declared_left_column() -> None:
    """比較對方向寫成右、左時，差值以右欄減左欄。"""
    upstream = fixtures._evaluate(fixtures._pair())
    forward = _diagnosis(upstream)
    reverse = _diagnosis(upstream, comparisons=(ChannelComparisonPair(left_role="right", right_role="left"),))
    for old, new in zip(forward.points, reverse.points, strict=True):
        assert (new.left_role, new.right_role) == (old.right_role, old.left_role)
        if old.state is ReflectionAsymmetryState.MEASURED:
            assert new.state is ReflectionAsymmetryState.MEASURED
            assert old.left_minus_right_db is not None
            assert new.left_minus_right_db == -old.left_minus_right_db


def test_scoring_layers_do_not_import_reflection_diagnostic() -> None:
    """原話：只做診斷，不進代價；排名與代價檔不得取這兩個模組。"""
    root = Path(__file__).resolve().parents[2] / "src" / "aosr" / "scoring"
    files = (root / "ranking.py", root / "category_registry.py", *root.glob("*_cost.py"))
    for path in files:
        imports = (node.module for node in ast.walk(ast.parse(path.read_text()))
                   if isinstance(node, ast.ImportFrom))
        assert all(module not in {"aosr.scoring.channel_matching_reflections",
                                  "aosr.scoring.channel_matching_reflections_contract"}
                   for module in imports)


def _two_receiver() -> CategoryEvaluation:
    room = {"Lx": 6.0, "Ly": 5.0, "Lz": 3.0}
    records = tuple(fixtures._record(role, y, receiver=receiver, room=room,
                                     receiver_y=1.2, receiver_x=x)
                    for role, y in (("left", 0.6), ("right", 1.8))
                    for receiver, x in (("main", 3.2), ("surround", 4.4)))
    return fixtures._evaluate(records)


def test_whole_missing_channel_makes_its_cells_unavailable_without_losing_main() -> None:
    """原話：確認沒有改成聲道資料缺失後，不留單側清單；主位不受影響。"""
    upstream = _two_receiver()
    before = _diagnosis(upstream, receiver_ids=("main", "surround"))
    one = next(row for row in before.one_sided if row.receiver_id == "surround")
    document = upstream.model_dump(mode="python")
    channel = next(ch for ch in document["payload"]["channels"]
                   if ch["role"] == one.absent_role and ch["receiver_id"] == "surround")
    channel["state"] = MetricState.UNAVAILABLE
    channel["reason_codes"] = (ReasonCode.REFLECTION_WINDOW_INCOMPLETE,)
    channel["coverage"] = "not_provable"
    for zone in channel["zones"]:
        zone["points"] = ()
    channel["total_window_energy_db"] = ()
    changed = CategoryEvaluation.model_validate(document)
    after = _diagnosis(changed, receiver_ids=("main", "surround"))
    assert after.state is MetricState.MEASURED
    assert tuple(p for p in before.points if p.receiver_id == "main") == tuple(
        p for p in after.points if p.receiver_id == "main")
    assert all(p.state is ReflectionAsymmetryState.UNAVAILABLE and
               ReasonCode.REFLECTION_WINDOW_INCOMPLETE in p.reason_codes
               for p in after.points if p.receiver_id == "surround")
    assert all(row.receiver_id != "surround" for row in after.one_sided)


def test_points_left_in_an_unavailable_channel_are_not_trusted() -> None:
    """原話：已證明兩邊時間窗完整才算單側；不可估聲道裡殘留的點不能當成有反射。"""
    upstream = _two_receiver()
    before = _diagnosis(upstream, receiver_ids=("main", "surround"))
    one = next(row for row in before.one_sided if row.receiver_id == "surround")
    document = upstream.model_dump(mode="python")
    channel = next(ch for ch in document["payload"]["channels"]
                   if ch["role"] == one.present.role and ch["receiver_id"] == "surround")
    channel["state"] = MetricState.UNAVAILABLE
    channel["reason_codes"] = (ReasonCode.REFLECTION_WINDOW_INCOMPLETE,)
    channel["coverage"] = "not_provable"
    after = _diagnosis(CategoryEvaluation.model_validate(document), receiver_ids=("main", "surround"))
    surround = [p for p in after.points if p.receiver_id == "surround"]
    assert surround
    assert all(p.state is ReflectionAsymmetryState.UNAVAILABLE and p.left is None and p.right is None
               and ReasonCode.REFLECTION_WINDOW_INCOMPLETE in p.reason_codes for p in surround)
    assert all(row.receiver_id != "surround" for row in after.one_sided)
    assert after.state is MetricState.MEASURED


def test_unavailable_channel_with_complete_window_is_not_confirmed_absence() -> None:
    """原話：確認沒有要兩邊時間窗已證明完整、那一支本身可估；整支不可估的聲道即使覆蓋寫完整、
    點上寫沒有反射，也只能記不可估（原因抄那一支的），不進單側清單。"""
    upstream = _two_receiver()
    before = _diagnosis(upstream, receiver_ids=("main", "surround"))
    one = next(row for row in before.one_sided if row.receiver_id == "surround")
    document = upstream.model_dump(mode="python")
    channel = next(ch for ch in document["payload"]["channels"]
                   if ch["role"] == one.absent_role and ch["receiver_id"] == "surround")
    assert channel["coverage"] == "complete"
    channel["state"] = MetricState.UNAVAILABLE
    channel["reason_codes"] = (ReasonCode.PLACEMENT_MISMATCH,)
    after = _diagnosis(CategoryEvaluation.model_validate(document), receiver_ids=("main", "surround"))
    cell = next(p for p in after.points
                if (p.receiver_id, p.zone, p.frequency_hz) == ("surround", one.zone, one.frequency_hz))
    assert cell.state is ReflectionAsymmetryState.UNAVAILABLE
    assert cell.reason_codes == (ReasonCode.PLACEMENT_MISMATCH,)
    assert all(row.receiver_id != "surround" for row in after.one_sided)
    assert after.state is MetricState.MEASURED


def test_zero_reflection_energy_counts_as_confirmed_absence() -> None:
    """零能量路徑與另一邊已量時仍是單側，而非補 0 dB。"""
    upstream = fixtures._evaluate(fixtures._pair())
    cell = next(p for p in _diagnosis(upstream).points if p.state is ReflectionAsymmetryState.MEASURED)
    document = upstream.model_dump(mode="python")
    right = next(ch for ch in document["payload"]["channels"] if ch["role"] == cell.right_role)
    zone = next(z for z in right["zones"] if z["zone"] == cell.zone)
    point = next(p for p in zone["points"] if p["frequency_hz"] == cell.frequency_hz)
    assert point["strongest_path_index"] is not None
    point["strongest_level_db"] = None
    point["strongest_state"] = MetricState.UNAVAILABLE
    point["strongest_reason_codes"] = (ReasonCode.ZERO_REFLECTION_ENERGY, ReasonCode.NO_REFLECTION_IN_ZONE_POINT)
    point["total_energy_db"] = {"value": None, "state": MetricState.UNAVAILABLE,
                                "reason_codes": (ReasonCode.ZERO_REFLECTION_ENERGY,)}
    changed = CategoryEvaluation.model_validate(document)
    found = next(p for p in _diagnosis(changed).points if (p.zone, p.frequency_hz)
                 == (cell.zone, cell.frequency_hz))
    assert found.state is ReflectionAsymmetryState.ONE_SIDED
    assert found.left_minus_right_db is None
    assert found.reason_codes == (ReasonCode.ZERO_REFLECTION_ENERGY, ReasonCode.NO_REFLECTION_IN_ZONE_POINT)


@pytest.mark.parametrize(("change", "expected"), (
    ("version", ReasonCode.EVALUATOR_VERSION_MISMATCH),
    ("candidate", ReasonCode.CANDIDATE_ID_MISMATCH),
    ("scene", ReasonCode.SCENE_FINGERPRINT_MISMATCH),
    ("upstream", ReasonCode.INSUFFICIENT_COVERAGE),
    ("role", ReasonCode.CHANNEL_ROLE_MISMATCH),
    ("speaker", ReasonCode.SPEAKER_ID_MISMATCH),
    ("receiver", ReasonCode.RECEIVER_ID_MISMATCH),
    ("primary", ReasonCode.RECEIVER_ID_MISMATCH),
    ("placement", ReasonCode.PLACEMENT_MISMATCH),
    ("pair_role", ReasonCode.CHANNEL_ROLE_MISMATCH),
    ("placement_speakers", ReasonCode.PLACEMENT_MISMATCH),
    ("placement_primary", ReasonCode.PLACEMENT_MISMATCH),
))
def test_identity_or_upstream_failure_only_marks_section_unavailable(
    change: str, expected: ReasonCode,
) -> None:
    """上游身分與可估性錯誤各有原因，不外溢到聲道匹配整類。

    擺位要兩份都列出每支喇叭與主位、同代號座標逐位相同；只比衝突的話，代號整組換掉也會被收下。
    """
    upstream = fixtures._evaluate(fixtures._pair())
    candidate_id, scene_fingerprint = upstream.candidate_id, upstream.scene_fingerprint
    channels: tuple[ChannelIdentity, ...] = _CHANNELS
    receiver_ids: tuple[str, ...] = ("main",)
    primary_receiver_id, placement = "main", upstream.placement
    comparisons: tuple[ChannelComparisonPair, ...] = _PAIR
    if change in {"version", "upstream"}:
        document = upstream.model_dump(mode="python")
        if change == "version":
            document["evaluator_version"] = "older-reflections"
        else:
            document["state"] = EvaluationState.UNAVAILABLE
            document["payload"] = None
            document["raw_quantities"] = ()
            document["reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
        upstream = CategoryEvaluation.model_validate(document)
    elif change == "candidate":
        candidate_id = "other-candidate"
    elif change == "scene":
        scene_fingerprint = "b" * 64
    elif change == "role":
        channels = (ChannelIdentity(role="other", speaker_id="left"), _CHANNELS[1])
    elif change == "speaker":
        channels = (ChannelIdentity(role="left", speaker_id="other"), _CHANNELS[1])
    elif change == "receiver":
        receiver_ids = ("other",)
    elif change == "primary":
        primary_receiver_id = "other"
    elif change == "pair_role":
        comparisons = (ChannelComparisonPair(left_role="left", right_role="center"),)
    elif change in {"placement_speakers", "placement_primary"}:
        document = upstream.placement.model_dump(mode="python")
        column = "speaker_positions_m" if change == "placement_speakers" else "receiver_positions_m"
        renamed = tuple((f"renamed-{name}", position) for name, position in document[column])
        assert renamed != tuple(document[column])
        document[column] = renamed
        placement = Placement.model_validate(document)
    else:
        document = upstream.placement.model_dump(mode="python")
        position = document["speaker_positions_m"][0]
        document["speaker_positions_m"] = ((position[0], (99.0, *position[1][1:])),
                                           *document["speaker_positions_m"][1:])
        placement = Placement.model_validate(document)
    result = reflection_asymmetry(
        upstream, candidate_id=candidate_id, scene_fingerprint=scene_fingerprint,
        channels=channels, comparisons=comparisons, receiver_ids=receiver_ids,
        primary_receiver_id=primary_receiver_id, placement=placement,
    )
    assert result.state is MetricState.UNAVAILABLE
    assert result.reason_codes == (expected,)
    assert result.points == ()


def test_missing_evaluation_has_explicit_reason_and_no_upstream_identity() -> None:
    """未交反射評估時，只有診斷節不可估且上游欄位留空。"""
    upstream = fixtures._evaluate(fixtures._pair())
    result = reflection_asymmetry(None, candidate_id=upstream.candidate_id,
                                  scene_fingerprint=upstream.scene_fingerprint,
                                  channels=_CHANNELS, comparisons=_PAIR,
                                  receiver_ids=("main",), primary_receiver_id="main",
                                  placement=upstream.placement)
    assert result.reason_codes == (ReasonCode.REFLECTIONS_EVALUATION_MISSING,)
    assert result.reflections_evaluator_version is None
    assert result.frequency_range_hz is None
    assert result.points == ()


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
    """原話：單側改為缺資料只讓那格 unavailable，聲道匹配整類仍已量。"""
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
    """原話：診斷不進代價；有交沒交反射的身分仍要分清。"""
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


def test_wrong_category_is_a_caller_error() -> None:
    """交錯評估類別是呼叫端錯誤，不能吞成不可估。"""
    wrong = matching._evaluate(matching._receivers(), matching._group(),
                               matching._channel_points(matching._receivers(), matching._group()))
    with pytest.raises(ValueError):
        _diagnosis(wrong)


def test_missing_whole_reflection_channel_keeps_matching_measured() -> None:
    """原話：單側存在不讓整類不可估；缺失的整支反射聲道也不能外溢。"""
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
    assert all(p.state is ReflectionAsymmetryState.UNAVAILABLE for p in section.points
               if p.receiver_id == "front")
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


def test_one_sided_list_matches_independent_payload_screen() -> None:
    """單側清單逐條等於直接從兩份反射 payload 篩出的原始資料。"""
    upstream = _one_sided()
    assert isinstance(upstream.payload, ReflectionsAndEchoPayload)
    section = _diagnosis(upstream)
    channels = {channel.role: channel for channel in upstream.payload.channels}
    manual = set()
    for left_zone, right_zone in zip(channels["left"].zones, channels["right"].zones, strict=True):
        assert left_zone.zone is right_zone.zone
        for lp, rp in zip(left_zone.points, right_zone.points, strict=True):
            assert lp.frequency_hz == rp.frequency_hz
            if (lp.strongest_level_db is None) == (rp.strongest_level_db is None):
                continue
            side_role = "left" if lp.strongest_level_db is not None else "right"
            point = lp if side_role == "left" else rp
            assert point.strongest_path_index is not None
            path = channels[side_role].reflections[point.strongest_path_index]
            manual.add(("main", left_zone.zone, point.frequency_hz, side_role,
                        point.strongest_level_db, point.strongest_delay_s,
                        path.source, path.source_index, path.wall_sequence))
    listed = {(item.receiver_id, item.zone, item.frequency_hz, item.present.role,
               item.present.level_db, item.present.delay_s, item.present.source,
               item.present.source_index, item.present.wall_sequence)
              for item in section.one_sided}
    assert listed == manual
