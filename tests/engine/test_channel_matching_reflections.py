"""反射左右差診斷：老闆第 3 格的原始 L−R、交換標籤與缺值語意。"""
from __future__ import annotations

import ast
from collections.abc import Sequence
import re
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.scoring.channel_matching import ChannelComparison, ChannelDefinition, ChannelGroup
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
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections
from aosr.scoring.reflections_contract import ReflectionsAndEchoPayload
from aosr.scoring.reflections_cost import cost_reflections_evaluation
from tests.engine import test_channel_matching as matching
from tests.engine import test_reflections as fixtures


_PAIR = (ChannelComparisonPair(left_role="left", right_role="right"),)
_CHANNELS = (ChannelIdentity(role="left", speaker_id="left"),
             ChannelIdentity(role="right", speaker_id="right"))


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
    """老闆決定（見決策紙）：最強單條反射聲級做 L−R，保留正負號。"""
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
    """老闆決定（見決策紙）：兩邊都低於門檻時，差異仍須保留。"""
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
    """老闆決定（見決策紙）：交換左右後差值反號；純換標籤，實體報表不動。"""
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
    """老闆決定（見決策紙）：單側有反射另列延遲及來源；雙側均無反射要分開記。"""
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
    """老闆決定（見決策紙）：交換左右後有反射側別交換，聲級不變、差值仍空。"""
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


@pytest.mark.parametrize("reasons", (
    (ReasonCode.INSUFFICIENT_COVERAGE,),
    (ReasonCode.INSUFFICIENT_COVERAGE, ReasonCode.NO_REFLECTION_IN_ZONE_POINT),
))
def test_unknown_point_reason_is_unavailable_and_leaves_one_sided_list(
    reasons: tuple[ReasonCode, ...],
) -> None:
    """老闆決定（見決策紙）：確認沒有改成資料缺失，改判不可估、不能留在單側清單。"""
    upstream = _one_sided()
    original = _diagnosis(upstream)
    one = original.one_sided[0]
    document = upstream.model_dump(mode="python")
    changed_point = False
    for channel in document["payload"]["channels"]:
        if channel["role"] != one.absent_role:
            continue
        for zone in channel["zones"]:
            if zone["zone"] != one.zone:
                continue
            for point in zone["points"]:
                if point["frequency_hz"] == one.frequency_hz:
                    point["strongest_reason_codes"] = reasons
                    point["strongest_state"] = MetricState.UNAVAILABLE
                    changed_point = True
    assert changed_point
    changed = CategoryEvaluation.model_validate(document)
    section = _diagnosis(changed)
    cell = next(p for p in section.points if (p.zone, p.frequency_hz) == (one.zone, one.frequency_hz))
    assert cell.state is ReflectionAsymmetryState.UNAVAILABLE
    assert cell.left_minus_right_db is None
    assert cell.reason_codes == reasons
    assert not any((row.zone, row.frequency_hz) == (one.zone, one.frequency_hz) for row in section.one_sided)
    assert section.state is MetricState.MEASURED


def test_threshold_only_changes_cost_not_raw_diagnosis(tmp_path: Path) -> None:
    """老闆決定（見決策紙）：只改超標門檻，原始診斷完全不變。"""
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


def _imports_reflection_diagnostic(source: str) -> bool:
    forbidden = {"channel_matching_reflections", "channel_matching_reflections_contract"}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.module in {f"aosr.scoring.{name}" for name in forbidden}:
                return True
            if node.module == "aosr.scoring" and any(
                alias.name in forbidden for alias in node.names
            ):
                return True
            # 被守的檔都住 aosr/scoring 同一層：`from .模組 import …` 與 `from . import 模組` 也算。
            if node.level >= 1 and (node.module in forbidden or (
                    node.module is None and any(alias.name in forbidden for alias in node.names))):
                return True
        elif isinstance(node, ast.Import) and any(
            alias.name in {f"aosr.scoring.{name}" for name in forbidden}
            for alias in node.names
        ):
            return True
    return False


def test_scoring_layers_do_not_import_reflection_diagnostic() -> None:
    """老闆決定（見決策紙）：只做診斷，不進代價。

    主對話約定：排名與代價檔不得取這兩個模組——這是守「不進代價」的一道匯入檢查，決策紙沒有另寫這一條。
    """
    root = Path(__file__).resolve().parents[2] / "src" / "aosr" / "scoring"
    files = (root / "ranking.py", root / "category_registry.py", *root.glob("*_cost.py"))
    for path in files:
        assert not _imports_reflection_diagnostic(path.read_text())


@pytest.mark.parametrize("module", (
    "channel_matching_reflections", "channel_matching_reflections_contract",
))
def test_reflection_import_guard_recognizes_all_import_forms(module: str) -> None:
    """三種絕對寫法與兩種同一層的相對寫法都必須被同一支守衛抓到。"""
    assert _imports_reflection_diagnostic(f"from aosr.scoring.{module} import Thing")
    assert _imports_reflection_diagnostic(f"from aosr.scoring import {module}")
    assert _imports_reflection_diagnostic(f"import aosr.scoring.{module}")
    assert _imports_reflection_diagnostic(f"from .{module} import Thing")
    assert _imports_reflection_diagnostic(f"from . import {module}")
    assert not _imports_reflection_diagnostic("from . import contract")
    assert not _imports_reflection_diagnostic("from aosr.scoring import contract")


def _two_receiver() -> CategoryEvaluation:
    room = {"Lx": 6.0, "Ly": 5.0, "Lz": 3.0}
    records = tuple(fixtures._record(role, y, receiver=receiver, room=room,
                                     receiver_y=1.2, receiver_x=x)
                    for role, y in (("left", 0.6), ("right", 1.8))
                    for receiver, x in (("main", 3.2), ("surround", 4.4)))
    return fixtures._evaluate(records)


def test_whole_missing_channel_makes_its_cells_unavailable_without_losing_main() -> None:
    """主對話約定（見決策紙）：聲道資料缺失不留單側清單；主位不受影響。"""
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
    surround = [p for p in after.points if p.receiver_id == "surround"]
    assert surround
    assert all(p.state is ReflectionAsymmetryState.UNAVAILABLE and
               ReasonCode.REFLECTION_WINDOW_INCOMPLETE in p.reason_codes
               for p in surround)
    assert all(row.receiver_id != "surround" for row in after.one_sided)


def test_points_left_in_an_unavailable_channel_are_not_trusted() -> None:
    """老闆決定（見決策紙）：已證明兩邊時間窗完整才算單側。主對話約定（見決策紙）：不可估聲道殘留的點不採信。"""
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
    """主對話約定（見決策紙）：確認沒有須整支已量且時間窗完整；整支不可估時逐格抄聲道原因。"""
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


def _renamed_ids(rows: Sequence[tuple[str, object]]) -> tuple[tuple[str, object], ...]:
    renamed = tuple((f"renamed-{name}", position) for name, position in rows)
    assert renamed != tuple(rows)
    return renamed


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
    ("upstream_placement_speakers", ReasonCode.PLACEMENT_MISMATCH),
    ("upstream_placement_primary", ReasonCode.PLACEMENT_MISMATCH),
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
    elif change.endswith(("placement_speakers", "placement_primary")):
        column = "speaker_positions_m" if change.endswith("speakers") else "receiver_positions_m"
        if change.startswith("upstream_"):
            # 改的是上游反射評估自己的擺位，聲道匹配這邊的擺位不動：兩份都要列齊才算對得上。
            document = upstream.model_dump(mode="python")
            document["placement"][column] = _renamed_ids(document["placement"][column])
            upstream = CategoryEvaluation.model_validate(document)
        else:
            document = upstream.placement.model_dump(mode="python")
            document[column] = _renamed_ids(document[column])
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
    assert (result.state, result.reason_codes, result.points) == (MetricState.UNAVAILABLE, (expected,), ())
    _assert_unavailable_copies_upstream(result, upstream)


def _assert_unavailable_copies_upstream(
    result: ReflectionAsymmetry, upstream: CategoryEvaluation,
) -> None:
    assert result.reflections_evaluator_version == upstream.evaluator_version
    assert result.reflections_settings_fingerprint == upstream.settings_fingerprint
    assert result.source_flags == upstream.flags
    if isinstance(upstream.payload, ReflectionsAndEchoPayload):
        assert result.frequency_range_hz == upstream.payload.frequency_range_hz
        assert result.window_upper_ms == upstream.payload.window_upper_ms
        assert result.primary_receiver_id == upstream.payload.primary_receiver_id


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


def test_wrong_category_is_a_caller_error() -> None:
    """交錯評估類別是呼叫端錯誤，不能吞成不可估。"""
    wrong = matching._evaluate(matching._receivers(), matching._group(),
                               matching._channel_points(matching._receivers(), matching._group()))
    with pytest.raises(ValueError):
        _diagnosis(wrong)


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


def test_real_evaluator_unavailable_channel_still_has_diagnostic_cells() -> None:
    """評估器產生的空點不可估聲道仍須逐格列原因。"""
    room = {"Lx": 6.0, "Ly": 5.0, "Lz": 3.0}
    records = tuple(fixtures._record(role, y, receiver=receiver, room=room,
                                     receiver_y=1.2, receiver_x=x)
                    for role, y in (("left", 0.6), ("right", 1.8))
                    for receiver, x in (("main", 3.2), ("surround", 4.4)))
    baseline = _diagnosis(fixtures._evaluate(records), receiver_ids=("main", "surround"))
    one = next(row for row in baseline.one_sided if row.receiver_id == "surround")
    changed_records = tuple(replace(record, third_octave_decay=None)
                            if record.receiver_id == "surround" and record.role == one.absent_role
                            else record for record in records)
    assert changed_records != records
    upstream = fixtures._evaluate(changed_records)
    assert isinstance(upstream.payload, ReflectionsAndEchoPayload)
    channel = next(ch for ch in upstream.payload.channels
                   if (ch.receiver_id, ch.role) == ("surround", one.absent_role))
    assert channel.state is MetricState.UNAVAILABLE
    assert channel.coverage == "complete"
    assert all(not zone.points for zone in channel.zones)
    section = _diagnosis(upstream, receiver_ids=("main", "surround"))
    cells = [point for point in section.points if point.receiver_id == "surround"]
    assert cells
    assert all(point.state is ReflectionAsymmetryState.UNAVAILABLE and
               point.reason_codes == channel.reason_codes for point in cells)


@pytest.mark.parametrize("duplicate", (False, True))
def test_both_absent_merges_raw_side_reasons_in_declaration_order(duplicate: bool) -> None:
    """兩邊確認沒有時，原始兩側原因碼去重且依宣告順序。"""
    records = []
    for record in fixtures._pair():
        table = record.report.path_table
        assert table is not None
        rows = tuple(row.model_copy(update={
            "relative_direct_energy": tuple(0.0 for _ in fixtures._AXIS)
        }) if row.order > 0 else row for row in table.rows)
        records.append(replace(record, report=record.report.model_copy(update={
            "path_table": table.model_copy(update={"rows": rows})
        })))
    upstream = fixtures._evaluate(tuple(records))
    assert isinstance(upstream.payload, ReflectionsAndEchoPayload)
    both = next(p for p in _diagnosis(upstream).points
                if p.state is ReflectionAsymmetryState.BOTH_ABSENT
                and ReasonCode.ZERO_REFLECTION_ENERGY in p.reason_codes)
    document = upstream.model_dump(mode="python")
    changed = set()
    for channel in document["payload"]["channels"]:
        for zone in channel["zones"]:
            if zone["zone"] != both.zone:
                continue
            for point in zone["points"]:
                if point["frequency_hz"] != both.frequency_hz:
                    continue
                if channel["role"] == both.left_role:
                    point["strongest_reason_codes"] = (
                        (ReasonCode.ZERO_REFLECTION_ENERGY, ReasonCode.NO_REFLECTION_IN_ZONE_POINT)
                        if duplicate else (ReasonCode.ZERO_REFLECTION_ENERGY,))
                else:
                    point.update(strongest_delay_s=None, strongest_path_index=None,
                                 strongest_reason_codes=(ReasonCode.NO_REFLECTION_IN_ZONE_POINT,),
                                 total_energy_db={"value": None, "state": MetricState.UNAVAILABLE,
                                                  "reason_codes": (ReasonCode.NO_REFLECTION_IN_ZONE_POINT,)})
                changed.add(channel["role"])
    assert changed == {both.left_role, both.right_role}
    changed_upstream = CategoryEvaluation.model_validate(document)
    assert isinstance(changed_upstream.payload, ReflectionsAndEchoPayload)
    raw = {ch.role: next(p for zone in ch.zones if zone.zone is both.zone
                         for p in zone.points if p.frequency_hz == both.frequency_hz)
           for ch in changed_upstream.payload.channels}
    expected = tuple(code for code in ReasonCode if code in set(
        (*raw[both.left_role].strongest_reason_codes,
         *raw[both.right_role].strongest_reason_codes)))
    found = next(p for p in _diagnosis(changed_upstream).points
                 if (p.zone, p.frequency_hz) == (both.zone, both.frequency_hz))
    assert found.state is ReflectionAsymmetryState.BOTH_ABSENT
    assert found.reason_codes == expected
    assert ReasonCode.ZERO_REFLECTION_ENERGY in found.reason_codes
    assert ReasonCode.NO_REFLECTION_IN_ZONE_POINT in found.reason_codes


def test_missing_side_does_not_copy_confirmed_absence_reason() -> None:
    """一側確認沒有、一側缺值，只帶缺值側原因。"""
    upstream = _one_sided()
    one = _diagnosis(upstream).one_sided[0]
    document = upstream.model_dump(mode="python")
    changed = False
    for channel in document["payload"]["channels"]:
        if channel["role"] != one.present.role:
            continue
        for zone in channel["zones"]:
            if zone["zone"] != one.zone:
                continue
            for point in zone["points"]:
                if point["frequency_hz"] == one.frequency_hz:
                    point["strongest_state"] = MetricState.UNAVAILABLE
                    point["strongest_reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
                    point["strongest_level_db"] = None
                    point["strongest_delay_s"] = None
                    point["strongest_path_index"] = None
                    point["total_energy_db"] = {
                        "value": None, "state": MetricState.UNAVAILABLE,
                        "reason_codes": (ReasonCode.INSUFFICIENT_COVERAGE,)}
                    changed = True
    assert changed
    found = next(p for p in _diagnosis(CategoryEvaluation.model_validate(document)).points
                 if (p.zone, p.frequency_hz) == (one.zone, one.frequency_hz))
    assert found.state is ReflectionAsymmetryState.UNAVAILABLE
    assert found.reason_codes == (ReasonCode.INSUFFICIENT_COVERAGE,)
    assert ReasonCode.NO_REFLECTION_IN_ZONE_POINT not in found.reason_codes
    assert ReasonCode.ZERO_REFLECTION_ENERGY not in found.reason_codes


def test_two_missing_sides_deduplicate_and_sort_reasons() -> None:
    """兩側缺值碼相反順序時，合併後去重並依宣告順序。"""
    upstream = fixtures._evaluate(fixtures._pair())
    cell = next(p for p in _diagnosis(upstream).points
                if p.state is ReflectionAsymmetryState.MEASURED)
    document = upstream.model_dump(mode="python")
    codes = (ReasonCode.INSUFFICIENT_COVERAGE, ReasonCode.REFLECTION_WINDOW_INCOMPLETE)
    changed = set()
    for channel in document["payload"]["channels"]:
        for zone in channel["zones"]:
            if zone["zone"] != cell.zone:
                continue
            for point in zone["points"]:
                if point["frequency_hz"] == cell.frequency_hz:
                    point["strongest_state"] = MetricState.UNAVAILABLE
                    point["strongest_reason_codes"] = codes if channel["role"] == "left" else codes[::-1]
                    point["strongest_level_db"] = None
                    point["strongest_delay_s"] = None
                    point["strongest_path_index"] = None
                    point["total_energy_db"] = {
                        "value": None, "state": MetricState.UNAVAILABLE,
                        "reason_codes": (ReasonCode.INSUFFICIENT_COVERAGE,)}
                    changed.add(channel["role"])
    assert changed == {"left", "right"}
    found = next(p for p in _diagnosis(CategoryEvaluation.model_validate(document)).points
                 if (p.zone, p.frequency_hz) == (cell.zone, cell.frequency_hz))
    assert found.state is ReflectionAsymmetryState.UNAVAILABLE
    assert found.reason_codes == tuple(code for code in ReasonCode if code in codes)


@pytest.mark.parametrize("change, expected", (
    ("version_upstream", ReasonCode.EVALUATOR_VERSION_MISMATCH),
    ("role_receiver", ReasonCode.CHANNEL_ROLE_MISMATCH),
    ("role_placement", ReasonCode.CHANNEL_ROLE_MISMATCH),
))
def test_identity_check_precedence(change: str, expected: ReasonCode) -> None:
    """同時兩錯時，先回版本或角色錯。"""
    upstream = fixtures._evaluate(fixtures._pair())
    channels = _CHANNELS
    receivers = ("main",)
    placement = upstream.placement
    if change == "version_upstream":
        document = upstream.model_dump(mode="python")
        document["evaluator_version"] = "older-reflections"
        document["state"] = EvaluationState.UNAVAILABLE
        document["payload"] = None
        document["raw_quantities"] = ()
        document["reason_codes"] = (ReasonCode.INSUFFICIENT_COVERAGE,)
        upstream = CategoryEvaluation.model_validate(document)
    else:
        channels = (ChannelIdentity(role="center", speaker_id="left"), _CHANNELS[1])
        if change == "role_receiver":
            receivers = ("other",)
        else:
            document = placement.model_dump(mode="python")
            name, coordinate = document["speaker_positions_m"][0]
            document["speaker_positions_m"] = ((name, (coordinate[0] + 0.1, *coordinate[1:])),
                                               *document["speaker_positions_m"][1:])
            placement = Placement.model_validate(document)
    result = reflection_asymmetry(
        upstream, candidate_id=upstream.candidate_id,
        scene_fingerprint=upstream.scene_fingerprint, channels=channels,
        comparisons=_PAIR, receiver_ids=receivers,
        primary_receiver_id="main", placement=placement)
    assert result.state is MetricState.UNAVAILABLE
    assert result.reason_codes == (expected,)


def test_two_comparison_pairs_follow_declared_order_and_sign() -> None:
    """比較對先於接收點排序，反向比較逐格反號。"""
    upstream = _two_receiver()
    pairs = (_PAIR[0], ChannelComparisonPair(left_role="right", right_role="left"))
    section = _diagnosis(upstream, comparisons=pairs, receiver_ids=("main", "surround"))
    assert section.state is MetricState.MEASURED
    first = [p for p in section.points if (p.left_role, p.right_role) == ("left", "right")]
    second = [p for p in section.points if (p.left_role, p.right_role) == ("right", "left")]
    assert first and second
    assert section.points == (*first, *second)
    for left, right in zip(first, second, strict=True):
        assert (left.receiver_id, left.zone, left.frequency_hz) == (
            right.receiver_id, right.zone, right.frequency_hz)
        assert left.state is right.state
        if left.state is ReflectionAsymmetryState.MEASURED:
            assert left.left_minus_right_db is not None
            assert right.left_minus_right_db == -left.left_minus_right_db


def test_unused_channel_role_still_requires_matching_reflection_role_set() -> None:
    """未參加比較的聲道角色也要與反射聲道集合一致。"""
    upstream = fixtures._evaluate(fixtures._pair())
    channels = (*_CHANNELS, ChannelIdentity(role="center", speaker_id="center"))
    section = _diagnosis(upstream, channels=channels)
    assert section.state is MetricState.UNAVAILABLE
    assert section.reason_codes == (ReasonCode.CHANNEL_ROLE_MISMATCH,)


def test_present_side_unavailable_with_complete_coverage_is_not_one_sided() -> None:
    """有反射側整支不可估，完整覆蓋與殘留點也不准單側。"""
    upstream = _two_receiver()
    one = next(row for row in _diagnosis(upstream, receiver_ids=("main", "surround")).one_sided
               if row.receiver_id == "surround")
    document = upstream.model_dump(mode="python")
    changed = False
    for channel in document["payload"]["channels"]:
        if (channel["receiver_id"], channel["role"]) == ("surround", one.present.role):
            assert channel["coverage"] == "complete"
            assert any(zone["points"] for zone in channel["zones"])
            channel["state"] = MetricState.UNAVAILABLE
            channel["reason_codes"] = (ReasonCode.PLACEMENT_MISMATCH,)
            changed = True
    assert changed
    section = _diagnosis(CategoryEvaluation.model_validate(document),
                         receiver_ids=("main", "surround"))
    found = next(p for p in section.points if (p.receiver_id, p.zone, p.frequency_hz)
                 == ("surround", one.zone, one.frequency_hz))
    assert section.state is MetricState.MEASURED
    assert found.state is ReflectionAsymmetryState.UNAVAILABLE
    assert found.reason_codes == (ReasonCode.PLACEMENT_MISMATCH,)
    assert not any((row.receiver_id, row.zone, row.frequency_hz)
                   == ("surround", one.zone, one.frequency_hz) for row in section.one_sided)
