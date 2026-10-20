"""第一階段讀回由曲線重新量測；比較只固定量法與材料，不固定待比較的房型擺位。"""
from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.quality_targets import QualityTargets
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.reporting import fem_slices, stage_one_result as results
from aosr.reporting.evaluation import ResultStanding
from aosr.scoring.stage_one_low_frequency import evaluate_stage_one_low_frequency
from aosr.scoring.stage_one_low_frequency_contract import StageOneLowFrequencyInput
from aosr.scoring.stage_one_score import SCORED_RANGE_KEY, WEIGHTS_KEY, score_stage_one, stage_one_score_settings
from tests.engine import _stage_one_cases as cases
from tests.engine._fem_cli_cases import forbid_fem
from tests.engine.test_scheme_pipeline import _run_control


@pytest.fixture
def saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, results.StageOneResult]:
    item = cases.scheme()
    result = results.run_stage_one(item, shards=cases.shards(monkeypatch, item), **cases.run_kwargs())
    path = tmp_path / "saved.json"
    results.save_stage_one_result(result, path)
    forbid_fem(monkeypatch)
    def no_shards(*args: object, **kwargs: object) -> object:
        raise AssertionError("讀回不可讀分片或重新求解")
    monkeypatch.setattr(fem_slices, "solve_slice", no_shards)
    monkeypatch.setattr(fem_slices, "energies_from_shards", no_shards)
    return path, result


def changed_registry(change: str) -> QualityTargets:
    registry = cases.run_kwargs()["registry"]
    section = registry.stage_one(cases.scheme().purpose)
    if change == "weights":
        tables = tuple(table.model_copy(update={"item": tuple(
            row.model_copy(update={"value": row.value * 3.0}) if row.name == "seat_robustness_rms_db" else row
            for row in table.item)}) if table.key == WEIGHTS_KEY else table for table in section.weight)
        section = section.model_copy(update={"weight": tables})
    else:
        entries = tuple(entry.model_copy(update={"value": (40.0, 200.0)})
                        if entry.key == SCORED_RANGE_KEY else entry for entry in section.setting)
        section = section.model_copy(update={"setting": entries})
    return registry.model_copy(update={"stage_one_sections": tuple(
        section if old.name == section.name else old for old in registry.stage_one_sections)})


def test_current_preserves_stored_result(saved: tuple[Path, results.StageOneResult]) -> None:
    path, stored = saved
    loaded = results.reload_stage_one_result(path, registry=cases.run_kwargs()["registry"], physics_identity=stored.physics_identity)
    assert loaded.standing is ResultStanding.CURRENT
    assert loaded.result == loaded.stored == stored
    with pytest.raises(FrozenInstanceError):
        setattr(loaded, "standing", ResultStanding.RERANKED)


def test_weights_only_rerank(saved: tuple[Path, results.StageOneResult]) -> None:
    path, stored = saved
    registry = changed_registry("weights")
    settings = stage_one_score_settings(registry, stored.scheme.purpose)
    assert settings.scored_range_hz == stored.score_settings.scored_range_hz
    assert settings.weights != stored.score_settings.weights
    loaded = results.reload_stage_one_result(path, registry=registry, physics_identity=stored.physics_identity)
    assert loaded.standing is ResultStanding.RERANKED
    assert loaded.result.score == score_stage_one(stored.evaluation, settings)
    assert loaded.result.score_settings == settings
    assert loaded.result.evaluation == stored.evaluation
    assert loaded.stored == stored
    assert path.read_text() == stored.model_dump_json()


def test_range_change_remeasures(saved: tuple[Path, results.StageOneResult]) -> None:
    path, stored = saved
    registry = changed_registry("range")
    settings = stage_one_score_settings(registry, stored.scheme.purpose)
    assert settings.weights == stored.score_settings.weights
    assert settings.scored_range_hz != stored.score_settings.scored_range_hz
    expected = evaluate_stage_one_low_frequency(StageOneLowFrequencyInput(
        candidate_id=stored.scheme.scheme_id, low_frequency_axis=stored.low_frequency_axis,
        speakers=stored.curves), scoring_range_hz=settings.scored_range_hz)
    loaded = results.reload_stage_one_result(path, registry=registry, physics_identity=stored.physics_identity)
    assert loaded.standing is ResultStanding.REMEASURED
    assert loaded.result.evaluation == expected
    assert loaded.result.score == score_stage_one(expected, settings)
    assert loaded.stored == stored
    results.StageOneResult.model_validate(loaded.result.model_dump())


def test_changed_raw_energy_passes_old_checks_then_remeasures(saved: tuple[Path, results.StageOneResult]) -> None:
    path, original = saved
    document = json.loads(path.read_text())
    curve = document["curves"][0]["primary"]
    index = next(index for index, hz in enumerate(curve["frequencies_hz"]) if hz >= 80.0)
    curve["fem_energy"][index] *= 7.0
    path.write_text(json.dumps(document))
    stored = results.load_stage_one_result(path)
    assert stored.curves != original.curves
    assert stored.evaluation == original.evaluation
    assert stored.score == original.score
    data = StageOneLowFrequencyInput(candidate_id=stored.scheme.scheme_id,
        low_frequency_axis=stored.low_frequency_axis, speakers=stored.curves)
    expected = evaluate_stage_one_low_frequency(data, scoring_range_hz=stored.score_settings.scored_range_hz)
    assert expected != stored.evaluation
    loaded = results.reload_stage_one_result(path, registry=cases.run_kwargs()["registry"], physics_identity=stored.physics_identity)
    assert loaded.standing is ResultStanding.REMEASURED
    assert loaded.result.evaluation == expected
    assert loaded.result.score == score_stage_one(expected, stored.score_settings)
    assert loaded.stored == stored


@pytest.mark.parametrize("registry_change", [None, "weights", "range"])
def test_physics_difference_takes_precedence(saved: tuple[Path, results.StageOneResult], registry_change: str | None) -> None:
    path, stored = saved
    registry = cases.run_kwargs()["registry"] if registry_change is None else changed_registry(registry_change)
    loaded = results.reload_stage_one_result(path, registry=registry, physics_identity="different")
    assert loaded.standing is ResultStanding.NEEDS_PHYSICS
    assert loaded.stored == stored
    settings = stage_one_score_settings(registry, stored.scheme.purpose)
    expected = evaluate_stage_one_low_frequency(StageOneLowFrequencyInput(candidate_id=stored.scheme.scheme_id,
        low_frequency_axis=stored.low_frequency_axis, speakers=stored.curves), scoring_range_hz=settings.scored_range_hz)
    assert loaded.result.evaluation == expected
    assert loaded.result.score == score_stage_one(expected, settings)
    assert loaded.result.physics_identity == stored.physics_identity


@pytest.mark.parametrize("version,reason", [("aosr.scheme_result.v999", "第二階段結果檔，請用 compare"),
                                            ("unknown", "認不得的結果檔格式")])
def test_reload_rejects_wrong_format(tmp_path: Path, version: str, reason: str) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps({"schema_version": version}))
    with pytest.raises(ValueError, match=reason):
        results.reload_stage_one_result(path, registry=cases.run_kwargs()["registry"], physics_identity="different")


def test_reload_rejects_stage_two_result(tmp_path: Path) -> None:
    path = tmp_path / "second.json"
    path.write_text(_run_control(cases.scheme()).model_dump_json())
    with pytest.raises(ValueError, match="第二階段結果檔，請用 compare"):
        results.reload_stage_one_result(path, registry=cases.run_kwargs()["registry"], physics_identity="different")


def test_reload_rejects_absent_stage_one_purpose(saved: tuple[Path, results.StageOneResult]) -> None:
    path, stored = saved
    registry = cases.run_kwargs()["registry"].model_copy(update={"stage_one_sections": ()})
    with pytest.raises(ValueError, match="第一階段.*用途|用途.*第一階段"):
        results.reload_stage_one_result(path, registry=registry, physics_identity=stored.physics_identity)


def test_reload_keeps_original_integrity_checks(saved: tuple[Path, results.StageOneResult]) -> None:
    path, stored = saved
    document = json.loads(path.read_text())
    document["score"]["screening_total"] += 1.0
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="逐位重算不同"):
        results.reload_stage_one_result(path, registry=cases.run_kwargs()["registry"], physics_identity=stored.physics_identity)


@pytest.mark.parametrize("field,label", [
    ("physics", "物理身分"), ("score_version", "合分版本"), ("evaluator_version", "評估器版本"),
    ("settings", "合分設定指紋"), ("channels", "聲道組指紋"), ("layout", "座位相對佈局"),
    ("axis", "低頻軸"), ("energy", "能量欄"), ("walls", "牆面與空氣假設"),
    ("density", "牆面與空氣假設"), ("speed", "牆面與空氣假設"),
])
def test_each_identity_field_blocks_comparison(saved: tuple[Path, results.StageOneResult], field: str, label: str) -> None:
    _, original = saved
    # 投影函式只取指定身分格；版本與能量欄用未來值，刻意不經今天的 Literal 形狀檢查。
    changed = original
    if field == "physics":
        changed = original.model_copy(update={"physics_identity": "different"})
    elif field in ("score_version", "settings"):
        key = "score_version" if field == "score_version" else "settings_fingerprint"
        changed = original.model_copy(update={"score": original.score.model_copy(update={key: "changed"})})
    elif field in ("evaluator_version", "energy"):
        key = "evaluator_version" if field == "evaluator_version" else "energy_column"
        changed = original.model_copy(update={"evaluation": original.evaluation.model_copy(update={key: "changed"})})
    elif field == "axis":
        changed = original.model_copy(update={"low_frequency_axis": LowFrequencyAxis.SEARCH_48})
    elif field == "channels":
        group = original.scheme.channel_group
        changed = original.model_copy(update={"scheme": original.scheme.model_copy(update={
            "channel_group": group.model_copy(update={"feature_match_tolerance_hz": group.feature_match_tolerance_hz + 1.0})})})
    elif field == "layout":
        points = original.scheme.receiver_set.points
        last = points[-1]
        moved = last.model_copy(update={"position_m": (last.position_m[0] + .1, *last.position_m[1:])})
        changed = original.model_copy(update={"scheme": original.scheme.model_copy(update={
            "receiver_set": original.scheme.receiver_set.model_copy(update={"points": (*points[:-1], moved)})})})
    else:
        scene = original.scheme.scene
        updates: dict[str, object]
        if field == "walls":
            assert scene.impedance_pa_s_per_m_by_wall is not None
            updates = {"impedance_pa_s_per_m_by_wall": {key: value * 2.0 for key, value in scene.impedance_pa_s_per_m_by_wall.items()}}
        else:
            key = "density_kg_m3" if field == "density" else "sound_speed_m_s"
            updates = {key: getattr(scene, key) * 1.1}
        changed = original.model_copy(update={"scheme": original.scheme.model_copy(update={"scene": scene.model_copy(update=updates)})})
    a, b = results.stage_one_comparison_identity(original), results.stage_one_comparison_identity(changed)
    assert a != b
    assert results.identity_differences(a, b) == (label,)


def test_identity_uses_ordered_hex_wall_and_air_values(saved: tuple[Path, results.StageOneResult]) -> None:
    _, stored = saved
    identity = results.stage_one_comparison_identity(stored)
    scene = stored.scheme.scene
    assert scene.impedance_pa_s_per_m_by_wall is not None
    expected = tuple(float(scene.impedance_pa_s_per_m_by_wall[wall.wall_name()]).hex() for wall in Wall.all())
    assert identity.wall_and_air_assumptions == (*expected, float(scene.density_kg_m3).hex(), float(scene.sound_speed_m_s).hex())
    with pytest.raises(FrozenInstanceError):
        setattr(identity, "physics_identity", "different")


@pytest.mark.parametrize("change", ["positions", "room"])
def test_room_size_and_absolute_positions_remain_comparable(saved: tuple[Path, results.StageOneResult], change: str) -> None:
    _, stored = saved
    scheme = stored.scheme
    if change == "positions":
        points = tuple(point.model_copy(update={"position_m": tuple(value + .2 for value in point.position_m)})
                       for point in scheme.receiver_set.points)
        scheme = scheme.model_copy(update={"receiver_set": scheme.receiver_set.model_copy(update={"points": points}),
            "speakers": {key: Point(point.x + .2, point.y + .2, point.z + .2) for key, point in scheme.speakers.items()}})
    else:
        room = scheme.scene.room_m
        scheme = scheme.model_copy(update={"scene": scheme.scene.model_copy(update={"room_m": Room(room.Lx + 1., room.Ly + 1., room.Lz + 1.)})})
    changed = results.StageOneResult.model_validate({**stored.model_dump(), "scheme": scheme})
    assert results.identity_differences(results.stage_one_comparison_identity(stored), results.stage_one_comparison_identity(changed)) == ()


def test_unscored_evaluation_change_still_remeasures(saved: tuple[Path, results.StageOneResult]) -> None:
    path, original = saved
    document = json.loads(path.read_text())
    document["evaluation"]["speakers"][0]["payload"]["own_line_slope_db_per_octave"] += 1.0
    path.write_text(json.dumps(document))
    stored = results.load_stage_one_result(path)
    assert stored.evaluation != original.evaluation
    assert stored.score == original.score
    loaded = results.reload_stage_one_result(path, registry=cases.run_kwargs()["registry"], physics_identity=stored.physics_identity)
    assert loaded.standing is ResultStanding.REMEASURED
    assert loaded.result.evaluation == original.evaluation
    assert loaded.result.score == original.score


def test_identity_keeps_wall_order_when_impedances_differ(saved: tuple[Path, results.StageOneResult]) -> None:
    _, stored = saved
    # 六面各不相同、也不照大小排，字典也故意不照牆序寫；換順序、排序或照插入順序取值都會對不上。
    values = {"yL": 3.0, "x0": 5.0, "floor": 6.0, "xL": 2.0, "ceiling": 1.0, "y0": 4.0}
    scene = stored.scheme.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": values})
    changed = stored.model_copy(update={"scheme": stored.scheme.model_copy(update={"scene": scene})})
    walls = results.stage_one_comparison_identity(changed).wall_and_air_assumptions[:6]
    assert walls == tuple(value.hex() for value in (6.0, 1.0, 5.0, 2.0, 4.0, 3.0))
