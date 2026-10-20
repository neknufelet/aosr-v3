"""第一階段只合併有限元素原值；評估、分數與結果格式各有獨立裁判。"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.frequency_axis import LowFrequencyAxis, low_frequency_axis_frequencies
from aosr.geometry.shoebox import Point, Wall
from aosr.reporting.evaluation import load_result
from aosr.reporting import fem_slices
from aosr.scoring.receiver_set import ReceiverRole
from aosr.scoring.stage_one_low_frequency_contract import StageOneLowFrequencyInput, StageOneSeatCurve, StageOneSpeakerCurves
from aosr.scoring.stage_one_score import score_stage_one_input, stage_one_score_settings
from tests.engine import _stage_one_cases as cases
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_scheme_pipeline import _many_fem, _run_control
from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path


def test_raw_energy_and_score_follow_independent_input(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("aosr.reporting.stage_one_result")
    item = cases.scheme()
    parts = cases.shards(monkeypatch, item)
    frequencies = low_frequency_axis_frequencies(LowFrequencyAxis.SEARCH)[0]
    walls = item.scene.impedance_pa_s_per_m_by_wall
    assert walls is not None
    expected = _many_fem(room=item.scene.room_m, sources=item.speakers,
        receivers={p.receiver_id: Point(*p.position_m)
                   for p in item.receiver_set.points},
        wall_impedances={wall: walls[wall.wall_name()] for wall in Wall.all()},
        frequencies_hz=frequencies, density_kg_m3=item.scene.density_kg_m3,
        sound_speed_m_s=item.scene.sound_speed_m_s)
    speakers = []
    for speaker in item.speakers:
        seats = [StageOneSeatCurve(receiver_id=p.receiver_id, direction_relative_to_primary=p.direction_relative_to_primary,
                 report_id=f"fem-{speaker}-{p.receiver_id}", frequencies_hz=frequencies,
                 fem_energy=expected[speaker, p.receiver_id]) for p in item.receiver_set.points]
        speakers.append(StageOneSpeakerCurves(speaker_id=speaker, primary=seats[0], surrounding=tuple(seats[1:])))
    data = StageOneLowFrequencyInput(candidate_id=item.scheme_id, low_frequency_axis=LowFrequencyAxis.SEARCH,
                                     speakers=tuple(speakers))
    received = []
    evaluate = module.evaluate_stage_one_low_frequency
    def capture(value: StageOneLowFrequencyInput, *, scoring_range_hz: tuple[float, float]) -> object:
        received.append(value)
        return evaluate(value, scoring_range_hz=scoring_range_hz)
    monkeypatch.setattr(module, "evaluate_stage_one_low_frequency", capture)
    kwargs = cases.run_kwargs()
    result = module.run_stage_one(item, shards=parts, **kwargs)
    assert received == [data]
    assert result.curves == data.speakers
    assert result.low_frequency_axis is LowFrequencyAxis.SEARCH
    assert result.score == score_stage_one_input(data, stage_one_score_settings(kwargs["registry"], item.purpose))
    assert result.evaluation.speakers == evaluate(data, scoring_range_hz=result.score.scored_range_hz).speakers
    assert result.schema_version == "aosr.stage_one_result.v1"
    assert result.scope == "stage_one_fem_only"


@pytest.mark.parametrize("damage,reason", [
    ("source", "全向"), ("furniture", "家具"), ("walls", "六面"),
    ("frequency", "頻率無關"), ("zones", "分區"),
    ("missing", "前後左右上下"), ("extra", "前後左右上下"), ("desk", "落地"),
    ("nonpositive", "正的"), ("other-seat", "前後左右上下"), ("odd-direction", "前後左右上下"),
])
def test_prerequisites_reject_each_invalid_case(damage: str, reason: str) -> None:
    module = importlib.import_module("aosr.reporting.stage_one_result")
    item = cases.scheme()
    if damage == "source":
        item = item.model_copy(update={"source_model": "product_default"})
    elif damage == "furniture":
        from aosr.reporting.scheme import FurnitureSpec
        from tests.engine._furniture_cases import relative_item
        item = item.model_copy(update={"furniture": (FurnitureSpec.model_validate(relative_item()),)})
    elif damage == "desk":
        from aosr.reporting.scheme import SpeakerSetup, SpeakerCabinet
        item = item.model_copy(update={"speaker_setup": SpeakerSetup(kind="bookshelf", mount="desk", representative=False,
            cabinet=SpeakerCabinet(width_m=.2, depth_m=.3, height_m=.4,
                acoustic_center_behind_front_m=.1, acoustic_center_above_bottom_m=.2))})
    elif damage == "nonpositive":
        # 方案本身收 0 與負阻抗，這一關只有前提檢查在擋。
        assert item.scene.impedance_pa_s_per_m_by_wall is not None
        walls = {wall: 0.0 for wall in item.scene.impedance_pa_s_per_m_by_wall}
        item = item.model_copy(update={"scene": item.scene.model_copy(update={"impedance_pa_s_per_m_by_wall": walls})})
    elif damage in ("other-seat", "odd-direction"):
        # 六個方向不重複、角色只有主位與周圍點時，只剩總點數或方向集合那一格擋得住。
        points = item.receiver_set.points
        if damage == "other-seat":
            points = (*points, points[-1].model_copy(update={
                "receiver_id": "sofa", "role": ReceiverRole.OTHER_SEAT, "direction_relative_to_primary": None}))
        else:
            points = (*points[:-1], points[-1].model_copy(update={"direction_relative_to_primary": "front-left"}))
        item = item.model_copy(update={"receiver_set": item.receiver_set.model_copy(update={"points": points})})
    elif damage in ("walls", "frequency", "zones"):
        assert item.scene.impedance_pa_s_per_m_by_wall is not None
        walls = dict(item.scene.impedance_pa_s_per_m_by_wall)
        changes: dict[str, object] = {"impedance_pa_s_per_m_by_wall": walls}
        if damage == "walls":
            walls["floor"] *= 2
        else:
            from aosr.physics.report_surfaces import WholeWallZone
            from tests.engine.test_measured_impedance import measured
            zone = WholeWallZone.model_validate({"area": "whole_wall", "material": measured()})
            changes["impedance_pa_s_per_m_by_wall"] = None
            changes["surface_zones_by_wall"] = {wall: (zone,) for wall in walls}
        item = item.model_copy(update={"scene": item.scene.model_copy(update=changes)})
    else:
        points = item.receiver_set.points
        points = points[:-1] if damage == "missing" else (*points, points[-1].model_copy(update={"receiver_id": "extra"}))
        item = item.model_copy(update={"receiver_set": item.receiver_set.model_copy(update={"points": points})})
    with pytest.raises(ValueError, match=reason):
        module.check_stage_one_prerequisites(item)


def test_declared_axis_must_match_fem_key(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("aosr.reporting.stage_one_result")
    item = cases.scheme()
    assert item.scene.impedance_pa_s_per_m_by_wall is not None
    parts = cases.shards(monkeypatch, item)
    changed = parts[0].fem_key.model_copy(update={"frequency_hex": tuple(
        (float.fromhex(value) + .001).hex() for value in parts[0].fem_key.frequency_hex)})
    with pytest.raises(ValueError, match="頻率"):
        module.stage_one_input(item, {}, changed)


def test_wall_assumption_uses_scheme_air_and_impedance() -> None:
    module = importlib.import_module("aosr.reporting.stage_one_result")
    item = cases.scheme()
    assert item.scene.impedance_pa_s_per_m_by_wall is not None
    scene = item.scene.model_copy(update={"density_kg_m3": 1.17, "sound_speed_m_s": 337.4,
        "impedance_pa_s_per_m_by_wall": {wall: 7312.9 for wall in item.scene.impedance_pa_s_per_m_by_wall}})
    assert module.wall_assumption_text(item.model_copy(update={"scene": scene})) == (
        "在這個牆面假設下：六面同一個頻率無關實數阻抗 18.5 ρc（ρc 是空氣特性阻抗）")


def test_stage_two_loader_rejects_stage_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("aosr.reporting.stage_one_result")
    item = cases.scheme()
    result = module.run_stage_one(item, shards=cases.shards(monkeypatch, item), **cases.run_kwargs())
    path = tmp_path / "result.json"
    module.save_stage_one_result(result, path)
    assert module.load_stage_one_result(path) == result
    with pytest.raises(ValueError, match="格式認不得"):
        load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")), directivity=DIRECTIVITY,
                    quality_targets_path=config_path("quality_targets.toml"), physics_identity=result.physics_identity)


def test_stage_one_loader_rejects_stage_two(tmp_path: Path) -> None:
    module = importlib.import_module("aosr.reporting.stage_one_result")
    result = _run_control(cases.scheme())
    path = tmp_path / "stage-two.json"
    path.write_text(result.model_dump_json())
    with pytest.raises(ValidationError):
        module.load_stage_one_result(path)


def test_energies_must_cover_exactly_every_speaker_and_seat(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("aosr.reporting.stage_one_result")
    item = cases.scheme()
    parts = cases.shards(monkeypatch, item)
    kwargs = cases.run_kwargs()
    energies = fem_slices.energies_from_shards(parts, scheme=item, capabilities=kwargs["capabilities"],
        directivity=kwargs["directivity"], physics_identity=kwargs["physics_identity"])
    module.stage_one_input(item, energies, parts[0].fem_key)
    first = next(iter(energies))
    extra = {**energies, ("center", first[1]): energies[first]}
    missing = {pair: value for pair, value in energies.items() if pair != first}
    for damaged in (extra, missing):
        with pytest.raises(ValueError, match="剛好包含"):
            module.stage_one_input(item, damaged, parts[0].fem_key)


@pytest.mark.parametrize("damage", ["wall", "speakers", "seats", "report-id", "frequency", "candidate"])
def test_result_file_that_disagrees_with_its_scheme_is_refused(
        monkeypatch: pytest.MonkeyPatch, damage: str) -> None:
    """結果檔各欄都要對得上它自己帶的方案；只改一格就拒收。"""
    module = importlib.import_module("aosr.reporting.stage_one_result")
    item = cases.scheme()
    result = module.run_stage_one(item, shards=cases.shards(monkeypatch, item), **cases.run_kwargs())
    data = result.model_dump(mode="json")
    module.StageOneResult.model_validate(data)
    first = data["curves"][0]
    if damage == "wall":
        data["wall_assumption_text"] += "。"
    elif damage == "speakers":
        data["curves"] = data["curves"][:1]
    elif damage == "seats":
        one, two = first["surrounding"][:2]
        one["direction_relative_to_primary"], two["direction_relative_to_primary"] = (
            two["direction_relative_to_primary"], one["direction_relative_to_primary"])
    elif damage == "report-id":
        first["primary"]["report_id"] = "fem-other-main"
    elif damage == "frequency":
        first["primary"]["frequencies_hz"][0] += 0.5
    else:
        data["evaluation"]["candidate_id"] = "other"
    with pytest.raises(ValidationError):
        module.StageOneResult.model_validate(data)
