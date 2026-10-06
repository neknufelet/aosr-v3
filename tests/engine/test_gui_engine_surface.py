"""GUI 所需的引擎公開邊界。"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import cast

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults, load_directivity_defaults
from aosr.config.paths import config_path
from aosr.reporting import pipeline
from aosr.reporting import validation
from aosr.physics import three_lane_report
from aosr.physics.report_source import default_source_model
from aosr.geometry.shoebox import Point
from aosr.reporting.display import level_db, impedance_multiple, LOW_FREQUENCY_DECAY_NOTE
from aosr.reporting.validation import validate_scheme
from aosr.reporting.result import SchemeResult, save_result


def _inputs() -> tuple[CapabilityTable, DirectivityDefaults]:
    return (load_capabilities(config_path("capabilities.toml")),
            load_directivity_defaults(config_path("directivity_defaults.toml")))


@pytest.mark.parametrize("change,expected", [
    ("missing_wall", "ceiling"), ("negative_impedance", "floor"),
    ("bad_scattering", "scattering_by_wall"),
    ("speaker_on_primary", "source_model.aim_m"),
    ("aim_outside", "aim_m"), ("bad_scheme", "scheme_id"),
])
def test_validation_matches_run_before_solver(change: str, expected: str,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    document = json.loads(Path("blueprint/scheme_reference_room.json").read_text())
    scene = document["scene"]
    if change == "missing_wall":
        del scene["impedance_pa_s_per_m_by_wall"]["ceiling"]
    elif change == "negative_impedance":
        scene["impedance_pa_s_per_m_by_wall"]["floor"] = -1
    elif change == "bad_scattering":
        scene["scattering_by_wall"] = {name: 2 for name in
                                       scene["impedance_pa_s_per_m_by_wall"]}
    elif change == "speaker_on_primary":
        document["speakers"]["left"] = {"x": 3.2, "y": 1.9, "z": 1.2}
    elif change == "bad_scheme":
        document["scheme_id"] = " "
    monkeypatch.setattr(three_lane_report, "solve_three_lane_reports",
                        lambda **kwargs: pytest.fail("求解不應開始"))
    capabilities, directivity = _inputs()
    if change == "aim_outside":
        class OutsideAim:
            def model_dump(self, *, mode: str) -> dict[str, object]:
                model = default_source_model(Point(3.2, 1.9, 1.2), directivity)
                document = model.model_dump(mode="json")
                document["aim_m"] = {"x": 1000.0, "y": 1.9, "z": 1.2}
                return document

        monkeypatch.setattr(validation, "default_source_model",
                            lambda aim, defaults: OutsideAim())
    problems = validate_scheme(document, capabilities=capabilities, directivity=directivity)
    assert problems and expected in "；".join(str(item) for item in problems)
    if change == "speaker_on_primary":
        assert any(item.path == "pairs.left.main.source_model.aim_m" for item in problems)
    with pytest.raises(ValueError) as exc:
        pipeline.run_scheme(document, capabilities=capabilities, directivity=directivity,
                            quality_targets_path=config_path("quality_targets.toml"),
                            engine_commit="test", program_fingerprint="calc-v1:" + "0" * 64, physics_identity="phys-v1:" + "0" * 64, run_date=date(2026, 9, 28))
    assert str(exc.value) == "；".join(str(item) for item in problems)


def test_shared_scene_problem_has_one_structured_path() -> None:
    document = json.loads(Path("blueprint/scheme_reference_room.json").read_text())
    document["scene"]["impedance_pa_s_per_m_by_wall"]["floor"] = -1
    capabilities, directivity = _inputs()
    problems = validate_scheme(document, capabilities=capabilities, directivity=directivity)
    assert len(problems) == len({(item.path, item.message) for item in problems})
    assert {item.path for item in problems} == {"scene.impedance_pa_s_per_m_by_wall.floor"}


def test_display() -> None:
    assert level_db(1.0) == 0.0
    assert level_db(100.0) == 20.0
    assert level_db(0.0) is None
    assert level_db(-1.0) is None
    assert "不計分" in LOW_FREQUENCY_DECAY_NOTE and "模態診斷報告" in LOW_FREQUENCY_DECAY_NOTE
    assert impedance_multiple(411.6, 1.2, 343.0) == pytest.approx(1.0)


def test_save_result_keeps_old_file_when_replace_fails(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        def model_dump_json(self) -> str:
            return "new"

    target = tmp_path / "result.json"
    target.write_text("old")

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        save_result(cast(SchemeResult, Result()), target)
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]


def test_save_result_cleans_partial_temp_when_write_fails(tmp_path: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    class Result:
        def model_dump_json(self) -> str:
            return "new payload"

    target = tmp_path / "result.json"
    target.write_text("old")
    real_fdopen = os.fdopen

    class PartialWriter:
        def __init__(self, descriptor: int, mode: str, encoding: str) -> None:
            self.handle = real_fdopen(descriptor, mode, encoding=encoding)

        def __enter__(self) -> "PartialWriter":
            return self

        def __exit__(self, *_args: object) -> None:
            self.handle.close()

        def write(self, value: str) -> None:
            self.handle.write(value[:3])
            raise OSError("partial write")

    monkeypatch.setattr(os, "fdopen", PartialWriter)
    with pytest.raises(OSError, match="partial write"):
        save_result(cast(SchemeResult, Result()), target)
    assert target.read_text() == "old"
    assert list(tmp_path.iterdir()) == [target]
