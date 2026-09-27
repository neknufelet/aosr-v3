"""方案邊界與報表能力查表的施工考卷。"""

from __future__ import annotations

from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.physics.report_io import load_input_document
from aosr.physics.report_source import default_source_model
from aosr.geometry.shoebox import Point
from aosr.physics.report_output import report_capability
from aosr.reporting.scheme import Scheme, load_scheme
from tests.engine._scoring_source_model_control import group, receivers, SPEAKERS
from tests.engine._source_model_control import SCENE_WITHOUT_SOURCE_MODEL


def _document() -> dict[str, object]:
    scene = {key: value for key, value in SCENE_WITHOUT_SOURCE_MODEL.items()
             if key not in {"source_m", "receiver_m"}}
    return {"schema_version": "aosr.scheme.v1", "scheme_id": "wall-1",
            "purpose": "dedicated_two_channel_listening_room", "scene": scene,
            "source_model": "omnidirectional", "speakers": SPEAKERS,
            "channel_group": group().model_dump(mode="json"),
            "receiver_set": receivers().model_dump(mode="json")}


@pytest.mark.parametrize("change,missing", [
    ({"source_model": None}, "source_model"),
    ({"source_model": "unknown"}, "source_model"),
    ({"schema_version": "wrong"}, "schema_version"),
])
def test_scheme_rejects_invalid_identity(change: dict[str, object], missing: str) -> None:
    document = _document() | change
    with pytest.raises(ValueError, match=missing):
        Scheme.model_validate(document)


@pytest.mark.parametrize("field", ["sound_speed_m_s", "density_kg_m3"])
def test_scheme_requires_medium(field: str) -> None:
    document = _document()
    original_scene = document["scene"]
    assert isinstance(original_scene, dict)
    scene = dict(original_scene)
    scene.pop(field)
    document["scene"] = scene
    with pytest.raises(ValueError, match=field):
        Scheme.model_validate(document)


def test_scheme_rejects_missing_or_unused_speaker() -> None:
    document = _document()
    document["speakers"] = {"left": SPEAKERS["left"]}
    with pytest.raises(ValueError, match="right"):
        Scheme.model_validate(document)
    document["speakers"] = SPEAKERS | {"extra": SPEAKERS["left"]}
    with pytest.raises(ValueError, match="extra"):
        Scheme.model_validate(document)


def test_scheme_rejects_single_channel() -> None:
    document = _document()
    document["channel_group"] = {
        "channels": [{"role": "left", "speaker_id": "left"}],
        "comparisons": [], "feature_match_tolerance_hz": 10.0,
    }
    with pytest.raises(ValueError, match="兩個聲道"):
        Scheme.model_validate(document)


def test_scheme_rejects_outside_speaker_and_receiver() -> None:
    document = _document()
    document["speakers"] = SPEAKERS | {"left": {"x": 7.0, "y": 1.0, "z": 1.0}}
    with pytest.raises(ValueError, match="left"):
        Scheme.model_validate(document)
    document = _document()
    original_set = document["receiver_set"]
    assert isinstance(original_set, dict)
    receiver_set = dict(original_set)
    points = list(receiver_set["points"])
    points[1] = points[1] | {"position_m": (-1.0, 1.0, 1.0)}
    receiver_set["points"] = points
    document["receiver_set"] = receiver_set
    with pytest.raises(ValueError, match="front"):
        Scheme.model_validate(document)


def test_reference_scheme_loads(path: Path = Path("blueprint/scheme_reference_room.json")) -> None:
    scheme = load_scheme(path)
    assert scheme.scheme_id == "reference-room-original"
    assert scheme.receiver_set.primary.position_m == (3.2, 1.9, 1.2)
    table = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    model = default_source_model(Point(*scheme.receiver_set.primary.position_m),
                                 directivity).model_dump(mode="json")
    for speaker in scheme.speakers.values():
        for receiver in scheme.receiver_set.points:
            document = scheme.scene.model_dump(mode="json", exclude_unset=True) | {
                "source_model": model,
                "source_m": {"x": speaker.x, "y": speaker.y, "z": speaker.z},
                "receiver_m": dict(zip(("x", "y", "z"), receiver.position_m, strict=True)),
            }
            assert load_input_document(document, table, directivity).receiver_m.as_tuple() == receiver.position_m


def test_report_capability_rejects_unsupported() -> None:
    table = load_capabilities(config_path("capabilities.toml"))
    assert report_capability(table).record is not None
    entry = table.for_entry("three_lane_report")
    changed = entry.model_copy(update={"capability": tuple(
        row.model_copy(update={"status": "unsupported", "evidence": ()})
        if row.room == "shoebox" and row.materials == "real_frequency_independent_impedance"
        else row for row in entry.capability)})
    unsupported = table.model_copy(update={"entry": tuple(
        changed if item.name == entry.name else item for item in table.entry)})
    with pytest.raises(ValueError, match="three_lane_report"):
        report_capability(unsupported)
