"""方案邊界與報表能力查表的施工考卷。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

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


def test_scheme_requires_source_model_even_when_key_is_absent() -> None:
    document = _document()
    del document["source_model"]
    with pytest.raises(ValueError, match="source_model"):
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


@pytest.mark.parametrize("change", [
    {"channels": [{"role": "left", "speaker_id": "left"},
                  {"role": "right", "speaker_id": "right"},
                  {"role": "center", "speaker_id": "center"}]},
    {"comparisons": [{"left_role": "left", "right_role": "right"},
                     {"left_role": "right", "right_role": "left"}]},
])
def test_scheme_rejects_extra_channel_or_comparison(change: dict[str, object]) -> None:
    document = _document()
    group_document = document["channel_group"]
    assert isinstance(group_document, dict)
    document["channel_group"] = group_document | change
    with pytest.raises(ValueError, match="剛好兩個聲道角色與一個比較對"):
        Scheme.model_validate(document)


@pytest.mark.parametrize("field", ["scheme_id", "purpose"])
def test_scheme_rejects_blank_identity(field: str) -> None:
    document = _document() | {field: "   "}
    with pytest.raises(ValueError, match="scheme_id 與 purpose 不可為空白"):
        Scheme.model_validate(document)


@pytest.mark.parametrize("axis,limit", [("x", 6.3), ("y", 4.1), ("z", 2.9)])
@pytest.mark.parametrize("point_kind", ["speaker", "receiver"])
def test_scheme_room_bounds_name_the_point(axis: str, limit: float, point_kind: str) -> None:
    for coordinate in (0.0, limit):
        document = _document()
        _replace_point(document, point_kind, axis, coordinate)
        Scheme.model_validate(document)
    for coordinate in (-0.01, limit + 0.01):
        document = _document()
        _replace_point(document, point_kind, axis, coordinate)
        with pytest.raises(ValidationError) as exc:
            Scheme.model_validate(document)
        point_name = "喇叭 left" if point_kind == "speaker" else "座位 front"
        assert point_name in exc.value.errors()[0]["msg"]


def _replace_point(document: dict[str, object], point_kind: str,
                   axis: str, coordinate: float) -> None:
    if point_kind == "speaker":
        document["speakers"] = SPEAKERS | {
            "left": SPEAKERS["left"] | {axis: coordinate}}
        return
    original_set = document["receiver_set"]
    assert isinstance(original_set, dict)
    receiver_set = dict(original_set)
    points = list(receiver_set["points"])
    position = dict(zip(("x", "y", "z"), points[1]["position_m"], strict=True))
    position[axis] = coordinate
    points[1] = points[1] | {"position_m": tuple(position.values())}
    receiver_set["points"] = points
    document["receiver_set"] = receiver_set


def test_reference_scheme_loads() -> None:
    path = Path(__file__).resolve().parents[2] / "blueprint/scheme_reference_room.json"
    scheme = load_scheme(path)
    assert scheme.scheme_id == "reference-room-original"
    assert scheme.receiver_set.primary.position_m == (3.2, 1.9, 1.2)
    # 聆聽區第一版只管主位 ±10 cm（#349）：周圍六點各只在一軸上離主位 0.1 m。
    primary = scheme.receiver_set.primary.position_m
    offsets = sorted(
        tuple(round(value - origin, 9) for value, origin in zip(point.position_m, primary, strict=True))
        for point in scheme.receiver_set.points if point.receiver_id != scheme.receiver_set.primary.receiver_id
    )
    assert offsets == sorted([(0.1, 0.0, 0.0), (-0.1, 0.0, 0.0), (0.0, 0.1, 0.0),
                              (0.0, -0.1, 0.0), (0.0, 0.0, 0.1), (0.0, 0.0, -0.1)])
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
