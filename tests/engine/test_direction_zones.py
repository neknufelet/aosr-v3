"""反射分區與兩聲道聆聽軸的幾何考卷。"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.direction_zones import (
    DirectionZone,
    ListeningAxisUndefined,
    ZoneLimits,
    classify,
    listening_angles,
    listening_axis,
    zone_limits,
)


_PURPOSE = "dedicated_two_channel_listening_room"


def test_boundaries_and_shallow_floor_follow_angles() -> None:
    limits, _ = zone_limits(load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE))
    assert classify(0.0, limits.vertical_min_abs_elevation_deg, limits) is DirectionZone.VERTICAL
    assert classify(0.0, -limits.vertical_min_abs_elevation_deg, limits) is DirectionZone.VERTICAL
    assert classify(limits.front_max_abs_azimuth_deg, 0.0, limits) is DirectionZone.LATERAL
    assert classify(limits.rear_min_abs_azimuth_deg, 0.0, limits) is DirectionZone.LATERAL
    assert classify(math.nextafter(limits.rear_min_abs_azimuth_deg, math.inf), 0.0, limits) is DirectionZone.REAR
    # 手算鏡像方向：水平向前 10、向下 1，仰角約 -5.7°，仍屬前向。
    azimuth, elevation = listening_angles((10.0, 0.0, -1.0), (1.0, 0.0))
    assert elevation > -limits.vertical_min_abs_elevation_deg
    assert classify(azimuth, elevation, limits) is DirectionZone.FRONT


def test_axis_rotation_and_left_right_sign() -> None:
    axis = listening_axis(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (0.0, -2.0, 0.0))
    assert axis == pytest.approx((0.0, 1.0))
    left = listening_angles((-1.0, 1.0, 0.0), axis)[0]
    right = listening_angles((1.0, 1.0, 0.0), axis)[0]
    assert left == pytest.approx(-right)
    assert left > 0.0
    rotated_axis = listening_axis(((0.0, -1.0, 0.0), (0.0, 1.0, 0.0)), (2.0, 0.0, 0.0))
    assert listening_angles((-1.0, 1.0, 0.0), axis)[0] == pytest.approx(
        listening_angles((-1.0, -1.0, 0.0), rotated_axis)[0]
    )


def test_axis_uses_primary_receiver_for_surrounding_points() -> None:
    speakers = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    main = (0.0, -2.0, 0.0)
    surrounding = (0.3, 0.5, 0.0)
    axis = listening_axis(speakers, main)
    direction_from_surrounding = (0.0, 1.0, 0.0)
    assert listening_angles(direction_from_surrounding, axis)[0] == pytest.approx(0.0)
    assert listening_axis(speakers, surrounding) == pytest.approx((0.0, -1.0))
    assert listening_angles(direction_from_surrounding, listening_axis(speakers, surrounding))[0] == 180.0


def test_exactly_opposite_direction_uses_positive_180_degrees() -> None:
    azimuth, elevation = listening_angles((0.0, -1.0, 0.0), (0.0, 1.0))
    assert azimuth == 180.0
    assert elevation == 0.0


def test_axis_follows_main_receiver_to_speaker_midpoint_on_both_sides() -> None:
    speakers = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    front_axis = listening_axis(speakers, (0.0, -2.0, 0.0))
    opposite_axis = listening_axis(speakers, (0.0, 2.0, 0.0))
    assert opposite_axis == pytest.approx(tuple(-component for component in front_axis))
    for receiver, axis in (((0.0, -2.0, 0.0), front_axis), ((0.0, 2.0, 0.0), opposite_axis)):
        toward_midpoint = (0.0 - receiver[0], 0.0 - receiver[1])
        assert sum(a * b for a, b in zip(axis, toward_midpoint)) > 0.0


def test_three_speakers_raise_listening_axis_undefined() -> None:
    speakers = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    with pytest.raises(ListeningAxisUndefined, match="聆聽軸定不出來"):
        listening_axis(speakers, (0.0, -2.0, 0.0))


@pytest.mark.parametrize("speakers,receiver", [
    (((0.0, 0.0, 0.0),), (0.0, -1.0, 0.0)),
    (((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), (0.0, -1.0, 0.0)),
    (((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (0.0, 0.0, 0.0)),
])
def test_axis_rejects_undefined_geometry(speakers: tuple[tuple[float, float, float], ...], receiver: tuple[float, float, float]) -> None:
    with pytest.raises(ListeningAxisUndefined, match="聆聽軸定不出來"):
        listening_axis(speakers, receiver)


@pytest.mark.parametrize("speakers,receiver", [
    (((0.0, 0.0, 0.0), (float("inf"), 0.0, 0.0)), (0.0, -1.0, 0.0)),
    (((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (0.0, float("nan"), 0.0)),
])
def test_axis_rejects_nonfinite_span_or_side(speakers: tuple[tuple[float, float, float], ...],
                                              receiver: tuple[float, float, float]) -> None:
    with pytest.raises(ListeningAxisUndefined, match="聆聽軸定不出來"):
        listening_axis(speakers, receiver)


def test_limits_load_three_baselines_and_validate_order() -> None:
    limits, statuses = zone_limits(load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE))
    assert set(statuses) == {
        "direction_zones.vertical_min_abs_elevation_deg",
        "direction_zones.front_max_abs_azimuth_deg",
        "direction_zones.rear_min_abs_azimuth_deg",
    }
    assert set(statuses.values()) == {"baseline"}
    with pytest.raises(ValueError, match="前向"):
        ZoneLimits(vertical_min_abs_elevation_deg=limits.vertical_min_abs_elevation_deg,
                   front_max_abs_azimuth_deg=limits.rear_min_abs_azimuth_deg,
                   rear_min_abs_azimuth_deg=limits.front_max_abs_azimuth_deg)


@pytest.mark.parametrize("degrees", (17.0, -73.0, 113.0))
def test_arbitrary_scene_rotation_preserves_relative_direction(degrees: float) -> None:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)

    def rotate(point: tuple[float, float, float]) -> tuple[float, float, float]:
        x, y, z = point
        return (x * cosine - y * sine, x * sine + y * cosine, z)

    speakers = ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    main = (0.0, -2.0, 0.0)
    direction = (-1.0, 1.0, 0.1)
    original = listening_angles(direction, listening_axis(speakers, main))
    rotated = listening_angles(rotate(direction), listening_axis(tuple(rotate(s) for s in speakers), rotate(main)))
    limits, _ = zone_limits(load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE))
    assert rotated == pytest.approx(original)
    assert classify(*original, limits) is DirectionZone.LATERAL
    assert classify(*rotated, limits) is classify(*original, limits)


def test_axis_uses_actual_speaker_midpoint_for_side() -> None:
    assert listening_axis(((0.0, 10.0, 0.0), (2.0, 10.0, 0.0)), (1.0, 11.0, 0.0)) == pytest.approx((0.0, -1.0))


def test_downward_elevation_is_negative() -> None:
    assert listening_angles((1.0, 0.0, -1.0), (1.0, 0.0))[1] == pytest.approx(-45.0)


@pytest.mark.parametrize("vector,axis", [
    ((0.0, 0.0, 0.0), (1.0, 0.0)),
    ((1.0, 0.0, 0.0), (0.0, 0.0)),
    ((float("inf"), 0.0, 0.0), (1.0, 0.0)),
])
def test_listening_angles_rejects_zero_or_nonfinite_vectors(vector: tuple[float, float, float],
                                                              axis: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="方向或聆聽軸必須是非零有限向量"):
        listening_angles(vector, axis)


@pytest.mark.parametrize("azimuth,elevation", [(float("nan"), 0.0), (0.0, float("inf"))])
def test_classify_rejects_nonfinite_angles(azimuth: float, elevation: float) -> None:
    limits, _ = zone_limits(load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE))
    with pytest.raises(ValueError, match="角度必須是有限數"):
        classify(azimuth, elevation, limits)


@pytest.mark.parametrize("field,value,message", [
    ("vertical_min_abs_elevation_deg", 0.0, "greater_than"),
    ("vertical_min_abs_elevation_deg", 90.0, "less_than"),
    ("front_max_abs_azimuth_deg", 0.0, "greater_than"),
    ("rear_min_abs_azimuth_deg", 180.0, "less_than"),
    ("front_max_abs_azimuth_deg", 140.0, "前向與後向界線"),
])
def test_zone_limits_rejects_each_bound(field: str, value: float, message: str) -> None:
    document = ZoneLimits(vertical_min_abs_elevation_deg=30.0, front_max_abs_azimuth_deg=40.0,
                          rear_min_abs_azimuth_deg=135.0).model_dump(mode="python")
    document[field] = value
    with pytest.raises(ValidationError, match=message):
        ZoneLimits.model_validate(document)


def _small_registry(tmp_path: Path, *, unit: str = "deg", status: str = "calibrated",
                    value: float | int = 30.0, omit: bool = False) -> Path:
    path = tmp_path / "quality_targets.toml"
    entries = (
        ("vertical_min_abs_elevation_deg", value, unit, status),
        ("front_max_abs_azimuth_deg", 40.0, "deg", "baseline"),
        ("rear_min_abs_azimuth_deg", 135.0, "deg", "baseline"),
    )
    lines = ['schema_version = 1', '[[purpose]]', f'name = "{_PURPOSE}"',
             'target = []', 'weight = []', 'qualification = []']
    for key, value, entry_unit, entry_status in entries:
        if omit and key == "rear_min_abs_azimuth_deg":
            continue
        lines.extend(['[[purpose.setting]]', f'key = "direction_zones.{key}"', f'value = {value}',
                      f'unit = "{entry_unit}"', f'status = "{entry_status}"',
                      'source_kind = "product_choice"', 'source = "test registry"'])
        if entry_status == "calibrated":
            lines.extend(['source_id = "local test"', 'source_version = "1"', 'locator = "entry 1"',
                          'conditions = "unit test"', 'frequency_range_hz = [1000.0, 8000.0]',
                          'context = "temporary fixture"', f'verification_digest = "sha256:{"0" * 64}"'])
    path.write_text("\n".join(lines) + "\n")
    return path


def test_zone_limits_reports_changed_registry_status(tmp_path: Path) -> None:
    purpose = load_quality_targets(_small_registry(tmp_path)).purpose(_PURPOSE)
    limits, statuses = zone_limits(purpose)
    assert limits.vertical_min_abs_elevation_deg == 30.0
    assert statuses["direction_zones.vertical_min_abs_elevation_deg"] == "calibrated"
    assert statuses["direction_zones.front_max_abs_azimuth_deg"] == "baseline"


def test_zone_limits_rejects_wrong_registry_unit(tmp_path: Path) -> None:
    purpose = load_quality_targets(_small_registry(tmp_path, unit="ms", status="baseline")).purpose(_PURPOSE)
    with pytest.raises(ValueError, match="必須是 deg 浮點設定"):
        zone_limits(purpose)


def test_zone_limits_rejects_nonfloat_registry_value(tmp_path: Path) -> None:
    purpose = load_quality_targets(_small_registry(tmp_path, status="baseline", value=30)).purpose(_PURPOSE)
    with pytest.raises(ValueError, match="必須是 deg 浮點設定"):
        zone_limits(purpose)


def test_zone_limits_rejects_missing_registry_key(tmp_path: Path) -> None:
    purpose = load_quality_targets(_small_registry(tmp_path, status="baseline", omit=True)).purpose(_PURPOSE)
    with pytest.raises(KeyError, match="rear_min_abs_azimuth_deg"):
        zone_limits(purpose)


def test_mirrored_reflections_share_zone() -> None:
    limits, _ = zone_limits(load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE))
    axis = listening_axis(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (0.0, -2.0, 0.0))
    left = listening_angles((-1.0, 1.0, 0.0), axis)
    right = listening_angles((1.0, 1.0, 0.0), axis)
    assert left[0] == pytest.approx(-right[0])
    assert classify(*left, limits) is classify(*right, limits)
