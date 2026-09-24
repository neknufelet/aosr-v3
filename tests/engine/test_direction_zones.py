"""反射分區與兩聲道聆聽軸的幾何考卷。"""
from __future__ import annotations

import math

import pytest

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
    axis = listening_axis(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (0.0, -2.0, 0.0))
    assert listening_angles((0.0, 1.0, 0.0), axis)[0] == pytest.approx(0.0)
    assert listening_angles((0.0, -1.0, 0.0), axis)[0] == pytest.approx(180.0)


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
    direction = (-1.0, 1.0, 1.0)
    original = listening_angles(direction, listening_axis(speakers, main))
    rotated = listening_angles(rotate(direction), listening_axis(tuple(rotate(s) for s in speakers), rotate(main)))
    limits, _ = zone_limits(load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE))
    assert rotated == pytest.approx(original)
    assert classify(*rotated, limits) is classify(*original, limits)


def test_mirrored_reflections_share_zone() -> None:
    limits, _ = zone_limits(load_quality_targets(config_path("quality_targets.toml")).purpose(_PURPOSE))
    axis = listening_axis(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), (0.0, -2.0, 0.0))
    left = listening_angles((-1.0, 1.0, 0.0), axis)
    right = listening_angles((1.0, 1.0, 0.0), axis)
    assert left[0] == pytest.approx(-right[0])
    assert classify(*left, limits) is classify(*right, limits)
