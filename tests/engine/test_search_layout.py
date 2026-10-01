"""搜尋擺位契約的考卷；抓缺少設定、參數換算與方案建構。"""

import importlib.util
from pathlib import Path


def test_layout_contract_is_available(tmp_path: Path) -> None:
    """新契約缺席必須紅，不能把收集錯誤當成紅燈證据。"""
    assert tmp_path.is_dir()
    for module in ("layout_settings", "layout", "constraints"):
        assert importlib.util.find_spec(f"aosr.search.{module}") is not None

import math
import shutil
from dataclasses import FrozenInstanceError, replace
from typing import Literal

import pytest
from pydantic import ValidationError

from aosr.geometry.shoebox import Point, distance
from aosr.reporting.scheme import Scheme, load_scheme
from aosr.search.layout import (
    UNIT_SPACE, LayoutParams, params_from_unit, place, standard_start,
    to_scheme, unit_from_params,
)
from aosr.search.layout_settings import Box, Cabinet, LayoutSettings, Span


@pytest.fixture
def project(tmp_path: Path) -> Scheme:
    source = next((Path(__file__).resolve().parents[2] / "blueprint").glob("scheme_reference_room.*"))
    target = tmp_path / "reference"
    shutil.copyfile(source, target)
    return load_scheme(target)


@pytest.fixture
def settings(tmp_path: Path) -> LayoutSettings:
    assert tmp_path.is_dir()
    return LayoutSettings(
        front_wall="x0", axis_offset_m=-0.1, speaker_height_m=1.2, ear_height_m=1.2,
        front_distance_m=Span(low=0.2, high=2.0), spacing_m=Span(low=0.2, high=2.0),
        listening_distance_m=Span(low=0.2, high=3.0),
        cabinet=Cabinet(width_m=0.2, depth_m=0.2, height_m=0.4),
    )


def test_reference_room_parameters_reproduce_reference_scheme(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
) -> None:
    result = to_scheme(project, place(project, settings, LayoutParams(1.0, 1.2, 2.2)), "reproduced")
    for key, point in project.speakers.items():
        assert result.speakers[key].as_tuple() == pytest.approx(point.as_tuple(), rel=0.0, abs=1e-12)
    for actual, original in zip(result.receiver_set.points, project.receiver_set.points, strict=True):
        assert actual.position_m == pytest.approx(original.position_m, rel=0.0, abs=1e-12)
        assert actual.model_dump(exclude={"position_m"}) == original.model_dump(exclude={"position_m"})
    assert result.channel_group == project.channel_group
    assert result.scene == project.scene
    assert result.source_model == project.source_model
    assert result.purpose == project.purpose
    assert result.receiver_set.layout_fingerprint == project.receiver_set.layout_fingerprint
    with (tmp_path / "roundtrip").open("w", encoding="utf-8") as handle:
        handle.write(result.model_dump_json())


@pytest.mark.parametrize("wall,axis,sign,across", [
    ("x0", 0, 1.0, 1), ("xL", 0, -1.0, 1),
    ("y0", 1, 1.0, 0), ("yL", 1, -1.0, 0),
])
def test_speakers_symmetric_and_seat_on_axis(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
    wall: Literal["x0", "xL", "y0", "yL"], axis: int, sign: float, across: int,
) -> None:
    chosen = LayoutSettings.model_validate(settings.model_dump() | {"front_wall": wall, "axis_offset_m": 0.25})
    placed = place(project, chosen, LayoutParams(1.0, 1.2, 1.5))
    left, right, main = placed.left.as_tuple(), placed.right.as_tuple(), placed.primary.as_tuple()
    centre = project.scene.room_m.length(across) / 2 + 0.25
    assert left[axis] == right[axis]
    assert left[axis] == (1.0 if sign > 0 else project.scene.room_m.length(axis) - 1.0)
    assert (main[axis] - left[axis]) * sign == pytest.approx(1.5)
    assert main[across] == centre
    assert (left[across] + right[across]) / 2 == centre
    assert abs(left[across] - centre) == pytest.approx(0.6)
    assert abs(right[across] - centre) == pytest.approx(0.6)
    assert left[2] == right[2] == chosen.speaker_height_m
    assert main[2] == chosen.ear_height_m
    assert tmp_path.is_dir()


@pytest.mark.parametrize("wall,facing", [
    ("x0", (-1.0, 0.0)), ("xL", (1.0, 0.0)),
    ("y0", (0.0, -1.0)), ("yL", (0.0, 1.0)),
])
def test_left_is_the_listeners_left_on_every_wall(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
    wall: Literal["x0", "xL", "y0", "yL"], facing: tuple[float, float],
) -> None:
    chosen = LayoutSettings.model_validate(settings.model_dump() | {"front_wall": wall})
    placed = place(project, chosen, LayoutParams(1.0, 1.2, 1.5))
    assert placed.facing == facing
    dx, dy = placed.left.x - placed.right.x, placed.left.y - placed.right.y
    assert facing[0] * dy - facing[1] * dx > 0
    renamed = project.model_dump()
    renamed["speakers"] = {"monitor-A": project.speakers["left"], "monitor-B": project.speakers["right"]}
    renamed["channel_group"] = project.channel_group.model_dump() | {"channels": (
        {"role": "right", "speaker_id": "monitor-B"}, {"role": "left", "speaker_id": "monitor-A"},
    )}
    result = to_scheme(Scheme.model_validate(renamed), placed, "renamed")
    assert result.speakers == {"monitor-A": placed.left, "monitor-B": placed.right}
    assert tmp_path.is_dir()


def test_same_wall_translation_keeps_layout_fingerprint(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
) -> None:
    first = to_scheme(project, place(project, settings, LayoutParams(1.0, 1.2, 2.2)), "first")
    second = to_scheme(project, place(project, settings, LayoutParams(1.25, 1.5, 1.5)), "second")
    assert first.receiver_set.layout_fingerprint == second.receiver_set.layout_fingerprint
    chosen = LayoutSettings.model_validate(settings.model_dump() | {"front_wall": "y0"})
    rotated = place(project, chosen, LayoutParams(1.0, 1.2, 1.5))
    front = dict(rotated.receivers)["front"]
    assert front.x == rotated.primary.x
    assert front.y < rotated.primary.y
    assert tmp_path.is_dir()


@pytest.mark.parametrize("original_turn,speakers", [
    (0, (Point(1.0, 1.5, 1.25), Point(1.0, 2.5, 1.25))),
    (1, (Point(3.5, 0.0, 1.25), Point(2.5, 0.0, 1.25))),
    (2, (Point(5.0, 2.5, 1.25), Point(5.0, 1.5, 1.25))),
    (3, (Point(2.5, 4.0, 1.25), Point(3.5, 4.0, 1.25))),
])
@pytest.mark.parametrize("wall,turn", [("x0", 0), ("y0", 1), ("xL", 2), ("yL", 3)])
def test_rotation_is_exact(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
    wall: Literal["x0", "xL", "y0", "yL"], turn: int,
    original_turn: int, speakers: tuple[Point, Point],
) -> None:
    # 二進位可精確表示的非對称偏移，隔離座標加減的捨入，專抓旋轉引入的失真。
    origin = (3.0, 2.0, 1.25)
    offsets = ((0.0, 0.0, 0.0), (-0.125, 0.25, 0.0625), (0.375, -0.5, -0.125))
    points = tuple(point.model_dump() | {"position_m": tuple(a + b for a, b in zip(origin, offset, strict=True))}
                   for point, offset in zip(project.receiver_set.points[:3], offsets, strict=True))
    receivers = project.receiver_set.model_validate({"points": points})
    changed = Scheme.model_validate(project.model_dump() | {
        "speakers": {"left": speakers[0], "right": speakers[1]},
        "receiver_set": receivers,
    })
    chosen = LayoutSettings.model_validate(settings.model_dump() | {
        "front_wall": wall, "axis_offset_m": 0.0, "ear_height_m": 1.25,
    })
    placed = place(changed, chosen, LayoutParams(1.0, 1.0, 1.0))
    for (_, actual), offset in zip(placed.receivers, offsets, strict=True):
        x, y, z = offset
        expected = ((x, y, z), (-y, x, z), (-x, -y, z), (y, -x, z))[(turn - original_turn) % 4]
        actual_offset = tuple(a - b for a, b in zip(actual.as_tuple(), placed.primary.as_tuple(), strict=True))
        # 零分量的正負零不代表幾何失真；其他位元須逐字相同。
        assert tuple((v if v else 0.0).hex() for v in actual_offset) == tuple((v if v else 0.0).hex() for v in expected)
    assert tmp_path.is_dir()


def test_unit_mapping_round_trips_and_rejects_out_of_range(tmp_path: Path, settings: LayoutSettings) -> None:
    assert set(UNIT_SPACE) == {"front_distance", "spacing", "listening_distance"}
    assert all(bounds == (0.0, 1.0) for bounds in UNIT_SPACE.values())
    for u in (0.0, 0.25, 0.75, 1.0):
        unit = dict.fromkeys(UNIT_SPACE, u)
        params = params_from_unit(unit, settings)
        assert unit_from_params(params, settings) == pytest.approx(unit, abs=1e-12)
        assert params.front_distance_m == pytest.approx(0.2 + u * 1.8)
        assert params.spacing_m == pytest.approx(0.2 + u * 1.8)
        assert params.listening_distance_m == pytest.approx(0.2 + u * 2.8)
    for bad in (-0.001, 1.001, math.inf, math.nan):
        for name in UNIT_SPACE:
            with pytest.raises(ValueError):
                params_from_unit(dict.fromkeys(UNIT_SPACE, 0.5) | {name: bad}, settings)
    with pytest.raises(ValueError):
        unit_from_params(LayoutParams(3.0, 1.0, 1.0), settings)
    with pytest.raises(ValueError):
        params_from_unit({}, settings)
    assert tmp_path.is_dir()


def test_standard_start_is_equilateral_or_none(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    params = standard_start(project, settings)
    assert params is not None
    assert params.front_distance_m == 1.0
    assert params.spacing_m == 1.2
    placed = place(project, settings, params)
    assert distance(placed.left, placed.right) == pytest.approx(distance(placed.left, placed.primary), rel=0.0, abs=1e-12)
    assert distance(placed.left, placed.right) == pytest.approx(distance(placed.right, placed.primary), rel=0.0, abs=1e-12)
    for field in ("front_distance_m", "spacing_m", "listening_distance_m"):
        changed = LayoutSettings.model_validate(settings.model_dump() | {field: {"low": 1.5, "high": 2.0}})
        assert standard_start(project, changed) is None
    assert tmp_path.is_dir()


@pytest.mark.parametrize("wall,expected", [("x0", 1.0), ("xL", 1.0), ("y0", 1.0), ("yL", 1.0)])
def test_standard_start_keeps_the_projects_own_front_distance_on_any_wall(
    tmp_path: Path, project: Scheme, settings: LayoutSettings,
    wall: Literal["x0", "xL", "y0", "yL"], expected: float,
) -> None:
    chosen = LayoutSettings.model_validate(settings.model_dump() | {
        "front_wall": wall, "front_distance_m": {"low": 0.1, "high": 6.0},
    })
    result = standard_start(project, chosen)
    assert result is not None
    assert result.front_distance_m == pytest.approx(expected)
    assert result.spacing_m == pytest.approx(1.2)
    assert tmp_path.is_dir()


def test_project_seat_off_axis_is_refused(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    moved = tuple(p.model_dump() | {"position_m": (p.position_m[0], p.position_m[1] + 0.01, p.position_m[2])}
                  for p in project.receiver_set.points)
    changed = Scheme.model_validate(project.model_dump() | {"receiver_set": {"points": moved}})
    with pytest.raises(ValueError):
        place(changed, settings, LayoutParams(1.0, 1.2, 2.2))
    assert tmp_path.is_dir()


@pytest.mark.parametrize("model,values", [
    (Span, {"low": 1.0, "high": 1.0}), (Span, {"low": math.nan, "high": 1.0}),
    (Span, {"low": 0.0, "high": math.inf}), (Span, {"low": 2.0, "high": 1.0}),
    (Cabinet, {"width_m": 0.0, "depth_m": 1.0, "height_m": 1.0}),
    (Cabinet, {"width_m": 1.0, "depth_m": 1.0, "height_m": 1.0, "acoustic_center_behind_front_m": 1.0}),
    (Cabinet, {"width_m": 1.0, "depth_m": 1.0, "height_m": 1.0, "acoustic_center_behind_front_m": -0.1}),
    (Cabinet, {"width_m": 1.0, "depth_m": 1.0, "height_m": 1.0, "acoustic_center_above_bottom_m": 1.1}),
])
def test_settings_reject_invalid_dimensions(tmp_path: Path, model: type[Span] | type[Cabinet], values: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(values)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("field,value", [
    ("speaker_height_m", 0.0), ("ear_height_m", -1.0), ("wall_gap_m", -0.1),
    ("axis_offset_m", math.inf), ("front_wall", "floor"), ("unexpected", 1),
    ("spacing_m", {"low": 0.0, "high": 1.0}),
    ("front_distance_m", {"low": -1.0, "high": 1.0}),
    ("listening_distance_m", {"low": 0.0, "high": 1.0}),
    ("speaker_areas", ()),
])
def test_layout_settings_reject_invalid_fields(
    tmp_path: Path, settings: LayoutSettings, field: str, value: object,
) -> None:
    with pytest.raises(ValidationError):
        LayoutSettings.model_validate(settings.model_dump() | {field: value})
    assert tmp_path.is_dir()


def test_settings_and_parameters_are_frozen(tmp_path: Path, settings: LayoutSettings) -> None:
    with pytest.raises(ValidationError):
        setattr(settings, "axis_offset_m", 1.0)
    with pytest.raises(ValidationError):
        Span.model_validate({"low": 0.0, "high": 1.0, "extra": 1.0})
    with pytest.raises(ValidationError):
        Box.model_validate({"x": {"low": 0.0, "high": 1.0}, "y": {"low": 0.0, "high": 1.0},
                            "z": {"low": 0.0, "high": 1.0}, "extra": 1})
    with pytest.raises(FrozenInstanceError):
        setattr(LayoutParams(1.0, 1.0, 1.0), "spacing_m", 2.0)
    for bad in (0.0, -1.0, math.inf, math.nan):
        with pytest.raises(ValueError):
            replace(LayoutParams(1.0, 1.0, 1.0), spacing_m=bad)
    assert settings.cabinet.acoustic_center_above_bottom_m is None
    assert settings.cabinet.acoustic_center_behind_front_m == 0.0
    assert tmp_path.is_dir()


def test_project_heights_are_fixed_inputs(tmp_path: Path, project: Scheme, settings: LayoutSettings) -> None:
    chosen = LayoutSettings.model_validate(settings.model_dump() | {"speaker_height_m": 1.0, "ear_height_m": 1.5})
    placed = place(project, chosen, LayoutParams(1.0, 1.2, 1.0))
    assert placed.left.z == placed.right.z == 1.0
    assert placed.primary.z == 1.5
    assert dict(placed.receivers)["up"].z == pytest.approx(1.6, rel=0.0, abs=1e-12)
    assert dict(placed.receivers)["down"].z == pytest.approx(1.4, rel=0.0, abs=1e-12)
    assert tmp_path.is_dir()
