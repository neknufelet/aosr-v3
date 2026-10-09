"""鎖定幾何的手算答案，含貼牆浮點反例；不以被測函式產生答案。"""
import json
import math
from pathlib import Path

import pytest

from aosr.geometry.shoebox import Point
from aosr.reporting.scheme import Scheme
from aosr.search import layout
from aosr.search.settings import SearchSettings
from tests.engine._seat_locked_cases import locked_settings
from tests.engine._search_store_cases import reference_project, settings_document


def test_default_serialization_and_strict_seat_lock(tmp_path: Path) -> None:
    document = settings_document()
    old = SearchSettings.model_validate(document)
    old_layout = old.canonical()["layout"]
    assert isinstance(old_layout, dict)
    assert "seat_locked" not in old_layout
    assert "seat_locked" not in json.loads(old.model_dump_json())["layout"]
    assert layout.space_for(old.layout) is layout.UNIT_SPACE
    explicit = SearchSettings.model_validate(document | {"layout": old.layout.model_dump() | {"seat_locked": False}})
    assert explicit.fingerprint == old.fingerprint
    assert locked_settings().fingerprint != old.fingerprint
    locked_layout = locked_settings().canonical()["layout"]
    assert isinstance(locked_layout, dict)
    assert "listening_distance_m" not in locked_layout
    assert "listening_distance_m" not in json.loads(locked_settings().model_dump_json())["layout"]
    with pytest.raises(ValueError):
        locked_settings(seat_locked=1)
    with pytest.raises(ValueError, match="listening_distance_m"):
        SearchSettings.model_validate(document | {"layout": old.layout.model_dump() | {"listening_distance_m": None}})
    assert tmp_path.is_dir()


def test_two_dimensional_conversion_copies_every_seat(tmp_path: Path) -> None:
    project = reference_project(tmp_path)
    settings = locked_settings().layout
    params = layout.params_from_unit({"front_distance": 0.5, "spacing": 0.5}, settings, project=project)
    # F=.25+.5*(2-.25)=1.125、S=.5+.5*(2-.5)=1.25、L=3.2-1.125。
    assert params == layout.LayoutParams(1.125, 1.25, 2.075)
    assert dict(layout.unit_from_params(params, settings)) == {"front_distance": 0.5, "spacing": 0.5}
    placement = layout.place(project, settings, params)
    assert placement.primary.as_tuple() == project.receiver_set.primary.position_m
    assert placement.receivers == tuple((r.receiver_id, Point(*r.position_m)) for r in project.receiver_set.points)
    assert (placement.left.y + placement.right.y) / 2 == project.receiver_set.primary.position_m[1]
    assert placement.left.x == 1.125 and placement.right.x == 1.125
    with pytest.raises(ValueError):
        layout.params_from_unit({"front_distance": 0.5, "spacing": 0.5, "listening_distance": 0.5}, settings, project=project)


def test_wall_seat_copy_prevents_last_bit_drift(tmp_path: Path) -> None:
    project = reference_project(tmp_path)
    receivers = project.receiver_set.model_dump()
    receivers["points"][0]["position_m"] = (3.581, 0.999, 1.2)
    receivers["points"][1]["position_m"] = (0.15, 0.999, 1.2)
    project = Scheme.model_validate(project.model_dump() | {"receiver_set": receivers})
    settings = locked_settings(axis_offset_m=0.999 - 2.0).layout
    front = 0.9411859570013019  # 指定摸底的主位尾差反例。
    params = layout.LayoutParams(front, 0.5, 3.581 - front)
    assert front + params.listening_distance_m != 3.581
    assert 3.581 + (0.15 - 3.581) != 0.15
    placed = layout.place(project, settings, params)
    assert placed.primary.as_tuple() == (3.581, 0.999, 1.2)
    # 喇叭中點橫向直接取主位：x0 牆左手是 -y，間距 0.5；舊算法「房寬/2＋偏移」在這組會差最後一位。
    assert 2.0 + (0.999 - 2.0) - 0.25 != 0.999 - 0.25
    assert placed.left.y == 0.999 - 0.25 and placed.right.y == 0.999 + 0.25
    assert dict(placed.receivers)[receivers["points"][1]["receiver_id"]].x == 0.15
    assert placed.receivers == tuple((r.receiver_id, Point(*r.position_m)) for r in project.receiver_set.points)


def test_locked_start_uses_derived_distance_and_sixty_degrees(tmp_path: Path) -> None:
    project = reference_project(tmp_path)
    settings = locked_settings(spacing_m={"low": 0.5, "high": 4.0}).layout
    start = layout.standard_start(project, settings)
    assert start is not None
    # 原中點 x=1；主位 x=3.2；L=2.2，60° 下 S=L/(sqrt(3)/2)。
    assert start == layout.LayoutParams(1.0, 2.2 / (math.sqrt(3.0) / 2.0), 2.2)
    blocked = locked_settings(base_angle_deg={"low": 60.0, "high": 70.0}).layout
    assert layout.standard_start(project, blocked) is None  # S<=2，L=2.2，角度達不到 60°。


def test_locked_start_ignores_original_off_axis_listening_distance(tmp_path: Path) -> None:
    project = reference_project(tmp_path)
    # 原喇叭中點 (1,1.5)，主位 (3.2,1.9)，原中點耳距 hypot(2.2,.4)；新中軸直接過主位。
    project = project.model_copy(update={"speakers": {"left": Point(1.0, 1.0, 1.2), "right": Point(1.0, 2.0, 1.2)}})
    settings = locked_settings(spacing_m={"low": 0.5, "high": 4.0}).layout
    start = layout.standard_start(project, settings)
    assert start == layout.LayoutParams(1.0, 2.2 / (math.sqrt(3.0) / 2.0), 2.2)


@pytest.mark.parametrize("wall,primary,expected", [
    ("x0", (3.2, 1.9, 1.2), 2.075), ("xL", (2.0, 1.9, 1.2), 2.875),
    ("y0", (3.2, 1.9, 1.2), 0.7749999999999999), ("yL", (3.2, 1.0, 1.2), 1.875),
])
def test_derived_listening_uses_one_axis_subtraction(
    tmp_path: Path, wall: str, primary: tuple[float, float, float], expected: float,
) -> None:
    project = reference_project(tmp_path)
    receivers = project.receiver_set.model_dump()
    receivers["points"][0]["position_m"] = primary
    project = Scheme.model_validate(project.model_dump() | {"receiver_set": receivers})
    settings = locked_settings(front_wall=wall).layout
    params = layout.params_from_unit({"front_distance": 0.5, "spacing": 0.5}, settings, project=project)
    assert params.listening_distance_m.hex() == expected.hex()
