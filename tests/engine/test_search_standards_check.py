"""擺位標準逐條檢查的獨立幾何答案；只在暫存樹讀寫。"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aosr.config.paths import CONFIG_DIR, config_path
from aosr.geometry.shoebox import Point, Room
from aosr.reporting.scheme import Scheme
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet
from tests.engine._precision_contracts import MUTANT_MARGIN, contract_value
from tests.engine._search_store_cases import reference_project

if TYPE_CHECKING:
    from aosr.config.placement_standards import PlacementStandards
    from aosr.search.standards_check import ChecklistRow, StandardsChecklist


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    source = next(path for path in CONFIG_DIR.iterdir() if path.stem == "placement_standards")
    target = tmp_path / source.stem
    target.write_bytes(config_path(source.name).read_bytes())
    return target


@pytest.fixture
def standards(registry_path: Path) -> PlacementStandards:
    from aosr.config.placement_standards import load_placement_standards

    return load_placement_standards(registry_path)


def make_scheme(tmp_path: Path, *, base: float = 2.0, h: float = 3.0,
                height: float = 2.0, ear: float = 2.0, radius: float = 0.25) -> Scheme:
    project = reference_project(tmp_path)
    return with_points(project, Point(4.0, 5.0, height), Point(4.0 + base, 5.0, height),
                       Point(4.0 + base / 2.0, 5.0 + h, ear), radius)


def with_points(project: Scheme, left: Point, right: Point,
                primary: Point, radius: float = 0.25) -> Scheme:
    roles = {channel.role: channel.speaker_id for channel in project.channel_group.channels}
    receivers = ReceiverSet(points=(
        ReceiverPoint(receiver_id="nearby", position_m=(primary.x, primary.y, primary.z + radius),
                      role=ReceiverRole.SURROUNDING, importance=1.0,
                      direction_relative_to_primary="up"),
        ReceiverPoint(receiver_id="main", position_m=primary.as_tuple(),
                      role=ReceiverRole.PRIMARY, importance=1.0),
    ))
    # 右喇叭先放進字典，座位清單主位也不是第一筆，避免拿排列順序冒充角色。
    return Scheme.model_validate(project.model_dump() | {
        "scene": project.scene.model_dump() | {"room_m": Room(12.0, 14.0, 8.0)},
        "speakers": {roles["right"]: right, roles["left"]: left},
        "receiver_set": receivers,
    })


def check(scheme: Scheme, standards: PlacementStandards,
          front_wall: str = "y0") -> StandardsChecklist:
    from aosr.search.standards_check import check_placement_standards

    return check_placement_standards(scheme, front_wall, standards,
                                     boundary_rel=contract_value("placement_standard_boundary"))


def row(checklist: StandardsChecklist, identity: str) -> ChecklistRow:
    return next(item for item in checklist.rows if item.id == identity)


def values(item: ChecklistRow) -> tuple[float, ...]:
    return tuple(actual.value for actual in item.actual)


def test_registry_loads_from_temporary_copy(tmp_path: Path) -> None:
    from aosr.config.placement_standards import load_placement_standards
    from aosr.config.paths import CONFIG_DIR, config_path

    source = next(path for path in CONFIG_DIR.iterdir() if path.stem == "placement_standards")
    target = tmp_path / source.stem
    target.write_bytes(config_path(source.name).read_bytes())
    standards = load_placement_standards(target)
    assert {entry.id for entry in standards.entries} >= {
        "itu_8_5_1_2_wall_distance", "ebu_a1_1_wall_distance",
    }


@pytest.mark.parametrize("base,itu,ebu", [
    (1.999999, "not_met", "not_met"), (2.0, "met", "not_met"),
    (2.000001, "met", "met"), (3.0, "met", "met"),
    (3.000001, "acceptable_in_suitable_rooms", "met"),
    (3.5, "acceptable_in_suitable_rooms", "met"),
    (3.999999, "acceptable_in_suitable_rooms", "met"),
    (4.0, "acceptable_in_suitable_rooms", "not_met"), (4.000001, "not_met", "not_met"),
])
def test_base_width_boundaries(tmp_path: Path, standards: PlacementStandards,
                               base: float, itu: str, ebu: str) -> None:
    checklist = check(make_scheme(tmp_path, base=base), standards)
    for identity, expected in (("itu_8_5_3_1_base_width", itu), ("ebu_a1_2_base_width", ebu)):
        item = row(checklist, identity)
        assert item.verdict == expected
        assert values(item) == pytest.approx((base,))


@pytest.mark.parametrize("difference,expected", [(0.0, "met"), (0.000001, "not_met"),
                                                 (-0.000001, "not_met")])
@pytest.mark.parametrize("role", ["left", "right"])
def test_itu_height_requires_each_centre_at_ear(tmp_path: Path, standards: PlacementStandards,
                                               difference: float, expected: str, role: str) -> None:
    scheme = make_scheme(tmp_path)
    left = Point(4.0, 5.0, 2.0 + (difference if role == "left" else 0.0))
    right = Point(6.0, 5.0, 2.0 + (difference if role == "right" else 0.0))
    item = row(check(with_points(scheme, left, right, Point(5.0, 8.0, 2.0)), standards),
               "itu_8_5_1_1_height")
    assert item.verdict == expected
    assert values(item) == pytest.approx((difference, 0.0) if role == "left" else (0.0, difference))
    assert tuple(actual.label for actual in item.actual) == ("左喇叭", "右喇叭")


@pytest.mark.parametrize("height,expected", [(1.2, "met"), (1.199999, "not_met")])
@pytest.mark.parametrize("role", ["left", "right"])
def test_ebu_speaker_height_boundary(tmp_path: Path, standards: PlacementStandards,
                                     height: float, expected: str, role: str) -> None:
    scheme = make_scheme(tmp_path)
    scheme = with_points(scheme, Point(4.0, 5.0, height if role == "left" else 2.0),
                         Point(6.0, 5.0, height if role == "right" else 2.0), Point(5.0, 8.0, 2.0))
    item = row(check(scheme, standards), "ebu_a1_1_speaker_height")
    assert item.verdict == expected
    assert values(item) == pytest.approx((height, 2.0) if role == "left" else (2.0, height))


@pytest.mark.parametrize("height,surface", [(0.6, "floor"), (7.4, "ceiling")])
def test_itu_six_surfaces_ebu_four_walls(tmp_path: Path, standards: PlacementStandards,
                                       height: float, surface: str) -> None:
    checklist = check(make_scheme(tmp_path, height=height), standards)
    itu = row(checklist, "itu_8_5_1_2_wall_distance")
    ebu = row(checklist, "ebu_a1_1_wall_distance")
    assert itu.verdict == "not_met"
    assert values(itu) == pytest.approx((0.6, 0.6))
    assert all(actual.closest_surface == surface for actual in itu.actual)
    assert ebu.verdict == "met"
    assert values(ebu) == (4.0, 5.0)
    assert tuple(actual.closest_surface for actual in ebu.actual) == ("x0", "y0")
    assert "離牆條件沒守：早期反射要另外控制並記錄做法" in itu.description


@pytest.mark.parametrize("gap,expected", [(1.0, "met"), (0.999999, "not_met")])
@pytest.mark.parametrize("surface", ["floor", "ceiling", "x0", "xL", "y0", "yL"])
def test_speaker_surface_boundaries(tmp_path: Path, standards: PlacementStandards,
                                    gap: float, expected: str, surface: str) -> None:
    scheme = make_scheme(tmp_path)
    coordinates = {"floor": (4.0, 5.0, gap), "ceiling": (4.0, 5.0, 8.0 - gap),
                   "x0": (gap, 5.0, 2.0), "xL": (12.0 - gap, 5.0, 2.0),
                   "y0": (4.0, gap, 2.0), "yL": (4.0, 14.0 - gap, 2.0)}
    scheme = with_points(scheme, Point(*coordinates[surface]), Point(6.0, 5.0, 2.0),
                         Point(5.0, 8.0, 2.0))
    checklist = check(scheme, standards)
    itu = row(checklist, "itu_8_5_1_2_wall_distance")
    assert itu.verdict == expected
    assert itu.actual[0].value == pytest.approx(gap)
    assert itu.actual[0].closest_surface == surface
    assert ("早期反射要另外控制並記錄做法" in itu.description) == (expected == "not_met")
    if surface not in {"floor", "ceiling"}:
        ebu = row(checklist, "ebu_a1_1_wall_distance")
        assert ebu.verdict == expected
        assert ebu.actual[0].value == pytest.approx(gap)
        assert ebu.actual[0].closest_surface == surface


@pytest.mark.parametrize("d,expected", [(2.0, "met"), (1.999999, "not_met")])
@pytest.mark.parametrize("side", ["left", "right"])
def test_listening_distance_lower_boundary(tmp_path: Path, standards: PlacementStandards,
                                           d: float, expected: str, side: str) -> None:
    scheme = make_scheme(tmp_path)
    primary = Point(4.0 if side == "left" else 6.0, 5.0 + d, 2.0)
    item = row(check(with_points(scheme, Point(4.0, 5.0, 2.0), Point(6.0, 5.0, 2.0), primary),
                     standards), "itu_8_5_3_2_listening_distance")
    assert item.verdict == expected
    expected_values = (d, math.sqrt(d * d + 4.0))
    assert values(item) == pytest.approx(expected_values if side == "left" else expected_values[::-1])


@pytest.mark.parametrize("d,expected", [(3.4, "met"), (3.400001, "not_met")])
def test_listening_distance_scaled_upper_boundary(tmp_path: Path, standards: PlacementStandards,
                                                  d: float, expected: str) -> None:
    # B=2；水平距離 1、耳高差 sqrt(D²-1)，因此三維 D 正好是題目给的值。
    scheme = make_scheme(tmp_path, h=0.0, ear=2.0 + math.sqrt(d * d - 1.0))
    item = row(check(scheme, standards), "itu_8_5_3_2_listening_distance")
    assert item.verdict == expected
    assert values(item) == pytest.approx((d, d))


@pytest.mark.parametrize("radius,expected", [(0.7, "met"), (0.700001, "not_met")])
def test_listening_area_boundary(tmp_path: Path, standards: PlacementStandards,
                                 radius: float, expected: str) -> None:
    item = row(check(make_scheme(tmp_path, ear=0.0, radius=radius), standards), "itu_8_5_3_3_area")
    assert item.verdict == expected
    assert values(item) == (radius,)


@pytest.mark.parametrize("angle,expected", [(10.0, "met"), (10.000001, "not_met")])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("side", ["left", "right"])
def test_inclination_absolute_boundary(tmp_path: Path, standards: PlacementStandards,
                                       angle: float, expected: str, sign: float, side: str) -> None:
    # 一支水平距離正好 1，另一支 sqrt(5)；離地零點作一端，避免高度相減的尾差。
    rise = math.tan(math.radians(angle))
    height, ear = (0.0, rise) if sign > 0.0 else (rise, 0.0)
    scheme = with_points(make_scheme(tmp_path), Point(4.0, 5.0, height), Point(6.0, 5.0, height),
                         Point(4.0 if side == "left" else 6.0, 6.0, ear))
    item = row(check(scheme, standards), "ebu_a1_1_inclination")
    assert item.verdict == expected
    other = math.degrees(math.atan(rise / math.sqrt(5.0)))
    assert values(item) == pytest.approx((angle, other) if side == "left" else (other, angle))


@pytest.mark.parametrize("front,primary,expected,surface", [
    ("x0", (10.5, 0.4, 2.0), 0.4, "y0"),
    ("xL", (0.4, 7.0, 2.0), 0.4, "x0"),
    ("y0", (6.0, 13.6, 2.0), 0.4, "yL"),
    ("yL", (11.6, 0.3, 2.0), 0.3, "y0"),
    ("x0", (0.2, 7.0, 2.0), 7.0, "y0"),
    ("xL", (11.8, 7.0, 2.0), 7.0, "y0"),
    ("y0", (6.0, 0.2, 2.0), 6.0, "x0"),
    ("yL", (6.0, 13.8, 2.0), 6.0, "x0"),
])
def test_listener_side_and_back_walls(tmp_path: Path, standards: PlacementStandards,
                                      front: str, primary: tuple[float, float, float],
                                      expected: float, surface: str) -> None:
    scheme = make_scheme(tmp_path)
    item = row(check(with_points(scheme, Point(4.0, 5.0, 2.0), Point(6.0, 5.0, 2.0),
                                 Point(*primary)), standards, front), "ebu_a1_1_listener_walls")
    assert values(item) == pytest.approx((expected,))
    assert item.actual[0].closest_surface == surface
    assert item.verdict == ("met" if expected >= 1.5 else "not_met")
    assert "只套主位" in item.description and "多人座位" in item.description


@pytest.mark.parametrize("gap,expected", [(1.5, "met"), (1.499999, "not_met")])
@pytest.mark.parametrize("front", ["x0", "xL", "y0", "yL"])
def test_listener_wall_boundary_ignores_surrounding(tmp_path: Path, standards: PlacementStandards,
                                                   gap: float, expected: str, front: str) -> None:
    primary = Point(gap, 7.0, 2.0) if front.startswith("y") else Point(6.0, gap, 2.0)
    scheme = with_points(make_scheme(tmp_path), Point(4.0, 5.0, 2.0), Point(6.0, 5.0, 2.0), primary)
    points = scheme.receiver_set.points
    near_wall = points[0].model_copy(update={"position_m": (0.1, 0.1, 2.0)})
    scheme = Scheme.model_validate(scheme.model_dump() | {
        "receiver_set": ReceiverSet(points=(near_wall, points[1])),
    })
    item = row(check(scheme, standards, front), "ebu_a1_1_listener_walls")
    assert item.verdict == expected
    assert values(item) == pytest.approx((gap,))


def test_value_only_rows_have_independent_geometry(tmp_path: Path, standards: PlacementStandards) -> None:
    checklist = check(make_scheme(tmp_path, base=2.0, h=1.0, height=3.0, ear=2.0), standards)
    expected = {"itu_8_5_3_3_angle": (90.0,), "ebu_a1_1_ear_height": (2.0,),
                "ebu_a1_2_angle_distance": (90.0, 1.0, 0.5), "ebu_a1_2_area": (0.25,)}
    for identity, actual in expected.items():
        item = row(checklist, identity)
        assert item.verdict == "value_only"
        assert values(item) == pytest.approx(actual)
    assert "圖說" in row(checklist, "ebu_a1_2_area").description
    assert "不一致" in row(checklist, "ebu_a1_2_area").description
    assert tuple(actual.unit for actual in row(checklist, "ebu_a1_2_angle_distance").actual) == (
        "deg", "m", "1",
    )


def test_base_width_and_distance_are_three_dimensional(tmp_path: Path, standards: PlacementStandards) -> None:
    scheme = with_points(make_scheme(tmp_path), Point(4.0, 5.0, 2.0), Point(6.0, 5.0, 3.5),
                         Point(5.0, 8.0, 2.0))
    checklist = check(scheme, standards)
    assert values(row(checklist, "itu_8_5_3_1_base_width")) == (2.5,)
    assert values(row(checklist, "itu_8_5_3_2_listening_distance")) == pytest.approx(
        (math.sqrt(10.0), 3.5),
    )
    assert values(row(checklist, "ebu_a1_2_angle_distance"))[1:] == pytest.approx((3.0, 1.2))


def test_horizontal_perpendicular_distance_for_oblique_base(tmp_path: Path,
                                                            standards: PlacementStandards) -> None:
    # 基線向量 (3,4)，主位位移 (5,2)：叉積絕對值 14、水平基寬 5，h=14/5。
    scheme = with_points(make_scheme(tmp_path), Point(3.0, 4.0, 2.0), Point(6.0, 8.0, 2.0),
                         Point(8.0, 6.0, 3.0))
    item = row(check(scheme, standards), "ebu_a1_2_angle_distance")
    assert item.verdict == "value_only"
    assert values(item) == pytest.approx((math.degrees(math.atan2(14.0, 6.0)), 2.8, 0.56))


def test_area_includes_every_declared_seat(tmp_path: Path, standards: PlacementStandards) -> None:
    scheme = make_scheme(tmp_path)
    additional = ReceiverPoint(receiver_id="another_seat", position_m=(5.6, 8.0, 2.8),
                               role=ReceiverRole.OTHER_SEAT, importance=1.0)
    scheme = Scheme.model_validate(scheme.model_dump() | {
        "receiver_set": ReceiverSet(points=(*scheme.receiver_set.points, additional)),
    })
    checklist = check(scheme, standards)
    assert values(row(checklist, "itu_8_5_3_3_area")) == pytest.approx((1.0,))
    assert row(checklist, "itu_8_5_3_3_area").verdict == "not_met"
    assert values(row(checklist, "ebu_a1_2_area")) == pytest.approx((1.0,))
    assert row(checklist, "ebu_a1_2_area").verdict == "value_only"


@pytest.mark.parametrize("front,primary,surface", [
    ("x0", (10.5, 7.0, 2.0), "xL"), ("xL", (1.5, 7.0, 2.0), "x0"),
    ("y0", (6.0, 12.5, 2.0), "yL"), ("yL", (6.0, 1.5, 2.0), "y0"),
])
def test_listener_back_wall_exact_boundary(tmp_path: Path, standards: PlacementStandards,
                                           front: str, primary: tuple[float, float, float],
                                           surface: str) -> None:
    scheme = with_points(make_scheme(tmp_path), Point(4.0, 5.0, 2.0), Point(6.0, 5.0, 2.0),
                         Point(*primary))
    item = row(check(scheme, standards, front), "ebu_a1_1_listener_walls")
    assert values(item) == (1.5,)
    assert item.verdict == "met" and item.actual[0].closest_surface == surface


# 每種判法各改一個門檻欄位：門檻寫死在程式的任何一種判法都會讓這題紅。
# 預設方案（make_scheme）：左右喇叭 (4,5,2)、(6,5,2)，主位 (5,8,2)，周圍點高 0.25 m，房間 12×14×8，前牆 y0；
# B＝2、D＝√10≈3.16、最近反射面是地板 2 m、最近牆 x0 4 m、主位到側牆／後牆最近 5 m、喇叭高 2 m。
@pytest.mark.parametrize("identity,kwargs,old,new,before,after", [
    ("itu_8_5_1_1_height", {}, "target = 0.0", "target = 0.5", "met", "not_met"),
    ("itu_8_5_1_2_wall_distance", {}, "lower = 1.0", "lower = 2.5", "met", "not_met"),
    ("itu_8_5_3_1_base_width", {}, "lower = 2.0", "lower = 2.5", "met", "not_met"),
    ("itu_8_5_3_1_base_width", {"base": 3.5}, "upper = 3.0", "upper = 3.6", "acceptable_in_suitable_rooms", "met"),
    ("itu_8_5_3_1_base_width", {"base": 3.5}, "acceptable_upper = 4.0", "acceptable_upper = 3.2",
     "acceptable_in_suitable_rooms", "not_met"),
    ("itu_8_5_3_2_listening_distance", {}, "lower = 2.0", "lower = 3.3", "met", "not_met"),
    ("itu_8_5_3_2_listening_distance", {}, "upper_factor = 1.7", "upper_factor = 1.5", "met", "not_met"),
    ("itu_8_5_3_3_area", {}, "upper = 0.7", "upper = 0.2", "met", "not_met"),
    ("ebu_a1_1_speaker_height", {"height": 1.25, "ear": 1.25}, "lower = 1.2", "lower = 1.3", "met", "not_met"),
    # 喇叭比耳高 0.5 m、水平 √10 m：仰角約 8.99°。
    ("ebu_a1_1_inclination", {"height": 2.5}, "upper = 10.0", "upper = 8.0", "met", "not_met"),
    ("ebu_a1_1_wall_distance", {}, "lower = 1.0", "lower = 4.5", "met", "not_met"),
    ("ebu_a1_1_listener_walls", {}, "lower = 1.5", "lower = 5.5", "met", "not_met"),
    ("ebu_a1_2_base_width", {}, "lower = 2.0", "lower = 1.9", "not_met", "met"),
    ("ebu_a1_2_base_width", {"base": 3.9}, "upper = 4.0", "upper = 3.8", "met", "not_met"),
])
def test_threshold_edit_changes_same_scheme(
    tmp_path: Path, registry_path: Path, standards: PlacementStandards, identity: str,
    kwargs: dict[str, float], old: str, new: str, before: str, after: str,
) -> None:
    from aosr.config.placement_standards import load_placement_standards

    scheme = make_scheme(tmp_path, **kwargs)
    first = row(check(scheme, standards), identity)
    text = registry_path.read_text()
    marker = f'id = "{identity}"'
    preceding, following = text.split(marker)
    entry, rest = following.split("[[entry]]", 1) if "[[entry]]" in following else (following, None)
    assert old in entry
    edited = entry.replace(old, new, 1)
    registry_path.write_text(preceding + marker + edited + ("" if rest is None else "[[entry]]" + rest))
    second = row(check(scheme, load_placement_standards(registry_path)), identity)
    assert first.verdict == before and second.verdict == after
    assert first.actual == second.actual


@pytest.mark.parametrize("invalid", ["duplicate", "missing_limit", "blank_quote", "mixed_units", "scaled_elsewhere",
                                     "distance_not_scaled"])
def test_invalid_registry_rejected(registry_path: Path, invalid: str) -> None:
    from aosr.config.placement_standards import load_placement_standards

    text = registry_path.read_text()
    if invalid == "duplicate":
        text = text.replace('id = "ebu_a1_2_base_width"', 'id = "itu_8_5_3_1_base_width"')
    elif invalid == "missing_limit":
        text = text.replace("upper = 3.0\n", "", 1)
    elif invalid == "blank_quote":
        loaded = tomllib.loads(text)
        quote = loaded["entry"][0]["quote"]
        text = text.replace(quote, "   ", 1)
    elif invalid == "mixed_units":
        # 夾角、垂距、比值三種單位混在一格，改成拿同一個上限比。
        marker = 'id = "ebu_a1_2_angle_distance"'
        preceding, following = text.split(marker)
        following = following.replace('comparison = "value_only"', 'comparison = "maximum"', 1)
        following = following.replace("target = 60.0\nreference_ratio = 0.9", "upper = 60.0", 1)
        text = preceding + marker + following
    elif invalid == "distance_not_scaled":
        # 反方向：喇叭到聽者的距離改用「至少多少」，上限要跟著基寬走的原文就丟了。
        marker = 'id = "itu_8_5_3_2_listening_distance"'
        preceding, following = text.split(marker)
        following = following.replace('comparison = "scaled_range"', 'comparison = "minimum"', 1)
        following = following.replace("upper_factor = 1.7\n", "", 1)
        text = preceding + marker + following
    else:
        # 耳高改用按基寬縮放的區間：基寬是公尺，只給喇叭到聽者的距離用。
        marker = 'id = "ebu_a1_1_ear_height"'
        preceding, following = text.split(marker)
        following = following.replace('comparison = "value_only"', 'comparison = "scaled_range"', 1)
        following = following.replace("target = 1.2", "lower = 1.0\nupper_factor = 1.0", 1)
        text = preceding + marker + following
    registry_path.write_text(text)
    with pytest.raises(ValueError):
        load_placement_standards(registry_path)


@pytest.mark.parametrize("comparison,thresholds", [
    ("equal", {}), ("minimum", {}), ("maximum", {}),
    ("closed_range", {"lower": 2.0}), ("open_range", {"upper": 4.0}),
    ("preferred_range", {"lower": 2.0, "upper": 3.0}),
    ("scaled_range", {"lower": 2.0}), ("value_only", {"lower": 1.0}),
    ("minimum", {"lower": 1.0, "upper": 2.0}),
    ("closed_range", {"lower": 3.0, "upper": 2.0}),
    ("preferred_range", {"lower": 2.0, "upper": 3.0, "acceptable_upper": 3.0}),
])
def test_comparison_requires_matching_threshold_fields(tmp_path: Path, standards: PlacementStandards,
                                                        comparison: str, thresholds: dict[str, float]) -> None:
    from aosr.config.placement_standards import PlacementStandard

    assert tmp_path.is_dir()
    document = standards.entries[0].model_dump() | {"comparison": comparison, "thresholds": thresholds}
    with pytest.raises(ValueError):
        PlacementStandard.model_validate(document)


def test_models_are_frozen_and_forbid_extra_fields(tmp_path: Path, standards: PlacementStandards) -> None:
    checklist = check(make_scheme(tmp_path), standards)
    models = (standards, standards.entries[0], standards.entries[0].thresholds,
              checklist, checklist.rows[0], checklist.rows[0].actual[0])
    for model in models:
        document = model.model_dump(by_alias=True)
        name = next(iter(type(model).model_fields))
        with pytest.raises(ValueError):
            setattr(model, name, getattr(model, name))
        with pytest.raises(ValueError):
            type(model).model_validate(document | {"unrecognised": "不可偷渡"})


def test_renderer_has_every_row_without_acoustic_comparison(tmp_path: Path,
                                                            standards: PlacementStandards) -> None:
    from aosr.search.standards_check import render_checklist_text

    checklist = check(make_scheme(tmp_path), standards)
    rendered = render_checklist_text(checklist)
    fixed = "這張表只列標準有沒有守：沒守的不淘汰，守了也不代表聲學較好"
    assert checklist.notice == fixed and fixed in rendered
    remaining = rendered.replace(fixed, "")
    assert all(word not in remaining for word in ("較好", "更好", "優於"))
    for item in checklist.rows:
        assert item.id in rendered and item.clause in rendered
        assert item.standard in rendered and str(item.pdf_page) in rendered
        assert item.quote in rendered and item.description in rendered
        assert item.verdict.label in rendered
        assert all(actual.unit in rendered for actual in item.actual)


def test_check_is_read_only(tmp_path: Path, standards: PlacementStandards,
                            monkeypatch: pytest.MonkeyPatch) -> None:
    scheme = make_scheme(tmp_path)
    before = scheme.model_dump()
    tree_before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("純函式不應開檔")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    checklist = check(scheme, standards)
    assert checklist.rows
    assert scheme.model_dump() == before
    monkeypatch.undo()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == tree_before


@pytest.mark.parametrize("front", ["floor", "ceiling", "unknown"])
def test_front_wall_must_be_vertical(tmp_path: Path, standards: PlacementStandards, front: str) -> None:
    with pytest.raises(ValueError):
        check(make_scheme(tmp_path), standards, front)


@pytest.mark.parametrize("right,primary", [
    ((4.0, 5.0, 3.0), (5.0, 8.0, 2.0)),
    ((6.0, 5.0, 2.0), (4.0, 5.0, 3.0)),
])
def test_undefined_horizontal_geometry_is_rejected(tmp_path: Path, standards: PlacementStandards,
                                                   right: tuple[float, float, float],
                                                   primary: tuple[float, float, float]) -> None:
    scheme = with_points(make_scheme(tmp_path), Point(4.0, 5.0, 2.0), Point(*right), Point(*primary))
    with pytest.raises(ValueError, match="退化"):
        check(scheme, standards)


def _base_scheme(tmp_path: Path, left_y: float, right_y: float) -> Scheme:
    """左右喇叭沿 y 排、同高；主位在連線中垂線上 3 m 外。"""
    project = reference_project(tmp_path)
    middle = (left_y + right_y) / 2.0
    return with_points(project, Point(4.0, left_y, 2.0), Point(4.0, right_y, 2.0), Point(7.0, middle, 2.0))


# 照 src/aosr/search/layout.py::place 的加減順序擺：中軸＝房寬／2，左右＝中軸 ± 間距／2（#614 複查實測的三組）。
@pytest.mark.parametrize("room_width,spacing,itu,ebu", [
    (3.02, 2.0, "met", "not_met"),
    (5.03, 3.0, "met", "met"),
    (4.04, 4.0, "acceptable_in_suitable_rooms", "not_met"),
])
def test_float_noise_on_base_width_boundaries(
    tmp_path: Path, standards: PlacementStandards, room_width: float, spacing: float, itu: str, ebu: str,
) -> None:
    """間距剛好設在門檻上時，座標加減的尾差不准把判定翻到另一邊；列出的實際值照原值。"""
    middle, half = room_width / 2.0, spacing / 2.0
    scheme = _base_scheme(tmp_path, middle + half, middle - half)
    checklist = check(scheme, standards)
    itu_row, ebu_row = row(checklist, "itu_8_5_3_1_base_width"), row(checklist, "ebu_a1_2_base_width")
    assert itu_row.actual[0].value != spacing  # 輸入真的帶尾差，否則這題沒有測到東西。
    assert itu_row.actual[0].value == (middle + half) - (middle - half)
    assert itu_row.verdict == itu and ebu_row.verdict == ebu


@pytest.mark.parametrize("identity,limit,fraction,inside,outside", [
    ("itu_8_5_3_1_base_width", 2.0, -1.0, "met", "not_met"),
    ("itu_8_5_3_1_base_width", 3.0, 1.0, "met", "acceptable_in_suitable_rooms"),
    ("ebu_a1_2_base_width", 2.0, 1.0, "not_met", "met"),
    ("ebu_a1_2_base_width", 4.0, -1.0, "not_met", "met"),
])
def test_boundary_snap_mutant_beyond_tolerance_is_red(
    tmp_path: Path, standards: PlacementStandards, identity: str, limit: float, fraction: float,
    inside: str, outside: str,
) -> None:
    """界線內（相對差 (1−δ)·T）當在門檻上判，界線外（(1+δ)·T）照原值判；T 只從登記簿讀，δ 是共用變異邊距。

    左喇叭放在 0.5 m：0.5 加基寬再減 0.5 在這個範圍是精確的浮點運算，基寬逐位等於擺進去的值。
    """
    limit_rel = contract_value("placement_standard_boundary")
    verdicts = []
    for scale in (1.0 - MUTANT_MARGIN, 1.0 + MUTANT_MARGIN):
        base = limit * (1.0 + fraction * scale * limit_rel)
        scheme = _base_scheme(tmp_path, 0.5 + base, 0.5)
        found = row(check(scheme, standards), identity)
        assert found.actual[0].value == base
        verdicts.append(found.verdict)
    assert verdicts == [inside, outside]


def _custom_scheme(tmp_path: Path, room: Room, left: Point, right: Point, primary: Point, seat: Point) -> Scheme:
    """自訂房間與一個周圍座位；其餘照 with_points。"""
    project = reference_project(tmp_path)
    roles = {channel.role: channel.speaker_id for channel in project.channel_group.channels}
    receivers = ReceiverSet(points=(
        ReceiverPoint(receiver_id="main", position_m=primary.as_tuple(), role=ReceiverRole.PRIMARY, importance=1.0),
        ReceiverPoint(receiver_id="nearby", position_m=seat.as_tuple(), role=ReceiverRole.SURROUNDING,
                      importance=1.0, direction_relative_to_primary="back"),
    ))
    return Scheme.model_validate(project.model_dump() | {
        "scene": project.scene.model_dump() | {"room_m": room},
        "speakers": {roles["left"]: left, roles["right"]: right},
        "receiver_set": receivers,
    })


def test_float_noise_on_listener_wall_minimum(tmp_path: Path, standards: PlacementStandards) -> None:
    """至少：房長 5.02、前牆 x0、主位 x＝1.0＋2.52，到後牆算成 1.4999999999999996，照原文 1.5 判守。"""
    primary = Point(1.0 + 2.52, 7.0, 1.2)
    scheme = _custom_scheme(tmp_path, Room(5.02, 14.0, 3.0), Point(1.0, 6.0, 1.2), Point(1.0, 8.0, 1.2),
                            primary, Point(primary.x, 7.0, 1.3))
    walls = row(check(scheme, standards, front_wall="x0"), "ebu_a1_1_listener_walls")
    assert walls.actual[0].value < 1.5 and walls.actual[0].closest_surface is not None
    assert walls.verdict == "met" and walls.actual[0].judged_as == 1.5


def test_float_noise_on_area_maximum(tmp_path: Path, standards: PlacementStandards) -> None:
    """至多：主位 x＝1.83、座位 x＝1.83＋0.7，半徑算成 0.7000000000000002，照原文 0.7 判守。"""
    primary = Point(1.83, 7.0, 1.2)
    scheme = _custom_scheme(tmp_path, Room(12.0, 14.0, 3.0), Point(5.0, 6.0, 1.2), Point(5.0, 8.0, 1.2),
                            primary, Point(1.83 + 0.7, 7.0, 1.2))
    area = row(check(scheme, standards, front_wall="xL"), "itu_8_5_3_3_area")
    assert area.actual[0].value > 0.7
    assert area.verdict == "met" and area.actual[0].judged_as == 0.7


def test_float_noise_on_scaled_upper(tmp_path: Path, standards: PlacementStandards) -> None:
    """按基寬縮放：B＝2.0、主位放在 D 剛好 1.7·B 的位置附近，取第一個算出來略大於 3.4 的點，照 3.4 判守。"""
    target, half = 3.4, 1.0
    x = math.sqrt(target ** 2 - half ** 2)
    while math.hypot(x, half) <= target:
        x = math.nextafter(x, math.inf)
    distance_value = math.hypot(x, half)
    assert distance_value - target <= target * contract_value("placement_standard_boundary")
    scheme = _custom_scheme(tmp_path, Room(12.0, 14.0, 8.0), Point(4.0, 5.0 + half, 2.0), Point(4.0, 5.0 - half, 2.0),
                            Point(4.0 + x, 5.0, 2.0), Point(4.0 + x, 5.0, 2.1))
    found = row(check(scheme, standards), "itu_8_5_3_2_listening_distance")
    above = [item for item in found.actual if item.value > target]
    assert above and all(item.judged_as == target for item in above)
    assert found.verdict == "met"


def test_float_noise_on_acceptable_upper_from_above(tmp_path: Path, standards: PlacementStandards) -> None:
    """可接受上限從上方：房寬 12.03、間距 4.0，基寬算成 4.000000000000001，ITU 照 4 判可接受、EBU 照 4 判沒守。"""
    middle = 12.03 / 2.0
    scheme = _base_scheme(tmp_path, middle + 2.0, middle - 2.0)
    checklist = check(scheme, standards)
    itu, ebu = row(checklist, "itu_8_5_3_1_base_width"), row(checklist, "ebu_a1_2_base_width")
    assert itu.actual[0].value > 4.0
    assert itu.verdict == "acceptable_in_suitable_rooms" and ebu.verdict == "not_met"
    # 兩條各自那一行都要加註（全文搜尋會被另一行的加註蓋過）。
    assert itu.actual[0].judged_as == 4.0 and ebu.actual[0].judged_as == 4.0
    from aosr.search.standards_check import render_checklist_text

    assert "照 4.0 判" in render_checklist_text(checklist)


@pytest.mark.parametrize("boundary_rel", [-1e-12, math.nan, math.inf])
def test_boundary_rel_must_be_finite_and_nonnegative(
    tmp_path: Path, standards: PlacementStandards, boundary_rel: float,
) -> None:
    from aosr.search.standards_check import check_placement_standards

    with pytest.raises(ValueError):
        check_placement_standards(make_scheme(tmp_path), "y0", standards, boundary_rel=boundary_rel)


def test_boundary_rel_has_no_default() -> None:
    """門檻邊界的範圍只住登記簿：檢查函式不准自帶預設值（呼叫端一定要從登記簿讀進來）。"""
    import inspect

    from aosr.search.standards_check import check_placement_standards

    parameter = inspect.signature(check_placement_standards).parameters["boundary_rel"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_value_only_rows_are_never_annotated(tmp_path: Path, standards: PlacementStandards) -> None:
    """只列值的條文不判，所以就算貼著參考值（正三角形的夾角算成 59.99999999999999 之類）也不加註。"""
    half = 1.0
    scheme = _custom_scheme(tmp_path, Room(12.0, 14.0, 8.0), Point(4.0, 5.0 + half, 2.0), Point(4.0, 5.0 - half, 2.0),
                            Point(4.0 + math.sqrt(3.0) * half, 5.0, 2.0), Point(4.0 + math.sqrt(3.0) * half, 5.0, 2.1))
    checklist = check(scheme, standards)
    angle = row(checklist, "itu_8_5_3_3_angle")
    assert angle.verdict == "value_only" and angle.actual[0].value == pytest.approx(60.0, abs=1e-9)
    assert angle.actual[0].value != 60.0  # 輸入真的帶尾差，否則這題沒有測到東西。
    for item in checklist.rows:
        if item.verdict == "value_only":
            assert all(value.judged_as is None for value in item.actual)


@pytest.mark.parametrize("change_factor", [False, True])
def test_scaled_annotation_names_factor_from_data(
    tmp_path: Path, registry_path: Path, standards: PlacementStandards, change_factor: bool,
) -> None:
    from aosr.config.placement_standards import load_placement_standards
    from aosr.search.standards_check import render_checklist_text

    identity = "itu_8_5_3_2_listening_distance"
    entry = next(item for item in standards.entries if item.id == identity)
    factor = entry.thresholds.upper_factor
    assert factor is not None
    if change_factor:
        text = registry_path.read_text()
        registry_path.write_text(text.replace(f"upper_factor = {factor}", f"upper_factor = {factor * 1.25}", 1))
        standards = load_placement_standards(registry_path)
        factor = next(item for item in standards.entries if item.id == identity).thresholds.upper_factor
        assert factor is not None
    base = 3.0
    limit = factor * base
    distance = limit * (1.0 + contract_value("placement_standard_boundary") / 2.0)
    depth = math.sqrt(distance**2 - (base / 2.0)**2)
    scheme = with_points(make_scheme(tmp_path), Point(4.0, 5.0, 2.0), Point(4.0 + base, 5.0, 2.0),
                         Point(4.0 + base / 2.0, 5.0 + depth, 2.0))
    checklist = check(scheme, standards)
    found = row(checklist, identity)
    assert found.verdict == "met" and all(item.judged_as == limit for item in found.actual)
    line = next(line for line in render_checklist_text(checklist).splitlines() if line.startswith(identity + " |"))
    assert f"照 {factor}×B（＝{limit}） 判" in line


def test_unscaled_annotation_keeps_numeric_threshold(tmp_path: Path, standards: PlacementStandards) -> None:
    from aosr.search.standards_check import render_checklist_text

    identity = "itu_8_5_3_1_base_width"
    lower = next(item for item in standards.entries if item.id == identity).thresholds.lower
    assert lower is not None
    base = lower * (1.0 - contract_value("placement_standard_boundary") / 2.0)
    checklist = check(make_scheme(tmp_path, base=base), standards)
    line = next(line for line in render_checklist_text(checklist).splitlines() if line.startswith(identity + " |"))
    assert f"照 {lower} 判" in line and "×B" not in line
