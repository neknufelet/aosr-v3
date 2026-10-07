"""#559 第四步：家具路徑列、手算方向／能量、牆面遮擋與主表界線。"""
from __future__ import annotations

import math
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from aosr.geometry.furniture import FaceDirection, Vec3
from aosr.geometry.shoebox import Point, Wall
from aosr.physics.furniture_paths import FurniturePath, single_bounce_furniture_paths
from aosr.physics.furniture_scene import furniture_pressure_with_directivity
from aosr.physics.geometric_lane import solve_geometric_early_lane
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.report_io import PathTableSection, SolverInputs
from aosr.physics.report_path_table import PathTableData, build_path_table, build_path_table_section
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.room_paths import RoomPath, image_source_paths
from aosr.physics.source_directivity import SourceModel, apply_pressure_factor
from tests.engine import _furniture_energy_cases as case, _furniture_third_order_case as third
from tests.engine.test_report_path_table import _path_table_inputs
from aosr.physics import report_io


def built_table(model: SourceModelSpec = case.OMNI, *,
                items: tuple[AbsoluteFurniture, ...] | None = (case.desk(),),
                furniture_rows: bool = True, order: int = 1, scattering: float = 0.36) -> PathTableData:
    return build_path_table(source_model=model, room=case.ROOM, source=case.SOURCE, receiver=case.RECEIVER,
        sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO_C, impedance_by_wall=case.WALLS,
        frequencies_hz=case.FREQUENCIES, scattering_coefficient=(scattering,) * len(case.FREQUENCIES),
        reflection_order_k=order, furniture=None if items is None else case.lane_inputs(items),
        furniture_rows=furniture_rows)


def hand_top_path() -> FurniturePath:
    # 第五支算例平移到房間；S=(2.4,2,1.05)、E=(3.6,2,1.05)，桌板 z=.5..6。
    face = next(face for face in case.lane_inputs((case.desk(),)).furniture[0].box.exposed_faces
                if face.direction == FaceDirection.TOP)
    return FurniturePath("desk", face, (3.0, 2.0, 0.6), (2.4, 2.0, 0.15),
        0.75, 0.75, 1.5, 1.5 / case.SPEED, 0.6, (0.8, 0.0, -0.6))


_OFF_AXIS = replace(case.analytic(), aim=Point(3.0, 3.0, 1.05))


def hand_direct_energy(model: SourceModelSpec) -> tuple[float, ...]:
    # 直達長 1.2；全向 D=1。兩參數公式 D²=A+(1-A)exp[-2β(1-cosθ)]。
    cosine = 1.0 if model.aim == case.RECEIVER else 0.6 / math.sqrt(1.36)
    curve = model.parameters
    if curve is None:
        return (1.0 / 1.2 ** 2,) * len(case.FREQUENCIES)
    out = []
    for f in case.FREQUENCIES:
        beta = curve.beta_limit / (1.0 + (curve.beta_corner_hz / f) ** curve.beta_exponent)
        floor_db = curve.power_floor_limit_db / (1.0 + (curve.power_floor_corner_hz / f) ** curve.power_floor_exponent)
        floor = 10.0 ** (floor_db / 10.0)
        out.append((floor + (1.0 - floor) * math.exp(-2.0 * beta * (1.0 - cosine))) / 1.2 ** 2)
    return tuple(out)


@pytest.mark.parametrize("model", (case.OMNI, case.analytic(), _OFF_AXIS))
def test_furniture_row_matches_hand_top_geometry_and_shared_pressure(model: SourceModelSpec) -> None:
    table = built_table(model)
    row = next(row for row in table.rows if row.wall_sequence == ("furniture",))
    direct = hand_direct_energy(model)
    p_f = furniture_pressure_with_directivity(hand_top_path(), case.FREQUENCIES,
        (1646.4,) * len(case.FREQUENCIES), rho_c=case.RHO_C, c=case.SPEED,
        model=SourceModel.OMNIDIRECTIONAL if model.parameters is None else SourceModel.TWO_PARAMETER,
        source=case.SOURCE, aim=model.aim, params=model.parameters)
    assert row.relative_direct_energy == pytest.approx(tuple(abs(p) ** 2 / basis for p, basis in zip(p_f, direct, strict=True)))
    assert (row.order, row.wall_sequence, row.furniture_id, row.furniture_face) == (1, ("furniture",), "desk", FaceDirection.TOP)
    assert row.reflection_point_m == pytest.approx((3.0, 2.0, 0.6))
    assert row.distance_m == pytest.approx(1.5)
    assert row.delay_s == pytest.approx(1.5 / case.SPEED)
    assert row.direction_vector == pytest.approx((-0.8, 0.0, -0.6))
    assert row.direction_angles.azimuth_deg == pytest.approx(180.0)
    assert row.direction_angles.elevation_deg == pytest.approx(math.degrees(math.atan2(-0.45, 0.6)))
    axis_cosine = 0.8 if model.aim == case.RECEIVER else 0.48 / math.sqrt(1.36)
    expected_angle = None if model.parameters is None else math.degrees(math.acos(axis_cosine))
    assert row.departure_off_axis_deg == (None if expected_angle is None else pytest.approx(expected_angle))
    changed = built_table(model, scattering=0.7)
    other = next(row for row in changed.rows if row.wall_sequence == ("furniture",))
    assert other.relative_direct_energy == row.relative_direct_energy
    assert all(row.furniture_id is None and row.furniture_face is None and row.reflection_point_m is None
               for row in table.rows if row.wall_sequence != ("furniture",))


def test_floor_is_removed_and_named_in_header() -> None:
    table = built_table()
    assert ("floor",) not in tuple(row.wall_sequence for row in table.rows)
    assert table.blocked_wall_paths == (("floor",),)
    section = PathTableSection.model_validate(asdict(table))
    assert section.blocked_wall_paths == (("floor",),)
    assert section.furniture_ids == ("desk",)
    assert section.furniture_model == "single_bounce_finite_size_v1"
    assert section.furniture_materials is not None
    assert tuple((m.furniture_id, m.material, m.unknown_bands_hz) for m in section.furniture_materials) == (
        ("desk", "wood", (63.0, 8000.0)),)


def _crosses_desk(start: Vec3, end: Vec3) -> bool:
    # 考卷自己的分軸夾區間；不呼叫產品的線段遮擋與 filter_room_paths。
    enter, leave = 0.0, 1.0
    margin = case.CONTACT_REL * 6.0
    for axis, (low, high) in enumerate(((2.6, 3.4), (1.2, 2.8), (0.5, 0.6))):
        step = end[axis] - start[axis]
        if step == 0.0:
            if not low + margin < start[axis] < high - margin:
                return False
            continue
        near, far = sorted(((low + margin - start[axis]) / step, (high - margin - start[axis]) / step))
        enter, leave = max(enter, near), min(leave, far)
        if enter >= leave:
            return False
    return True


def test_blocked_wall_paths_keep_original_enumeration_order() -> None:
    original = image_source_paths(case.ROOM, case.SOURCE, case.RECEIVER, case.SPEED, max_order=3)
    blocked: list[tuple[str, ...]] = []
    kept: list[tuple[str, ...]] = []
    for path in original:
        vertices = (case.SOURCE.as_tuple(), *(bounce.point for bounce in reversed(path.bounces)), case.RECEIVER.as_tuple())
        sequence = tuple(wall for bounce in path.bounces for wall in bounce.walls)
        target = blocked if any(_crosses_desk(start, end) for start, end in zip(vertices, vertices[1:])) else kept
        target.append(sequence)
    table = built_table(order=3)
    assert blocked and any(len(sequence) > 1 for sequence in blocked)
    assert table.blocked_wall_paths == tuple(blocked)
    assert tuple(row.wall_sequence for row in table.rows if row.furniture_id is None) == tuple(kept)


def test_third_order_two_pieces_rows_match_hand_pressures_and_own_blocked_list() -> None:
    """非預設介質、曲線、對準點與逐件阻抗；被擋清單由考卷自己判，含只被聲源那段擋住的二階以上路徑。"""
    inputs = third.lane_inputs(third.FREQUENCIES)
    table = build_path_table(source_model=third.MODEL, room=third.ROOM, source=third.SOURCE,
        receiver=third.RECEIVER, sound_speed_m_s=third.SPEED, rho_c_pa_s_per_m=third.RHO_C,
        impedance_by_wall={wall: third.WALLS[wall.wall_name()] for wall in Wall.all()},
        frequencies_hz=third.FREQUENCIES, scattering_coefficient=(third.SCATTERING,) * len(third.FREQUENCIES),
        reflection_order_k=3, furniture=inputs)
    paths = image_source_paths(third.ROOM, third.SOURCE, third.RECEIVER, third.SPEED, max_order=3,
        materials=third.materials(third.FREQUENCIES))
    kept, blocked, source_leg_only = third.split_blocked(paths, inputs)
    assert source_leg_only

    def sequence(path: RoomPath) -> tuple[str, ...]:
        return tuple(wall for bounce in path.bounces for wall in bounce.walls)

    assert table.blocked_wall_paths == tuple(sequence(path) for path in blocked)
    assert tuple(row.wall_sequence for row in table.rows if row.furniture_id is None) == tuple(map(sequence, kept))
    direct = apply_pressure_factor([path for path in kept if path.order == 0], third.RECEIVER, third.FREQUENCIES,
        SourceModel.TWO_PARAMETER, source=third.SOURCE, aim=third.AIM, params=third.CURVE)[0]
    impedances = {item.furniture_id: values for item, values in third.pieces(third.FREQUENCIES)}
    axis = tuple((a - s) / math.dist(third.AIM.as_tuple(), third.SOURCE.as_tuple())
                 for a, s in zip(third.AIM.as_tuple(), third.SOURCE.as_tuple(), strict=True))
    furniture_paths = single_bounce_furniture_paths(third.SOURCE.as_tuple(), third.RECEIVER.as_tuple(),
        inputs.furniture, c=third.SPEED, margin_m=inputs.margin_m)
    assert {path.furniture_id for path in furniture_paths} == {"back", "glassdesk"}
    for path in furniture_paths:
        row = next(row for row in table.rows if (row.furniture_id, row.furniture_face) == (path.furniture_id, path.face_direction))
        p_f = furniture_pressure_with_directivity(path, third.FREQUENCIES, impedances[path.furniture_id],
            rho_c=third.RHO_C, c=third.SPEED, model=SourceModel.TWO_PARAMETER, source=third.SOURCE,
            aim=third.AIM, params=third.CURVE)
        assert row.relative_direct_energy == pytest.approx(
            tuple(abs(p) ** 2 / abs(d) ** 2 for p, d in zip(p_f, direct.path_pressure, strict=True)), rel=1e-12)
        arrival = tuple(h - r for h, r in zip(path.hit, third.RECEIVER.as_tuple(), strict=True))
        assert row.direction_vector == pytest.approx(tuple(v / math.hypot(*arrival) for v in arrival), rel=1e-12)
        leaving = tuple(h - s for h, s in zip(path.hit, third.SOURCE.as_tuple(), strict=True))
        cosine = sum(v * a for v, a in zip(leaving, axis, strict=True)) / math.hypot(*leaving)
        assert row.departure_off_axis_deg == pytest.approx(math.degrees(math.acos(cosine)), abs=1e-9)
        distance = math.hypot(*leaving) + math.hypot(*arrival)
        assert row.distance_m == pytest.approx(distance, rel=1e-12)
        assert row.delay_s == pytest.approx(distance / third.SPEED, rel=1e-12)
        assert row.direction_angles.azimuth_deg == pytest.approx(math.degrees(math.atan2(arrival[1], arrival[0])), abs=1e-9)
        assert row.direction_angles.elevation_deg == pytest.approx(
            math.degrees(math.atan2(arrival[2], math.hypot(arrival[0], arrival[1]))), abs=1e-9)
        assert row.reflection_point_m == pytest.approx(path.hit, abs=1e-12)
    # 玻璃桌頂面 z=0.74：聲源對頂面平面的鏡像連到接收點，跟平面的交點就是反射點（不靠被測列舉）。
    image = (third.SOURCE.x, third.SOURCE.y, 2.0 * 0.74 - third.SOURCE.z)
    t = (0.74 - image[2]) / (third.RECEIVER.z - image[2])
    top = next(row for row in table.rows if (row.furniture_id, row.furniture_face) == ("glassdesk", FaceDirection.TOP))
    assert top.reflection_point_m == pytest.approx(
        tuple(a + t * (b - a) for a, b in zip(image, third.RECEIVER.as_tuple(), strict=True)), abs=1e-12)
    # 兩件不同材質：表頭各件的未知頻帶照材質登記簿手抄（皮革 63／8000 Hz、玻璃另缺 125 Hz）。
    assert table.furniture_materials is not None
    assert tuple((m.furniture_id, m.material, m.unknown_bands_hz) for m in table.furniture_materials) == (
        ("back", "leather", (63.0, 8000.0)), ("glassdesk", "glass", (63.0, 125.0, 8000.0)))


def test_blocked_direct_raises_named_value_error() -> None:
    blocker = case.desk("blocking-desk").model_copy(update={"height_m": 1.0})
    with pytest.raises(ValueError, match="直達.*blocking-desk"):
        built_table(items=(blocker,))


@pytest.mark.parametrize("model", (case.OMNI, case.analytic(), _OFF_AXIS))
def test_path_energy_times_direct_is_energy_lane_furniture_pressure(model: SourceModelSpec) -> None:
    furniture = case.lane_inputs((case.desk(),))
    # s=1 消掉牆面鏡面；桌面只一個有效反射，早期反射能量就是 |p_f|²。
    lane = solve_geometric_early_lane(source_model=model, room=case.ROOM, source=case.SOURCE, receiver=case.RECEIVER,
        sound_speed_m_s=case.SPEED, rho_c_pa_s_per_m=case.RHO_C, frequencies_hz=case.FREQUENCIES,
        impedance_by_wall={wall.wall_name(): z for wall, z in case.WALLS.items()},
        scattering_by_wall={wall.wall_name(): 1.0 for wall in case.WALLS}, reflection_order_k=1, furniture=furniture)
    row = next(row for row in built_table(model, scattering=1.0).rows if row.furniture_id == "desk")
    assert tuple(relative * direct for relative, direct in zip(row.relative_direct_energy, lane.direct_energy, strict=True)) == pytest.approx(lane.reflected_energy)


def test_wall_only_mode_still_filters_but_never_adds_furniture_rows() -> None:
    table = built_table(furniture_rows=False)
    assert all(row.furniture_id is None for row in table.rows)
    assert ("floor",) not in tuple(row.wall_sequence for row in table.rows)
    assert table.blocked_wall_paths == (("floor",),)
    assert table.furniture_ids == ("desk",)
    PathTableSection.model_validate(asdict(table))


def test_furniture_parameter_is_required_and_keyword_only() -> None:
    import inspect

    parameter = inspect.signature(build_path_table).parameters["furniture"]
    assert parameter.default is inspect.Parameter.empty
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_rows_follow_all_walls_in_id_and_face_order() -> None:
    # a 的 -y 面在 y=2.5；z 桌面的 top 在 z=.6。宣告順序跟代號序相反。
    tall = case.desk("a").model_copy(update={"bottom_center_m": (3.0, 3.0, 0.5),
        "width_m": 2.0, "depth_m": 1.0, "height_m": 2.0})
    table = built_table(items=(case.desk("z"), tall))
    keys = tuple((row.furniture_id, row.furniture_face) for row in table.rows if row.furniture_id is not None)
    assert keys == (("a", FaceDirection.Y_MINUS), ("z", FaceDirection.TOP))
    first = next(i for i, row in enumerate(table.rows) if row.furniture_id is not None)
    assert all(row.furniture_id is None for row in table.rows[:first])
    assert all(row.furniture_id is not None for row in table.rows[first:])
    assert table == built_table(items=(tall, case.desk("z")))


def _solved_furniture() -> SolverInputs:
    return report_io.solver_inputs(_path_table_inputs())._replace(room=case.ROOM, source=case.SOURCE,
        receiver=case.RECEIVER, sound_speed_m_s=case.SPEED, density_kg_m3=case.DENSITY,
        impedance_by_wall=case.WALLS, reflection_order_k=1, furniture=(case.desk(),))


@pytest.mark.parametrize("explicit_contact", (False, True))
def test_section_forwards_input_furniture_and_contact(explicit_contact: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.physics import report_path_table as module

    def forbidden(_: object) -> float:
        pytest.fail("已提供接觸尺時不得重讀登記簿")

    if explicit_contact:
        monkeypatch.setattr(module, "furniture_contact_rel", forbidden)
    raw = SimpleNamespace(geometric_lane=SimpleNamespace(frequencies_hz=case.FREQUENCIES, scattering=(0.36,) * len(case.FREQUENCIES)))
    section = build_path_table_section(raw, _solved_furniture(), contact_rel=case.CONTACT_REL if explicit_contact else None)
    assert section.furniture_ids == ("desk",)
    assert section.blocked_wall_paths == (("floor",),)
    row = next(row for row in section.rows if row.furniture_id == "desk")
    assert row.reflection_point_m == pytest.approx((3.0, 2.0, 0.6))


def test_section_without_furniture_never_reads_furniture_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.physics import report_path_table as module, furniture_scene

    def forbidden(_: object) -> float:
        pytest.fail("無家具時不得讀材質或接觸登記簿")

    monkeypatch.setattr(module, "furniture_contact_rel", forbidden)
    monkeypatch.setattr(module, "load_furniture_materials", forbidden)
    monkeypatch.setattr(furniture_scene, "load_furniture_materials", forbidden)
    raw = SimpleNamespace(geometric_lane=SimpleNamespace(frequencies_hz=case.FREQUENCIES, scattering=(0.36,) * len(case.FREQUENCIES)))
    section = build_path_table_section(raw, _solved_furniture()._replace(furniture=None))
    assert section.furniture_ids is None
    assert all(row.furniture_id is None for row in section.rows)


def test_invisible_input_furniture_keeps_its_registered_material_and_unknown_bands() -> None:
    remote = case.desk("remote:glass").model_copy(update={"material": "glass", "width_m": 0.4,
        "depth_m": 0.4, "bottom_center_m": (5.5, 3.5, 0.5)})
    section = PathTableSection.model_validate(asdict(built_table(items=(remote,))))
    assert all(row.furniture_id is None for row in section.rows)
    assert section.furniture_ids == ("remote:glass",)
    assert section.furniture_model == "single_bounce_finite_size_v1"
    assert section.furniture_materials is not None
    assert tuple((m.furniture_id, m.material, m.unknown_bands_hz) for m in section.furniture_materials) == (
        ("remote:glass", "glass", (63.0, 125.0, 8000.0)),)
