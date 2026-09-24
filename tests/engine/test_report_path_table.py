"""票 #360 路徑表考卷：六欄、幾何方向、逐細軸能量與表頭。"""

from __future__ import annotations

import math

import pytest

from blueprint import reference_room_geometry as geom

from aosr.config.capabilities import load_capabilities
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.config.paths import config_path
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io
from aosr.physics.report_path_table import PathTableData, build_path_table
from aosr.physics.report_output import output_from_report
from aosr.physics.three_lane_report import ThreeLaneReport


def test_report_contract_exposes_the_opt_in_path_table() -> None:
    assert hasattr(report_io, "PathRow")
    assert "path_table" in report_io.ReportOutput.model_fields


def _path_table_inputs() -> report_io.ReportInput:
    walls = {wall.wall_name(): 1600.0 for wall in Wall.all()}
    return report_io.load_input_document({
        "room_m": {"Lx": 5.0, "Ly": 6.0, "Lz": 7.0},
        "source_m": {"x": 1.0, "y": 2.0, "z": 3.0},
        "receiver_m": {"x": 3.0, "y": 4.0, "z": 5.0},
        "sound_speed_m_s": 320.0,
        "density_kg_m3": 1.25,
        "impedance_pa_s_per_m_by_wall": walls,
    }, load_capabilities(config_path("capabilities.toml")))


def _with_one_field_changed(
    field: str, solved: report_io.SolverInputs
) -> report_io.SolverInputs:
    """只換一格；新增一格而這裡沒補，下面那題會先紅在這裡。"""
    if field == "room":
        return solved._replace(room=Room(5.0, 6.0, 8.0))
    if field == "source":
        return solved._replace(source=Point(1.5, 2.0, 3.0))
    if field == "receiver":
        return solved._replace(receiver=Point(2.0, 4.0, 5.0))
    if field == "sound_speed_m_s":
        return solved._replace(sound_speed_m_s=330.0)
    if field == "density_kg_m3":
        return solved._replace(density_kg_m3=1.3)
    if field == "impedance_by_wall":
        return solved._replace(impedance_by_wall={**solved.impedance_by_wall, Wall.X0: 900.0})
    if field == "scattering_by_wall":
        return solved._replace(scattering_by_wall={wall: 0.2 for wall in Wall.all()})
    if field == "reflection_order_k":
        return solved._replace(reflection_order_k=solved.reflection_order_k + 1)
    if field == "low_frequency_axis":
        return solved._replace(low_frequency_axis=next(
            axis for axis in LowFrequencyAxis if axis is not solved.low_frequency_axis
        ))
    raise AssertionError(f"SolverInputs 多了一格 {field}，這裡要補一個換掉的值")


@pytest.mark.parametrize("field", report_io.SolverInputs._fields)
def test_output_rejects_path_table_inputs_differing_in_any_field(field: str) -> None:
    """路徑表的求解輸入要跟這份報表的輸入同一份：任何一格不同都拒收，訊息指名那一格。"""
    inputs = _path_table_inputs()
    solved = report_io.solver_inputs(inputs)
    other = _with_one_field_changed(field, solved)
    report = object.__new__(ThreeLaneReport)
    object.__setattr__(report, "reflection_order_k", inputs.reflection_order_k)
    object.__setattr__(report, "low_frequency_axis", inputs.low_frequency_axis)

    with pytest.raises(ValueError, match=f"path_table_inputs 的 {field} "):
        output_from_report(
            report,
            inputs=inputs,
            with_points=False,
            path_table_inputs=other,
        )


@pytest.fixture(scope="module")
def path_table() -> PathTableData:
    room = Room(10.0, 10.0, 10.0)
    source = Point(2.0, 3.0, 4.0)
    receiver = Point(1.0, 1.0, 1.0)
    sound_speed = 343.0
    density = 1.2
    impedance = {wall: 4.0 * density * sound_speed for wall in Wall.all()}
    frequencies = (100.0, 200.0)
    return build_path_table(
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=sound_speed,
        rho_c_pa_s_per_m=density * sound_speed,
        impedance_by_wall=impedance,
        frequencies_hz=frequencies,
        scattering_coefficient=tuple(0.2 for _frequency in frequencies),
        reflection_order_k=1,
    )


def test_path_table_has_six_fields_and_required_header(
    path_table: PathTableData,
) -> None:
    assert set(report_io.PathRow.model_fields) == {
        "order",
        "wall_sequence",
        "delay_s",
        "distance_m",
        "direction_vector",
        "direction_angles",
        "relative_direct_energy",
    }
    assert path_table.reflection_order_k == 1
    assert path_table.scattering_coefficient == pytest.approx(
        tuple(0.2 for _frequency in path_table.frequencies_hz)
    )
    assert path_table.includes_speaker_directivity is False
    direct = next(row for row in path_table.rows if row.order == 0)
    assert direct.wall_sequence == ()
    assert direct.relative_direct_energy == tuple(
        1.0 for _frequency in path_table.frequencies_hz
    )


def test_path_direction_keeps_raw_geometric_azimuth_and_elevation(
    path_table: PathTableData,
) -> None:
    direct = next(row for row in path_table.rows if row.order == 0)
    distance = math.sqrt(14.0)
    assert direct.direction_vector == pytest.approx(
        (1.0 / distance, 2.0 / distance, 3.0 / distance)
    )
    assert direct.direction_angles.azimuth_deg == pytest.approx(
        math.degrees(math.atan2(2.0, 1.0))
    )
    assert direct.direction_angles.elevation_deg == pytest.approx(
        math.degrees(math.atan2(3.0, math.sqrt(5.0)))
    )


def test_path_energy_uses_each_path_pressure_with_scattering_retention() -> None:
    rho_c = 1.2 * 343.0
    table = build_path_table(
        room=Room(10.0, 10.0, 10.0),
        source=Point(2.0, 1.0, 1.0),
        receiver=Point(4.0, 1.0, 1.0),
        sound_speed_m_s=343.0,
        rho_c_pa_s_per_m=rho_c,
        impedance_by_wall={wall: 4.0 * rho_c for wall in Wall.all()},
        frequencies_hz=(100.0, 200.0),
        scattering_coefficient=(0.2, 0.2),
        reflection_order_k=1,
    )

    direct = next(row for row in table.rows if row.wall_sequence == ())
    assert direct.relative_direct_energy == (1.0, 1.0)

    # x0 鏡像源在 x=-2：直達距離 2 m、反射距離 6 m、法向入射的
    # R=(4ρc−ρc)/(4ρc+ρc)=3/5。因此每個頻點的逐路徑相對能量是
    # (1−0.2) × |(3/5)/6|² ÷ |1/2|² = 4/125；沒有把同階別條路徑相加。
    x0 = next(row for row in table.rows if row.wall_sequence == ("x0",))
    assert x0.relative_direct_energy == pytest.approx((4.0 / 125.0, 4.0 / 125.0))


def test_path_delays_follow_a_legitimate_sound_speed_change() -> None:
    """同一份場景合法換聲速：每條路徑的延遲照「同一段距離÷新聲速」跟著變，路徑一條不多不少。"""
    def table(sound_speed: float) -> PathTableData:
        frequencies = (100.0, 200.0)
        return build_path_table(
            room=Room(10.0, 10.0, 10.0),
            source=Point(2.0, 3.0, 4.0),
            receiver=Point(1.0, 1.0, 1.0),
            sound_speed_m_s=sound_speed,
            rho_c_pa_s_per_m=1.2 * sound_speed,
            impedance_by_wall={wall: 1600.0 for wall in Wall.all()},
            frequencies_hz=frequencies,
            scattering_coefficient=tuple(0.2 for _frequency in frequencies),
            reflection_order_k=2,
        )

    slow, fast = table(320.0), table(343.0)

    assert [row.wall_sequence for row in fast.rows] == [row.wall_sequence for row in slow.rows]
    for slow_row, fast_row in zip(slow.rows, fast.rows, strict=True):
        assert fast_row.delay_s == pytest.approx(slow_row.delay_s * 320.0 / 343.0)
        assert fast_row.distance_m == slow_row.distance_m


def test_path_distance_is_the_image_source_geometry_and_delay_times_speed(
    path_table: PathTableData,
) -> None:
    """第 1 格第②點的「距離」：鏡像聲源到接收點的直線距離，另用鏡像幾何手算對答案。

    房 10×10×10、聲源 (2,3,4)、接收點 (1,1,1)：直達是 √14；一階鏡像把聲源對那一面
    牆翻過去（x0 翻成 x=−2、xL 翻成 x=18，y、z 同理）。每一列也要滿足距離＝延遲×聲速
    （fixture 的聲速是 343）。
    """
    source, receiver, size = (2.0, 3.0, 4.0), (1.0, 1.0, 1.0), 10.0
    mirrored = {
        ("x0",): (-source[0], source[1], source[2]),
        ("xL",): (2.0 * size - source[0], source[1], source[2]),
        ("y0",): (source[0], -source[1], source[2]),
        ("yL",): (source[0], 2.0 * size - source[1], source[2]),
        ("floor",): (source[0], source[1], -source[2]),
        ("ceiling",): (source[0], source[1], 2.0 * size - source[2]),
        (): source,
    }
    by_walls = {row.wall_sequence: row for row in path_table.rows}
    assert set(by_walls) == set(mirrored)
    for walls, image in mirrored.items():
        assert by_walls[walls].distance_m == pytest.approx(math.dist(image, receiver))
    assert by_walls[()].distance_m == pytest.approx(math.sqrt(14.0))
    for row in path_table.rows:
        assert row.distance_m == pytest.approx(row.delay_s * 343.0)


def test_every_path_distance_matches_independent_image_geometry_up_to_third_order() -> None:
    """二階、三階的距離也要對：拿獨立的鏡像幾何（``blueprint/reference_room_geometry.py``，
    只 import 標準庫、不共用 ``src/`` 的程式）列出第 3 階以內每一個鏡像聲源，算它到接收點的
    直線距離；路徑表每一條路徑的距離排好序要跟它一一相等，延遲也要是距離÷聲速。

    只驗一階的話，把二階以上的距離算成兩倍考卷也不會紅（#360 找碴）。
    """
    room, source, receiver, sound_speed = (7.0, 5.0, 3.0), (1.3, 2.1, 1.2), (4.9, 3.4, 1.7), 343.0
    table = build_path_table(
        room=Room(*room),
        source=Point(*source),
        receiver=Point(*receiver),
        sound_speed_m_s=sound_speed,
        rho_c_pa_s_per_m=1.2 * sound_speed,
        impedance_by_wall={wall: 1600.0 for wall in Wall.all()},
        frequencies_hz=(100.0, 200.0),
        scattering_coefficient=(0.2, 0.2),
        reflection_order_k=3,
    )
    donor_room = geom.Room(lx=room[0], ly=room[1], lz=room[2], c=sound_speed)
    expected = sorted(
        math.dist(geom.image_from_identity(donor_room, identity, source), receiver)
        for identity in geom.enumerate_identities(3)
    )
    actual = sorted(row.distance_m for row in table.rows)
    assert actual == pytest.approx(expected)
    assert {row.order for row in table.rows} == {0, 1, 2, 3}
    for row in table.rows:
        assert row.delay_s == pytest.approx(row.distance_m / sound_speed)
