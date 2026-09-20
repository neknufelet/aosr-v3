"""票 #360 路徑表考卷：六欄、幾何方向、逐細軸能量與表頭。"""

from __future__ import annotations

import math

import pytest

from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io
from aosr.physics.report_path_table import PathTableData, build_path_table


def test_report_contract_exposes_the_opt_in_path_table() -> None:
    assert hasattr(report_io, "PathRow")
    assert "path_table" in report_io.ReportOutput.model_fields


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
