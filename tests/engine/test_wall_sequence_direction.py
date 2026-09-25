"""票 #475：牆序列從接收點那一側往回列（第一面是最後碰到的牆），不是時間順序。"""

from __future__ import annotations

import math

import pytest

from aosr.geometry.shoebox import WALL_SEQUENCE_ORDER, Point, Room, Wall
from aosr.physics import report_io
from aosr.physics.report_path_table import PathTableData, build_path_table
from aosr.physics.room_path_output import _human_table
from aosr.scoring.channel_matching_reflections_contract import ReflectionSide
from aosr.scoring.reflections_contract import ReflectionPath

# 票上的幾何：房長 10 m、聲源 x=2、接收點 x=8，另外兩軸同高同寬，x 方向的路徑就是一條線。
_ROOM = Room(10.0, 6.0, 4.0)
_SOURCE = Point(2.0, 3.0, 2.0)
_RECEIVER = Point(8.0, 3.0, 2.0)
_WALL_X = {"x0": 0.0, "xL": _ROOM.Lx}


def _table() -> PathTableData:
    rho_c = 1.2 * 343.0
    return build_path_table(
        room=_ROOM, source=_SOURCE, receiver=_RECEIVER, sound_speed_m_s=343.0,
        rho_c_pa_s_per_m=rho_c,
        impedance_by_wall={wall: 4.0 * rho_c for wall in Wall.all()},
        frequencies_hz=(100.0,), scattering_coefficient=(0.2,), reflection_order_k=2,
    )


def _length_if_listed_in_time_order(walls: tuple[str, ...]) -> float:
    """另一條路：把牆序列當成聲音依時間碰到的順序，從聲源一路走到接收點的長度。"""
    stops = [_SOURCE.x, *(_WALL_X[wall] for wall in walls), _RECEIVER.x]
    return math.fsum(abs(after - before) for before, after in zip(stops, stops[1:]))


def test_wall_sequence_lists_the_last_bounce_first() -> None:
    """('x0','xL') 那一條照時間是先碰 xL、再碰 x0：8+10+8＝26 m；('xL','x0') 是 2+10+2＝14 m。"""
    rows = {row.wall_sequence: row for row in _table().rows}
    assert rows[("x0", "xL")].distance_m == pytest.approx(26.0)
    assert rows[("xL", "x0")].distance_m == pytest.approx(14.0)
    along_x = [walls for walls in rows if walls and set(walls) <= set(_WALL_X)]
    assert any(len(walls) > 1 for walls in along_x)
    for walls in along_x:
        # 反過來讀（最後一面先走）長度才對得上；正著讀在二階那兩條會錯。
        assert rows[walls].distance_m == pytest.approx(_length_if_listed_in_time_order(walls[::-1]))
    assert rows[("x0", "xL")].distance_m != pytest.approx(_length_if_listed_in_time_order(("x0", "xL")))


def test_every_wall_sequence_field_carries_the_same_direction_sentence() -> None:
    """報表路徑表、反射評估、反射左右差三處欄位說明都是同一句；人看的表頭不再寫「時序」。"""
    for model in (report_io.PathRow, ReflectionPath, ReflectionSide):
        assert model.model_fields["wall_sequence"].description == WALL_SEQUENCE_ORDER
    header = _human_table([]).splitlines()[0]
    assert "接收點往回" in header
    assert "時序" not in header
