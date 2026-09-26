"""把既有鏡像路徑收成報表用的逐路徑資料，不做評估或打分。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 只在型別檢查時看得到，執行期由呼叫端延後匯入，避免與 report_io 成環
    from aosr.physics.report_io import PathTableSection, SolverInputs

from dataclasses import dataclass

from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.amplitude import Materials
from aosr.physics.room_paths import RoomPath, image_source_paths
from aosr.physics.report_source import SourceModelKind, SourceModelSpec, require_omnidirectional


@dataclass(frozen=True)
class DirectionAnglesData:
    azimuth_deg: float
    elevation_deg: float


@dataclass(frozen=True)
class PathRowData:
    order: int
    wall_sequence: tuple[str, ...]
    delay_s: float
    distance_m: float
    direction_vector: tuple[float, float, float]
    direction_angles: DirectionAnglesData
    departure_off_axis_deg: float | None
    relative_direct_energy: tuple[float, ...]


@dataclass(frozen=True)
class PathTableData:
    reflection_order_k: int
    frequencies_hz: tuple[float, ...]
    scattering_coefficient: tuple[float, ...]
    source_model_kind: SourceModelKind
    rows: tuple[PathRowData, ...]


def _direction(path: RoomPath, receiver: Point) -> tuple[
    tuple[float, float, float], DirectionAnglesData
]:
    dx = path.image[0] - receiver.x
    dy = path.image[1] - receiver.y
    dz = path.image[2] - receiver.z
    vector = (dx / path.dist_m, dy / path.dist_m, dz / path.dist_m)
    return vector, DirectionAnglesData(
        azimuth_deg=math.degrees(math.atan2(dy, dx)),
        elevation_deg=math.degrees(math.atan2(dz, math.hypot(dx, dy))),
    )


def _path_row(
    path: RoomPath, receiver: Point, direct_energy: tuple[float, ...],
    scattering_coefficient: tuple[float, ...],
) -> PathRowData:
    """逐列算一條路徑；到達方向保留原有定義。"""
    vector, angles = _direction(path, receiver)
    relative = tuple(
        (1.0 - scattering) ** path.order * abs(pressure) ** 2 / basis
        for pressure, basis, scattering in zip(
            path.path_pressure, direct_energy, scattering_coefficient, strict=True,
        )
    )
    return PathRowData(
        order=path.order,
        wall_sequence=tuple(wall for bounce in path.bounces for wall in bounce.walls),
        delay_s=path.delay_s,
        distance_m=path.dist_m,
        direction_vector=vector,
        direction_angles=angles,
        departure_off_axis_deg=None,
        relative_direct_energy=relative,
    )


def build_path_table(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    sound_speed_m_s: float,
    rho_c_pa_s_per_m: float,
    impedance_by_wall: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
    scattering_coefficient: tuple[float, ...],
    reflection_order_k: int,
    source_model: SourceModelSpec,
) -> PathTableData:
    """算每條路徑的 ``(1−散射)^階數 × |壓力|² ÷ 直達能量``。

    每條路徑各自取模平方，已乘散射留存且相對同頻點直達能量，不含同階內干涉。報表既有
    的逐階能量是先把同階複數壓力相加再取模平方；兩者刻意不同，不可拿來互相驗證。
    """
    require_omnidirectional(source_model)
    if len(frequencies_hz) != len(scattering_coefficient):
        raise ValueError("路徑表的頻率軸與散射係數數量不同")
    materials = Materials(
        rho_c=rho_c_pa_s_per_m,
        frequencies_hz=frequencies_hz,
        walls={
            wall.wall_name(): tuple(
                complex(impedance_by_wall[wall]) for _frequency in frequencies_hz
            )
            for wall in Wall.all()
        },
    )
    paths = image_source_paths(
        room,
        source,
        receiver,
        sound_speed_m_s,
        max_order=reflection_order_k,
        materials=materials,
    )
    direct = next(path for path in paths if path.order == 0)
    direct_energy = tuple(abs(value) ** 2 for value in direct.path_pressure)
    rows = tuple(_path_row(path, receiver, direct_energy, scattering_coefficient) for path in paths)
    return PathTableData(
        reflection_order_k=reflection_order_k,
        frequencies_hz=frequencies_hz,
        scattering_coefficient=scattering_coefficient,
        source_model_kind=source_model.kind,
        rows=rows,
    )


def build_path_table_section(report: object, inputs: SolverInputs) -> PathTableSection:
    """只在呼叫端要求時，從同一組求解輸入重建逐路徑資料。"""
    lane = report.geometric_lane  # type: ignore[attr-defined]  # expires=2026-12-08 reason=output_from_report 已驗過 ThreeLaneReport
    built = build_path_table(
        room=inputs.room,
        source=inputs.source,
        receiver=inputs.receiver,
        sound_speed_m_s=inputs.sound_speed_m_s,
        rho_c_pa_s_per_m=inputs.density_kg_m3 * inputs.sound_speed_m_s,
        impedance_by_wall=inputs.impedance_by_wall,
        frequencies_hz=lane.frequencies_hz,
        scattering_coefficient=lane.scattering,
        reflection_order_k=inputs.reflection_order_k,
        source_model=inputs.source_model,
    )
    from aosr.physics.report_io import PathRow, PathTableSection

    return PathTableSection(
        reflection_order_k=built.reflection_order_k,
        frequencies_hz=built.frequencies_hz,
        scattering_coefficient=built.scattering_coefficient,
        source_model_kind=built.source_model_kind,
        rows=tuple(PathRow.model_validate(asdict(row)) for row in built.rows),
    )
