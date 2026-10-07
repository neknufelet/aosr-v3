"""把既有鏡像路徑收成報表用的逐路徑資料，不做評估或打分。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # 只在型別檢查時看得到，執行期由呼叫端延後匯入，避免與 report_io 成環
    from aosr.physics.report_io import PathTableSection, SolverInputs

from dataclasses import dataclass

from aosr.config.furniture_materials import FurnitureMaterialName, load_furniture_materials
from aosr.config.paths import config_path
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.geometry.furniture import FaceDirection, Vec3
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics.amplitude import Materials
from aosr.physics.furniture_paths import FurniturePath, direct_path_blockers, filter_room_paths, single_bounce_furniture_paths
from aosr.physics.furniture_scene import FurnitureLaneInputs, furniture_lane_inputs, furniture_pressure_with_directivity
from aosr.physics.room_paths import RoomPath, image_source_paths
from aosr.physics.report_source import SourceModelKind, SourceModelSpec, directivity_to_apply
from aosr.physics.source_directivity import (
    SourceModel, apply_pressure_factor, departure_direction, off_axis_degrees, speaker_axis,
)


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
    furniture_id: str | None = None
    furniture_face: FaceDirection | None = None
    reflection_point_m: Vec3 | None = None


@dataclass(frozen=True)
class FurnitureMaterialData:
    furniture_id: str
    material: FurnitureMaterialName
    unknown_bands_hz: tuple[float, ...]


@dataclass(frozen=True)
class PathTableData:
    reflection_order_k: int
    frequencies_hz: tuple[float, ...]
    scattering_coefficient: tuple[float, ...]
    source_model_kind: SourceModelKind
    rows: tuple[PathRowData, ...]
    furniture_ids: tuple[str, ...] | None = None
    furniture_model: str | None = None
    blocked_wall_paths: tuple[tuple[str, ...], ...] | None = None
    furniture_materials: tuple[FurnitureMaterialData, ...] | None = None


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
    source: Point, source_model: SourceModelSpec,
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
        departure_off_axis_deg=_departure_angle(path, source, receiver, source_model),
        relative_direct_energy=relative,
    )


def _departure_angle(
    path: RoomPath, source: Point, receiver: Point, source_model: SourceModelSpec,
) -> float | None:
    directivity = directivity_to_apply(source_model)
    if directivity is None:
        return None
    return off_axis_degrees(departure_direction(path, receiver), speaker_axis(source, directivity[1]))


def _furniture_row(path: FurniturePath, receiver: Point, direct_energy: tuple[float, ...],
                   source: Point, source_model: SourceModelSpec, frequencies_hz: tuple[float, ...],
                   impedance: tuple[float, ...], rho_c: float, c: float) -> PathRowData:
    """家具只在這裡換成一列：同能量路 p_f、不乘整房散射、方向取最後一段。"""
    directivity = directivity_to_apply(source_model)
    pressure = furniture_pressure_with_directivity(path, frequencies_hz, impedance, rho_c=rho_c, c=c,
        model=SourceModel.OMNIDIRECTIONAL if directivity is None else SourceModel.TWO_PARAMETER,
        source=source, aim=None if directivity is None else directivity[1],
        params=None if directivity is None else directivity[0])
    dx, dy, dz = (hit - coordinate for hit, coordinate in zip(path.hit, receiver.as_tuple(), strict=True))
    vector = (dx / path.d_refl, dy / path.d_refl, dz / path.d_refl)
    angles = DirectionAnglesData(math.degrees(math.atan2(dy, dx)), math.degrees(math.atan2(dz, math.hypot(dx, dy))))
    departure = (None if directivity is None else
        off_axis_degrees(path.departure_direction, speaker_axis(source, directivity[1])))
    return PathRowData(order=1, wall_sequence=("furniture",), delay_s=path.delay_s, distance_m=path.dist_m,
        direction_vector=vector, direction_angles=angles, departure_off_axis_deg=departure,
        relative_direct_energy=tuple(abs(p) ** 2 / basis for p, basis in zip(pressure, direct_energy, strict=True)),
        furniture_id=path.furniture_id, furniture_face=path.face_direction, reflection_point_m=path.hit)


def _furniture_rows(furniture: FurnitureLaneInputs, source: Point, receiver: Point,
                    direct_energy: tuple[float, ...], source_model: SourceModelSpec,
                    frequencies_hz: tuple[float, ...], rho_c: float, c: float) -> tuple[PathRowData, ...]:
    """共用列舉器按家具代號與上、下、+x、-x、+y、-y 面序排，不插到牆面中間。"""
    impedances = {item.furniture_id: values for item, values in furniture.impedance_by_item}
    paths = single_bounce_furniture_paths(source.as_tuple(), receiver.as_tuple(), furniture.furniture,
                                        c=c, margin_m=furniture.margin_m)
    return tuple(_furniture_row(path, receiver, direct_energy, source, source_model,
        frequencies_hz, impedances[path.furniture_id], rho_c, c) for path in paths)


def _filtered_walls(paths: tuple[RoomPath, ...], source: Point, receiver: Point,
                    furniture: FurnitureLaneInputs | None) -> tuple[list[RoomPath], tuple[tuple[str, ...], ...] | None]:
    """先過濾並核直達存在，再按原輸入順序把被擋身分換成牆名序列。"""
    kept, blocked = (paths, ()) if furniture is None else filter_room_paths(
        paths, source.as_tuple(), receiver.as_tuple(), furniture.furniture, margin_m=furniture.margin_m)
    if not any(path.order == 0 for path in kept):
        blockers = () if furniture is None else direct_path_blockers(
            {"聲源": source.as_tuple()}, {"接收點": receiver.as_tuple()}, furniture.furniture,
            margin_m=furniture.margin_m)[("聲源", "接收點")]
        raise ValueError("直達路徑不存在或被家具擋住，不符合擺位要求：" + "、".join(blockers))
    blocked_ids = set(blocked)
    sequences = None if furniture is None else tuple(
        tuple(wall for bounce in path.bounces for wall in bounce.walls) for path in paths if path.identity in blocked_ids)
    return list(kept), sequences


def _furniture_materials(furniture: FurnitureLaneInputs | None) -> tuple[FurnitureMaterialData, ...] | None:
    """沒家具不讀登記簿；每件輸入家具都留下材質，包含沒有有效反射的家具。"""
    if furniture is None:
        return None
    items = sorted(furniture.absolute_furniture or (), key=lambda item: item.furniture_id)
    if tuple(item.furniture_id for item in items) != tuple(sorted(item.furniture_id for item in furniture.furniture)):
        raise ValueError("家具場景與逐頻阻抗的輸入家具代號不同")
    if not items:
        return None
    materials = dict(load_furniture_materials(config_path("furniture_materials.toml")).materials())
    return tuple(FurnitureMaterialData(item.furniture_id, item.material, materials[item.material].unknown_bands_hz)
                 for item in items)


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
    furniture: FurnitureLaneInputs | None,
    furniture_rows: bool = True,
) -> PathTableData:
    """算每條路徑的 ``(1−散射)^階數 × |壓力|² ÷ 直達能量``。

    家具列用共用 p_f，不乘整房散射。furniture_rows=False 只過濾牆面，供時間窗補算表使用。
    每條路徑各自取模平方，牆面已乘散射留存且相對同頻點直達能量，不含同階內干涉。報表既有
    的逐階能量是先把同階複數壓力相加再取模平方；兩者刻意不同，不可拿來互相驗證。
    """
    if len(frequencies_hz) != len(scattering_coefficient):
        raise ValueError("路徑表的頻率軸與散射係數數量不同")
    metadata = _furniture_materials(furniture)
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
    paths, blocked = _filtered_walls(tuple(paths), source, receiver, furniture)
    directivity = directivity_to_apply(source_model)
    if directivity is not None:
        curve, aim = directivity
        paths = apply_pressure_factor(
            paths, receiver, frequencies_hz, SourceModel.TWO_PARAMETER,
            source=source, aim=aim, params=curve,
        )
    direct = next(path for path in paths if path.order == 0)
    direct_energy = tuple(abs(value) ** 2 for value in direct.path_pressure)
    rows = tuple(_path_row(path, receiver, direct_energy, scattering_coefficient, source, source_model)
                 for path in paths)
    if furniture is not None and furniture_rows:
        rows += _furniture_rows(furniture, source, receiver, direct_energy, source_model,
                               frequencies_hz, rho_c_pa_s_per_m, sound_speed_m_s)
    return PathTableData(
        reflection_order_k=reflection_order_k,
        frequencies_hz=frequencies_hz,
        scattering_coefficient=scattering_coefficient,
        source_model_kind=source_model.kind,
        rows=rows,
        furniture_ids=None if metadata is None else tuple(item.furniture_id for item in metadata),
        furniture_model=None if metadata is None else "single_bounce_finite_size_v1",
        blocked_wall_paths=None if metadata is None else blocked,
        furniture_materials=metadata,
    )


def build_path_table_section(report: object, inputs: SolverInputs, *, contact_rel: float | None = None) -> PathTableSection:
    """只在呼叫端要求時，從同一組求解輸入重建逐路徑資料。"""
    lane = report.geometric_lane  # type: ignore[attr-defined]  # expires=2026-12-08 reason=output_from_report 已驗過 ThreeLaneReport
    if inputs.furniture and contact_rel is None:
        contact_rel = furniture_contact_rel(default_precision_contracts_path())
    furniture = furniture_lane_inputs(inputs.furniture, (inputs.room.Lx, inputs.room.Ly, inputs.room.Lz),
        lane.frequencies_hz, inputs.density_kg_m3 * inputs.sound_speed_m_s, contact_rel=contact_rel)
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
        furniture=furniture,
    )
    from aosr.physics.report_io import PathTableSection

    return PathTableSection.model_validate(asdict(built))
