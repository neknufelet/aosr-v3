"""三路報表的細軸與密軸幾何路包裝；共用材料、散射與家具輸入。"""
from __future__ import annotations

from collections.abc import Mapping

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ, GEOMETRIC_BAND_FREQUENCIES_HZ
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.materials.construction_models import ConstructionNote, SurfaceMaterial
from aosr.physics.wall_reflections import lane_reflection_sources
from aosr.physics.geometric_lane import (
    GeometricEarlyResult, GeometricLaneResult, solve_geometric_early_lane, solve_geometric_lane,
)
from aosr.physics.furniture_scene import FurnitureLaneInputs, FurnitureReportInputs, furniture_lane_inputs
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.late_energy import LateEnergyOrderResult
from aosr.physics.report_source import SourceModelSpec
from aosr.physics.three_lane_report_materials import _scattering_by_name


def _solve_geometric_report_lane(
    *,
    source_model: SourceModelSpec,
    room: Room,
    source: Point,
    receiver: Point,
    scattering_by_wall: Mapping[Wall, float] | None,
    rho_c_pa_s_per_m: float,
    sound_speed_m_s: float,
    reflection_order_k: int,
    frequencies_hz: tuple[float, ...] = GEOMETRIC_LANE_FREQUENCIES_HZ,
    late_result: LateEnergyOrderResult | None = None,
    furniture: tuple[AbsoluteFurniture, ...] | None = None,
    contact_rel: float | None = None,
    furniture_inputs: FurnitureLaneInputs | None = None,
    materials_by_wall: Mapping[Wall, SurfaceMaterial],
    density_kg_m3: float | None = None,
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None,
) -> GeometricLaneResult:
    if furniture and contact_rel is None:
        raise ValueError("有家具時必須提供 contact_rel（家具幾何接觸界線）")
    sources = lane_reflection_sources(materials_by_wall, frequencies_hz,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m, sound_speed_m_s=sound_speed_m_s, density_kg_m3=density_kg_m3,
        notes_by_wall=notes_by_wall)
    return solve_geometric_lane(
        source_model=source_model,
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=sound_speed_m_s,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        frequencies_hz=frequencies_hz,
        impedance_by_wall=sources.normal_rows(), reflection_sources=sources,
        scattering_by_wall=_scattering_by_name(scattering_by_wall),
        reflection_order_k=reflection_order_k,
        late_result=late_result,
        furniture=furniture_inputs if furniture_inputs is not None else furniture_lane_inputs(
            furniture, (room.Lx, room.Ly, room.Lz), frequencies_hz, rho_c_pa_s_per_m, contact_rel=contact_rel),
    )



def _solve_dense_geometric_report_lane(
    *,
    source_model: SourceModelSpec,
    room: Room,
    source: Point,
    receiver: Point,
    scattering_by_wall: Mapping[Wall, float] | None,
    rho_c_pa_s_per_m: float,
    sound_speed_m_s: float,
    reflection_order_k: int,
    furniture: tuple[AbsoluteFurniture, ...] | None = None,
    contact_rel: float | None = None,
    furniture_inputs: FurnitureLaneInputs | None = None,
    materials_by_wall: Mapping[Wall, SurfaceMaterial],
    density_kg_m3: float | None = None,
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None,
) -> GeometricEarlyResult:
    if furniture and contact_rel is None:
        raise ValueError("有家具時必須提供 contact_rel（家具幾何接觸界線）")
    sources = lane_reflection_sources(materials_by_wall, GEOMETRIC_BAND_FREQUENCIES_HZ,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m, sound_speed_m_s=sound_speed_m_s, density_kg_m3=density_kg_m3,
        notes_by_wall=notes_by_wall)
    return solve_geometric_early_lane(
        source_model=source_model,
        room=room,
        source=source,
        receiver=receiver,
        sound_speed_m_s=sound_speed_m_s,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        frequencies_hz=GEOMETRIC_BAND_FREQUENCIES_HZ,
        impedance_by_wall=sources.normal_rows(), reflection_sources=sources,
        scattering_by_wall=_scattering_by_name(scattering_by_wall),
        reflection_order_k=reflection_order_k,
        furniture=furniture_inputs if furniture_inputs is not None else furniture_lane_inputs(
            furniture, (room.Lx, room.Ly, room.Lz), GEOMETRIC_BAND_FREQUENCIES_HZ,
            rho_c_pa_s_per_m, contact_rel=contact_rel),
    )



def _solve_both_geometric_report_lanes(
    *,
    source_model: SourceModelSpec,
    room: Room,
    source: Point,
    receiver: Point,
    scattering_by_wall: Mapping[Wall, float] | None,
    rho_c_pa_s_per_m: float,
    sound_speed_m_s: float,
    reflection_order_k: int,
    report_frequencies_hz: tuple[float, ...] = GEOMETRIC_LANE_FREQUENCIES_HZ,
    late_result: LateEnergyOrderResult | None = None,
    furniture: tuple[AbsoluteFurniture, ...] | None = None,
    contact_rel: float | None = None,
    furniture_inputs: FurnitureReportInputs | None = None,
    materials_by_wall: Mapping[Wall, SurfaceMaterial],
    density_kg_m3: float | None = None,
    notes_by_wall: dict[Wall, tuple[ConstructionNote, ...]] | None = None,
) -> tuple[GeometricLaneResult, GeometricEarlyResult]:
    """細軸幾何路與 0.5 Hz 密軸早期路各解一次，回傳兩者。

    兩路吃的是**同一個** ``reflection_order_k``；頻帶平均那一支再比一次三者相等
    （:func:`~aosr.physics.geometric_lane.average_geometric_lane_to_bands_with_dense_early`）。
    """
    geometric = _solve_geometric_report_lane(
        source_model=source_model,
        room=room,
        source=source,
        receiver=receiver,
        materials_by_wall=materials_by_wall, density_kg_m3=density_kg_m3, notes_by_wall=notes_by_wall,
        scattering_by_wall=scattering_by_wall,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        sound_speed_m_s=sound_speed_m_s,
        reflection_order_k=reflection_order_k,
        frequencies_hz=report_frequencies_hz,
        late_result=late_result,
        furniture=furniture, contact_rel=contact_rel,
        furniture_inputs=None if furniture_inputs is None else furniture_inputs.fine,
    )
    dense_early = _solve_dense_geometric_report_lane(
        source_model=source_model,
        room=room,
        source=source,
        receiver=receiver,
        materials_by_wall=materials_by_wall, density_kg_m3=density_kg_m3, notes_by_wall=notes_by_wall,
        scattering_by_wall=scattering_by_wall,
        rho_c_pa_s_per_m=rho_c_pa_s_per_m,
        sound_speed_m_s=sound_speed_m_s,
        reflection_order_k=reflection_order_k,
        furniture=furniture, contact_rel=contact_rel,
        furniture_inputs=None if furniture_inputs is None else furniture_inputs.dense,
    )
    return geometric, dense_early

