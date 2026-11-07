"""#562 正式報表的逐區晚期接線；物理算法仍在既有牆面響應與晚期求解器。"""
from __future__ import annotations

from dataclasses import replace

from aosr.config.frequency_axis import LATE_DECAY_FREQUENCIES_HZ
from aosr.config.three_lane_crossover import SCHROEDER_T60_BANDS_HZ
from aosr.geometry.shoebox import Room
from aosr.physics import three_lane_report as report
from aosr.physics.furniture_scene import furniture_absorption_area_m2, furniture_adjusted_absorption, furniture_absorption_fraction
from aosr.physics.geometric_lane import solve_geometric_late_energy
from aosr.physics.late_energy import LateEnergyOrderResult
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.report_surfaces import ZonesByWall
from aosr.physics.wall_surfaces import WallSurfaceResponse, zoned_wall_surface_response


def prepare_zoned_late_family(
    *, room: Room, zones: ZonesByWall, density_kg_m3: float, sound_speed_m_s: float,
    report_frequencies_hz: tuple[float, ...], reflection_order_k: int,
    furniture: tuple[AbsoluteFurniture, ...] | None,
) -> tuple[WallSurfaceResponse, dict[float, float], float, LateEnergyOrderResult, report._ReportLateDecay]:
    """交接採面積等效吸音，輻射度與衰減保留逐區分布；無分區不走這一支。"""
    axis = tuple(sorted(set(report_frequencies_hz) | set(LATE_DECAY_FREQUENCIES_HZ) | set(SCHROEDER_T60_BANDS_HZ)))
    surfaces = zoned_wall_surface_response(room, zones, axis, density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
    rho_c = density_kg_m3*sound_speed_m_s
    late_surfaces = surfaces
    if furniture:
        af = furniture_absorption_area_m2(furniture, axis, rho_c)
        base = surfaces.equivalent.absorption_by_wall
        adjusted = furniture_adjusted_absorption(room, axis, base, af)
        # 同一個整房 ε 套逐區剩餘反射率，沿用家具決策紙與原整面公式。
        fractions = furniture_absorption_fraction(room, axis, base, af)
        zone_rows = {name: tuple(replace(cell, absorption=tuple(a+(1.0-a)*e for a, e in
            zip(cell.absorption, fractions, strict=True))) for cell in cells)
            for name, cells in surfaces.absorption_by_zone.items()}
        late_surfaces = replace(surfaces, equivalent=replace(surfaces.equivalent, absorption_by_wall=adjusted),
                                absorption_by_zone=zone_rows)
    t60, f_s = report._eyring_t60_and_schroeder(room, late_surfaces.band_rows(SCHROEDER_T60_BANDS_HZ))
    late = solve_geometric_late_energy(room=room, rho_c_pa_s_per_m=rho_c,
        frequencies_hz=report_frequencies_hz, absorption_by_wall=late_surfaces.named_rows(report_frequencies_hz),
        absorption_by_zone=late_surfaces.named_zone_rows(report_frequencies_hz), reflection_order_k=reflection_order_k)
    decay = report._solve_report_late_decay(room=room, absorption_by_wall=late_surfaces.named_rows(LATE_DECAY_FREQUENCIES_HZ),
        absorption_by_zone=late_surfaces.named_zone_rows(LATE_DECAY_FREQUENCIES_HZ),
        rho_c_pa_s_per_m=rho_c, sound_speed_m_s=sound_speed_m_s)
    return surfaces.equivalent, t60, f_s, late, decay
