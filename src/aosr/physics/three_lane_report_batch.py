"""單份與候選報表共用的房間準備、逐位置接合流程。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aosr.config.frequency_axis import (
    GEOMETRIC_BAND_FREQUENCIES_HZ,
    LowFrequencyAxis,
    low_frequency_axis_frequencies,
)
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import three_lane_report as report
from aosr.physics.crossover import CrossoverWeights, crossover_weights
from aosr.physics.geometric_lane import solve_geometric_late_energy
from aosr.physics.furniture_paths import direct_path_blockers
from aosr.physics.furniture_scene import FurnitureReportInputs, furniture_report_inputs
from aosr.physics.report_furniture import AbsoluteFurniture
from aosr.physics.late_energy import LateEnergyOrderResult
from aosr.physics.report_source import SourceModelSpec, directivity_to_apply


@dataclass(frozen=True)
class _Shared:
    room: Room
    wall_impedances: dict[Wall, float]
    scattering_by_wall: Mapping[Wall, float] | None
    sound_speed_m_s: float
    density_kg_m3: float
    rho_c_pa_s_per_m: float
    capability: report.ReportCapability
    reflection_order_k: int
    low_frequency_axis: LowFrequencyAxis
    fem_frequencies_hz: tuple[float, ...]
    report_frequencies_hz: tuple[float, ...]
    t60: dict[float, float]
    f_s_hz: float
    full_weights: CrossoverWeights
    dense_weights: CrossoverWeights
    late_result: LateEnergyOrderResult
    decay: report._ReportLateDecay
    source_model: SourceModelSpec
    furniture: FurnitureReportInputs | None = None


def _prepare(
    *, source_model: SourceModelSpec, room: Room, sound_speed_m_s: float, density_kg_m3: float,
    impedance_by_wall: Mapping[Wall, object],
    scattering_by_wall: Mapping[Wall, float] | None,
    capability: report.ReportCapability | None,
    reflection_order_k: int, low_frequency_axis: LowFrequencyAxis,
    furniture: tuple[AbsoluteFurniture, ...] | None = None, contact_rel: float | None = None,
) -> _Shared:
    """只依賴房間的材料、權重、晚期混響及衰減各做一次。"""
    if furniture and contact_rel is None:
        raise ValueError("有家具時必須提供 contact_rel（家具幾何接觸界線）")
    walls = report._wall_impedances(impedance_by_wall)
    fem_frequencies, report_frequencies = low_frequency_axis_frequencies(low_frequency_axis)
    rho_c = density_kg_m3 * sound_speed_m_s
    t60, f_s_hz = report._eyring_t60_and_schroeder(room, walls, rho_c)
    late_result = solve_geometric_late_energy(
        room=room,
        rho_c_pa_s_per_m=rho_c,
        frequencies_hz=report_frequencies,
        impedance_by_wall={wall.wall_name(): complex(value) for wall, value in walls.items()},
        reflection_order_k=reflection_order_k,
    )
    decay = report._solve_report_late_decay(
        room=room, wall_impedances=walls,
        rho_c_pa_s_per_m=rho_c, sound_speed_m_s=sound_speed_m_s,
    )
    return _Shared(
        source_model=source_model, room=room, wall_impedances=walls, scattering_by_wall=scattering_by_wall,
        sound_speed_m_s=sound_speed_m_s, density_kg_m3=density_kg_m3,
        rho_c_pa_s_per_m=rho_c,
        capability=capability if capability is not None else report._unchecked_capability(),
        reflection_order_k=reflection_order_k, low_frequency_axis=low_frequency_axis,
        fem_frequencies_hz=fem_frequencies, report_frequencies_hz=report_frequencies,
        t60=t60, f_s_hz=f_s_hz,
        full_weights=crossover_weights(report_frequencies, f_s_hz),
        dense_weights=crossover_weights(GEOMETRIC_BAND_FREQUENCIES_HZ, f_s_hz),
        late_result=late_result, decay=decay,
        furniture=furniture_report_inputs(furniture, (room.Lx, room.Ly, room.Lz),
            report_frequencies, GEOMETRIC_BAND_FREQUENCIES_HZ, rho_c, contact_rel=contact_rel),
    )


def _pair_report(
    shared: _Shared, source: Point, receiver: Point, fem_energy: tuple[float, ...],
) -> report.ThreeLaneReport:
    """每一對只求兩軸鏡像法，並以同一段接合、組表程式產生報表。"""
    geometric, dense_early = report._solve_both_geometric_report_lanes(
        source_model=shared.source_model,
        room=shared.room, source=source, receiver=receiver,
        wall_impedances=shared.wall_impedances,
        scattering_by_wall=shared.scattering_by_wall,
        rho_c_pa_s_per_m=shared.rho_c_pa_s_per_m,
        sound_speed_m_s=shared.sound_speed_m_s,
        reflection_order_k=shared.reflection_order_k,
        report_frequencies_hz=shared.report_frequencies_hz,
        late_result=shared.late_result,
        furniture=None if shared.furniture is None else shared.furniture.fine.absolute_furniture,
        contact_rel=None if shared.furniture is None else shared.furniture.contact_rel,
        furniture_inputs=shared.furniture,
    )
    if len(fem_energy) != len(shared.fem_frequencies_hz):
        raise ValueError("有限元素能量數量必須等於當次有限元素頻率軸")
    fem_by_frequency = dict(zip(shared.fem_frequencies_hz, fem_energy, strict=True))
    points = report.stitch_energy_points(
        frequencies_hz=shared.report_frequencies_hz,
        fem_energy_by_frequency=fem_by_frequency,
        geometric_lane=geometric,
        geometric_indices=tuple(range(len(shared.report_frequencies_hz))),
        weights=shared.full_weights,
    )
    return report._report_result(
        capability=shared.capability,
        low_frequency_axis=shared.low_frequency_axis,
        f_s_hz=shared.f_s_hz, t60=shared.t60,
        fem_frequencies=shared.fem_frequencies_hz,
        fem_energy=fem_energy, full_weights=shared.full_weights,
        geometric=geometric, dense_early=dense_early,
        dense_weights=shared.dense_weights,
        late_decay=shared.decay.result,
        decay_unavailable_by_center_hz=shared.decay.unavailable_by_center_hz,
        points=points, fem_by_frequency=fem_by_frequency,
    )


def _injected_energies(
    energies: Mapping[tuple[str, str], Sequence[float]],
    sources: Mapping[str, Point], receivers: Mapping[str, Point], n: int,
) -> dict[tuple[str, str], tuple[float, ...]]:
    """完整注入才可接合，正規化為純 float tuple；缺項不自行求解。"""
    if set(energies) != {(source, receiver) for source in sources for receiver in receivers}:
        raise ValueError("fem_energies 鍵集合必須恰好等於所有喇叭與座位組合")
    if any(not isinstance(values, Sequence) or isinstance(values, (str, bytes))
           for values in energies.values()):
        raise ValueError("fem_energies 必須是數值序列，不能是 str 或 bytes")
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           for values in energies.values() for value in values):
        raise ValueError("fem_energies 每個值必須是 int 或 float，不能是 bool")
    try:
        normalized = {pair: tuple(float(value) for value in values)
                      for pair, values in energies.items()}
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("fem_energies 必須是有限非負數值序列") from exc
    if any(len(values) != n for values in normalized.values()):
        raise ValueError("fem_energies 長度必須等於有限元素頻率軸")
    if any(not math.isfinite(value) or math.copysign(1.0, value) < 0.0
           for values in normalized.values() for value in values):
        raise ValueError("fem_energies 必須是有限非負能量")
    return normalized


def _check_direct_paths(shared: _Shared, sources: Mapping[str, Point], receivers: Mapping[str, Point]) -> None:
    """有限元素前逐對拒收所有被擋直達，列齊喇叭、座位與家具代號。"""
    if shared.furniture is None:
        return
    lane = shared.furniture.fine
    blockers = direct_path_blockers({name: point.as_tuple() for name, point in sources.items()},
        {name: point.as_tuple() for name, point in receivers.items()}, lane.furniture, margin_m=lane.margin_m)
    reasons = [f"喇叭 {source} → 座位 {receiver}：家具 {'、'.join(items)}"
               for (source, receiver), items in blockers.items() if items]
    if reasons:
        raise ValueError("直達路徑被家具擋住，不符合擺位要求；" + "；".join(reasons))


def solve_reports(
    *, source_model: SourceModelSpec, room: Room, sources: Mapping[str, Point], receivers: Mapping[str, Point],
    sound_speed_m_s: float, density_kg_m3: float,
    impedance_by_wall: Mapping[Wall, object],
    scattering_by_wall: Mapping[Wall, float] | None,
    capability: report.ReportCapability | None,
    reflection_order_k: int, low_frequency_axis: LowFrequencyAxis,
    batch_fem: bool,
    fem_energies: Mapping[tuple[str, str], Sequence[float]] | None = None,
    furniture: tuple[AbsoluteFurniture, ...] | None = None, contact_rel: float | None = None,
) -> dict[tuple[str, str], report.ThreeLaneReport]:
    """單份與候選共用準備和逐對接合，僅有限元素入口依模式選擇。"""
    if fem_energies is not None and not batch_fem:
        raise ValueError("注入 fem_energies 必須使用 batch_fem=True")
    if not sources or not receivers:
        raise ValueError("sources 與 receivers 都不能是空的")
    injected = (None if fem_energies is None else _injected_energies(
        fem_energies, sources, receivers, len(low_frequency_axis_frequencies(low_frequency_axis)[0])))
    directivity = directivity_to_apply(source_model)
    if directivity is not None:
        for name, source in sources.items():
            if directivity[1] == source:
                raise ValueError(f"source_model.aim_m 與聲源 {name} 重合")
    shared = _prepare(
        source_model=source_model, room=room, sound_speed_m_s=sound_speed_m_s, density_kg_m3=density_kg_m3,
        impedance_by_wall=impedance_by_wall, scattering_by_wall=scattering_by_wall,
        capability=capability, reflection_order_k=reflection_order_k,
        low_frequency_axis=low_frequency_axis,
        furniture=furniture, contact_rel=contact_rel,
    )
    _check_direct_paths(shared, sources, receivers)
    if injected is not None:
        energies = injected
    elif batch_fem:
        energies = report._solve_fem_energies(
            room=room, sources=sources, receivers=receivers,
            wall_impedances=shared.wall_impedances,
            frequencies_hz=shared.fem_frequencies_hz,
            density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s,
        )
    else:
        energies = {
            (source_name, receiver_name): report._solve_fem_energy(
                room=room, source=source, receiver=receiver,
                wall_impedances=shared.wall_impedances,
                frequencies_hz=shared.fem_frequencies_hz,
                density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s,
            )
            for source_name, source in sources.items()
            for receiver_name, receiver in receivers.items()
        }
    return {
        (source_name, receiver_name): _pair_report(
            shared, source, receiver, energies[(source_name, receiver_name)]
        )
        for source_name, source in sources.items()
        for receiver_name, receiver in receivers.items()
    }
