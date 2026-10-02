"""小房間共用分解考卷：現算舊演算法，答案不跨機器保存。"""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

from aosr import runtime
from aosr.geometry.shoebox import Point
from aosr.physics import fem_helmholtz as fem
from tests.engine.test_three_lane_reports_batch import (
    DENSITY, FEM_RECEIVERS, FEM_SOURCES, SOUND_SPEED, WALLS,
)

if TYPE_CHECKING:
    from aosr.reporting.scheme import Scheme

FREQUENCIES = (23.0, 43.0, 73.0, 113.0, 173.0, 263.0)
Candidates = Mapping[str, tuple[Mapping[str, Point], Mapping[str, Point]]]
Energies = dict[str, dict[tuple[str, str], tuple[float, ...]]]


def candidates() -> Candidates:
    return {
        name: (
            {key: Point(point.x + offset, point.y, point.z)
             for key, point in FEM_SOURCES.items()},
            {key: Point(point.x, point.y + offset, point.z)
             for key, point in FEM_RECEIVERS.items()},
        )
        for name, offset in (("trial|left", 0.0), ("trial", 0.03), ("third", 0.06))
    }


def old_energies(operators: fem.P2Operators, items: Candidates) -> Energies:
    """改動前的 factor=True、後續 refactor、一維 solve 與逐純量能量。"""
    runtime.preload_mkl()
    runtime.set_pardiso_threads()
    from pydiso.mkl_solver import MKLPardisoSolver

    result: Energies = {}
    for candidate, (sources, receivers) in items.items():
        loads = {name: fem.point_source_load(operators, point)
                 for name, point in sources.items()}
        probes = {name: fem._point_operator(operators, point)
                  for name, point in receivers.items()}
        pressures: dict[tuple[str, str], list[complex]] = {
            (source, receiver): [] for source in sources for receiver in receivers}
        solver: MKLPardisoSolver | None = None
        for frequency in FREQUENCIES:
            system = fem.assemble_helmholtz_system(
                operators, frequency_hz=frequency, wall_impedances=WALLS,
                density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
            if solver is None:
                solver = MKLPardisoSolver(
                    system, matrix_type="complex_symmetric", factor=True)
            else:
                solver.refactor(system)
            for source, load in loads.items():
                solution = solver.solve(load)
                for receiver, probe in probes.items():
                    pressures[source, receiver].append(complex((probe @ solution)[0]))
        result[candidate] = {
            pair: tuple(float(abs(value) ** 2)
                        for value in np.asarray(values, dtype=np.complex128))
            for pair, values in pressures.items()}
    return result


def energy_hex(values: Mapping[tuple[str, str], tuple[float, ...]]) -> dict[tuple[str, str], tuple[str, ...]]:
    return {pair: tuple(value.hex() for value in energies)
            for pair, energies in values.items()}


def small_scheme(name: str = "trial") -> 'Scheme':
    from aosr.reporting.scheme import Scheme, Scene
    from aosr.scoring.channel_group import ChannelGroup, ChannelDefinition, ChannelComparison
    from aosr.scoring.receiver_set import ReceiverSet, ReceiverPoint, ReceiverRole
    from tests.engine.test_three_lane_reports_batch import FEM_ROOM

    return Scheme(
        schema_version="aosr.scheme.v1", scheme_id=name, purpose="music_stereo",
        scene=Scene(room_m=FEM_ROOM, sound_speed_m_s=SOUND_SPEED, density_kg_m3=DENSITY,
                    impedance_pa_s_per_m_by_wall={wall.wall_name(): value for wall, value in WALLS.items()}),
        source_model="omnidirectional", speakers=dict(FEM_SOURCES),
        channel_group=ChannelGroup(
            channels=tuple(ChannelDefinition(role=key, speaker_id=key) for key in FEM_SOURCES),
            comparisons=(ChannelComparison(left_role="left", right_role="right"),),
            feature_match_tolerance_hz=1.0),
        receiver_set=ReceiverSet(points=tuple(
            ReceiverPoint(receiver_id=key, position_m=point.as_tuple(), importance=1.0,
                          role=ReceiverRole.PRIMARY if key == "front" else ReceiverRole.SURROUNDING,
                          direction_relative_to_primary=None if key == "front" else "back")
            for key, point in FEM_RECEIVERS.items())))
