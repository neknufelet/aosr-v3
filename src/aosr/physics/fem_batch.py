"""多候選共用一張正式網格與頻域分解；能量逐個純量計算保留末位。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from aosr.config.fem_lane import FEM_ELEMENTS_PER_WAVELENGTH, FEM_MESH_RANDOM_SEED
from aosr.config.frequency_axis import FEM_GEOMETRIC_CROSSOVER_CAP_HZ
from aosr.geometry.shoebox import Point, Room
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.physics.fem_helmholtz import (
    WallImpedances, assemble_p2_operators, point_source_load, solve_frequency_responses_many,
)

Candidates = Mapping[str, tuple[Mapping[str, Point], Mapping[str, Point]]]


def solve_fem_energies_many(
    *, room: Room, candidates: Candidates, wall_impedances: WallImpedances,
    frequencies_hz: Sequence[float], density_kg_m3: float, sound_speed_m_s: float,
    frequency_indices: Sequence[int] | None = None,
) -> dict[str, dict[tuple[str, str], tuple[float, ...]]]:
    """建一次正式 300 Hz 網格與 P2，各候選只讀自己的座位，不組二維右側。

    複數聲壓先由求解器存為 complex128，再逐元素 float(abs(value) ** 2)；
    numpy 整批 abs 與平方的最後一位不同，不能代換這條純量公式。
    """
    if not candidates or any(not sources or not receivers
                             for sources, receivers in candidates.values()):
        raise ValueError("候選、聲源與座位都不能是空的")
    mesh = generate_shoebox_mesh(
        room, max_frequency_hz=FEM_GEOMETRIC_CROSSOVER_CAP_HZ,
        elements_per_wavelength=FEM_ELEMENTS_PER_WAVELENGTH,
        sound_speed_m_s=sound_speed_m_s, random_seed=FEM_MESH_RANDOM_SEED)
    operators = assemble_p2_operators(mesh)
    pressures = solve_frequency_responses_many(
        operators,
        right_hand_sides={(candidate, name): point_source_load(operators, source)
                          for candidate, (sources, _) in candidates.items()
                          for name, source in sources.items()},
        receivers={(candidate, name): receiver
                   for candidate, (_, receivers) in candidates.items()
                   for name, receiver in receivers.items()},
        pairs=tuple(((candidate, source), (candidate, receiver))
                    for candidate, (sources, receivers) in candidates.items()
                    for source in sources for receiver in receivers),
        frequencies_hz=frequencies_hz, wall_impedances=wall_impedances,
        density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s,
        frequency_indices=frequency_indices)
    return {
        candidate: {(source, receiver): tuple(float(abs(value) ** 2) for value in
                        pressures[(candidate, source), (candidate, receiver)])
                    for source in sources for receiver in receivers}
        for candidate, (sources, receivers) in candidates.items()}
