"""錨定分析、一維逐源解與逐純量能量必須守住舊演算法逐位答案。"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr import runtime
from aosr.geometry.shoebox import Room
from aosr.geometry.shoebox_mesh import ShoeboxMesh, generate_shoebox_mesh
from aosr.physics import fem_helmholtz as fem
from tests.engine import _fem_shared_cases as cases
from tests.engine.test_three_lane_reports_batch import (
    DENSITY, FEM_RECEIVERS, FEM_ROOM, FEM_SOURCES, SOUND_SPEED, WALLS,
    operators as operators,
)


def _responses(operators: fem.P2Operators, indices: Sequence[int] | None) -> dict[tuple[str, str], NDArray[np.complex128]]:
    return fem.solve_frequency_responses_many(
        operators, right_hand_sides={name: fem.point_source_load(operators, point)
                                    for name, point in FEM_SOURCES.items()},
        receivers=FEM_RECEIVERS, frequencies_hz=cases.FREQUENCIES,
        wall_impedances=WALLS, density_kg_m3=DENSITY,
        sound_speed_m_s=SOUND_SPEED, frequency_indices=indices)


@pytest.mark.parametrize("indices", ((), (1, 1), (2, 1), (-1,), (len(cases.FREQUENCIES),)))
def test_indices_rejected(operators: fem.P2Operators, indices: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        _responses(operators, indices)


def _track_discretization(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    from aosr.physics import fem_batch

    mesh = generate_shoebox_mesh
    p2 = fem.assemble_p2_operators
    calls: list[str] = []

    def build(room: Room, *, max_frequency_hz: float, elements_per_wavelength: int,
              sound_speed_m_s: float, random_seed: int) -> ShoeboxMesh:
        calls.append("mesh")
        return mesh(room, max_frequency_hz=max_frequency_hz,
                    elements_per_wavelength=elements_per_wavelength,
                    sound_speed_m_s=sound_speed_m_s, random_seed=random_seed)

    def assemble(discretization: ShoeboxMesh) -> fem.P2Operators:
        calls.append("p2")
        return p2(discretization)

    monkeypatch.setattr(fem_batch, "generate_shoebox_mesh", build)
    monkeypatch.setattr(fem_batch, "assemble_p2_operators", assemble)
    return calls


@pytest.mark.parametrize("slices", (1, 2, 3, len(cases.FREQUENCIES)))
def test_solver_anchor_and_call_order(
    operators: fem.P2Operators, monkeypatch: pytest.MonkeyPatch, slices: int,
) -> None:
    runtime.preload_mkl()
    from pydiso import mkl_solver
    from aosr.reporting.fem_slices import slice_indices
    from aosr.physics.fem_batch import solve_fem_energies_many

    builds = _track_discretization(monkeypatch)
    real = mkl_solver.MKLPardisoSolver
    assemble = fem.assemble_helmholtz_system
    systems: dict[int, int] = {}
    events: list[tuple[str, int]] = []

    def tracked(ops: fem.P2Operators, *, frequency_hz: float,
                wall_impedances: fem.WallImpedances, density_kg_m3: float,
                sound_speed_m_s: float) -> csr_matrix[np.complex128]:
        system = assemble(ops, frequency_hz=frequency_hz, wall_impedances=wall_impedances,
                          density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
        systems[id(system)] = cases.FREQUENCIES.index(frequency_hz)
        events.append(("assemble", cases.FREQUENCIES.index(frequency_hz)))
        return system

    class Tracked:
        def __init__(self, system: csr_matrix[np.complex128], *, matrix_type: str, factor: bool) -> None:
            assert not factor
            assert systems[id(system)] == 0
            assert events[-1] == ("assemble", 0)
            self.anchor = system
            self.solver = real(system, matrix_type=matrix_type, factor=factor)
            events.append(("construct", 0))

        def refactor(self, system: csr_matrix[np.complex128]) -> None:
            index = systems[id(system)]
            if index == 0:
                assert system is self.anchor
            events.append(("refactor", index))
            self.solver.refactor(system)

        def solve(self, load: NDArray[np.complex128]) -> NDArray[np.complex128]:
            assert load.shape == (operators.basis.N,)
            assert events[-1][0] in {"refactor", "solve"}
            events.append(("solve", events[-1][1]))
            return self.solver.solve(load)

    monkeypatch.setattr(fem, "assemble_helmholtz_system", tracked)
    monkeypatch.setattr(mkl_solver, "MKLPardisoSolver", Tracked)
    for part in range(slices):
        events.clear()
        builds.clear()
        indices = slice_indices(len(cases.FREQUENCIES), slices, part)
        solve_fem_energies_many(
            room=FEM_ROOM, candidates=cases.candidates(), wall_impedances=WALLS,
            frequencies_hz=cases.FREQUENCIES, density_kg_m3=DENSITY,
            sound_speed_m_s=SOUND_SPEED, frequency_indices=indices)
        assert builds == ["mesh", "p2"]
        calls = [event for event in events if event[0] != "assemble"]
        expected = [("construct", 0)] + [event for i in indices
                    for event in [("refactor", i)] + [("solve", i)] * sum(len(sources) for sources, _ in cases.candidates().values())]
        assert calls == expected


@pytest.mark.parametrize("structure", ("indices", "indptr"))
def test_sparsity_positions_guarded(
    operators: fem.P2Operators, monkeypatch: pytest.MonkeyPatch, structure: str,
) -> None:
    assemble = fem.assemble_helmholtz_system

    def shifted(ops: fem.P2Operators, *, frequency_hz: float,
                wall_impedances: fem.WallImpedances, density_kg_m3: float,
                sound_speed_m_s: float) -> csr_matrix[np.complex128]:
        system = assemble(ops, frequency_hz=frequency_hz, wall_impedances=wall_impedances,
                          density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s)
        if frequency_hz != cases.FREQUENCIES[0]:
            if structure == "indices":
                system.indices = np.roll(system.indices, 1)
            else:
                system.indptr[1] += 1
        return system

    monkeypatch.setattr(fem, "assemble_helmholtz_system", shifted)
    with pytest.raises(ValueError, match="非零"):
        _responses(operators, (1,))


def test_pairs_and_hashable_keys(operators: fem.P2Operators) -> None:
    keys = (("trial|left", "right"), ("trial", "left|right"))
    loads = {key: fem.point_source_load(operators, point)
             for key, point in zip(keys, FEM_SOURCES.values(), strict=True)}
    receivers = {key: point for key, point in zip(keys, FEM_RECEIVERS.values(), strict=True)}
    def solve(pairs: Sequence[tuple[tuple[str, str], tuple[str, str]]] | None) -> dict[
        tuple[tuple[str, str], tuple[str, str]], NDArray[np.complex128]
    ]:
        return fem.solve_frequency_responses_many(
            operators, right_hand_sides=loads, receivers=receivers,
            frequencies_hz=cases.FREQUENCIES, wall_impedances=WALLS,
            density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED, pairs=pairs)

    all_pairs = solve(None)
    requested = tuple((key, key) for key in keys)
    selected = solve(requested)
    assert set(selected) == set(requested)
    for pair in requested:
        assert np.array_equal(selected[pair], all_pairs[pair])
    with pytest.raises(ValueError):
        solve(((keys[0], ("missing", "key")),))
    with pytest.raises(ValueError):
        solve(((("missing", "source"), keys[0]),))


@pytest.mark.parametrize("slices", (1, 2, 3, len(cases.FREQUENCIES)))
@pytest.mark.parametrize("selection", ("single", "all", "shuffled", "subset"))
def test_many_energies_equal_old_hex(
    operators: fem.P2Operators, slices: int, selection: str,
) -> None:
    from aosr.physics.fem_batch import solve_fem_energies_many
    from aosr.reporting.fem_slices import slice_indices

    items = dict(cases.candidates())
    if selection == "single":
        items = dict(list(items.items())[:1])
    elif selection == "shuffled":
        items = dict(reversed(list(items.items())))
    elif selection == "subset":
        items = dict(list(items.items())[1:])
    expected = cases.old_energies(operators, items)
    collected: cases.Energies = {name: {pair: () for pair in values}
                                for name, values in expected.items()}
    for part in range(slices):
        actual = solve_fem_energies_many(
            room=FEM_ROOM, candidates=items, wall_impedances=WALLS,
            frequencies_hz=cases.FREQUENCIES, density_kg_m3=DENSITY,
            sound_speed_m_s=SOUND_SPEED,
            frequency_indices=slice_indices(len(cases.FREQUENCIES), slices, part))
        assert set(actual) == set(items)
        for name, values in actual.items():
            assert set(values) == set(expected[name])
            for pair, energy in values.items():
                collected[name][pair] += energy
    assert {name: cases.energy_hex(values) for name, values in collected.items()} == {
        name: cases.energy_hex(values) for name, values in expected.items()}


def test_noncontiguous_frequency_subset_matches_old_hex(operators: fem.P2Operators) -> None:
    indices = (0, 2, len(cases.FREQUENCIES) - 1)
    actual = _responses(operators, indices)
    expected = cases.old_energies(operators, {"trial": (FEM_SOURCES, FEM_RECEIVERS)})["trial"]
    for pair, pressure in actual.items():
        assert tuple(float(abs(value) ** 2).hex() for value in pressure) == tuple(
            expected[pair][index].hex() for index in indices)


def test_candidate_and_point_ids_do_not_collide(operators: fem.P2Operators) -> None:
    from aosr.physics.fem_batch import solve_fem_energies_many

    sources = tuple(FEM_SOURCES.values())
    receivers = tuple(FEM_RECEIVERS.values())
    items = {
        "trial|left": ({"right": sources[0]}, {"front|back": receivers[0]}),
        "trial": ({"left|right": sources[1]}, {"left|front|back": receivers[1]}),
    }
    expected = cases.old_energies(operators, items)
    actual = solve_fem_energies_many(
        room=FEM_ROOM, candidates=items, wall_impedances=WALLS,
        frequencies_hz=cases.FREQUENCIES, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
    assert {name: cases.energy_hex(values) for name, values in actual.items()} == {
        name: cases.energy_hex(values) for name, values in expected.items()}


def test_default_single_wrapper_matches_old_hex(operators: fem.P2Operators) -> None:
    from aosr.physics.three_lane_report import _solve_fem_energies

    expected = cases.old_energies(operators, {"trial": (FEM_SOURCES, FEM_RECEIVERS)})["trial"]
    actual = _solve_fem_energies(
        room=FEM_ROOM, sources=FEM_SOURCES, receivers=FEM_RECEIVERS, wall_impedances=WALLS,
        frequencies_hz=cases.FREQUENCIES, density_kg_m3=DENSITY, sound_speed_m_s=SOUND_SPEED)
    assert cases.energy_hex(actual) == cases.energy_hex(expected)
