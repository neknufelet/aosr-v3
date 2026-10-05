"""模態參與對直接頻域解的獨立考卷；只驗方向與代數，不訂 FEM 精度門檻。"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from aosr.geometry.shoebox import Point, Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.physics.fem_helmholtz import P2Operators, assemble_p2_operators, point_source_load, solve_frequency_responses
from aosr.physics.fem_modal import FemModalSpectrum, solve_fem_modes
from aosr.physics.fem_modal_participation import ModalExpansion, prepare_modal_expansion
from aosr.physics.modal_convention import ModalKind
from tests.engine.test_fem_modal import C, RHO, ROUNDING


SOURCE = Point(0.35, 0.42, 0.38)
RECEIVER = Point(2.45, 1.11, 0.72)
LENGTHS = (3.0, 1.7, 1.2)


def _spectrum(beta: float, cap: float = 270, *, retain: bool = True) -> tuple[FemModalSpectrum, P2Operators]:
    mesh = generate_shoebox_mesh(Room(*LENGTHS), max_frequency_hz=270,
                                 elements_per_wavelength=2, sound_speed_m_s=C, random_seed=1)
    spectrum = solve_fem_modes(mesh, wall_impedances=_walls(beta), density_kg_m3=RHO,
                               sound_speed_m_s=C, frequency_max_hz=cap, retain_shapes=retain)
    return spectrum, assemble_p2_operators(mesh)


def _walls(beta: float) -> dict[Wall, float | None]:
    return {wall: RHO * C / beta if beta else None for wall in Wall.all()}


def _prepare(spectrum: FemModalSpectrum, operators: P2Operators, beta: float) -> ModalExpansion:
    return prepare_modal_expansion(spectrum, operators, wall_impedances=_walls(beta),
                                   density_kg_m3=RHO, sound_speed_m_s=C)


@pytest.fixture(scope="module")
def damped() -> tuple[FemModalSpectrum, P2Operators, ModalExpansion]:
    spectrum, ops = _spectrum(0.05)
    return spectrum, ops, _prepare(spectrum, ops, 0.05)


@pytest.mark.parametrize("beta", [0.0, 0.02])
def test_modal_sum_approaches_direct_solution_with_higher_cutoff(beta: float) -> None:
    spectra = [_spectrum(beta, cap) for cap in (85, 175, 270)]
    frequencies = [25.0, spectra[-1][0].modal_table[0].frequency_hz * 0.99, 68.0]
    ops = spectra[-1][1]
    direct = solve_frequency_responses(ops, right_hand_side=point_source_load(ops, SOURCE),
        receiver=RECEIVER, frequencies_hz=frequencies, wall_impedances=_walls(beta),
        density_kg_m3=RHO, sound_speed_m_s=C)
    errors = []
    for spectrum, operators in spectra:
        lookup = _prepare(spectrum, operators, beta).at_positions([SOURCE], [RECEIVER])
        errors.append(abs(lookup.pressures(frequencies)[0, 0] - direct))
        if beta == 0:
            static = next(row for row in lookup.entries if row.mode.kind is ModalKind.STATIC)
            assert static.residue_k == 0j
            assert static.double_pole_k2 is not None
            assert static.relative_db is None
    assert np.all(errors[-1] < errors[0])
    assert np.linalg.norm(errors[-1]) < np.linalg.norm(errors[-2])


def test_reciprocity_and_arbitrary_complex_shape_scaling(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion]) -> None:
    spectrum, ops, expansion = damped
    forward = expansion.at_positions([SOURCE], [RECEIVER])
    reverse = expansion.at_positions([RECEIVER], [SOURCE])
    scaled = replace(spectrum, solutions=tuple(replace(row, shape=row.shape * (0.3 + 1.7j))
        if row.shape is not None else row for row in spectrum.solutions))
    rescaled = _prepare(scaled, ops, 0.05).at_positions([SOURCE], [RECEIVER])
    for left, right, changed in zip(forward.entries, reverse.entries, rescaled.entries, strict=True):
        assert left.residue_k == pytest.approx(right.residue_k, rel=ROUNDING, abs=ROUNDING)
        assert left.residue_k == pytest.approx(changed.residue_k, rel=ROUNDING, abs=ROUNDING)
        assert left.resonance_magnitude == pytest.approx(changed.resonance_magnitude, rel=ROUNDING)
    assert np.allclose(forward.pressures([30, 57, 110]), rescaled.pressures([30, 57, 110]),
                       rtol=ROUNDING, atol=ROUNDING)


def test_node_suppresses_source_and_receiver_for_rigid_axial_mode() -> None:
    spectrum, ops = _spectrum(0)
    expansion = _prepare(spectrum, ops, 0)
    antinode = Point(0.1, 0.85, 0.6)
    node = Point(LENGTHS[0] / 2, 0.85, 0.6)
    mode_index = min(range(len(spectrum.solutions)), key=lambda i: abs(spectrum.solutions[i].frequency_hz - C / (2 * LENGTHS[0])))
    lookup = expansion.at_positions([antinode, node], [antinode, node])
    residues = {(row.source_index, row.receiver_index): abs(row.residue_k)
                for row in lookup.entries if row.mode_index == mode_index}
    assert residues[1, 0] < residues[0, 0]
    assert residues[0, 1] < residues[0, 0]
    assert residues[1, 1] < residues[1, 0]
    # 網格加細時節點相對參與繼續下降；不把粗網格節點誤差釘成精度值。
    mesh = generate_shoebox_mesh(Room(*LENGTHS), max_frequency_hz=270,
                                 elements_per_wavelength=3, sound_speed_m_s=C, random_seed=1)
    fine = solve_fem_modes(mesh, wall_impedances=_walls(0), density_kg_m3=RHO,
                           sound_speed_m_s=C, frequency_max_hz=85, retain_shapes=True)
    rows = _prepare(fine, assemble_p2_operators(mesh), 0).at_positions([antinode, node], [antinode]).entries
    axial = [row for row in rows if row.mode.kind is ModalKind.RESONANCE]
    assert abs(axial[1].residue_k / axial[0].residue_k) < residues[1, 0] / residues[0, 0]


def _half_power_width(frequencies: NDArray[np.float64], pressure: NDArray[np.complex128]) -> tuple[float, float]:
    power = abs(pressure)**2
    peak = int(np.argmax(power))
    half = power[peak] / 2
    left = np.flatnonzero(power[:peak] < half)[-1]
    right = peak + np.flatnonzero(power[peak:] < half)[0]
    lower = float(np.interp(half, power[left:left + 2], frequencies[left:left + 2]))
    upper = float(np.interp(half, power[right - 1:right + 1][::-1], frequencies[right - 1:right + 1][::-1]))
    return upper - lower, float(abs(pressure[peak]))


def test_isolated_mode_dense_sweep_decay_and_peak_converge() -> None:
    errors = []
    isolation = []
    for beta in (0.01, 0.005, 0.0025):
        spectrum, ops = _spectrum(beta, 175)
        mode = spectrum.modal_table[0]
        width = mode.omega.imag / math.pi
        gap = min(abs(other.frequency_hz - mode.frequency_hz) for other in spectrum.modal_table if other is not mode)
        assert gap > 10 * width  # 孤立案例的幾何選擇，並非 FEM 精度門檻。
        isolation.append(gap / width)
        scan_errors = []
        for points in (31, 61, 121, 241, 257):
            frequencies = np.linspace(mode.frequency_hz - 2 * width, mode.frequency_hz + 2 * width, points)
            direct = solve_frequency_responses(ops, right_hand_side=point_source_load(ops, SOURCE), receiver=RECEIVER,
                frequencies_hz=frequencies, wall_impedances=_walls(beta), density_kg_m3=RHO, sound_speed_m_s=C)
            measured_width, peak = _half_power_width(frequencies, direct)
            scan_errors.append(abs(measured_width / width - 1))
        assert all(fine < coarse for coarse, fine in zip(scan_errors, scan_errors[1:]))
        row = next(row for row in _prepare(spectrum, ops, beta).at_positions([SOURCE], [RECEIVER]).entries
                   if row.mode is mode)
        assert row.resonance_magnitude is not None
        measured_decay = math.pi * measured_width
        measured_t60 = 3 * math.log(10) / measured_decay
        assert mode.t60_s is not None
        errors.append(np.array([abs(measured_decay / mode.omega.imag - 1),
                                abs(measured_t60 / mode.t60_s - 1), abs(peak / row.resonance_magnitude - 1)]))
    assert all(fine > coarse for coarse, fine in zip(isolation, isolation[1:]))
    assert np.all(errors[-1] < errors[0])
    assert np.all(errors[-1] < errors[-2])


def test_many_positions_reuse_modes_and_record_each_lookup(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion], monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, expansion = damped
    from pydiso import mkl_solver

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("換位置不能求解或分解")

    monkeypatch.setattr(mkl_solver, "MKLPardisoSolver", fail)
    batched = expansion.at_positions([SOURCE, RECEIVER], [RECEIVER, SOURCE])
    for source_index, source in enumerate((SOURCE, RECEIVER)):
        for receiver_index, receiver in enumerate((RECEIVER, SOURCE)):
            single = expansion.at_positions([source], [receiver])
            assert np.array_equal(batched.pressures([40, 70])[source_index, receiver_index], single.pressures([40, 70])[0, 0])
            assert single.position_seconds > 0
    assert batched.position_seconds > 0


def test_nonresonant_rows_are_not_ranked_and_pair_db_is_relative(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion]) -> None:
    _, _, expansion = damped
    rows = expansion.at_positions([SOURCE, RECEIVER], [RECEIVER]).entries
    for source_index in range(2):
        resonances = [row for row in rows if row.source_index == source_index and row.mode.kind is ModalKind.RESONANCE]
        strongest = max(row.resonance_magnitude for row in resonances if row.resonance_magnitude is not None)
        for row in resonances:
            assert row.resonance_magnitude is not None
            assert row.relative_db == pytest.approx(20 * math.log10(row.resonance_magnitude / strongest), rel=ROUNDING, abs=ROUNDING)
    nonresonant = [row for row in rows if row.mode.kind is not ModalKind.RESONANCE]
    assert {row.mode.kind for row in nonresonant} == {ModalKind.STATIC, ModalKind.NONOSCILLATING_DECAY}
    assert all(row.relative_db is None and row.resonance_magnitude is None for row in nonresonant)


@pytest.mark.parametrize("frequencies", [[], [0], [-1], [math.nan], [math.inf]])
def test_invalid_response_frequencies_rejected(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion], frequencies: list[float]) -> None:
    with pytest.raises(ValueError):
        damped[2].at_positions([SOURCE], [RECEIVER]).pressures(frequencies)


@pytest.mark.parametrize("sources,receivers", [([], [RECEIVER]), ([SOURCE], []),
    ([Point(4, 1, 1)], [RECEIVER]), ([Point(math.nan, 1, 1)], [RECEIVER]),
    ([SOURCE], [Point(1, math.inf, 1)])])
def test_invalid_positions_rejected(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion], sources: list[Point], receivers: list[Point]) -> None:
    with pytest.raises(ValueError):
        damped[2].at_positions(sources, receivers)


def test_shapes_required_and_wrong_operators_rejected(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion]) -> None:
    spectrum, ops, _ = damped
    missing = replace(spectrum, solutions=tuple(replace(row, shape=None) for row in spectrum.solutions))
    with pytest.raises(ValueError, match="retain_shapes"):
        _prepare(missing, ops, 0.05)
    with pytest.raises(ValueError, match="同一"):
        _prepare(spectrum, replace(ops, mass=ops.mass * 2), 0.05)


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
@pytest.mark.parametrize("parameter", ["density_kg_m3", "sound_speed_m_s"])
def test_invalid_physics_rejected(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion], value: float, parameter: str) -> None:
    spectrum, ops, _ = damped
    parameters = {"density_kg_m3": RHO, "sound_speed_m_s": C}
    parameters[parameter] = value
    with pytest.raises(ValueError):
        prepare_modal_expansion(spectrum, ops, wall_impedances=_walls(0.05), **parameters)


@pytest.mark.parametrize("case", ["empty", "space", "zero_shape", "nonfinite_shape", "shape_dimension", "walls"])
def test_malformed_spectrum_or_mismatched_walls_rejected(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion], case: str) -> None:
    spectrum, ops, _ = damped
    beta = 0.05
    if case == "empty":
        spectrum = replace(spectrum, solutions=())
    elif case == "space":
        spectrum = replace(spectrum, degrees_of_freedom=ops.basis.N + 1)
    elif case == "walls":
        beta = 0.1
    else:
        row = spectrum.modal_table[0]
        shape = np.zeros(ops.basis.N, dtype=np.complex128)
        if case == "nonfinite_shape":
            shape[0] = math.nan
        if case == "shape_dimension":
            shape = shape[:-1]
        spectrum = replace(spectrum, solutions=(replace(row, shape=shape),))
    with pytest.raises(ValueError):
        _prepare(spectrum, ops, beta)


def test_defective_static_bilinear_norm_rejected(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion]) -> None:
    spectrum, ops, _ = damped
    shape = np.zeros(ops.basis.N, dtype=np.complex128)
    shape[:2] = (1, 1j)
    diagonal = np.ones(ops.basis.N)
    diagonal[:2] = 0
    artificial = replace(ops, stiffness=csr_matrix(np.diag(diagonal)), mass=csr_matrix(np.eye(ops.basis.N)))
    static = next(row for row in spectrum.solutions if row.kind is ModalKind.STATIC)
    defective = replace(spectrum, solutions=(replace(static, shape=shape, residual=0.0),))
    with pytest.raises(ArithmeticError, match="雙線性範數"):
        _prepare(defective, artificial, 0)


def test_nonorthogonal_repeated_mode_rejected(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion]) -> None:
    spectrum, ops, _ = damped
    mode = spectrum.modal_table[0]
    repeated = replace(spectrum, solutions=(mode, mode))
    with pytest.raises(ArithmeticError, match="重根"):
        _prepare(repeated, ops, 0.05)


def test_interpolation_failure_propagates_unchanged(damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion], monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.physics import fem_modal_participation

    failure = RuntimeError("原始 P2 插值失敗")

    def fail(*args: object, **kwargs: object) -> None:
        raise failure

    monkeypatch.setattr(fem_modal_participation, "point_source_load", fail)
    with pytest.raises(RuntimeError) as caught:
        damped[2].at_positions([SOURCE], [RECEIVER])
    assert caught.value is failure


def test_undamped_poles_are_infinite_and_exact_pole_pressure_rejected() -> None:
    spectrum, ops = _spectrum(0, 85)
    result = _prepare(spectrum, ops, 0).at_positions([SOURCE], [RECEIVER])
    row = next(row for row in result.entries if row.mode.kind is ModalKind.RESONANCE)
    assert row.resonance_magnitude == math.inf
    assert row.relative_db == 0.0
    with pytest.raises(ArithmeticError, match="極點"):
        result.pressures([row.mode.frequency_hz])


def test_resonance_magnitude_is_the_full_term_at_the_damped_frequency(
        damped: tuple[FemModalSpectrum, P2Operators, ModalExpansion]) -> None:
    """共振大小＝這一個模態的完整雙支項，在「有阻尼的共振頻率」Re ω/(2π) 取絕對值（考卷自己照定義算）。"""
    _, _, expansion = damped
    rows = [row for row in expansion.at_positions([SOURCE], [RECEIVER]).entries
            if row.mode.kind is ModalKind.RESONANCE]
    assert rows
    for row in rows:
        pole = row.mode.omega / C
        k = row.mode.omega.real / C
        expected = abs(row.residue_k / (k - pole) - row.residue_k.conjugate() / (k + pole.conjugate()))
        assert row.resonance_magnitude == pytest.approx(expected, rel=ROUNDING)
