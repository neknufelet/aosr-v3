"""有限元素模態的結構與收斂考卷；沒有有限元素對真值的精度門檻。"""
from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from aosr.geometry.shoebox import Room, Wall
from aosr.geometry.shoebox_mesh import generate_shoebox_mesh
from aosr.physics.fem_modal import FemModalSpectrum, solve_fem_modes
from aosr.physics.modal_convention import ModalKind
from aosr.physics.modal_rectangle_truth import RectangleModalProblem, trace_modes


# 凍結的小案例條件；不是產品物理常數。
LENGTHS = (2.0, 1.7, 1.3)
C = 343.0
RHO = 1.2
CAP = 165.0
# 矩陣運算、稀疏分解與迭代累積的浮點舍入預算，不是離散化精度契約。
ROUNDING = 2.0**18 * np.finfo(float).eps


def _rigid_frequencies(lengths: tuple[float, float, float], cap: float) -> list[float]:
    limits = [range(math.floor(2 * cap * length / C) + 1) for length in lengths]
    return [frequency for index in product(*limits)
            if (frequency := C / 2 * math.sqrt(sum((n / size)**2 for n, size in zip(index, lengths)))) <= cap]


def _solve(beta: tuple[float, float, float], density: int,
           lengths: tuple[float, float, float] = LENGTHS, cap: float = CAP,
           *, shapes: bool = False) -> FemModalSpectrum:
    mesh = generate_shoebox_mesh(Room(*lengths), max_frequency_hz=cap,
                                 elements_per_wavelength=density,
                                 sound_speed_m_s=C, random_seed=1)
    walls = {wall: None if beta[wall.axis()] == 0 else RHO * C / beta[wall.axis()]
             for wall in Wall.all()}
    return solve_fem_modes(mesh, wall_impedances=walls, density_kg_m3=RHO,
                           sound_speed_m_s=C, frequency_max_hz=cap,
                           reference_frequencies_hz=_rigid_frequencies(lengths, cap + 60),
                           retain_shapes=shapes)


def _paired_errors(spectrum: FemModalSpectrum, truth: list[complex]) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray([row.omega for row in spectrum.solutions if row.kind is not ModalKind.STATIC])
    expected = np.asarray([value for value in truth if value != 0j])
    distances = abs(actual[:, None] - expected[None, :]) / abs(expected[None, :])
    rows, columns = linear_sum_assignment(distances)
    assert set(rows) == set(range(len(actual)))
    assert set(columns) == set(range(len(expected)))
    assert all(row.residual < ROUNDING for row in spectrum.solutions)
    return actual[rows], expected[columns]


def _assert_static(spectrum: FemModalSpectrum) -> None:
    statics = [row for row in spectrum.solutions if row.kind is ModalKind.STATIC]
    assert [row.omega for row in statics] == [0j]
    assert statics[0].t60_s is None and statics[0].q is None
    assert spectrum.zero_rad_s > 0
    assert all(row.kind is ModalKind.RESONANCE for row in spectrum.modal_table)


def test_rigid_modes_match_independent_formula_and_converge() -> None:
    truth = [complex(2 * math.pi * f) for f in _rigid_frequencies(LENGTHS, CAP)]
    errors = []
    for density in (3, 5, 7):
        spectrum = _solve((0.0, 0.0, 0.0), density, shapes=True)
        _assert_static(spectrum)
        actual, expected = _paired_errors(spectrum, truth)
        assert np.all(actual.imag == 0)
        assert all(row.t60_s is not None and math.isinf(row.t60_s)
                   and row.q is not None and math.isinf(row.q) for row in spectrum.modal_table)
        assert all(row.shape is not None for row in spectrum.solutions)
        errors.append(float(np.linalg.norm((actual.real - expected.real) / expected.real)))
    assert all(fine < coarse for coarse, fine in zip(errors, errors[1:]))
    # P2 的低頻特徵值收斂階通常約四階；只驗有加速下降，不釘精度值。
    assert errors[-1] / errors[-2] < 5 / 7


def test_rigid_degenerate_frequencies_keep_all_independent_modes() -> None:
    lengths = (1.5, 1.5, 1.5)
    cap = 145.0
    truth = [complex(2 * math.pi * f) for f in _rigid_frequencies(lengths, cap)]
    spectrum = _solve((0.0, 0.0, 0.0), 4, lengths, cap)
    _assert_static(spectrum)
    _paired_errors(spectrum, truth)


@pytest.mark.parametrize("beta", [1e-6, 1e-7])
def test_weak_damping_is_not_erased_by_static_jordan_noise(beta: float) -> None:
    spectrum = _solve((beta, beta, beta), 3)
    _assert_static(spectrum)
    assert [row.kind for row in spectrum.solutions if row.frequency_hz == 0] == [
        ModalKind.STATIC, ModalKind.NONOSCILLATING_DECAY]
    assert all(row.t60_s is not None and math.isfinite(row.t60_s)
               and row.q is not None and math.isfinite(row.q) for row in spectrum.modal_table)
    truth = trace_modes(RectangleModalProblem(LENGTHS, C, RHO, (beta, beta, beta)), frequency_max_hz=CAP)
    _paired_errors(spectrum, [mode.omega for mode in truth.modes])


def test_unresolvable_damped_zero_cluster_reports_failure() -> None:
    # 平台的舍入不同：能分辨就留兩支，不能分辨就明確報錯，禁止吞掉非零支。
    try:
        spectrum = _solve((1e-8, 1e-8, 1e-8), 3)
    except ArithmeticError as failure:
        assert "不可分辨靜態與非振盪衰減" in str(failure)
    else:
        assert [row.kind for row in spectrum.solutions if row.frequency_hz == 0] == [
            ModalKind.STATIC, ModalKind.NONOSCILLATING_DECAY]


@pytest.mark.parametrize("beta", [(0.25, 0.25, 0.25), (0.10, 0.17, 0.23)])
def test_impedance_complex_frequency_decay_t60_q_converge(beta: tuple[float, float, float]) -> None:
    truth = trace_modes(RectangleModalProblem(LENGTHS, C, RHO, beta),
                        frequency_max_hz=CAP, seed_frequency_max_hz=CAP + 60)
    assert all(mode.reached_target for mode in truth.modes)
    expected_modes = [mode for mode in truth.modes if mode.quantities.frequency_hz <= CAP]
    decay = [mode for mode in expected_modes if mode.quantities.kind is ModalKind.NONOSCILLATING_DECAY]
    assert [mode.index for mode in decay] == [(0, 0, 0)]
    errors = []
    for density in (3, 5):
        spectrum = _solve(beta, density)
        _assert_static(spectrum)
        actual, expected = _paired_errors(spectrum, [mode.omega for mode in expected_modes])
        decays = [row for row in spectrum.solutions if row.kind is ModalKind.NONOSCILLATING_DECAY]
        # 不振盪解回報的實部是零（舍入雜訊只留在 raw_omega），頻率也是零。
        assert decays and all(row.omega.real == 0 and row.frequency_hz == 0 for row in decays)
        assert all(row.shape is None for row in spectrum.solutions)
        oscillating = expected.real > spectrum.zero_rad_s
        re_error = np.linalg.norm((actual.real[oscillating] - expected.real[oscillating]) / expected.real[oscillating])
        im_error = np.linalg.norm((actual.imag - expected.imag) / expected.imag)
        rows = [row for row in spectrum.solutions if row.kind is not ModalKind.STATIC]
        t60 = np.asarray([row.t60_s for row in rows], dtype=float)
        quality = np.asarray([row.q for row in rows], dtype=float)
        t60_error = np.linalg.norm(t60 / (3 * math.log(10) / expected.imag) - 1)
        q_error = np.linalg.norm(quality[oscillating]
                                 / (expected.real[oscillating] / (2 * expected.imag[oscillating])) - 1)
        assert np.all(quality[~oscillating] == 0)
        errors.append(np.array([re_error, im_error, t60_error, q_error]))
    assert np.all(errors[-1] < errors[0])
