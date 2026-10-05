"""半解析阻尼的獨立考卷：不以被測超越方程或換算函式算期望。

壁面動量式 jωρv=-∇p，向外 v=p/Z，故 ∂n p+jkβp=0。
一維管的反射係數 r=(1-β)/(1+β)，往返閉合 r²e^(-2jkL)=1，
故 e^(-2jkL)=((1+β)/(1-β))²；0≤β<1 時
k=nπ/L+j log((1+β)/(1-β))/L。小阻尼由能量積分得
Im ω = c Σ β_i (2 if n_i>0 else 1)/L_i + O(β²)。
推導：將剛性 p₀ 代入弱式的一階項，δk=jΣβ∫牆p₀²/(2∫房p₀²)。
每軸 ∫X₀²=L/(2 if n>0 else 1)，兩牆 X₀² 合計為 2，故得到上述斜率。
零指標分岔的非零根則為 Im ω=2c Σ β_i/L_i+O(β³)，不是振盪式。
比較容差用浮點舍入放大或微擾截斷階數；沒有有限元素精度門檻。
"""
from __future__ import annotations

import cmath
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import root

from aosr.config.physics_constants import PhysicsConstants
from aosr.physics.modal_convention import ModalKind, angular_frequency, classify_omega, modal_quantities
from aosr.physics.modal_rectangle_truth import (
    ContinuationOutcome,
    RectangleModalProblem,
    trace_mode,
    trace_modes,
    rigid_indices,
    BranchRelation,
    RectangleTruth,
)


LENGTHS = (6.0, 4.0, 3.0)
C = 343.0
RHO = 1.2
# 牛頓迭代與三角函式累積舍入，取 float64 epsilon 的 2^18 倍。
ROUNDING = 2.0**18 * np.finfo(float).eps
# 手算換算只有幾次乘除、對數與 π，十六倍 epsilon 足以涵蓋舍入。
HAND_ROUNDING = 16 * np.finfo(float).eps


def _problem(beta: tuple[float, float, float]) -> RectangleModalProblem:
    return RectangleModalProblem(LENGTHS, C, RHO, beta)


@pytest.mark.parametrize("index", [(1, 0, 0), (0, 2, 1), (3, 2, 1)])
def test_zero_admittance_recovers_independent_rigid_frequency(
    index: tuple[int, int, int],
) -> None:
    expected = C / 2 * math.sqrt(sum((n / length) ** 2 for n, length in zip(index, LENGTHS)))
    rigid = trace_mode(_problem((0.0, 0.0, 0.0)), index)
    tiny = trace_mode(_problem((1e-8, 2e-8, 3e-8)), index)
    assert rigid.quantities.frequency_hz == pytest.approx(expected, rel=ROUNDING)
    assert tiny.quantities.frequency_hz == pytest.approx(expected, rel=ROUNDING)
    assert tiny.omega.imag > 0


def _shape(q: complex, n: int, length: float, x: float) -> tuple[complex, complex]:
    """獨立中心座標 cos/sin 形狀；奇函數除軸波數以保留 k_i=0 極限。"""
    wave = cmath.sqrt(q)
    s = x - length / 2
    if n % 2 == 0:
        return cmath.cos(wave * s), -wave * cmath.sin(wave * s)
    if abs(wave) < 1e-12:
        return complex(s), 1 + 0j
    return cmath.sin(wave * s) / wave, cmath.cos(wave * s)


@pytest.mark.parametrize("index", [(0, 0, 0), (1, 0, 0), (2, 1, 3), (0, 2, 0)])
def test_all_six_original_boundary_conditions_and_dispersion(
    index: tuple[int, int, int],
) -> None:
    beta = (0.25, 0.12, 0.07)
    mode = trace_mode(_problem(beta), index)
    assert mode.reached_target
    assert sum(mode.axis_wave_numbers_squared) == pytest.approx(mode.wave_number**2, rel=ROUNDING)
    for axis, (length, admittance) in enumerate(zip(LENGTHS, beta)):
        for face, normal in ((0.0, -1.0), (length, 1.0)):
            coordinates = [length_i * 0.37 for length_i in LENGTHS]
            coordinates[axis] = face
            shapes = [_shape(q, n, size, x) for q, n, size, x in zip(
                mode.axis_wave_numbers_squared, index, LENGTHS, coordinates,
            )]
            other = 1 + 0j
            for i, (value, _) in enumerate(shapes):
                if i != axis:
                    other *= value
            value, derivative = shapes[axis]
            derivative_term = normal * derivative * other
            impedance_term = 1j * mode.wave_number * admittance * value * other
            scale = max(abs(derivative_term), abs(impedance_term), 1.0)
            assert abs(derivative_term + impedance_term) / scale < ROUNDING


@pytest.mark.parametrize("n", [0, 1, 2, 5])
@pytest.mark.parametrize("beta", [0.1, 0.7])
def test_one_dimensional_closed_form(n: int, beta: float) -> None:
    mode = trace_mode(_problem((beta, 0.0, 0.0)), (n, 0, 0))
    expected = n * math.pi / LENGTHS[0] + 1j * math.log((1 + beta) / (1 - beta)) / LENGTHS[0]
    assert mode.reached_target
    assert mode.wave_number == pytest.approx(expected, rel=ROUNDING, abs=ROUNDING)


@pytest.mark.parametrize("index", [(1, 0, 0), (2, 1, 3), (0, 2, 0)])
def test_small_damping_slope_converges_to_independent_energy_perturbation(
    index: tuple[int, int, int],
) -> None:
    weights = (1.0, 0.7, 0.4)
    slope = C * sum(w * (2 if n else 1) / size for w, n, size in zip(weights, index, LENGTHS))
    errors = []
    for epsilon in (0.01, 0.005, 0.0025):
        beta = tuple(epsilon * w for w in weights)
        mode = trace_mode(_problem((beta[0], beta[1], beta[2])), index)
        errors.append(abs(mode.omega.imag / epsilon / slope - 1))
    # β 是實數時 k(−β) 是 k(β) 的共軛，Im k 只含 β 的奇次方，一階式的相對誤差是 O(β²)：
    # β 每減半誤差掉約 4 倍；取 3 倍當下界，不釘單次近似值。斜率錯了（β 多乘一個倍數）誤差會停在常數。
    assert all(later < earlier / 3 for earlier, later in zip(errors, errors[1:]))


def test_zero_branch_is_decay_continuation_and_static_root_remains() -> None:
    rigid = trace_mode(_problem((0.0, 0.0, 0.0)), (0, 0, 0))
    damped = trace_mode(_problem((0.25, 0.25, 0.25)), (0, 0, 0))
    assert rigid.quantities.kind is ModalKind.STATIC
    assert damped.quantities.kind is ModalKind.NONOSCILLATING_DECAY
    assert damped.outcome is ContinuationOutcome.NONOSCILLATING
    assert damped.points[0].wave_number == 0j
    assert damped.points[-1].parameter == 1.0
    assert all(point.wave_number.imag > 0 for point in damped.points[1:])
    # 靜態根另外記帳、零指標的衰減支不進模態表（不是房間共振）。
    truth = trace_modes(_problem((0.25, 0.25, 0.25)), indices=[(0, 0, 0)])
    assert modal_quantities(angular_frequency(truth.static_wave_number, C)).kind is ModalKind.STATIC
    assert not truth.modal_table
    tiny = trace_mode(_problem((1e-5, 2e-5, 3e-5)), (0, 0, 0))
    expected = 2 * C * sum(beta / length for beta, length in zip((1e-5, 2e-5, 3e-5), LENGTHS))
    # 零支的斜率式相對截斷 O(β²)，此處 β≤3e-5，留十倍量級餘裕。
    assert tiny.omega.imag == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("index", [(0, 0, 0), (1, 0, 0), (0, 2, 0), (3, 2, 1)])
def test_halved_maximum_step_preserves_label_and_root(index: tuple[int, int, int]) -> None:
    problem = _problem((0.7, 0.4, 0.2))
    coarse = trace_mode(problem, index, max_step=0.05)
    fine = trace_mode(problem, index, max_step=0.025)
    assert coarse.reached_target and fine.reached_target
    assert coarse.wave_number == pytest.approx(fine.wave_number, rel=ROUNDING)
    assert coarse.axis_wave_numbers_squared == pytest.approx(fine.axis_wave_numbers_squared, rel=ROUNDING)
    assert all(b.parameter - a.parameter <= 0.05 + ROUNDING for a, b in zip(coarse.points, coarse.points[1:]))


def test_frequency_cap_retains_trace_of_crossing_branch() -> None:
    problem = _problem((0.25, 0.25, 0.25))
    index = (0, 0, 5)
    rigid_hz = C / 2 * index[2] / LENGTHS[2]
    mode = trace_mode(problem, index, frequency_max_hz=rigid_hz)
    assert mode.outcome is ContinuationOutcome.ABOVE_FREQUENCY_LIMIT
    assert mode.quantities.frequency_hz > rigid_hz
    batch = trace_modes(problem, indices=[index, (0, 0, 0)], frequency_max_hz=rigid_hz)
    assert {m.index for m in batch.modes} == {index, (0, 0, 0)}
    assert all(m.quantities.kind is ModalKind.RESONANCE for m in batch.modal_table)
    assert mode.index not in {m.index for m in batch.modal_table}


def test_impedance_conversion_uses_supplied_case_physics() -> None:
    physics = PhysicsConstants(air_density_kg_m3=1.7, sound_speed_m_s=321.0,
                               ref_pressure_pa=2e-5, air_viscosity_pa_s=1.8e-5)
    problem = RectangleModalProblem.from_impedances(LENGTHS, physics, (4 * physics.rho_c, None, 10 * physics.rho_c))
    assert problem.admittances == (0.25, 0.0, 0.1)
    assert problem.sound_speed_m_s == physics.sound_speed_m_s
    assert problem.density_kg_m3 == physics.air_density_kg_m3


@pytest.mark.parametrize("omega,kind", [
    (0j, ModalKind.STATIC), (3j, ModalKind.NONOSCILLATING_DECAY),
    (4 + 3j, ModalKind.RESONANCE), (4 + 0j, ModalKind.RESONANCE),
])
def test_classification(omega: complex, kind: ModalKind) -> None:
    assert classify_omega(omega) is kind


def test_frequency_t60_and_q_hand_calculated() -> None:
    assert angular_frequency(2 + 3j, 300.0) == 600 + 900j
    quantities = modal_quantities(20 * math.pi + 2j)
    assert quantities.frequency_hz == pytest.approx(10, rel=HAND_ROUNDING)
    assert quantities.t60_s == pytest.approx(1.5 * math.log(10), rel=HAND_ROUNDING)
    assert quantities.q == pytest.approx(5 * math.pi, rel=HAND_ROUNDING)
    lossless = modal_quantities(20 * math.pi + 0j)
    assert lossless.t60_s == math.inf and lossless.q == math.inf
    decay = modal_quantities(2j)
    assert decay.t60_s == pytest.approx(1.5 * math.log(10), rel=HAND_ROUNDING)
    assert decay.q == 0.0
    static = modal_quantities(0j)
    assert static.t60_s is None and static.q is None


@pytest.mark.parametrize("omega", [-1 + 2j, 1 - 2j, complex(math.nan, 0)])
def test_convention_rejects_negative_branch_growth_and_nonfinite(omega: complex) -> None:
    with pytest.raises(ValueError):
        modal_quantities(omega)


def test_direct_jump_loses_index_against_independent_wall_equations() -> None:
    """原始邊界式交給另一求根器：一步從 (0,2,0) 跳到 (2,0,0)。"""
    beta = (0.9, 0.9, 0.9)
    index = (0, 2, 0)

    def independent_boundary(x: np.ndarray[tuple[int, ...], np.dtype[np.float64]]) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        axes = x[:3] + 1j * x[3:]
        wave = cmath.sqrt(complex(sum(axes)))
        equations = []
        for q, n, size, b in zip(axes, index, LENGTHS, beta):
            value, derivative = _shape(complex(q), n, size, size)
            equations.append(derivative + 1j * wave * b * value)
        values = np.asarray(equations)
        return np.concatenate((values.real, values.imag))

    rigid_axes = [(n * math.pi / size)**2 for n, size in zip(index, LENGTHS)]
    direct = root(independent_boundary, np.asarray([*rigid_axes, 0.0, 0.0, 0.0]), tol=1e-11)
    assert direct.success
    # 另一求根器的原始邊界殘差，仍只容許 float64 舍入放大量。
    assert max(abs(independent_boundary(direct.x))) < ROUNDING
    direct_axes = direct.x[:3] + 1j * direct.x[3:]
    wanted = trace_mode(_problem(beta), index)
    finer = trace_mode(_problem(beta), index, max_step=0.025)
    neighbor = trace_mode(_problem(beta), (2, 0, 0))
    assert wanted.reached_target and neighbor.reached_target and finer.reached_target
    assert wanted.axis_wave_numbers_squared == pytest.approx(finer.axis_wave_numbers_squared, rel=ROUNDING)
    assert direct_axes == pytest.approx(neighbor.axis_wave_numbers_squared, rel=ROUNDING)
    assert not np.allclose(direct_axes, wanted.axis_wave_numbers_squared, rtol=ROUNDING, atol=ROUNDING)


def test_rigid_enumeration_preserves_degenerate_shapes_and_independent_indices() -> None:
    problem = _problem((0.0, 0.0, 0.0))
    cap = 90.0
    expected = {(x, y, z) for x in range(8) for y in range(8) for z in range(8)
                if C / 2 * math.sqrt((x / 6)**2 + (y / 4)**2 + (z / 3)**2) <= cap}
    assert set(rigid_indices(problem, cap)) == expected
    truth = trace_modes(problem, frequency_max_hz=cap)
    assert {mode.index for mode in truth.modes} == expected
    assert {mode.index for mode in truth.modal_table} == expected - {(0, 0, 0)}
    assert not any(relation.kind == "coalesced" for relation in truth.relations)
    # (2,0,0) 和 (0,0,1) 同頻不同形狀，必須保留指標，不能按頻率去重。
    assert {(2, 0, 0), (0, 0, 1)} <= {mode.index for mode in truth.modal_table}


def test_singular_heavy_damping_is_reported_without_fake_completion() -> None:
    mode = trace_mode(_problem((0.5, 0.5, 0.5)), (1, 0, 0))
    assert not mode.reached_target
    assert mode.outcome is ContinuationOutcome.UNRESOLVED
    assert mode.failure
    assert 0 < mode.points[-1].parameter < 1
    # 停在過阻尼轉折：實部掉到接近零（振盪要變成不振盪），不是任意的求根失敗。
    assert mode.points[-1].wave_number.real < 1e-2 * mode.points[-1].wave_number.imag
    assert all(point.residual < ROUNDING for point in mode.points)
    truth = trace_modes(_problem((0.5, 0.5, 0.5)), indices=[(1, 0, 0)])
    assert not truth.modal_table


@pytest.mark.parametrize("invalid", [-0.1, math.inf, math.nan])
def test_admittance_input_is_checked(invalid: float) -> None:
    with pytest.raises(ValueError):
        _problem((invalid, 0.0, 0.0))


def test_invalid_enumeration_requests_fail_explicitly() -> None:
    problem = _problem((0.1, 0.1, 0.1))
    with pytest.raises(ValueError):
        trace_mode(problem, (1, 0, 0), max_step=1)
    with pytest.raises(ValueError):
        trace_modes(problem)
    with pytest.raises(ValueError):
        trace_modes(problem, indices=[(1, 0, 0), (1, 0, 0)])
    with pytest.raises(ValueError):
        trace_modes(problem, frequency_max_hz=100, seed_frequency_max_hz=90)


def test_coalescence_ledger_keeps_every_parent_out_of_the_modal_table() -> None:
    """併根帳的獨立輸入：直接構造同形狀的兩份標記，不假裝找到新物理併根；兩條都留在帳上、都不進模態表。"""
    root_trace = trace_mode(_problem((0.1, 0.0, 0.0)), (2, 0, 0))
    parents = ((2, 0, 0), (4, 0, 0))
    modes = tuple(replace(root_trace, index=index, outcome=ContinuationOutcome.COALESCED) for index in parents)
    ledger = RectangleTruth(modes, (BranchRelation(parents, "coalesced"),), None)
    assert {mode.index for mode in ledger.modes} == set(parents)
    assert not ledger.modal_table
    assert all(mode.outcome is ContinuationOutcome.COALESCED for mode in ledger.modes)


def test_largest_allowed_step_keeps_the_same_root_as_fine_steps() -> None:
    """步長只影響快慢、不准換根：大阻尼下用最大步長追 (4,1,0)，要跟細步長同一個根。

    主對話 2026-10-05 植錯時找到的例子：拿掉「根一步移動太多就縮步」那道檢查，這一題最大步長會落到鄰近的根。
    """
    problem = _problem((0.9, 0.9, 0.9))
    fine = trace_mode(problem, (4, 1, 0), max_step=0.0125)
    largest = trace_mode(problem, (4, 1, 0), max_step=0.25)
    assert fine.reached_target and largest.reached_target
    assert largest.axis_wave_numbers_squared == pytest.approx(fine.axis_wave_numbers_squared, rel=ROUNDING, abs=ROUNDING)


@pytest.mark.parametrize(("omega", "kind"), [
    # 第 0 步有限元素實際算出的雜訊值（複查拿來的例子）：靜態根、帶負雜訊虛部的無阻尼模態。
    (6.66e-13 - 1.86e-13j, ModalKind.STATIC), (1.64e-12 + 2.14e-12j, ModalKind.STATIC),
    (2 * math.pi * 28.58 - 1.1e-9j, ModalKind.RESONANCE), (1e-12 + 51.97j, ModalKind.NONOSCILLATING_DECAY),
])
def test_caller_supplied_zero_scale_classifies_numerical_noise(omega: complex, kind: ModalKind) -> None:
    assert classify_omega(omega, zero_rad_s=1e-8) is kind
    assert modal_quantities(omega, zero_rad_s=1e-8).kind is kind


def test_rigid_enumeration_keeps_a_mode_exactly_at_the_cap() -> None:
    """上限剛好等於某個剛性頻率時也要列進來（複查找到的捨入例子 L=3.3、n=23）。"""
    problem = RectangleModalProblem((3.3, 1.0, 1.0), C, RHO, (0.0, 0.0, 0.0))
    cap = C / 2 * 23 / 3.3
    assert (23, 0, 0) in rigid_indices(problem, cap)
