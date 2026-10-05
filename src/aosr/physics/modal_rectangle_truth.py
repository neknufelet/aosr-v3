"""長方形房、每軸兩牆同一實阻抗：三複數未知數的耦合延拓真值。

時間 e^(+jωt)，jωρv=-∇p、外法向 v_n=p/Z 給 ∂n p+jkβp=0。
分部積分 Helmholtz 式得到 K+jkCt-k²M，與既有頻域組裝同號。
中心座標 s=x-L/2：偶指標 X=cos(k_x s)，奇指標 X=sin(k_x s)/k_x。
令 a=L/2、t=(a k_x)²，S=sin(sqrt(t))/sqrt(t)、C=cos(sqrt(t))。
每軸方程為偶 tS-jkβaC=0、奇 C+jkβaS=0；三軸的 k 同時滿足
k²=Σ t_i/a_i²。使用 t 避免 n_i=0 時軸波數平方根的假奇異性。
三條方程與完整耦合 Jacobian（牛頓法的導數矩陣）同時求解，不逐軸找根。

(0,0,0) 在 β=0 為重零起點：靜態根另外保留；用 u_i=t_i/k 消去
靜態因子，k=Σ u_i/a_i²，方程 u_i S(ku_i)-jβ_i a_i C(ku_i)=0。
它在 u=0 有非奇異導數，能由零連續追出非零純衰減支，沒有預設根的種類。

延拓 β(s)=sβ_target：每步以前根預測，同時比整步與兩個半步的牛頓根。
殘差、不一致或根移動過大便拒絕並折半；易收斂才放大 1.5 倍，上限 max_step。
達最小步仍失敗就保留最後成功點並標 unresolved，不把失敗當模態消失。
奇異併根附近此參數延拓不能保證穿越；沒有改用弧長延拓或有限元素。

頻率掃描枚舉「剛性起點」至 seed_frequency_max_hz（預設為頻率上限），
保留全部延拓結局，modal_table 才依阻尼後頻率與種類篩選。額外起點範圍由
呼叫端指定；此有限枚舉不是任意大阻尼下全頻譜完備性的證明。
物理量必須由呼叫端傳入；產品用設定層 PhysicsConstants，凍結案例用案例值。
"""
from __future__ import annotations

import cmath
import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations, product

import numpy as np
from numpy.typing import NDArray

from aosr.config.physics_constants import PhysicsConstants
from aosr.physics.modal_convention import ModalKind, ModalQuantities, angular_frequency, modal_quantities


ROOT_RESIDUAL_TOL = 2e-13
PATH_AGREEMENT_TOL = 1e-8
ROOT_MOVEMENT_TOL = 0.2
BRANCH_MATCH_TOL = 1e-8
_MIN_STEP = 2.0**-24
_MAX_ACCEPTED_STEPS = 4096
ModeIndex = tuple[int, int, int]
ComplexTriple = tuple[complex, complex, complex]
ComplexVector = NDArray[np.complex128]


@dataclass(frozen=True)
class RectangleModalProblem:
    """長寬高 m、聲速 m/s、密度 kg/m³、三軸牆對導納 β=ρc/Z。"""

    lengths_m: tuple[float, float, float]
    sound_speed_m_s: float
    density_kg_m3: float
    admittances: tuple[float, float, float]

    def __post_init__(self) -> None:
        positive = (*self.lengths_m, self.sound_speed_m_s, self.density_kg_m3)
        if len(self.lengths_m) != 3 or len(self.admittances) != 3:
            raise ValueError("長度與導納各需三軸")
        if any(not math.isfinite(x) or x <= 0 for x in positive):
            raise ValueError("長度、聲速、密度必須為正有限實數")
        if any(not math.isfinite(b) or b < 0 for b in self.admittances):
            raise ValueError("導納必須非負有限實數；零表示剛性")

    @classmethod
    def from_impedances(
        cls, lengths_m: tuple[float, float, float], physics: PhysicsConstants,
        impedances_pa_s_m: tuple[float | None, float | None, float | None],
    ) -> RectangleModalProblem:
        """比聲阻抗 Z（Pa·s/m），None 表示剛性，使用傳入的物理條件。"""
        if any(z is not None and (not math.isfinite(z) or z <= 0) for z in impedances_pa_s_m):
            raise ValueError("比聲阻抗必須為正有限實數或 None")
        beta = tuple(0.0 if z is None else physics.rho_c / z for z in impedances_pa_s_m)
        return cls(lengths_m, physics.sound_speed_m_s, physics.air_density_kg_m3,
                   (beta[0], beta[1], beta[2]))


class ContinuationOutcome(Enum):
    """起點身份與最後可驗結果；跨界與純衰減不被默默刪掉。"""

    TRACKED = "tracked"
    NONOSCILLATING = "nonoscillating"
    STATIC = "static"
    ABOVE_FREQUENCY_LIMIT = "above_frequency_limit"
    COALESCED = "coalesced"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ContinuationPoint:
    """導納比例、原始波數與軸波數平方（rad²/m²）、牛頓殘差。"""

    parameter: float
    wave_number: complex
    axis_wave_numbers_squared: ComplexTriple
    residual: float
    iterations: int


@dataclass(frozen=True)
class ModeTrace:
    """一個剛性指標的完整接受步紀錄；失敗資訊不換成假解。"""

    index: ModeIndex
    sound_speed_m_s: float
    points: tuple[ContinuationPoint, ...]
    outcome: ContinuationOutcome
    failure: str | None = None

    @property
    def reached_target(self) -> bool:
        return self.points[-1].parameter == 1.0

    @property
    def wave_number(self) -> complex:
        return self.points[-1].wave_number

    @property
    def axis_wave_numbers_squared(self) -> ComplexTriple:
        return self.points[-1].axis_wave_numbers_squared

    @property
    def axis_wave_numbers(self) -> ComplexTriple:
        """軸波數 rad/m；± 的形狀只差整體符號，取主平方根。"""
        q = self.axis_wave_numbers_squared
        return cmath.sqrt(q[0]), cmath.sqrt(q[1]), cmath.sqrt(q[2])

    @property
    def omega(self) -> complex:
        return angular_frequency(self.wave_number, self.sound_speed_m_s)

    @property
    def quantities(self) -> ModalQuantities:
        return modal_quantities(self.omega)


@dataclass(frozen=True)
class BranchRelation:
    """頻率排序交叉不等於指標交換；同形狀同根才標併根。"""

    indices: tuple[ModeIndex, ModeIndex]
    kind: str


@dataclass(frozen=True)
class RectangleTruth:
    """保留所有起點的帳與獨立靜態解；僅共振才進模態表。"""

    modes: tuple[ModeTrace, ...]
    relations: tuple[BranchRelation, ...]
    frequency_max_hz: float | None
    static_wave_number: complex = 0j

    @property
    def modal_table(self) -> tuple[ModeTrace, ...]:
        """併根的每一條都不進表：兩條會振盪的起點在實數參數上剛好併在目標點不是常態，
        多半是某條跳了根；不挑一條留下、把另一條藏成「併掉不算漏」，兩條都留在帳上待查（複查）。"""
        merged = {index for relation in self.relations if relation.kind == "coalesced" for index in relation.indices}
        return tuple(mode for mode in self.modes if mode.reached_target
                     and mode.index not in merged
                     and mode.quantities.kind is ModalKind.RESONANCE
                     and (self.frequency_max_hz is None
                          or mode.quantities.frequency_hz <= self.frequency_max_hz))


def _triple(values: ComplexVector) -> ComplexTriple:
    return complex(values[0]), complex(values[1]), complex(values[2])


def _sinc_cos(t: complex) -> tuple[complex, complex, complex, complex]:
    """整函數 S(t)、C(t) 與導數；原點用泰勒式避免相消。"""
    if abs(t) < 1e-4:
        sine = 1 - t / 6 + t**2 / 120 - t**3 / 5040 + t**4 / 362880
        cosine = 1 - t / 2 + t**2 / 24 - t**3 / 720 + t**4 / 40320
        derivative = -1 / 6 + t / 60 - t**2 / 1680 + t**3 / 90720
    else:
        root = cmath.sqrt(t)
        sine = cmath.sin(root) / root
        cosine = cmath.cos(root)
        derivative = (cosine - sine) / (2 * t)
    return sine, cosine, derivative, -sine / 2


def _wave_and_axes(
    state: ComplexVector, half_lengths: NDArray[np.float64], zero_branch: bool,
) -> tuple[complex, ComplexVector]:
    if zero_branch:
        k = complex(np.sum(state / half_lengths**2))
        return k, k * state / half_lengths**2
    axes = state / half_lengths**2
    return cmath.sqrt(complex(np.sum(axes))), axes


def _equations(
    state: ComplexVector, problem: RectangleModalProblem, index: ModeIndex, parameter: float,
) -> tuple[ComplexVector, NDArray[np.complex128]]:
    """三條方程及完整導數；總波數的導數流入每一列每一欄。"""
    a = np.asarray(problem.lengths_m, dtype=float) / 2
    beta = np.asarray(problem.admittances) * parameter
    zero_branch = index == (0, 0, 0)
    k, _ = _wave_and_axes(state, a, zero_branch)
    values = np.zeros(3, dtype=np.complex128)
    jacobian = np.zeros((3, 3), dtype=np.complex128)
    for i, n in enumerate(index):
        t = complex(k * state[i]) if zero_branch else complex(state[i])
        sine, cosine, ds, dc = _sinc_cos(t)
        if zero_branch:
            values[i] = (state[i] * sine - 1j * beta[i] * a[i] * cosine) / a[i]
            dt = state[i] / a**2
            dt[i] += k
            jacobian[i] = (state[i] * ds - 1j * beta[i] * a[i] * dc) * dt / a[i]
            jacobian[i, i] += sine / a[i]
        elif n % 2 == 0:
            scale = max(1.0, (n * math.pi / 2)**2)
            values[i] = (t * sine - 1j * k * beta[i] * a[i] * cosine) / scale
            jacobian[i] = -1j * beta[i] * a[i] * cosine / (2 * k * a**2 * scale)
            jacobian[i, i] += (sine + t * ds - 1j * k * beta[i] * a[i] * dc) / scale
        else:
            values[i] = cosine + 1j * k * beta[i] * a[i] * sine
            jacobian[i] = 1j * beta[i] * a[i] * sine / (2 * k * a**2)
            jacobian[i, i] += dc + 1j * k * beta[i] * a[i] * ds
    return values, jacobian


def _newton(
    start: ComplexVector, problem: RectangleModalProblem, index: ModeIndex, parameter: float,
) -> tuple[ComplexVector, float, int]:
    """複數聯立牛頓，回溯使殘差下降；無法收斂則交給延拓縮步。"""
    state = start.copy()
    for iteration in range(24):
        values, jacobian = _equations(state, problem, index, parameter)
        residual = float(np.max(np.abs(values)))
        if not math.isfinite(residual):
            raise ArithmeticError("非有限方程殘差")
        if residual <= ROOT_RESIDUAL_TOL:
            return state, residual, iteration
        try:
            delta = np.linalg.solve(jacobian, -values)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError("牛頓導數矩陣奇異") from exc
        for backtrack in range(12):
            candidate = state + delta * 2.0**-backtrack
            next_values, _ = _equations(candidate, problem, index, parameter)
            if float(np.max(np.abs(next_values))) < residual:
                state = candidate
                break
        else:
            raise ArithmeticError("牛頓回溯無法減少殘差")
    raise ArithmeticError("牛頓迭代上限仍未收斂")


def _point(
    parameter: float, state: ComplexVector, problem: RectangleModalProblem, index: ModeIndex,
    residual: float, iterations: int,
) -> ContinuationPoint:
    a = np.asarray(problem.lengths_m, dtype=float) / 2
    wave, axes = _wave_and_axes(state, a, index == (0, 0, 0))
    return ContinuationPoint(parameter, wave, _triple(axes), residual, iterations)


def _advance(
    state: ComplexVector, problem: RectangleModalProblem, index: ModeIndex,
    parameter: float, step: float,
) -> tuple[ComplexVector, tuple[ContinuationPoint, ContinuationPoint], int]:
    full, _, _ = _newton(state, problem, index, parameter + step)
    half, residual, iterations = _newton(state, problem, index, parameter + step / 2)
    middle = _point(parameter + step / 2, half, problem, index, residual, iterations)
    end, residual, final_iterations = _newton(half, problem, index, parameter + step)
    scale = np.maximum(1.0, np.abs(state))
    if float(np.max(np.abs(full - end) / scale)) > PATH_AGREEMENT_TOL:
        raise ArithmeticError("整步與兩個半步落在不同根")
    if float(np.max(np.abs(end - state) / scale)) > ROOT_MOVEMENT_TOL:
        raise ArithmeticError("根移動過大，縮步保留指標")
    final = _point(parameter + step, end, problem, index, residual, final_iterations)
    return end, (middle, final), max(iterations, final_iterations)


def _outcome(point: ContinuationPoint, speed: float, cap: float | None) -> ContinuationOutcome:
    if point.parameter != 1.0:
        return ContinuationOutcome.UNRESOLVED
    quantities = modal_quantities(angular_frequency(point.wave_number, speed))
    if quantities.kind is ModalKind.STATIC:
        return ContinuationOutcome.STATIC
    if quantities.kind is ModalKind.NONOSCILLATING_DECAY:
        return ContinuationOutcome.NONOSCILLATING
    if cap is not None and quantities.frequency_hz > cap:
        return ContinuationOutcome.ABOVE_FREQUENCY_LIMIT
    return ContinuationOutcome.TRACKED


def trace_mode(
    problem: RectangleModalProblem, index: ModeIndex, *, max_step: float = 0.05,
    frequency_max_hz: float | None = None,
) -> ModeTrace:
    """以剛性指標逐小步追根；每步都用上一步根作牛頓初值。"""
    if len(index) != 3 or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in index):
        raise ValueError("指標必須為三個非負整數")
    if not math.isfinite(max_step) or not 0 < max_step <= 0.25:
        raise ValueError("max_step 必須介於零與 0.25，不能一步跳到目標")
    _check_cap(frequency_max_hz)
    state = np.asarray([(n * math.pi / 2)**2 for n in index], dtype=np.complex128)
    points = [_point(0.0, state, problem, index, 0.0, 0)]
    if not any(problem.admittances):
        points.append(_point(1.0, state, problem, index, 0.0, 0))
    step = min(max_step, 0.025)
    failure = None
    while points[-1].parameter < 1.0:
        if len(points) > 2 * _MAX_ACCEPTED_STEPS:
            failure = "接受步數達上限，不能宣稱已追到目標"
            break
        parameter = points[-1].parameter
        step = min(step, 1.0 - parameter)
        try:
            end, accepted, iterations = _advance(state, problem, index, parameter, step)
        except (ArithmeticError, OverflowError, ValueError) as exc:
            step /= 2
            if step < _MIN_STEP:
                failure = str(exc)
                break
            continue
        state = end
        points.extend(accepted)
        if iterations <= 5:
            step = min(max_step, step * 1.5)
    outcome = _outcome(points[-1], problem.sound_speed_m_s, frequency_max_hz)
    return ModeTrace(index, problem.sound_speed_m_s, tuple(points), outcome, failure)


def _check_cap(cap: float | None) -> None:
    if cap is not None and (not math.isfinite(cap) or cap <= 0):
        raise ValueError("頻率上限必須為正有限數")


def rigid_indices(problem: RectangleModalProblem, frequency_max_hz: float) -> tuple[ModeIndex, ...]:
    """枚舉剛性解析起點，簡併仍各保留一個指標。"""
    _check_cap(frequency_max_hz)
    # 多列一格再用頻率篩：上限剛好等於某個剛性頻率時，floor 的捨入不能把它丟掉（複查）。
    bounds = [range(math.floor(2 * frequency_max_hz * size / problem.sound_speed_m_s) + 2)
              for size in problem.lengths_m]
    indices = []
    for index in product(*bounds):
        frequency = problem.sound_speed_m_s / 2 * math.sqrt(sum(
            (n / size)**2 for n, size in zip(index, problem.lengths_m, strict=True)))
        if frequency <= frequency_max_hz:
            indices.append((index[0], index[1], index[2]))
    return tuple(indices)


def _relations(modes: tuple[ModeTrace, ...]) -> tuple[BranchRelation, ...]:
    """不用頻率近接去重；區分形狀併根與排序交叉（指標仍保存）。"""
    relations = []
    for left, right in combinations(modes, 2):
        if not left.reached_target or not right.reached_target:
            continue
        qa, qb = np.asarray(left.axis_wave_numbers_squared), np.asarray(right.axis_wave_numbers_squared)
        same_parity = all(a % 2 == b % 2 for a, b in zip(left.index, right.index))
        distance = float(np.max(np.abs(qa - qb) / np.maximum(1.0, np.maximum(abs(qa), abs(qb)))))
        if same_parity and distance <= BRANCH_MATCH_TOL:
            relations.append(BranchRelation((left.index, right.index), "coalesced"))
        before = left.points[0].wave_number.real - right.points[0].wave_number.real
        after = left.wave_number.real - right.wave_number.real
        if before * after < 0 and abs(before) > BRANCH_MATCH_TOL and abs(after) > BRANCH_MATCH_TOL:
            relations.append(BranchRelation((left.index, right.index), "frequency_order_crossing"))
    return tuple(relations)


def trace_modes(
    problem: RectangleModalProblem, *, indices: Iterable[ModeIndex] | None = None,
    frequency_max_hz: float | None = None, seed_frequency_max_hz: float | None = None,
    max_step: float = 0.05,
) -> RectangleTruth:
    """指標或頻率起點清單擇一；跨出界線也保留在 modes 的追蹤帳裡。"""
    _check_cap(frequency_max_hz)
    _check_cap(seed_frequency_max_hz)
    if indices is None:
        seed_cap = seed_frequency_max_hz if seed_frequency_max_hz is not None else frequency_max_hz
        if seed_cap is None:
            raise ValueError("必須提供指標或頻率上限")
        if frequency_max_hz is not None and seed_cap < frequency_max_hz:
            raise ValueError("剛性起點範圍不能小於要求的頻率上限")
        indices = rigid_indices(problem, seed_cap)
    elif seed_frequency_max_hz is not None:
        raise ValueError("指定指標時不再指定剛性起點上限")
    labels = tuple(indices)
    if len(set(labels)) != len(labels):
        raise ValueError("指標不能重複")
    modes = tuple(trace_mode(problem, index, max_step=max_step,
                             frequency_max_hz=frequency_max_hz) for index in labels)
    relations = _relations(modes)
    merged = {index for relation in relations if relation.kind == "coalesced" for index in relation.indices}
    modes = tuple(replace(mode, outcome=ContinuationOutcome.COALESCED) if mode.index in merged else mode
                  for mode in modes)
    return RectangleTruth(modes, relations, frequency_max_hz)
