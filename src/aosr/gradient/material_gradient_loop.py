"""第三階段牆面材料梯度搜尋的後半段（#561 庚二）：在可行解上的候選確認迴圈、停止原因與接續。

找法照決策紙 ``stage-three-gradient-objective-low-frequency-lane.md``「決定」第 3 條：在可行解重算梯度，
用「梯度 × 實際規格跳距（帶方向）」估每面換規格的變化、提出少量候選；入選候選從目前解出發、所有面的空腔
做有限步數的梯度重調、再用整房目標比，基準方案也用同樣步數重調，改善超過門檻才接受；在設定的候選數與
計算預算內沒有找到超過門檻的可行替換就停；候選數、輪數、求值次數都有上限，停下時記原因。求值次數上限
限的是帳本記的求值次數（不含前段、不含接續時丟掉的半輪與重播）；機器實際算的次數可以更多。

候選的**產生**（:func:`candidates_from_gradient`）跟候選的**確認、停止與帳本**（:func:`confirmation_round`、
:func:`run_rounds`）分開：今天候選只有「同一面、同一家族、換到相鄰的合法厚度」一種，不換樣板（紙第 4 條）。
本模組只執行替身低頻目標的候選比較；尚未實作決策紙第 2 條要求的候選確認與終點大峰檢查，待完整模型確認階段補齊。

純函式，沒有命令列、沒有行程、不印東西；同一起點重跑逐位相同（秒數除外）。帳本是唯一的接續來源：
:func:`resume_rounds` 讀回帳本、丟掉最後一列決定之後的半輪、從那裡重跑，跑完的帳本跟一口氣跑完的逐值相同。
"""
from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from aosr.config.material_gradient import DescentSettings, MaterialGradientSettings
from aosr.gradient.material_gradient import (
    LOOP_STOP_MAX_EVALUATIONS, LOOP_STOP_MAX_ROUNDS, LOOP_STOP_NO_CANDIDATES, LOOP_STOP_NO_IMPROVEMENT,
    ChainObjective, DescentResult, FloatVector, LoopStopReason, SurfaceDesign, checked_theta, depth_room,
    compiled_evaluator, is_feasible, retune_cavities,
)
from aosr.gradient.material_gradient_ledger import (
    CandidateJump, CandidateOutcome, MaterialGradientHeader, MaterialGradientRow, RoundDecision, append_rows,
    decision_row, descent_rows, jump_row, read_ledger, truncate_rows,
)


Direction = Literal["thinner", "thicker"]
_DIRECTIONS: tuple[tuple[Direction, int], ...] = (("thinner", -1), ("thicker", 1))


class Candidate(NamedTuple):
    """一個候選：哪一面、往薄或往厚、跳到的厚度與空腔（沒有空腔層是 None）、整個可行位移 Δθ、預測變化（梯度 · Δθ）。"""

    surface_id: str
    direction: Direction
    thickness_m: float
    cavity_m: float | None
    delta: tuple[float, ...]
    predicted_change: float

    @property
    def candidate_id(self) -> str:
        """帳本用的代號，例如 ``x0:thicker``。"""
        return f"{self.surface_id}:{self.direction}"


class ConfirmedCandidate(NamedTuple):
    """確認過的候選：跳點 θ、它的空腔重調、重調後的值、改善（基準重調後的值減它）。"""

    candidate: Candidate
    jump_theta: FloatVector
    retune: DescentResult
    value: float
    improvement: float


class RoundResult(NamedTuple):
    """一輪的結果。``opened`` 為假時預算不夠、這一輪沒開（沒有求值，θ 原樣，值與分項是 None）。

    ``stop_reason`` 是這一輪自己的結論：「求值到上限」（沒開）、「沒有候選」、「已確認候選內沒有超過門檻」，
    接受了某個候選時是 None。``theta`` 是這一輪交出去的 θ：接受了就是那個候選重調後的、沒接受就是重調後的基準、
    沒有候選或沒開就是原樣。``bound`` 是開這一輪前算的最壞求值次數。
    """

    opened: bool
    bound: int
    start_value: float | None
    candidates: tuple[Candidate, ...]
    baseline: DescentResult | None
    confirmed: tuple[ConfirmedCandidate, ...]
    accepted: Candidate | None
    theta: FloatVector
    value: float | None
    components: tuple[float, float] | None
    evaluations: int
    seconds: float
    stop_reason: LoopStopReason | None


class RoundSummary(NamedTuple):
    """帳本裡一輪的摘要：輪號、交出去的值、基準重調後的值、每個候選（代號, 重調後的值, 改善）、接受了誰、
    這一輪與累計求值次數、下一輪最壞要幾次、停止原因。"""

    round: int
    value: float
    baseline_value: float | None
    outcomes: tuple[tuple[str, float, float], ...]
    accepted: str | None
    evaluations: int
    total_evaluations: int
    next_round_bound: int | None
    stop_reason: str | None


class LoopResult(NamedTuple):
    """整場：最終 θ（雙精度，從帳本讀回）、每輪摘要、停止原因、總求值次數。"""

    theta: FloatVector
    rounds: tuple[RoundSummary, ...]
    stop_reason: LoopStopReason
    evaluations: int


# ---- 候選的產生 ----

def candidates_from_gradient(theta_feasible: jax.Array | FloatVector, gradient: jax.Array | FloatVector,
                             designs: Sequence[SurfaceDesign], settings: MaterialGradientSettings,
                             ) -> tuple[Candidate, ...]:
    """在可行 θ 用「梯度 · 整個可行位移」估每面換到相鄰合法厚度的變化，取最負的前 ``max_per_round`` 個。

    兩個以上合法厚度的面往薄一格、往厚一格各一個（到底、到頂那一邊沒有）；只有一個合法厚度的面沒有候選。
    位移是整個可行位移：厚度跳到相鄰合法值，空腔照取規格同一條規則切到「上限減新厚度」（切到下界仍放不下就不是候選）。
    排序：預測變化由小到大、同值照面序、再照薄先厚後。在 θ 的精度下算。
    """
    values = checked_theta(theta_feasible, designs)
    if not is_feasible(values, designs):
        raise ValueError("候選要從取過規格的可行 θ 產生")
    slope = np.asarray(gradient)
    if slope.shape != values.shape or not np.all(np.isfinite(slope)):
        raise ValueError(f"梯度必須是有限、形狀 {values.shape} 的陣列")
    slope = slope.astype(values.dtype)
    dtype = values.dtype.type
    found: list[tuple[float, int, int, Candidate]] = []
    offset = 0
    for surface_index, design in enumerate(designs):
        if design.material is None or design.max_total_depth_m is None:
            continue
        legal = [dtype(value) for value in design.legal_thicknesses_m]
        has_cavity = design.cavity_variable is not None
        current = legal.index(values[offset])
        for order, (direction, shift) in enumerate(_DIRECTIONS):
            if not 0 <= current + shift < len(legal):
                continue
            thickness = legal[current + shift]
            room = depth_room(dtype(design.max_total_depth_m), thickness)
            cavity_low = dtype(design.lower[1]) if has_cavity else dtype(0.0)
            if room < cavity_low:
                continue
            delta = np.zeros_like(values)
            delta[offset] = thickness - values[offset]
            cavity: float | None = None
            if has_cavity:
                new_cavity = min(values[offset + 1], room)
                delta[offset + 1] = new_cavity - values[offset + 1]
                cavity = float(new_cavity)
            predicted = float(np.dot(slope, delta))
            found.append((predicted, surface_index, order, Candidate(
                design.surface_id, direction, float(thickness), cavity, tuple(float(x) for x in delta), predicted)))
        offset += len(design.variables)
    found.sort(key=lambda item: item[:3])
    return tuple(item[3] for item in found[:settings.candidates.max_per_round])


def apply_candidate(theta_feasible: jax.Array | FloatVector, candidate: Candidate,
                    designs: Sequence[SurfaceDesign]) -> FloatVector:
    """跳到候選點：那一面的厚度與空腔直接換成候選記的值（不做 θ＋Δθ 的浮點加法），回可行 θ（精度不變）。"""
    values = checked_theta(theta_feasible, designs).copy()
    dtype = values.dtype.type
    offset = 0
    for design in designs:
        if design.surface_id == candidate.surface_id and design.material is not None:
            values[offset] = dtype(candidate.thickness_m)
            if design.cavity_variable is not None and candidate.cavity_m is not None:
                values[offset + 1] = dtype(candidate.cavity_m)
            if not is_feasible(values, designs):
                raise ValueError(f"候選 {candidate.candidate_id} 跳過去不可行")
            return values
        offset += len(design.variables)
    raise ValueError(f"候選的面 {candidate.surface_id} 不在設計清單裡")


# ---- 求值預算 ----

def line_search_trials(settings: DescentSettings) -> int:
    """一步線搜尋最多試幾次：從 ``initial_step`` 起每次乘 ``backtrack_ratio``、不小於 ``min_step`` 的試探個數。"""
    trials, trial = 0, settings.initial_step
    while trial >= settings.min_step:
        trials += 1
        trial *= settings.backtrack_ratio
    return trials


def round_evaluation_bound(settings: MaterialGradientSettings, candidate_count: int) -> int:
    """一輪最壞要求幾次目標：梯度一次；基準與每個候選各一段重調，每段起點一次＋最多 K 步 × 每步最多試探數。

    沒有候選的輪只有梯度那一次。預算照這個最壞值判：開得了的輪一定跑得完，帳本記的求值次數不會超過上限——
    限的是帳本記的求值次數（不含前段、不含接續時丟掉的半輪與重播）；機器實際算的次數可以更多。
    """
    if candidate_count < 0:
        raise ValueError("候選數不可為負")
    if candidate_count == 0:
        return 1
    per_retune = 1 + settings.retune.steps * line_search_trials(settings.descent)
    return 1 + (candidate_count + 1) * per_retune


def _candidate_count(theta: FloatVector, designs: Sequence[SurfaceDesign], settings: MaterialGradientSettings) -> int:
    """這個 θ 上一輪會確認幾個候選；候選有沒有只看厚度位置與總深度，跟梯度無關，所以用零梯度數。"""
    return len(candidates_from_gradient(theta, np.zeros_like(theta), designs, settings))


# ---- 一輪 ----

def confirmation_round(objective: ChainObjective, theta_feasible: jax.Array | FloatVector,
                       designs: Sequence[SurfaceDesign], settings: MaterialGradientSettings, *,
                       budget_left: int) -> RoundResult:
    """在目前可行 θ 算梯度（一次求值）、產候選，基準與每個候選各自從自己的起點做最多 K 步空腔重調再比。

    基準＝目前 θ 原樣重調（不問前面重調過沒有）；候選＝跳到候選點再重調；改善＝基準重調後的值減候選重調後的值。
    最好的候選改善「超過」門檻才接受、交出它重調後的 θ；都沒超過就交出重調後的基準。沒有候選時不重調、θ 原樣。
    最壞求值次數（:func:`round_evaluation_bound`）大於 ``budget_left`` 就不開這一輪、不求值，記「求值到上限」。
    """
    theta = checked_theta(theta_feasible, designs)
    if not is_feasible(theta, designs):
        raise ValueError("候選確認要從取過規格的可行 θ 開始")
    if isinstance(budget_left, bool) or budget_left < 0:
        raise ValueError("剩下的求值次數必須是非負整數")
    bound = round_evaluation_bound(settings, _candidate_count(theta, designs, settings))
    if bound > budget_left:
        return RoundResult(False, bound, None, (), None, (), None, theta, None, None, 0, 0.0, LOOP_STOP_MAX_EVALUATIONS)
    clock = time.perf_counter()
    (value, aux), gradient = compiled_evaluator(objective)(jnp.asarray(theta))
    start_value = float(value)
    candidates = candidates_from_gradient(theta, np.asarray(gradient), designs, settings)
    if not candidates:
        components = (float(aux.mean_own_line_ripple_rms_db), float(aux.mean_seat_robustness_rms_db))
        return RoundResult(True, bound, start_value, (), None, (), None, theta, start_value, components, 1,
                           time.perf_counter() - clock, LOOP_STOP_NO_CANDIDATES)
    baseline = retune_cavities(objective, theta, designs, settings)
    base_value = baseline.steps[-1].value
    confirmed = []
    for candidate in candidates:
        jumped = apply_candidate(theta, candidate, designs)
        retuned = retune_cavities(objective, jumped, designs, settings)
        confirmed.append(ConfirmedCandidate(candidate, jumped, retuned, retuned.steps[-1].value,
                                            base_value - retuned.steps[-1].value))
    best = max(confirmed, key=lambda item: item.improvement)
    accepted = best if best.improvement > settings.candidates.improvement_threshold else None
    chosen = accepted.retune if accepted is not None else baseline
    evaluations = 1 + baseline.evaluations + sum(item.retune.evaluations for item in confirmed)
    return RoundResult(True, bound, start_value, candidates, baseline, tuple(confirmed),
                       accepted.candidate if accepted is not None else None, chosen.theta, chosen.steps[-1].value,
                       chosen.steps[-1].components, evaluations, time.perf_counter() - clock,
                       None if accepted is not None else LOOP_STOP_NO_IMPROVEMENT)


# ---- 整場與接續 ----

class _LoopStart(NamedTuple):
    theta: FloatVector
    value: float
    components: tuple[float, float]
    round: int
    total: int
    next_step: int


def _jump_model(item: ConfirmedCandidate) -> CandidateJump:
    candidate = item.candidate
    return CandidateJump(direction=candidate.direction, thickness_m=candidate.thickness_m, cavity_m=candidate.cavity_m,
                         delta_hex=tuple(value.hex() for value in candidate.delta),
                         predicted_change=candidate.predicted_change)


def _segment_rows(result: RoundResult, round_index: int, first_step: int) -> list[list[MaterialGradientRow]]:
    """一輪開了之後的列，照寫入的批次切：基準一批、每個候選（跳點＋重調）一批。"""
    if result.baseline is None:
        return []
    batches = [list(descent_rows(result.baseline, "baseline_retune", first_step, round=round_index))]
    step = first_step + len(batches[0])
    for item in result.confirmed:
        start = item.retune.steps[0]
        batch = [jump_row(step, round_index, item.candidate.candidate_id, tuple(float(x) for x in item.jump_theta),
                          start.value, start.components, _jump_model(item)),
                 *descent_rows(item.retune, "candidate_retune", step + 1, round=round_index,
                               candidate=item.candidate.candidate_id)]
        batches.append(batch)
        step += len(batch)
    return batches


def _loop_stop(result: RoundResult, rounds_done: int, total: int, designs: Sequence[SurfaceDesign],
               settings: MaterialGradientSettings) -> tuple[LoopStopReason | None, int | None]:
    """一輪做完後整場停不停：這一輪自己的結論優先，再看輪數，最後看下一輪最壞要的次數放不放得進剩下的預算。"""
    if result.stop_reason is not None:
        return result.stop_reason, None
    if rounds_done >= settings.candidates.max_rounds:
        return LOOP_STOP_MAX_ROUNDS, None
    bound = round_evaluation_bound(settings, _candidate_count(result.theta, designs, settings))
    if total + bound > settings.candidates.max_evaluations:
        return LOOP_STOP_MAX_EVALUATIONS, bound
    return None, bound


def _continue_rounds(objective: ChainObjective, start: _LoopStart, designs: Sequence[SurfaceDesign],
                     settings: MaterialGradientSettings, ledger_path: Path, header: MaterialGradientHeader) -> None:
    theta, value, components = start.theta, start.value, start.components
    round_index, total, step = start.round, start.total, start.next_step
    while True:
        result = confirmation_round(objective, theta, designs, settings,
                                    budget_left=settings.candidates.max_evaluations - total)
        for batch in _segment_rows(result, round_index, step):
            append_rows(ledger_path, header, batch)
            step += len(batch)
        total += result.evaluations
        if result.opened:
            reason, bound = _loop_stop(result, round_index + 1, total, designs, settings)
            theta = result.theta
            if result.baseline is not None and result.value is not None and result.components is not None:
                # 有重調才換值；沒有候選的輪 θ 原樣，值與分項沿用起點那一列（帳本核的就是這件事）。
                value, components = result.value, result.components
        else:
            reason, bound = LOOP_STOP_MAX_EVALUATIONS, result.bound
        decision = RoundDecision(
            baseline_value=result.baseline.steps[-1].value if result.baseline is not None else None,
            outcomes=tuple(CandidateOutcome(candidate=item.candidate.candidate_id,
                                            predicted_change=item.candidate.predicted_change,
                                            value=item.value, improvement=item.improvement)
                           for item in result.confirmed),
            threshold=settings.candidates.improvement_threshold,
            accepted=result.accepted.candidate_id if result.accepted is not None else None,
            round_evaluations=result.evaluations, total_evaluations=total, next_round_bound=bound)
        append_rows(ledger_path, header, [decision_row(step, round_index, tuple(float(x) for x in theta), value,
                                                       components, result.seconds, decision, reason)])
        step += 1
        round_index += 1
        if reason is not None:
            return


def _summaries(rows: Sequence[MaterialGradientRow]) -> tuple[RoundSummary, ...]:
    return tuple(RoundSummary(
        int(row.round or 0), row.value, row.decision.baseline_value,
        tuple((item.candidate, item.value, item.improvement) for item in row.decision.outcomes),
        row.decision.accepted, row.decision.round_evaluations, row.decision.total_evaluations,
        row.decision.next_round_bound, row.stop_reason)
        for row in rows if row.decision is not None)


def _finished(rows: Sequence[MaterialGradientRow]) -> LoopResult | None:
    """帳本最後一列是帶停止原因的決定列就是跑完了，從帳本組出結果；否則 None。"""
    if not rows or rows[-1].phase != "round_decision" or rows[-1].stop_reason is None:
        return None
    last = rows[-1]
    if last.decision is None:
        return None
    reason = next(item for item in (LOOP_STOP_NO_IMPROVEMENT, LOOP_STOP_NO_CANDIDATES, LOOP_STOP_MAX_ROUNDS,
                                    LOOP_STOP_MAX_EVALUATIONS) if item == last.stop_reason)
    return LoopResult(np.asarray(last.theta(), dtype=np.float64), _summaries(rows), reason,
                      last.decision.total_evaluations)


def _read_result(ledger_path: Path, header: MaterialGradientHeader) -> LoopResult:
    _, rows = read_ledger(ledger_path, expected_header=header)
    result = _finished(rows)
    if result is None:
        raise ValueError("帳本沒有以帶停止原因的決定列結尾")
    return result


def _pre_loop_start(rows: Sequence[MaterialGradientRow], designs: Sequence[SurfaceDesign]) -> _LoopStart:
    """還沒有任何決定列：從前段（下降、取規格、重調）最後一列起跑；那一列要可行、要是收好的一段。"""
    pre_loop = [row for row in rows if row.round is None]
    if not pre_loop:
        raise ValueError("帳本沒有前段（取規格或重調）的列，候選確認迴圈不知道從哪個 θ 起跑")
    last = pre_loop[-1]
    if last.phase == "descent" or (last.phase == "retune" and last.stop_reason is None):
        raise ValueError(f"帳本前段最後一列是 {last.phase}、不是取完規格或重調完的可行點，不能起跑")
    theta = np.asarray(last.theta(), dtype=np.float64)
    if not is_feasible(theta, designs):
        raise ValueError("帳本前段最後一列的 θ 不可行")
    return _LoopStart(theta, last.value, last.components, 0, 0, len(pre_loop))


def run_rounds(objective: ChainObjective, theta_feasible: jax.Array | FloatVector, designs: Sequence[SurfaceDesign],
               settings: MaterialGradientSettings, ledger_path: Path, header: MaterialGradientHeader) -> LoopResult:
    """一輪接一輪直到停，每一段下降與每一輪的決定都接在帳本後面寫（每批 flush 與 fsync）。

    ``ledger_path`` 要已經寫好前段（庚一的連續下降、取規格、重調），最後一列的 θ 要逐位等於 ``theta_feasible``——
    帳本是接續的唯一來源，起點要在帳本裡。停止原因是封閉集合（``LOOP_STOP_REASONS``）。回的結果從帳本讀回組成。
    求值次數上限限的是帳本記的求值次數（不含前段、不含接續時丟掉的半輪與重播）；機器實際算的次數可以更多。
    """
    theta = checked_theta(theta_feasible, designs)
    _, rows = read_ledger(ledger_path, expected_header=header)
    if any(row.round is not None for row in rows):
        raise ValueError("帳本已經有候選確認的列，要接續請用 resume_rounds")
    start = _pre_loop_start(rows, designs)
    if [value.hex() for value in np.asarray(theta, dtype=np.float64)] != list(rows[-1].theta_hex):
        raise ValueError("起點 θ 要逐位等於帳本前段最後一列的 θ")
    _continue_rounds(objective, start._replace(theta=theta), designs, settings, ledger_path, header)
    return _read_result(ledger_path, header)


def _check_should_continue(decision: RoundDecision, start: _LoopStart, designs: Sequence[SurfaceDesign],
                           settings: MaterialGradientSettings) -> None:
    """最後一列決定沒有停止原因時，照設定重算一次「該不該停」：輪數到上限、或下一輪最壞值放不進剩下的預算、
    或記的下一輪最壞值跟重算的不同，都是帳本跟設定矛盾，拒收、不開輪（#561 審查 B2）。"""
    limits = settings.candidates
    if start.round >= limits.max_rounds:
        raise ValueError(f"帳本最後一列決定（第 {start.round - 1} 輪）沒有停止原因，但照設定已到輪數上限 "
                         f"{limits.max_rounds}（應記「{LOOP_STOP_MAX_ROUNDS}」）；帳本與設定矛盾，不接續")
    bound = round_evaluation_bound(settings, _candidate_count(start.theta, designs, settings))
    if start.total + bound > limits.max_evaluations:
        raise ValueError(f"帳本最後一列決定沒有停止原因，但累計 {start.total} 次加下一輪最壞 {bound} 次超過求值上限 "
                         f"{limits.max_evaluations}（應記「{LOOP_STOP_MAX_EVALUATIONS}」）；帳本與設定矛盾，不接續")
    if decision.next_round_bound != bound:
        raise ValueError(f"帳本記的下一輪最壞求值次數 {decision.next_round_bound} 跟照設定重算的 {bound} 不同，不接續")


def resume_rounds(objective: ChainObjective, designs: Sequence[SurfaceDesign], settings: MaterialGradientSettings,
                  ledger_path: Path, header: MaterialGradientHeader) -> LoopResult:
    """讀回帳本接著跑：標頭（替身、面、變數、設定指紋）跟這一次不一樣就拒收。

    最後一列決定之前的輪算完成，之後沒寫完的半輪丟掉（帳本截到那一列）重跑；沒有任何決定列就從前段最後一列起跑。
    已經以停止原因收尾的帳本不再求值，直接交回。θ 從帳本讀回是雙精度；累計求值次數接帳本記的（丟掉的半輪不算）。
    最後一列決定沒有停止原因、但照設定該停（輪數或預算）的帳本拒收，不開輪。
    """
    _, rows = read_ledger(ledger_path, expected_header=header)
    finished = _finished(rows)
    if finished is not None:
        return finished
    decisions = [index for index, row in enumerate(rows) if row.phase == "round_decision"]
    if not decisions:
        start = _pre_loop_start(rows, designs)
        truncate_rows(ledger_path, header, start.next_step)
    else:
        last = rows[decisions[-1]]
        if last.decision is None or last.round is None:
            raise ValueError("決定列缺內容")
        start = _LoopStart(np.asarray(last.theta(), dtype=np.float64), last.value, last.components, last.round + 1,
                           last.decision.total_evaluations, decisions[-1] + 1)
        _check_should_continue(last.decision, start, designs, settings)
        truncate_rows(ledger_path, header, decisions[-1] + 1)
    _continue_rounds(objective, start, designs, settings, ledger_path, header)
    return _read_result(ledger_path, header)
