"""換樣板整段與接續（#561 庚三）：一輪接一輪直到停、每輪整輪算完才寫進 ``templates.jsonl``；接續讀回帳本接著跑。

接在庚二之後：起點是庚二帳本收尾那一列的 θ 與起點配置（:class:`TemplateStart`）。帳本是唯一的接續來源：
:func:`resume_template_rounds` 讀回帳本（標頭跟這一次不一樣就拒收），最後一列決定之前的輪算完成、之後沒寫完的半輪
丟掉（帳本截到那一列）重跑；已收尾就直接交回；決定列沒停止原因、但照設定該停（輪數、總時間、預算）的帳本拒收
（:func:`~aosr.gradient.template_ledger.check_template_rows` 給了設定就核）。非起點的配置一律從帳本列的 ``surfaces``
重建設計與目標，不看別處記的 preset。同一起點重跑逐位相同（秒數除外）；總時間只看帳本記的各輪秒數加總，不看牆鐘。

求值次數上限限的是帳本記的換樣板求值次數（不含庚二、不含接續時丟掉的半輪與重播）；機器實際算的次數可以更多。
給了 ``should_stop`` 就在每次求值之前查一次，為真拋 ``SearchStopped``：一輪的列等整輪算完才寫，所以帳本停在上一列決定。
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np

from aosr.gradient.material_gradient import FloatVector, StopCheck, SurfaceDesign, design_names, is_feasible
from aosr.gradient.template_ledger import (
    TemplateDecision, TemplateHeader, TemplateOutcome, TemplateRow, TemplateStart, append_template_rows,
    create_template_ledger, decision_row, decisions_seconds, quick_row, read_template_ledger, segment_rows,
    truncate_template_rows,
)
from aosr.gradient.template_rounds import (
    TEMPLATE_STOP_MAX_EVALUATIONS, TEMPLATE_STOP_MAX_ROUNDS, TEMPLATE_STOP_MAX_SECONDS, TEMPLATE_STOP_REASONS,
    Surfaces, TemplateContext, TemplatePoint, TemplateRoundResult, TemplateStopReason, designs_for_surfaces,
    next_round_bound, surfaces_of, template_round,
)


class TemplateResult(NamedTuple):
    """整段（從帳本讀回）：最後的配置、變數名、θ（雙精度）、值與兩分項、停止原因、做完幾輪、累計求值次數、總秒數。"""

    surfaces: Surfaces
    variable_names: tuple[str, ...]
    theta: FloatVector
    value: float
    components: tuple[float, float]
    stop_reason: TemplateStopReason
    rounds: int
    evaluations: int
    seconds: float


class _Cursor(NamedTuple):
    """接著跑的位置：這一輪起點那一點（配置、θ 的 hex、值、兩分項）、第幾輪、累計求值次數、累計秒數、下一列的步序。"""

    point: TemplateStart
    round: int
    total: int
    seconds: float
    next_step: int


def finished_result(rows: Sequence[TemplateRow]) -> TemplateResult | None:
    """帳本最後一列是帶停止原因的決定列就是跑完了，從帳本組出結果；否則 None。"""
    if not rows or rows[-1].phase != "template_decision" or rows[-1].stop_reason is None:
        return None
    last = rows[-1]
    if last.decision is None:
        return None
    reason = next(item for item in TEMPLATE_STOP_REASONS if item == last.stop_reason)
    return TemplateResult(last.surfaces, last.variable_names, np.asarray(last.theta(), dtype=np.float64), last.value,
                          last.components, reason, sum(1 for row in rows if row.decision is not None),
                          last.decision.total_evaluations, decisions_seconds(rows))


def _batches(result: TemplateRoundResult, start: Sequence[SurfaceDesign], round_index: int,
             first_step: int) -> list[list[TemplateRow]]:
    """一輪開了之後的列，照寫入的批次切：快速評分一批、基準一批（配置是這一輪起點的 ``start``）、每個候選一批。"""
    if result.baseline is None:
        return []
    batches = [[quick_row(first_step + offset, round_index, score) for offset, score in enumerate(result.quick)]]
    step = first_step + len(batches[0])
    batches.append(list(segment_rows(result.baseline, "baseline_retune", step, round=round_index,
                                     surfaces=surfaces_of(start), variable_names=design_names(start))))
    step += len(batches[-1])
    for item in result.retuned:
        batches.append(list(segment_rows(item.retune, "candidate_retune", step, round=round_index,
                                         surfaces=item.quick.surfaces, variable_names=item.quick.variable_names,
                                         candidate=item.quick.candidate_id)))
        step += len(batches[-1])
    return batches


def _stop(result: TemplateRoundResult, round_index: int, total: int, seconds: float,
          context: TemplateContext) -> tuple[TemplateStopReason | None, int | None]:
    """一輪做完後整段停不停：沒開的輪照它自己的原因；開了的照「本輪結論 → 輪數 → 總時間 → 下一輪預算」的順序。
    總時間是帳本記的各決定列秒數加總（含這一輪），不看牆鐘。"""
    limits = context.settings.templates
    if not result.opened:
        reason = result.stop_reason
        if reason is None:
            raise ValueError("沒開的輪一定有原因")
        return reason, (result.bound if reason == TEMPLATE_STOP_MAX_EVALUATIONS else None)
    if result.stop_reason is not None:
        return result.stop_reason, None
    if round_index + 1 >= limits.max_rounds:
        return TEMPLATE_STOP_MAX_ROUNDS, None
    if seconds > limits.max_seconds:
        return TEMPLATE_STOP_MAX_SECONDS, None
    bound = next_round_bound(surfaces_of(result.point.designs), context.settings)
    if total + bound > limits.max_evaluations:
        return TEMPLATE_STOP_MAX_EVALUATIONS, bound
    return None, bound


def _decision(result: TemplateRoundResult, threshold: float, total: int, bound: int | None) -> TemplateDecision:
    baseline = result.baseline.steps[-1].value if result.baseline is not None else None
    return TemplateDecision(
        baseline_value=baseline, quick_count=len(result.quick),
        outcomes=tuple(TemplateOutcome(candidate=item.quick.candidate_id, quick_value=item.quick.value,
                                       value=item.value, improvement=item.improvement) for item in result.retuned),
        threshold=threshold, accepted=result.accepted.quick.candidate_id if result.accepted is not None else None,
        round_evaluations=result.evaluations, total_evaluations=total, next_round_bound=bound)


def _point_of(start: TemplateStart, context: TemplateContext) -> TemplatePoint:
    """帳本記的那一點（配置、θ 的 hex、值、分項）→ 重建設計；變數名跟 θ 長度要對得上、θ 要可行。"""
    designs = designs_for_surfaces(start.surfaces, context)
    theta = np.asarray([float.fromhex(value) for value in start.theta_hex], dtype=np.float64)
    if len(theta) != len(design_names(designs)) or not is_feasible(theta, designs):
        raise ValueError("帳本記的配置與 θ 對不上重建的設計，或 θ 不可行")
    return TemplatePoint(designs, theta, start.value, start.components)


def _continue(context: TemplateContext, path: Path, header: TemplateHeader, origin: TemplateStart,
              cursor: _Cursor, should_stop: StopCheck | None) -> None:
    point = _point_of(cursor.point, context)
    objective = context.objective_for(point.designs)
    round_index, total, seconds, step = cursor.round, cursor.total, cursor.seconds, cursor.next_step
    settings = context.settings
    while True:
        result = template_round(objective, point, context, budget_left=settings.templates.max_evaluations - total,
                                should_stop=should_stop)
        for batch in _batches(result, point.designs, round_index, step):
            append_template_rows(path, header, batch, settings=settings, start=origin)
            step += len(batch)
        total += result.evaluations
        seconds += result.seconds
        reason, bound = _stop(result, round_index, total, seconds, context)
        chosen = result.point
        row = decision_row(step, round_index, surfaces=surfaces_of(chosen.designs),
                           variable_names=design_names(chosen.designs),
                           theta_hex=tuple(float(value).hex() for value in chosen.theta), value=chosen.value,
                           components=chosen.components, seconds=result.seconds,
                           decision=_decision(result, settings.candidates.improvement_threshold, total, bound),
                           stop_reason=reason)
        append_template_rows(path, header, [row], settings=settings, start=origin)
        if reason is not None:
            return
        point, objective, round_index, step = chosen, result.objective, round_index + 1, step + 1


def _read(path: Path, header: TemplateHeader, context: TemplateContext, origin: TemplateStart) -> TemplateResult:
    _, rows = read_template_ledger(path, expected_header=header, settings=context.settings, start=origin)
    result = finished_result(rows)
    if result is None:
        raise ValueError("換樣板帳本沒有以帶停止原因的決定列結尾")
    return result


def run_template_rounds(context: TemplateContext, path: Path, header: TemplateHeader, origin: TemplateStart, *,
                        should_stop: StopCheck | None = None) -> TemplateResult:
    """開始：帳本獨佔建檔（已有就丟錯）、寫標頭，從 ``origin``（庚二收尾那一點）一輪接一輪跑到停；回從帳本讀回的結果。

    ``header`` 的起點配置要等於 ``origin`` 的配置。停止原因是封閉集合（``TEMPLATE_STOP_REASONS``）。
    """
    if header.initial_surfaces != origin.surfaces:
        raise ValueError("換樣板帳本標頭的起點配置跟起點不同")
    create_template_ledger(path, header)
    _continue(context, path, header, origin, _Cursor(origin, 0, 0, 0.0, 0), should_stop)
    return _read(path, header, context, origin)


def resume_template_rounds(context: TemplateContext, path: Path, header: TemplateHeader, origin: TemplateStart, *,
                           should_stop: StopCheck | None = None) -> TemplateResult:
    """接續：讀回帳本（標頭、設定、起點都要對），已收尾直接交回；否則截到最後一列決定（沒有決定就截到標頭）接著跑。

    累計求值次數與秒數接帳本記的（丟掉的半輪不算）；決定列沒停止原因、照設定卻該停的帳本在讀回時就拒收。
    """
    _, rows = read_template_ledger(path, expected_header=header, settings=context.settings, start=origin)
    finished = finished_result(rows)
    if finished is not None:
        return finished
    decisions = [index for index, row in enumerate(rows) if row.decision is not None]
    if not decisions:
        cursor = _Cursor(origin, 0, 0, 0.0, 0)
    else:
        last = rows[decisions[-1]]
        if last.decision is None:
            raise ValueError("決定列缺內容")
        cursor = _Cursor(TemplateStart(last.surfaces, last.theta_hex, last.value, last.components), last.round + 1,
                         last.decision.total_evaluations, decisions_seconds(rows[:decisions[-1] + 1]),
                         decisions[-1] + 1)
    truncate_template_rows(path, header, cursor.next_step)
    _continue(context, path, header, origin, cursor, should_stop)
    return _read(path, header, context, origin)
