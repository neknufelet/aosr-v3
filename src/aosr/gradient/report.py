"""牆面材料梯度搜尋的人看報告（#561 辛一）：只讀資料夾裡的 ``project.json``、``status.json``、帳本，不求值、不寫檔。

每一段（下降幾步、取規格前後值、重調幾步）、每一輪的決定、累計求值次數、停止原因、最後的可行點（每面 preset、
厚度 mm、空腔 mm）、值與分項；最後一行固定是終點完整模型重算還沒做（辛二）。帳本照 ``read_ledger`` 逐列驗，
壞列照樣報錯。最後可行點：有決定列取最後一列決定，否則取前段收好的重調末列，再否則取規格那一列。
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from aosr.gradient.material_gradient_ledger import MaterialGradientHeader, MaterialGradientRow, read_ledger
from aosr.gradient.run import (
    LEDGER_FILE, STATUS_FILE, STOP_FILE, GradientStatus, ledger_progress, read_project, surface_specs,
)

FULL_MODEL_LINE: Final = "終點完整模型重算：未做（辛二）"
_MM: Final = 1000.0


def _segment(rows: Sequence[MaterialGradientRow], phase: str) -> list[MaterialGradientRow]:
    return [row for row in rows if row.phase == phase and row.round is None]


def _segment_lines(label: str, segment: Sequence[MaterialGradientRow]) -> list[str]:
    if not segment:
        return [f"{label}：還沒寫進帳本"]
    last = segment[-1]
    closing = (f"，停止原因「{last.stop_reason}」、這段求值 {last.evaluations} 次" if last.stop_reason is not None
               else "（沒寫完）")
    return [f"{label}：{len(segment) - 1} 步，值 {segment[0].value:.6f} → {last.value:.6f}{closing}"]


def _pre_loop_lines(rows: Sequence[MaterialGradientRow]) -> list[str]:
    lines = ["前段："]
    descent, snaps, retune = _segment(rows, "descent"), _segment(rows, "snap"), _segment(rows, "retune")
    lines += ["  " + line for line in _segment_lines("連續下降", descent)]
    if snaps:
        before = f"{descent[-1].value:.6f}" if descent else "？"
        blocked = [choice.surface_id for choice in snaps[0].snap or () if choice.cavity_clipped or choice.thickness_retreated]
        lines.append(f"  取規格：值 {before} → {snaps[0].value:.6f}"
                     f"（被總深度擋過的面：{'、'.join(blocked) if blocked else '沒有'}）")
    else:
        lines.append("  取規格：還沒寫進帳本")
    lines += ["  " + line for line in _segment_lines("空腔重調", retune)]
    return lines


def _round_lines(rows: Sequence[MaterialGradientRow]) -> list[str]:
    lines = []
    for row in rows:
        decision = row.decision
        if decision is None:
            continue
        lines.append(f"第 {row.round} 輪：交出的值 {row.value:.6f}；這輪求值 {decision.round_evaluations} 次、"
                     f"累計 {decision.total_evaluations} 次")
        if decision.baseline_value is not None:
            lines.append(f"  基準重調後 {decision.baseline_value:.6f}（門檻 {decision.threshold}）")
        for outcome in decision.outcomes:
            lines.append(f"  候選 {outcome.candidate}：預測 {outcome.predicted_change:+.6f}、重調後 {outcome.value:.6f}、"
                         f"改善 {outcome.improvement:+.6f}")
        accepted = f"接受 {decision.accepted}" if decision.accepted is not None else "沒有接受"
        reason = f"；停止原因「{row.stop_reason}」" if row.stop_reason is not None else ""
        lines.append(f"  決定：{accepted}{reason}")
    pending = rows[-1] if rows and rows[-1].round is not None and rows[-1].decision is None else None
    if pending is not None:
        lines.append(f"第 {pending.round} 輪：進行中（還沒有決定列，接續時這半輪會丟掉重跑）")
    return lines


def _feasible(rows: Sequence[MaterialGradientRow]) -> MaterialGradientRow | None:
    decisions = [row for row in rows if row.decision is not None]
    if decisions:
        return decisions[-1]
    closed = [row for row in _segment(rows, "retune") if row.stop_reason is not None]
    if closed:
        return closed[-1]
    snaps = _segment(rows, "snap")
    return snaps[-1] if snaps else None


def _point_lines(header: MaterialGradientHeader, row: MaterialGradientRow | None) -> list[str]:
    if row is None:
        return ["最後的可行點：還沒有（取規格還沒寫進帳本）"]
    lines = [f"最後的可行點（帳本第 {row.step} 列 {row.phase}）："]
    for spec in surface_specs(header, row.theta()):
        if spec.preset_id is None:
            lines.append(f"  {spec.surface_id}：剛性面")
            continue
        cavity = "沒有空腔層" if spec.cavity_m is None else f"空腔 {spec.cavity_m * _MM:.3f} mm"
        thickness = f"{spec.thickness_m * _MM:.3f}" if spec.thickness_m is not None else "？"
        lines.append(f"  {spec.surface_id}：{spec.preset_id}，棉板 {thickness} mm，{cavity}")
    lines.append(f"  值 {row.value:.6f}；分項：主位起伏 {row.components[0]:.6f} dB、座位穩健性 {row.components[1]:.6f} dB")
    return lines


def _status_line(folder: Path) -> str:
    try:
        status = GradientStatus.model_validate_json((folder / STATUS_FILE).read_bytes())
    except (OSError, ValidationError) as error:
        return f"狀態：讀不到狀態檔（{type(error).__name__}）"
    note = f"；{status.message}" if status.message else ""
    return f"狀態：{status.state}（子行程 {status.pid}，這次開始 {status.started_at}、最後寫 {status.updated_at}{note}）"


def render_report(folder: Path) -> str:
    """整份報告文字，最後一行固定是 :data:`FULL_MODEL_LINE`。"""
    project = read_project(folder)
    lines = [f"梯度搜尋 {project.gradient_id}（用途 {project.purpose}，開始 {project.started_at}）", _status_line(folder)]
    if (folder / STOP_FILE).exists():
        lines.append("停止記號：在（下一次求值之前會停；resume 會先清掉它）")
    ledger = folder / LEDGER_FILE
    if not ledger.exists():
        lines.append("帳本：還沒有")
        return "\n".join([*lines, FULL_MODEL_LINE]) + "\n"
    header, rows = read_ledger(ledger)
    evaluations, rounds, _, reason = ledger_progress(rows)
    lines += _pre_loop_lines(rows)
    lines += _round_lines(rows)
    lines.append(f"累計求值次數（帳本記的、不含前段）：{evaluations}；做完 {rounds} 輪")
    lines.append(f"停止原因：{reason}" if reason is not None else "停止原因：還沒收尾")
    lines += _point_lines(header, _feasible(rows))
    lines.append(FULL_MODEL_LINE)
    return "\n".join(lines) + "\n"
