"""牆面材料梯度搜尋的人看報告（#561 辛一、辛三）：只讀資料夾裡的 ``project.json``、``status.json``、帳本、方案檔、
``result.json`` 與 ``endpoint.json``，不求值、不寫檔。

每一段（下降幾步、取規格前後值、重調幾步）、每一輪的決定、累計求值次數、停止原因、最後的可行點（每面 preset、
厚度 mm、空腔 mm；不貼的面印方案原本那面牆的材料）、值與分項。帳本照 ``read_ledger`` 逐列驗，壞列照樣報錯。
最後可行點：有決定列取最後一列決定，否則取前段收好的重調末列，再否則取規格那一列。
不貼的面讀 ``project.json`` 記的方案檔（讀，不核雜湊；方案檔事後換過，這裡印的是現在那一份）。
最後一節是終點：``endpoint.json`` 綁的搜尋身分、帳本檔與結果檔摘要都等於現在的就印終點數字，任一件對不上印過期，
沒有 ``endpoint.json`` 最後一行是 :data:`FULL_MODEL_LINE`。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from aosr.geometry.shoebox import Wall
from aosr.gradient.endpoint_run import ENDPOINT_FILE, EndpointRecord, read_endpoint, read_result
from aosr.gradient.material_gradient_ledger import MaterialGradientHeader, MaterialGradientRow, read_ledger
from aosr.gradient.run import (
    LEDGER_FILE, RESULT_FILE, STATUS_FILE, STOP_FILE, GradientProject, GradientStatus, file_sha256, ledger_progress,
    project_wall_materials, read_project, surface_specs,
)
from aosr.materials.construction_models import Assembly, ConstantImpedance, SurfaceMaterial
from aosr.reporting.scheme import load_scheme

FULL_MODEL_LINE: Final = "終點完整模型重算：未做（跑 endpoint <資料夾>）"
STALE_ENDPOINT_LINE: Final = "終點報告過期：endpoint.json 跟現在的帳本、結果檔或搜尋身分對不上，要重跑 endpoint"
ENDPOINT_HEADING: Final = "終點完整模型重算（"
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


def material_text(material: SurfaceMaterial) -> str:
    """原牆材料的一句話：固定阻抗印數值（Pa·s/m），構造與實測阻抗表印種類。"""
    if isinstance(material, ConstantImpedance):
        return f"固定阻抗 {material.impedance_pa_s_m:.3f} Pa·s/m"
    if isinstance(material, Assembly):
        return f"構造（房間側起 {'、'.join(layer.kind for layer in material.layers)}；背後 {material.backing}）"
    return f"實測阻抗表（種類 {material.kind}）"


def _kept_texts(project: GradientProject) -> dict[str, str]:
    """不貼的面那一句：方案檔那一面原本的牆；方案檔讀不到時每面都說讀不到（報告照印，不中斷）。"""
    try:
        materials = project_wall_materials(project, load_scheme(Path(project.scheme_path)))
    except (OSError, ValueError) as error:
        return {wall.wall_name(): f"不貼（保留原牆；方案檔讀不到：{type(error).__name__}）" for wall in Wall.all()}
    return {wall.wall_name(): f"不貼（保留原牆：{material_text(material)}）" for wall, material in materials.items()}


def _point_lines(header: MaterialGradientHeader, row: MaterialGradientRow | None,
                 kept: dict[str, str]) -> list[str]:
    if row is None:
        return ["最後的可行點：還沒有（取規格還沒寫進帳本）"]
    lines = [f"最後的可行點（帳本第 {row.step} 列 {row.phase}）："]
    for spec in surface_specs(header, row.theta()):
        if spec.preset_id is None:
            lines.append(f"  {spec.surface_id}：{kept.get(spec.surface_id, '不貼（保留原牆；方案沒有這一面）')}")
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


def _dbs(values: Mapping[str, float | None]) -> str:
    return "、".join(f"{name} {'量不到' if value is None else f'{value:.3f} dB'}" for name, value in values.items())


def _peak(value: float | None) -> str:
    return "量不到" if value is None else f"{value:.3f} dB"


def _endpoint_section(record: EndpointRecord) -> list[str]:
    report, baseline, gain, seconds = record.report, record.baseline, record.improvement, record.seconds
    dropped = "、".join(report.dropped_cavities) if report.dropped_cavities else "沒有"
    return [
        f"{ENDPOINT_HEADING}{record.finished_at}，引擎 {record.engine_commit or '沒記'}）：",
        f"  替身值 {report.reduced_total:.6f}；完整模型值 {report.full_total:.6f}"
        f"（主位起伏 {report.full_components[0]:.6f} dB、座位穩健性 {report.full_components[1]:.6f} dB）",
        f"  差（完整減替身）{report.total_difference:+.6f}；上界 {report.total_difference_bound:.6f}",
        f"  每座位最大逐頻差（替身對完整模型）：{_dbs(report.frequency_db_by_seat)}；"
        f"全部最大 {report.max_frequency_db:.6f} dB",
        f"  大峰（完整模型）：{_dbs(report.big_peak_db_by_speaker)}；較差 {_peak(report.big_peak_db)}",
        f"  基準（方案原本六面，完整模型）：值 {baseline.full_total:.6f}（主位起伏 {baseline.full_components[0]:.6f} dB、"
        f"座位穩健性 {baseline.full_components[1]:.6f} dB）；大峰 {_dbs(baseline.big_peak_db_by_speaker)}，"
        f"較差 {_peak(baseline.big_peak_db)}",
        f"  改善（基準減終點，完整模型）：{gain.total:+.6f}（主位起伏 {gain.components[0]:+.6f} dB、"
        f"座位穩健性 {gain.components[1]:+.6f} dB）",
        f"  拿掉的空腔（小於最小空腔不放空氣層）：{dropped}",
        f"  匯出方案：{record.exported_scheme}（SHA-256 {record.exported_scheme_sha256}）",
        f"  秒數：匯出算子 {seconds.export_operators:.1f}、終點重算 {seconds.endpoint:.1f}、基準 {seconds.baseline:.1f}",
    ]


def stale_reasons(folder: Path, project: GradientProject, record: EndpointRecord) -> list[str]:
    """endpoint.json 對現在這個資料夾還算不算數：四件都要對，回對不上的那幾件（空的就是有效）。

    搜尋身分（endpoint.json 與 result.json 的 ``gradient_id`` 都等於 project.json 的）、現在帳本檔的 SHA-256 等於終點綁的、
    現在 result.json 檔的 SHA-256 等於終點讀的那一份。只讀檔、量摘要，不求值。
    """
    reasons = []
    if record.gradient_id != project.gradient_id:
        reasons.append("endpoint.json 的 gradient_id 不是這次搜尋的")
    ledger, result = folder / LEDGER_FILE, folder / RESULT_FILE
    if not ledger.exists() or file_sha256(ledger) != record.result_ledger_sha256:
        reasons.append("帳本檔跟終點綁的不同")
    if not result.exists() or file_sha256(result) != record.result_sha256:
        reasons.append("result.json 跟終點讀的不同")
    elif read_result(folder).gradient_id != project.gradient_id:
        reasons.append("result.json 的 gradient_id 不是這次搜尋的")
    return reasons


def _endpoint_lines(folder: Path, project: GradientProject) -> list[str]:
    """有 endpoint.json 且 :func:`stale_reasons` 是空的就印終點一節；有對不上的印過期（指名哪幾件）；
    沒有 endpoint.json 印「未做」那一行。"""
    if not (folder / ENDPOINT_FILE).exists():
        return [FULL_MODEL_LINE]
    record = read_endpoint(folder)
    reasons = stale_reasons(folder, project, record)
    if reasons:
        return [STALE_ENDPOINT_LINE, f"  對不上的：{'、'.join(reasons)}"]
    return _endpoint_section(record)


def render_report(folder: Path) -> str:
    """整份報告文字；最後是終點一節（或過期、或「未做」那一行）。"""
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
    lines += _point_lines(header, _feasible(rows), _kept_texts(project))
    lines += _endpoint_lines(folder, project)
    return "\n".join(lines) + "\n"
