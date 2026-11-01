"""牆面材料梯度搜尋的人看報告（#561 辛一、辛三、庚三）：只讀資料夾裡的 ``project.json``、``status.json``、兩本帳、方案檔、
``result.json`` 與 ``endpoint.json``，不求值、不寫檔。

每一段（下降幾步、取規格前後值、重調幾步）、每一輪的決定、累計求值次數、停止原因、最後的可行點（每面 preset、
厚度 mm、空腔 mm；不貼的面印方案原本那面牆的材料）、值與分項。帳本照 ``read_ledger`` 逐列驗，壞列照樣報錯。
換樣板一節（庚三）：每輪列全部快速評分的值，沒進重調的標「只快速評分」、進重調的標「重調過」（重調後的值與改善），
換過去那面沒有空腔變數的另標「無可重調變數」；基準重調後的值、門檻、決定、累計求值次數、停止原因照字面。換樣板帳本讀不開
時這一節印讀不開的原因（不中斷，終點那一節照樣判過期）。這裡的值都是替身算的；完整模型確認只在終點那一節。
最後可行點：換樣板帳本有決定列取它最後一列決定（配置照那一列），否則庚二帳本有決定列取最後一列決定，否則取前段收好的
重調末列，再否則取規格那一列。
不貼的面讀 ``project.json`` 記的方案檔（讀，不核雜湊；方案檔事後換過，這裡印的是現在那一份）。
最後一節是終點：``endpoint.json`` 綁的搜尋身分、兩本帳與結果檔摘要都等於現在的、兩本帳都有收尾、結果檔宣稱的兩本帳就是
現在這兩本，才印終點數字，任一件對不上印過期，
沒有 ``endpoint.json`` 最後一行是 :data:`FULL_MODEL_LINE`。

#561 壬一（老闆 2026-10-30：「第三場一定要同時列舊分數與新分數，避免只因加了權重、排名改變，就誤以為聲學真的改善」）：
每一列印「舊合分／新合分／代理」，終點段（完整模型）另印真大峰與上界 T·ln N，並列兩行：「代理 − 內部最大殘差」
（夾擠的對象是內部最大殘差：應在 0～上界，超出——兩端各放寬終點報告記的均方根零判地板 ``rms_floor_db``——就印
:data:`PROXY_MISMATCH`，只報不判）與「真大峰」（偵測器的最高峰，只保證不大於內部最大殘差；低於它表示那個最大值
不是嚴格局部極大，例如緊貼更高的端點）。
報告只讀、不接續，所以加代理之前的資料夾（庚二帳本 v4、換樣板帳本 v1、結果 v3、終點 v2，示範房那一種）照樣讀得回，
那些列印「舊合分」並註明沒有代理；接續與終點仍拒收它們。
#990：狀態之後一行印安裝方式（:func:`mount_line`，中文、不印代號）；貼在原牆上時最後可行點每個貼樣板的面另印它背後那面原牆。
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ValidationError

from aosr.config.material_gradient import MOUNT_ON_WALL
from aosr.geometry.shoebox import Wall
from aosr.gradient.endpoint import EndpointReport
from aosr.gradient.endpoint_run import ENDPOINT_FILE, EndpointBaseline, EndpointRecord, read_endpoint, read_result
from aosr.gradient.material_gradient_ledger import (
    LEGACY_LEDGER_VERSION, MaterialGradientHeader, MaterialGradientRow, read_ledger,
)
from aosr.gradient.run import (
    LEDGER_FILE, RESULT_FILE, STATUS_FILE, STOP_FILE, TEMPLATES_FILE, GradientProject, GradientStatus, Layout,
    file_sha256, ledger_progress, project_wall_materials, read_project, surface_specs,
)
from aosr.gradient.template_ledger import TemplateRow, decisions_seconds, read_template_ledger, template_progress
from aosr.materials.construction_models import Assembly, ConstantImpedance, Layer, MountedOnWall, SurfaceMaterial
from aosr.reporting.scheme import load_scheme

FULL_MODEL_LINE: Final = "終點完整模型重算：未做（跑 endpoint <資料夾>）"
STALE_ENDPOINT_LINE: Final = "終點報告過期：endpoint.json 跟現在的帳本、結果檔或搜尋身分對不上，要重跑 endpoint"
TEMPLATE_HEADING: Final = "換樣板（替身快速評分與重調；完整模型確認只在終點那一節）："
QUICK_ONLY: Final = "只快速評分"
RETUNED: Final = "重調過"
NOT_RETUNABLE: Final = "無可重調變數"
ENDPOINT_HEADING: Final = "終點完整模型重算（"
LEGACY_SCORE_NOTE: Final = "加大峰代理之前的帳本：值就是舊合分，沒有新合分與代理（接續會拒收、要重跑）"
LEGACY_ROW_MARK: Final = "（無代理）"
LEGACY_ENDPOINT_NOTE: Final = "加大峰代理之前的終點：只有舊合分，沒有新合分、代理與上界"
MOUNT_ON_WALL_LINE: Final = "安裝方式：貼在原牆上（貼樣板的面新增層背後接方案原本那面牆，不是剛背；換樣板沿用）"
MOUNT_RIGID_LINE: Final = "安裝方式：剛背（樣板照原樣，新增層背後是剛性牆）"
MOUNTED_TEXT: Final = "貼在原牆上"
PROXY_MISMATCH: Final = "代理與內部最大殘差對不上"
_MM: Final = 1000.0


def _scores(row: MaterialGradientRow | TemplateRow) -> str:
    """一列的分數：「舊合分／新合分／代理」；加代理之前的列只有舊合分（它的值）。"""
    proxy = row.proxy_db()
    if proxy is None or row.screening_total is None:
        return f"舊合分 {row.value:.6f}{LEGACY_ROW_MARK}"
    return f"舊合分 {row.screening_total:.6f}／新合分 {row.value:.6f}／代理 {proxy:.3f} dB"


def _last_of(rows: Sequence[MaterialGradientRow] | Sequence[TemplateRow], phase: str, round_index: int | None,
             candidate: str | None) -> MaterialGradientRow | TemplateRow | None:
    found = [row for row in rows if (row.phase, row.round, row.candidate) == (phase, round_index, candidate)]
    return found[-1] if found else None


def _scores_of(row: MaterialGradientRow | TemplateRow | None, value: float) -> str:
    """決定裡記的值（新合分）配上它出處那一列的三個數；找不到那一列只印值。"""
    return _scores(row) if row is not None and row.value == value else f"值 {value:.6f}"


def _segment(rows: Sequence[MaterialGradientRow], phase: str) -> list[MaterialGradientRow]:
    return [row for row in rows if row.phase == phase and row.round is None]


def _segment_lines(label: str, segment: Sequence[MaterialGradientRow]) -> list[str]:
    if not segment:
        return [f"{label}：還沒寫進帳本"]
    last = segment[-1]
    closing = (f"，停止原因「{last.stop_reason}」、這段求值 {last.evaluations} 次" if last.stop_reason is not None
               else "（沒寫完）")
    return [f"{label}：{len(segment) - 1} 步，{_scores(segment[0])} → {_scores(last)}{closing}"]


def _pre_loop_lines(rows: Sequence[MaterialGradientRow]) -> list[str]:
    lines = ["前段："]
    descent, snaps, retune = _segment(rows, "descent"), _segment(rows, "snap"), _segment(rows, "retune")
    lines += ["  " + line for line in _segment_lines("連續下降", descent)]
    if snaps:
        before = _scores(descent[-1]) if descent else "？"
        blocked = [choice.surface_id for choice in snaps[0].snap or () if choice.cavity_clipped or choice.thickness_retreated]
        lines.append(f"  取規格：{before} → {_scores(snaps[0])}"
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
        lines.append(f"第 {row.round} 輪：交出 {_scores(row)}；這輪求值 {decision.round_evaluations} 次、"
                     f"累計 {decision.total_evaluations} 次")
        if decision.baseline_value is not None:
            baseline = _last_of(rows, "baseline_retune", row.round, None)
            lines.append(f"  基準重調後 {_scores_of(baseline, decision.baseline_value)}（門檻 {decision.threshold}）")
        for outcome in decision.outcomes:
            retuned = _last_of(rows, "candidate_retune", row.round, outcome.candidate)
            lines.append(f"  候選 {outcome.candidate}：預測 {outcome.predicted_change:+.6f}、重調後 "
                         f"{_scores_of(retuned, outcome.value)}、改善 {outcome.improvement:+.6f}")
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


def _word(model: type[BaseModel], field: str, value: str) -> str:
    """從材料模型欄位描述取代號的中文（「代號 表示／是 中文」），不另建翻譯表；底層不能用網頁那一份。"""
    match = re.search(rf"(?<![A-Za-z_]){re.escape(value)}\s*(?:表示|是)\s*([^，；）]+)", model.model_fields[field].description or "")
    if match is None:
        raise ValueError(f"材料模型描述缺少 {value!r} 的中文說明")
    return match.group(1).strip()


def _layers_text(layers: Sequence[Layer]) -> str:
    return "、".join(_word(type(layer), "kind", layer.kind) for layer in layers)


def material_text(material: SurfaceMaterial) -> str:
    """原牆材料的一句話（中文，不印種類代號）：固定阻抗印數值（Pa·s/m），構造印各層與背後，實測阻抗表印涵蓋範圍；
    貼在原牆上的構造另印它的原牆。"""
    if isinstance(material, ConstantImpedance):
        return f"固定阻抗 {material.impedance_pa_s_m:.3f} Pa·s/m"
    if isinstance(material, Assembly):
        return f"多層構造（房間側起 {_layers_text(material.layers)}；背後{_word(Assembly, 'backing', material.backing)}）"
    if isinstance(material, MountedOnWall):
        return f"貼在原牆上的構造（房間側起 {_layers_text(material.layers)}；原牆 {material_text(material.wall)}）"
    return material.coverage_text()


def _kept_texts(project: GradientProject) -> dict[str, str]:
    """不貼的面那一句：方案檔那一面原本的牆；方案檔讀不到時每面都說讀不到（報告照印，不中斷）。"""
    try:
        materials = project_wall_materials(project, load_scheme(Path(project.scheme_path)))
    except (OSError, ValueError) as error:
        return {wall.wall_name(): f"不貼（保留原牆；方案檔讀不到：{type(error).__name__}）" for wall in Wall.all()}
    return {wall.wall_name(): f"不貼（保留原牆：{material_text(material)}）" for wall, material in materials.items()}


def mount_line(project: GradientProject) -> str:
    """安裝方式那一行（中文、不印代號）。"""
    return MOUNT_ON_WALL_LINE if project.mount == MOUNT_ON_WALL else MOUNT_RIGID_LINE


def _mounted_texts(project: GradientProject) -> dict[str, str]:
    """貼在原牆上時，貼樣板的面那一行後面加的那一句（背後那面原牆）；剛背是空的。方案檔讀不到時每面都說讀不到。"""
    if project.mount != MOUNT_ON_WALL:
        return {}
    try:
        materials = project_wall_materials(project, load_scheme(Path(project.scheme_path)))
    except (OSError, ValueError) as error:
        return {wall.wall_name(): f"，{MOUNTED_TEXT}（方案檔讀不到：{type(error).__name__}）" for wall in Wall.all()}
    return {wall.wall_name(): f"，{MOUNTED_TEXT}（原牆：{material_text(material)}）" for wall, material in materials.items()}


def _point_lines(layout: Layout, row: MaterialGradientRow | TemplateRow | None,
                 kept: dict[str, str], mounted: Mapping[str, str]) -> list[str]:
    if row is None:
        return ["最後的可行點：還沒有（取規格還沒寫進帳本）"]
    book = "換樣板帳本" if isinstance(row, TemplateRow) else "帳本"
    lines = [f"最後的可行點（{book}第 {row.step} 列 {row.phase}）："]
    for spec in surface_specs(layout, row.theta()):
        if spec.preset_id is None:
            lines.append(f"  {spec.surface_id}：{kept.get(spec.surface_id, '不貼（保留原牆；方案沒有這一面）')}")
            continue
        cavity = "沒有空腔層" if spec.cavity_m is None else f"空腔 {spec.cavity_m * _MM:.3f} mm"
        thickness = f"{spec.thickness_m * _MM:.3f}" if spec.thickness_m is not None else "？"
        lines.append(f"  {spec.surface_id}：{spec.preset_id}，棉板 {thickness} mm，{cavity}{mounted.get(spec.surface_id, '')}")
    lines.append(f"  {_scores(row)}；分項：主位起伏 {row.components[0]:.6f} dB、座位穩健性 {row.components[1]:.6f} dB")
    return lines


def _quick_label(row: TemplateRow, retuned: Mapping[str, tuple[float, float]], block: Sequence[TemplateRow]) -> str:
    """一個快速評分候選的標記：沒進重調「只快速評分」、進了「重調過」（重調後的分數、改善）；換過去那面沒有空腔變數另標。"""
    if row.candidate is None or row.candidate not in retuned:
        return QUICK_ONLY + ("" if row.swapped_retunable else f"、{NOT_RETUNABLE}")
    value, improvement = retuned[row.candidate]
    note = "" if row.swapped_retunable else f"（換過去那面{NOT_RETUNABLE}，重調照步數走其他面的空腔；整房都沒有空腔時是同一點重複求值）"
    last = _last_of(block, "candidate_retune", row.round, row.candidate)
    return f"{RETUNED}{note}：重調後 {_scores_of(last, value)}、改善 {improvement:+.6f}"


def _template_round_lines(rows: Sequence[TemplateRow]) -> list[str]:
    lines: list[str] = []
    for round_index in sorted({row.round for row in rows}):
        block = [row for row in rows if row.round == round_index]
        decision_row = block[-1] if block[-1].decision is not None else None
        decision = decision_row.decision if decision_row is not None else None
        retuned = {outcome.candidate: (outcome.value, outcome.improvement)
                   for outcome in (decision.outcomes if decision is not None else ())}
        lines.append(f"換樣板第 {round_index} 輪：" + ("進行中（還沒有決定列，接續時這半輪會丟掉重跑）" if decision is None
                                                    else f"這輪求值 {decision.round_evaluations} 次、累計 {decision.total_evaluations} 次"))
        for row in block:
            if row.phase == "quick_score":
                lines.append(f"  {row.candidate}：快速評分 {_scores(row)}，{_quick_label(row, retuned, block)}")
        if decision_row is None or decision is None:
            continue
        if decision.baseline_value is not None:
            baseline = _last_of(block, "baseline_retune", round_index, None)
            lines.append(f"  基準（本輪起點配置）重調後 {_scores_of(baseline, decision.baseline_value)}"
                         f"（門檻 {decision.threshold}）")
        accepted = f"接受 {decision.accepted}" if decision.accepted is not None else "沒有接受"
        if decision.quick_count == 0:
            accepted = "沒開（沒有快速評分）"
        reason = f"；停止原因「{decision_row.stop_reason}」" if decision_row.stop_reason is not None else ""
        lines.append(f"  決定：{accepted}；交出 {_scores(decision_row)}{reason}")
    return lines


def _template_lines(folder: Path) -> tuple[list[str], TemplateRow | None]:
    """換樣板一節與它最後一列決定（沒有就 None）；帳本沒有或讀不開時印一行原因。"""
    path = folder / TEMPLATES_FILE
    if not path.exists():
        return [TEMPLATE_HEADING, "  換樣板帳本：還沒有"], None
    try:
        header, rows = read_template_ledger(path, legacy=True)
    except (OSError, ValueError) as error:
        return [TEMPLATE_HEADING, f"  換樣板帳本讀不開：{type(error).__name__}：{error}"], None
    evaluations, rounds, reason = template_progress(rows)
    legacy = [f"  換樣板帳本 {header.ledger_version}：{LEGACY_SCORE_NOTE}"] if any(row.is_legacy() for row in rows) else []
    lines = [TEMPLATE_HEADING, *legacy, *("  " + line for line in _template_round_lines(rows)),
             f"  換樣板累計求值次數（換樣板帳本記的，不含庚二）：{evaluations}；做完 {rounds} 輪；"
             f"各輪秒數加總 {decisions_seconds(rows):.1f}",
             f"  換樣板停止原因：{reason}" if reason is not None else "  換樣板停止原因：還沒收尾"]
    decisions = [row for row in rows if row.decision is not None]
    return lines, (decisions[-1] if decisions else None)


def _status_line(folder: Path) -> str:
    try:
        status = GradientStatus.model_validate_json((folder / STATUS_FILE).read_bytes())
    except (OSError, ValidationError) as error:
        return f"狀態：讀不到狀態檔（{type(error).__name__}）"
    note = f"；{status.message}" if status.message else ""
    return (f"狀態：{status.state}（子行程 {status.pid}，這次開始 {status.started_at}、最後寫 {status.updated_at}；"
            f"換樣板求值 {status.template_evaluations} 次、做完 {status.template_rounds_done} 輪{note}）")


def _dbs(values: Mapping[str, float | None]) -> str:
    return "、".join(f"{name} {'量不到' if value is None else f'{value:.3f} dB'}" for name, value in values.items())


def _peak(value: float | None) -> str:
    return "量不到" if value is None else f"{value:.3f} dB"


def proxy_lines(label: str, proxy_db: float | None, interior_max_db: float | None, big_peak_db: float | None,
                bound_db: float | None, floor_db: float) -> list[str]:
    """兩行：「代理 − 內部最大殘差 ＝ … dB（應在 0～上界之間）」，超出（兩端各放寬 ``floor_db``）加印
    :data:`PROXY_MISMATCH`、不判；「真大峰 … dB（偵測器……）」。加代理之前的終點只印一行註記。"""
    if proxy_db is None or interior_max_db is None or bound_db is None:
        return [f"  {label}代理：{LEGACY_ENDPOINT_NOTE}"]
    gap = proxy_db - interior_max_db
    inside = -floor_db <= gap <= bound_db + floor_db
    return [
        f"  {label}代理 − 內部最大殘差 ＝ {gap:+.6f} dB（應在 0～上界 {bound_db:.6f} dB 之間）"
        + ("" if inside else f"；{PROXY_MISMATCH}"),
        f"  {label}真大峰 {_peak(big_peak_db)}（偵測器；低於內部最大殘差 {interior_max_db:.3f} dB 表示那個最大值"
        "不是嚴格局部極大）",
    ]


def _full_scores(total: float, objective_total: float | None, proxy_db: float | None, big_peak_db: float | None,
                 bound_db: float | None) -> str:
    """完整模型那一組：舊合分／新合分／代理／真大峰／上界；加代理之前的終點只有舊合分與真大峰。"""
    if objective_total is None or proxy_db is None or bound_db is None:
        return f"舊合分 {total:.6f}／真大峰 {_peak(big_peak_db)}{LEGACY_ROW_MARK}"
    return (f"舊合分 {total:.6f}／新合分 {objective_total:.6f}／代理 {proxy_db:.3f} dB／真大峰 {_peak(big_peak_db)}"
            f"／上界 {bound_db:.3f} dB")


def _reduced_scores(report: EndpointReport) -> str:
    if report.reduced_objective_total is None or report.reduced_proxy_db is None:
        return f"舊合分 {report.reduced_total:.6f}"
    return (f"舊合分 {report.reduced_total:.6f}／新合分 {report.reduced_objective_total:.6f}／"
            f"代理 {report.reduced_proxy_db:.3f} dB")


def _baseline_scores(baseline: EndpointBaseline, bound_db: float | None) -> str:
    return _full_scores(baseline.full_total, baseline.full_objective_total, baseline.full_proxy_db,
                        baseline.big_peak_db, bound_db)


def _endpoint_section(record: EndpointRecord) -> list[str]:
    report, baseline, gain, seconds = record.report, record.baseline, record.improvement, record.seconds
    dropped = "、".join(report.dropped_cavities) if report.dropped_cavities else "沒有"
    bound = report.proxy_gap_bound_db
    new_gain = ("" if gain.objective_total is None or gain.proxy_db is None
                else f"；新合分 {gain.objective_total:+.6f}、代理 {gain.proxy_db:+.3f} dB")
    return [
        f"{ENDPOINT_HEADING}{record.finished_at}，引擎 {record.engine_commit or '沒記'}）："
        + ("" if bound is not None else f"{LEGACY_ENDPOINT_NOTE}"),
        f"  替身：{_reduced_scores(report)}",
        f"  完整模型：{_full_scores(report.full_total, report.full_objective_total, report.full_proxy_db, report.big_peak_db, bound)}"
        f"（主位起伏 {report.full_components[0]:.6f} dB、座位穩健性 {report.full_components[1]:.6f} dB）",
        *proxy_lines("終點", report.full_proxy_db, report.full_interior_max_db, report.big_peak_db, bound,
                     report.rms_floor_db),
        f"  舊合分的差（完整減替身）{report.total_difference:+.6f}；上界 {report.total_difference_bound:.6f}",
        f"  每座位最大逐頻差（替身對完整模型）：{_dbs(report.frequency_db_by_seat)}；"
        f"全部最大 {report.max_frequency_db:.6f} dB",
        f"  真大峰（完整模型）：{_dbs(report.big_peak_db_by_speaker)}；較差 {_peak(report.big_peak_db)}",
        f"  基準（方案原本六面，完整模型）：{_baseline_scores(baseline, bound)}（主位起伏 {baseline.full_components[0]:.6f} dB、"
        f"座位穩健性 {baseline.full_components[1]:.6f} dB）；真大峰 {_dbs(baseline.big_peak_db_by_speaker)}",
        *proxy_lines("基準", baseline.full_proxy_db, baseline.full_interior_max_db, baseline.big_peak_db, bound,
                     report.rms_floor_db),
        f"  改善（基準減終點，完整模型）：舊合分 {gain.total:+.6f}（主位起伏 {gain.components[0]:+.6f} dB、"
        f"座位穩健性 {gain.components[1]:+.6f} dB）{new_gain}",
        f"  拿掉的空腔（小於最小空腔不放空氣層）：{dropped}",
        f"  匯出方案：{record.exported_scheme}（SHA-256 {record.exported_scheme_sha256}）",
        f"  秒數：匯出算子 {seconds.export_operators:.1f}、終點重算 {seconds.endpoint:.1f}、基準 {seconds.baseline:.1f}",
    ]


def stale_reasons(folder: Path, project: GradientProject, record: EndpointRecord) -> list[str]:
    """endpoint.json 對現在這個資料夾還算不算數：九件都要對，回對不上的那幾件（空的就是有效）。

    搜尋身分（endpoint.json 與 result.json 的 ``gradient_id`` 都等於 project.json 的）、現在帳本檔的 SHA-256 等於終點綁的、
    現在帳本有收尾（最後一列是帶停止原因與決定的 ``round_decision``）、現在 result.json 檔的 SHA-256 等於終點讀的那一份、
    result.json 宣稱出自的帳本（``ledger_sha256``）就是現在這本；庚三再三件：現在換樣板帳本檔的 SHA-256 等於終點綁的、
    換樣板帳本有收尾（讀得開、最後一列是帶停止原因的決定列）、result.json 宣稱的換樣板帳本（``template_ledger_sha256``）
    就是現在這本。各檔的摘要都對上、彼此也一致才算數；幾份檔一起偽造到內部一致擋不住（本機摘要不是簽章）。
    只讀檔、量摘要，不求值。
    """
    reasons = []
    if record.gradient_id != project.gradient_id:
        reasons.append("endpoint.json 的 gradient_id 不是這次搜尋的")
    ledger, result = folder / LEDGER_FILE, folder / RESULT_FILE
    current = file_sha256(ledger) if ledger.exists() else None
    if current != record.result_ledger_sha256:
        reasons.append("帳本檔跟終點綁的不同")
    rows = read_ledger(ledger, legacy=True)[1] if current is not None else ()
    last = rows[-1] if rows else None
    if last is None or last.phase != "round_decision" or last.stop_reason is None or last.decision is None:
        reasons.append("帳本沒收尾")
    templates = folder / TEMPLATES_FILE
    current_templates = file_sha256(templates) if templates.exists() else None
    if current_templates != record.result_template_ledger_sha256:
        reasons.append("換樣板帳本檔跟終點綁的不同")
    if current_templates is None or not _templates_closed(templates):
        reasons.append("換樣板帳本沒收尾")
    if not result.exists():
        return [*reasons, "result.json 跟終點讀的不同"]
    if file_sha256(result) != record.result_sha256:
        reasons.append("result.json 跟終點讀的不同")
    claimed = read_result(folder, legacy=True)
    if claimed.gradient_id != project.gradient_id:
        reasons.append("result.json 的 gradient_id 不是這次搜尋的")
    if claimed.ledger_sha256 != current:
        reasons.append("結果宣稱的帳本不是現在這本")
    if claimed.template_ledger_sha256 != current_templates:
        reasons.append("結果宣稱的換樣板帳本不是現在這本")
    return reasons


def _templates_closed(path: Path) -> bool:
    try:
        rows = read_template_ledger(path, legacy=True)[1]
    except (OSError, ValueError):
        return False
    return template_progress(rows)[2] is not None


def _endpoint_lines(folder: Path, project: GradientProject) -> list[str]:
    """有 endpoint.json 且 :func:`stale_reasons` 是空的就印終點一節；有對不上的印過期（指名哪幾件）；
    沒有 endpoint.json 印「未做」那一行；endpoint.json 或 result.json 讀不開（例如庚三之前的舊版本）也印過期、附原因。"""
    if not (folder / ENDPOINT_FILE).exists():
        return [FULL_MODEL_LINE]
    try:
        record = read_endpoint(folder, legacy=True)
        reasons = stale_reasons(folder, project, record)
    except (OSError, ValueError) as error:
        return [STALE_ENDPOINT_LINE, f"  讀不開：{type(error).__name__}：{error}"]
    if reasons:
        return [STALE_ENDPOINT_LINE, f"  對不上的：{'、'.join(reasons)}"]
    return _endpoint_section(record)


def render_report(folder: Path) -> str:
    """整份報告文字；最後是終點一節（或過期、或「未做」那一行）。"""
    project = read_project(folder)
    lines = [f"梯度搜尋 {project.gradient_id}（用途 {project.purpose}，開始 {project.started_at}）", _status_line(folder),
             mount_line(project)]
    if (folder / STOP_FILE).exists():
        lines.append("停止記號：在（下一次求值之前會停；resume 會先清掉它）")
    ledger = folder / LEDGER_FILE
    if not ledger.exists():
        lines.append("帳本：還沒有")
        return "\n".join([*lines, FULL_MODEL_LINE]) + "\n"
    header, rows = read_ledger(ledger, legacy=True)
    evaluations, rounds, _, reason = ledger_progress(rows)
    if header.ledger_version == LEGACY_LEDGER_VERSION:
        lines.append(f"帳本 {header.ledger_version}：{LEGACY_SCORE_NOTE}")
    lines += _pre_loop_lines(rows)
    lines += _round_lines(rows)
    lines.append(f"累計求值次數（帳本記的、不含前段）：{evaluations}；做完 {rounds} 輪")
    lines.append(f"停止原因：{reason}" if reason is not None else "停止原因：還沒收尾")
    template_lines, template_last = _template_lines(folder)
    lines += template_lines
    layout: Layout = template_last if template_last is not None else header
    lines += _point_lines(layout, template_last if template_last is not None else _feasible(rows), _kept_texts(project),
                          _mounted_texts(project))
    lines += _endpoint_lines(folder, project)
    return "\n".join(lines) + "\n"
