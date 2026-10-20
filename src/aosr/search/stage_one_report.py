"""只讀第一階段搜尋帳本、狀態與候選結果；不求解。"""
from __future__ import annotations

from dataclasses import dataclass

from aosr.reporting.stage_one_result import StageOneResult, load_stage_one_result, wall_assumption_text
from aosr.scoring.contract import Feature
from aosr.scoring.contract_base import Flag
from aosr.scoring.stage_one_low_frequency_contract import IDEAL_SOURCE_ASSUMPTION_TEXT, StageOneSpeakerPayload
from aosr.search import ledger
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore


@dataclass(frozen=True)
class StageOneReportRow:
    """保留帳本與讀回錯誤，壞掉的一列不妨礙其他候選。"""

    trial_number: int
    row: ledger.LedgerRow
    result: StageOneResult | None
    error: str | None


@dataclass(frozen=True)
class StageOneSearchReport:
    """第一階段完整名次與不可排名列。"""

    wall_text: str
    status: SearchStatus
    ranked: tuple[StageOneReportRow, ...]
    excluded: tuple[StageOneReportRow, ...]


def _read(store: SearchStore, row: ledger.LedgerRow) -> StageOneReportRow:
    try:
        result = load_stage_one_result(store.candidate_path(row.trial_number))
        return StageOneReportRow(row.trial_number, row, result, None)
    except (OSError, ValueError) as error:
        return StageOneReportRow(row.trial_number, row, None, f"結果讀回失敗：{error}")


def build_stage_one_report(store: SearchStore) -> StageOneSearchReport:
    """排名採帳本合分，分項與峰谷只讀已存結果。"""
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    rows = ledger.read_for(store).rows
    scored = sorted((row for row in rows if row.outcome == "scored"),
                    key=lambda row: (row.score if row.score is not None else float("inf"), row.trial_number))
    excluded = sorted((row for row in rows if row.outcome == "excluded"), key=lambda row: row.trial_number)
    return StageOneSearchReport(wall_assumption_text(store.project), status,
        tuple(_read(store, row) for row in scored), tuple(_read(store, row) for row in excluded))


def _feature(feature: Feature) -> str:
    width = "量不到" if feature.width_octave is None else f"{feature.width_octave:.2f} 八度"
    if Flag.FEATURE_NARROWER_THAN_AXIS in feature.flags:
        width += "（窄於頻率軸）"
    if Flag.FEATURE_BOUNDARY_INCOMPLETE in feature.flags:
        width += "（邊界不完整）"
    return f"{feature.center_frequency_hz:.2f} Hz，{feature.depth_db:.2f} dB，{width}"


def _selected(payload: StageOneSpeakerPayload, index: int | None) -> str:
    return "無" if index is None else _feature(payload.features[index])


def _speaker_lines(result: StageOneResult) -> list[str]:
    lines = []
    for speaker in result.evaluation.speakers:
        payload = speaker.payload
        if payload is None:
            lines.append(f"  {speaker.speaker_id}：不可估；原因 {','.join(code.value for code in speaker.reason_codes)}")
        else:
            lines.append(f"  {speaker.speaker_id}：主項 {payload.own_line_ripple_rms_db:.6f} dB；"
                f"穩健性 {payload.seat_robustness_rms_db:.6f} dB；"
                f"最強峰 {_selected(payload, payload.strongest_peak_index)}；"
                f"最深谷 {_selected(payload, payload.deepest_dip_index)}")
    return lines


def render_stage_one_text(report: StageOneSearchReport) -> str:
    """明列所有得分候選，峰谷高度對每支喇叭自己那條直線量。"""
    status = report.status
    reasons = "、".join(f"{reason} {count}" for reason, count in sorted(status.illegal_reasons.items())) or "無"
    lines = [report.wall_text, IDEAL_SOURCE_ASSUMPTION_TEXT, "低頻拖尾不進自動排名，入選後要另外檢查",
        f"停止原因：{status.state}；{status.message}",
        f"問過 {status.asked}；算完 {status.computed}；不合法 {status.illegal}（各原因：{reasons}）；"
        f"不可估 {status.excluded.get('unassessed', 0)}；不能同表 {status.excluded.get('incomparable', 0)}",
        "名次｜試算編號｜離前牆／喇叭間距／聆聽距離（公尺）｜合分",
        "每支喇叭：主項／穩健性；最強峰／最深谷（頻率 Hz、高度 dB 對自己那條直線、寬度八度）"]
    for rank, item in enumerate(report.ranked, 1):
        meters = item.row.params_m
        lines.append(f"{rank}｜試算 {item.trial_number}｜{meters['front_distance']:.6f}／"
            f"{meters['spacing']:.6f}／{meters['listening_distance']:.6f}｜{item.row.score}")
        if item.error:
            lines.append(f"試算 {item.trial_number}：{item.error}")
        elif item.result is not None:
            lines.extend(_speaker_lines(item.result))
    lines.append("不可排名候選（不給名次）")
    for item in report.excluded:
        label = "不可估" if item.row.reason == "unassessed" else "不能同表"
        if item.error:
            detail = item.error
        elif item.result is not None:
            detail = ",".join(code.value for code in item.result.score.reason_codes)
        else:
            detail = "缺少結果"
        lines.append(f"試算 {item.trial_number}：{label}；{detail}")
    return "\n".join(lines) + "\n"
