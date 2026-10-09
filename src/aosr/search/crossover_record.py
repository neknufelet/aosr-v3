"""交接提醒的獨立摘要、原子存讀與共用中文；不保存換接結果。"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Literal, Self, TypeAlias

from pydantic import BaseModel, model_validator

from aosr.reporting.display import speaker_label
from aosr.search.labels import trial_label as trial_label

from aosr.search.outer_status import OuterConclusion, OuterSnapshot, snapshot_of
from aosr.search.refine import RefineRow
from aosr.search.report_comparison import read_refinement_rows
from aosr.search.run import SearchStatus
from aosr.search.store import FROZEN, SearchStore

Verdict: TypeAlias = Literal["sensitive", "stable", "unverified"]
VERDICTS: dict[str, str] = {
    "sensitive": "排名對交接方式敏感，不宜只憑小分差選唯一第一名",
    "stable": "在已測接法下，排名穩定",
    "unverified": "交接影響尚未驗證",
}
TITLE = "300 Hz 交接敏感度（不改名次）"
INTRO = "只看換接法時名次會不會變，正式接法、分數與名次不變"
DISTANCE_NOTE = "距離近不等於聲音接近；這裡只看算法選擇對排名的影響，不是誤差界，也不是聽感比較"
INCOMPLETE = "上次沒做完（可能進行中或被中斷）"
STALE = "檢查之後細算表又變了，這份是舊的"
STOPPED_REASON = "已停止，沒有算完；人手接續後會再試"
STOPPED_NOTE = "交接敏感度檢查被停止，搜尋結果不受影響"


class RowStamp(BaseModel):
    model_config = FROZEN
    trial_number: int | None
    outcome: str
    total_cost: float | None
    cost_hex: str | None


class CostRow(BaseModel):
    model_config = FROZEN
    trial_number: int | None
    total_cost: float


class ExcludedRow(BaseModel):
    model_config = FROZEN
    trial_number: int | None
    reason_text: str


class VariantRecord(BaseModel):
    model_config = FROZEN
    key: str
    label: str
    basis: str
    truncated: bool = False
    tested: bool = False
    reason_text: str = ""
    ranking: tuple[CostRow, ...] = ()
    excluded: tuple[ExcludedRow, ...] = ()
    official_rank: int | None = None
    speaker_distance_cm: dict[str, float] = {}
    primary_distance_cm: float | None = None

    @model_validator(mode="after")
    def distances_together(self) -> Self:
        if bool(self.speaker_distance_cm) != (self.primary_distance_cm is not None):
            raise ValueError("喇叭距離與主位距離必須同時提供或同時省略")
        return self


class CrossoverSummary(BaseModel):
    model_config = FROZEN
    schema_version: Literal["aosr.crossover_sensitivity.v1"] = "aosr.crossover_sensitivity.v1"
    conclusion: OuterConclusion | None = None
    snapshot: OuterSnapshot = OuterSnapshot()
    rows: tuple[RowStamp, ...] = ()
    rows_fingerprint: str = ""
    completed: bool = False
    state: Literal["running", "done", "skipped", "failed", "stopped"] = "running"
    verdict: Verdict = "unverified"
    reason_text: str = ""
    official_best: int | None = None
    variants: tuple[VariantRecord, ...] = ()


def row_stamps(rows: tuple[RefineRow, ...]) -> tuple[RowStamp, ...]:
    return tuple(RowStamp(trial_number=r.trial_number, outcome=r.outcome, total_cost=r.total_cost,
        cost_hex=None if r.total_cost is None else r.total_cost.hex()) for r in rows)


def rows_fingerprint(rows: tuple[RowStamp, ...]) -> str:
    return hashlib.sha256("\n".join(r.model_dump_json() for r in rows).encode()).hexdigest()


def fresh_summary(store: SearchStore, status: SearchStatus) -> CrossoverSummary:
    stamps = row_stamps(read_refinement_rows(store))
    return CrossoverSummary(conclusion=status.outer.conclusion, snapshot=snapshot_of(status), rows=stamps,
        rows_fingerprint=rows_fingerprint(stamps))


def is_stale(summary: CrossoverSummary, current: CrossoverSummary) -> bool:
    return (summary.snapshot != current.snapshot or summary.conclusion != current.conclusion
            or summary.rows != current.rows or summary.rows_fingerprint != current.rows_fingerprint)


def summary_path(folder: Path) -> Path:
    return folder / "crossover-sensitivity" / "summary.json"


def read_summary(folder: Path) -> CrossoverSummary | None:
    path = summary_path(folder)
    return CrossoverSummary.model_validate_json(path.read_bytes()) if path.exists() else None


def write_summary(folder: Path, summary: CrossoverSummary) -> None:
    path = summary_path(folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = mkstemp(dir=path.parent, prefix="summary-write-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(summary.model_dump_json() + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)




def _variant_lines(variant: VariantRecord, *, seat_locked: bool = False) -> tuple[str, ...]:
    clipping = "被 300 Hz 截斷" if variant.truncated else "未被截斷"
    if variant.key == "legacy" and "做不出交接帶" in variant.reason_text:
        clipping = "不適用"
    lines = [f"{variant.label}：{variant.basis}；{clipping}；{'已測' if variant.tested else '未測'}"]
    if variant.reason_text:
        lines.append(f"{variant.label}：{variant.reason_text}")
    if variant.ranking:
        rank = "未進此接法的排名" if variant.official_rank is None else f"第 {variant.official_rank} 名"
        lines.append(f"{variant.label}第一名：{trial_label(variant.ranking[0].trial_number)}；正式第一名在此接法：{rank}")
    if variant.speaker_distance_cm:
        distances = "；".join(f"{speaker_label(key)}相距 {value:.1f} 公分" for key, value in variant.speaker_distance_cm.items())
        lines.append(f"與正式第一名的距離：{distances}；主位相距 {variant.primary_distance_cm:.1f} 公分"
                     + ("（座位鎖定，主位不動）" if seat_locked else ""))
    lines.extend(f"{variant.label}／{trial_label(row.trial_number)}：{row.reason_text}；未進此接法的排名"
                 for row in variant.excluded)
    return tuple(lines)


def summary_lines(summary: CrossoverSummary | None, *, seat_locked: bool = False) -> tuple[str, ...]:
    if summary is None:
        return (INTRO, VERDICTS["unverified"], "未開始（搜尋正常收尾後才補）", DISTANCE_NOTE)
    temporary = "（暫時）" if summary.conclusion != "complete" else ""
    lines = [INTRO, VERDICTS[summary.verdict] + temporary]
    if not summary.completed:
        lines.append(INCOMPLETE)
    if summary.reason_text:
        lines.append(summary.reason_text)
    for variant in summary.variants:
        lines.extend(_variant_lines(variant, seat_locked=seat_locked))
    return tuple(line.replace("\r", " ").replace("\n", "；") for line in (*lines, DISTANCE_NOTE))
