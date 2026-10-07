"""交接摘要的唯讀投影；報告與網頁共用，不讀完整結果、不重評、不寫檔。"""
from __future__ import annotations

from pydantic import BaseModel, ValidationError

from aosr.search.crossover_record import (
    INTRO, STALE, TITLE, CrossoverSummary, fresh_summary, is_stale, read_summary, summary_lines,
)
from aosr.search.run import SearchStatus
from aosr.search.store import FROZEN, SearchStore


class CrossoverReport(BaseModel):
    model_config = FROZEN
    lines: tuple[str, ...] = summary_lines(None)
    warning: bool = False


def crossover_report(store: SearchStore, status: SearchStatus) -> CrossoverReport:
    try:
        summary: CrossoverSummary | None = read_summary(store.path)
        lines = summary_lines(summary)
        if summary is not None and is_stale(summary, fresh_summary(store, status)):
            lines = (*lines, STALE)
        return CrossoverReport(lines=lines, warning=summary is not None and summary.state == "failed")
    except ValidationError:
        return CrossoverReport(lines=(INTRO, "交接敏感度摘要讀不到：JSON 資料損壞或欄位不完整"), warning=True)
    except (OSError, ValueError, KeyError) as error:
        return CrossoverReport(lines=(INTRO, f"交接敏感度摘要讀不到：{error}"), warning=True)


def crossover_text(report: CrossoverReport) -> str:
    return "\n".join((TITLE, *(line.replace("\r", " ").replace("\n", "；") for line in report.lines)))
