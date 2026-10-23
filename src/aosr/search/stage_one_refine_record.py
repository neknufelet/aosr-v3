"""第一階段細算摘要、原子保存與獨立讀回核對。"""
from __future__ import annotations
import os
import tempfile
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.reporting.stage_one_result import StageOneComparisonIdentity, StageOneResult, load_stage_one_result
from aosr.search.refine_run import _job
from aosr.search.run import ComputeFailed
from aosr.search.stage_one_selection_contract import FROZEN, StageOneSelection, big_peak
from aosr.search.stage_one_worker import check_stage_one_result
from aosr.search.store import SearchStore

PENDING_CHECK = "入選後檢查見下一段"
# 第一階段細算的三種收尾各對一個外圈結論；外圈快照只記第二階段的細算，細算重跑後要拿這張表核結論過期沒有。
REFINE_CONCLUSIONS: dict[str, Literal["complete", "refine_failed", "refine_interrupted"]] = {
    "done": "complete", "failed": "refine_failed", "interrupted": "refine_interrupted"}


class RefinedSeat(BaseModel):
    """單份驗證結果的狀態；對照保留數值但不佔重排名額。"""

    model_config = FROZEN
    trial_number: int
    result_file: str
    state: Literal["pending", "scored", "unassessed", "incomparable", "unreadable"] = "pending"
    reason: str = ""
    score: float | None = None
    big_peak_db: float | None = None
    seconds: float | None = None
    refine_rank: int | None = None


class StageOneRefineStatus(BaseModel):
    """可完整重建選席與接續；比較身分只屬於驗證軸。"""

    model_config = FROZEN
    selection: StageOneSelection
    results: tuple[RefinedSeat, ...]
    comparison_trial: int | None = None
    comparison_identity: StageOneComparisonIdentity | None = None
    state: Literal["selected", "done", "interrupted", "failed"]
    message: str


def save_stage_one_refine(store: SearchStore, status: StageOneRefineStatus) -> StageOneRefineStatus:
    """同目錄暫存、同步後原子替換；失敗只清這次自己的暫存檔。"""
    path = store.path / "stage_one_refine.json"
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(status.model_dump_json())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return status


def read_stage_one_refined_result(store: SearchStore, number: int) -> StageOneResult:
    """細算自己的讀回入口：出處、驗證代號、軸、方案及搜尋快照都要相符。"""
    job = _job(store, number)
    result = load_stage_one_result(job.result_path)
    check_stage_one_result(result, job, store.search_id, store.identity)
    if result.low_frequency_axis != LowFrequencyAxis.VERIFICATION or result.scheme != job.scheme:
        raise ComputeFailed("細算結果的驗證軸或方案跟工作不同")
    try:
        big_peak(result)
    except ValueError as error:
        raise ComputeFailed(f"細算結果的峰谷摘要壞了：{error}") from error
    return result
