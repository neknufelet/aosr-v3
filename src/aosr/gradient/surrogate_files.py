"""替身建模服務（``aosr.gradient.surrogate_build``）寫的檔、離開碼與「要等多久」的推估（#1098）。

這一支不載入 JAX 與物理（網頁行程照它讀進度）。一次建模＝一個工作資料夾：
- ``progress.json``：階段、完整解次數對預算、秩、每一輪的殘差與秒數、留出驗證解到第幾點（每一輪換名寫一次）；
- ``result.json``：建完才有：判決（放進格子／留出驗證沒過／閘門拒收）、停止原因、逐座位的兩個差、兩把尺；
- ``stage/``：先在這裡建好替身四件（模型、manifest、留出驗證收據、完整模型指紋）與訓練說明，閘門核過才整個換名進格子。
"""
from __future__ import annotations

import math
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

PROGRESS_FILE: Final = "progress.json"
RESULT_FILE: Final = "result.json"
STAGE_DIR: Final = "stage"
MODEL_FILE: Final = "surrogate.npz"
TRAINING_FILE: Final = "training.json"
PROGRESS_VERSION: Final = "aosr.gradient.surrogate_progress.v1"
RESULT_VERSION: Final = "aosr.gradient.surrogate_result.v1"

EXIT_INSTALLED: Final = 0
EXIT_FAILED: Final = 1
EXIT_STOPPED: Final = 3
EXIT_NOT_USABLE: Final = 4

Phase = Literal["preparing", "building", "validating", "checking", "installing"]
Verdict = Literal["installed", "not_validated", "gate_refused"]
ESTIMATE_LABEL: Final = "推估"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class RoundRecord(_Frozen):
    """一輪量完候選殘差的那一刻：到這裡解了幾次、秩、抽到的最大殘差、從建模開始過了幾秒。"""

    round: int = Field(ge=0)
    full_solves: int = Field(ge=0)
    rank: int = Field(ge=0)
    max_residual: float
    elapsed_s: float = Field(ge=0.0)


class SurrogateProgress(_Frozen):
    schema_version: Literal["aosr.gradient.surrogate_progress.v1"] = PROGRESS_VERSION
    phase: Phase
    started_at: float
    phase_started_at: float
    updated_at: float
    workers: int = Field(ge=1)
    dense_threads: int | None = None
    patches: int = Field(ge=1)
    dofs: int | None = None
    axis_points: int | None = None
    training_sets: int | None = None
    holdout_sets: int | None = None
    max_full_solves: int = Field(ge=1)
    stop_residual: float
    full_solves: int = Field(default=0, ge=0)
    rank: int | None = None
    rounds: tuple[RoundRecord, ...] = ()
    snapshot_seconds: float | None = None
    stop_reason: str | None = None
    build_seconds: float | None = None
    validation_done: int | None = None
    validation_total: int | None = None


class SurrogateResult(_Frozen):
    schema_version: Literal["aosr.gradient.surrogate_result.v1"] = RESULT_VERSION
    verdict: Verdict
    message: str
    stop_reason: str
    full_solves: int
    rank: int
    build_seconds: float
    validation_seconds: float | None = None
    frequency_db_by_seat: dict[str, float] = Field(default_factory=dict)
    band_db_by_seat: dict[str, float] = Field(default_factory=dict)
    rulers: dict[str, float] = Field(default_factory=dict)
    within_contract: bool | None = None
    slot: str | None = None
    retired: str | None = None
    finished_at: float


def _per_round(progress: SurrogateProgress) -> tuple[float, int] | None:
    """最近一輪的秒數與那一輪補了幾次完整解（一輪都還沒有就 None）。"""
    rounds = progress.rounds
    if not rounds:
        return None
    if len(rounds) == 1:
        return rounds[0].elapsed_s, rounds[0].full_solves
    last, before = rounds[-1], rounds[-2]
    return last.elapsed_s - before.elapsed_s, last.full_solves - before.full_solves


def _per_solve(progress: SurrogateProgress) -> float | None:
    if progress.snapshot_seconds is None or progress.full_solves == 0:
        return None
    return progress.snapshot_seconds / progress.full_solves


def estimate(progress: SurrogateProgress, now: float) -> dict[str, object]:
    """還要等多久（推估）：照這一跑已經量到的秒數外推，不是保證。

    建模段：最近一輪的秒數 × 預算內最多還要幾輪（每輪補的次數照最近一輪），加上留出驗證那一段（點數 × 建模量到的
    每次完整解秒數）；輪越後越慢（秩越大，投影與殘差越貴），殘差先達標會提早停，所以兩頭都可能偏。
    留出驗證段：照這一段已解的速度外推剩下的點。
    """
    elapsed = now - progress.started_at
    base: dict[str, object] = {"label": ESTIMATE_LABEL, "elapsed_s": elapsed, "remaining_s": None}
    validation_points = (None if progress.holdout_sets is None or progress.axis_points is None
                         else progress.holdout_sets * progress.axis_points)
    per_solve = _per_solve(progress)
    if progress.phase == "preparing":
        return base | {"basis": "還在匯出完整模型與算材料阻抗；建模第一輪做完才量得到速度"}
    if progress.phase == "building":
        measured = _per_round(progress)
        if measured is None or per_solve is None or validation_points is None:
            return base | {"basis": "建模第一輪還沒做完（起點那一批完整解在算），量到速度才推得出來"}
        seconds, solved = measured
        left = max(0, progress.max_full_solves - progress.full_solves)
        rounds_left = 0 if left == 0 else math.ceil(left / max(1, solved))
        build_left = rounds_left * seconds
        validation_left = validation_points * per_solve
        return base | {"remaining_s": build_left + validation_left, "build_remaining_s": build_left,
                       "validation_remaining_s": validation_left,
                       "basis": (f"最近一輪 {seconds:.0f} 秒 × 預算內最多再 {rounds_left} 輪，加留出驗證 {validation_points} 點 × "
                                 f"每次完整解 {per_solve:.2f} 秒；輪越後越慢，殘差先達標會提早停")}
    if progress.phase == "validating":
        done, total = progress.validation_done or 0, progress.validation_total or validation_points
        if total is None:
            return base | {"basis": "留出驗證點數還沒記下"}
        if done > 0:
            rate = (now - progress.phase_started_at) / done
            basis = f"留出驗證已解 {done}／{total} 點，照這一段的速度外推"
        elif per_solve is not None:
            rate = per_solve
            basis = f"留出驗證 {total} 點 × 建模量到的每次完整解 {per_solve:.2f} 秒"
        else:
            return base | {"basis": "留出驗證剛開始，還量不到速度"}
        remaining = (total - done) * rate
        return base | {"remaining_s": remaining, "validation_remaining_s": remaining, "basis": basis}
    return base | {"remaining_s": 0.0, "basis": "建完、驗完，正在用搜尋開跑那一道閘門核對並放進格子"}
