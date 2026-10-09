"""擺位穩定性唯一報告資料：凍結摘要、逐點進度、原子存讀與判舊。"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator

from aosr.reporting.result import PlacementShiftName, PurposeSettings
from aosr.reporting.scheme import Scheme
from aosr.reporting.validation import SchemeProblem
from aosr.search.constraints import Violation
from aosr.search.crossover_record import RowStamp
from aosr.search.outer_status import OuterConclusion, OuterSnapshot
from aosr.search.placement_stability import FinalistSelection, Outcome, PointOutcome, StabilityArithmetic
from aosr.search.placement_stability_events import BoundaryDistance, FurnitureEvent
from aosr.search.store import FROZEN, SearchIdentity

DIRECTORY = "placement-stability"
STOPPED_REASON = "已停止，沒有算完；人手接續後會再試"
STOPPED_NOTE = "擺位穩定性計算被停止，搜尋結果不受影響"
IDENTITY_SKIP = "不補：身分跟這場搜尋不同"
BATCH_ABORTED = "同批被中止"


class IdentityStamp(BaseModel):
    model_config = FROZEN
    physics_identity: str
    program_fingerprint: str
    purpose_settings: PurposeSettings

    @classmethod
    def of(cls, identity: SearchIdentity) -> IdentityStamp:
        return cls(physics_identity=identity.physics_identity, program_fingerprint=identity.program_fingerprint,
                   purpose_settings=identity.purpose_settings)


class PointRecord(BaseModel):
    model_config = FROZEN
    trial_number: int | None = Field(ge=0)
    shift_name: PlacementShiftName
    scheme_hash: str | None = None
    outcome: Outcome | None = None
    reason_text: str = ""
    violations: tuple[Violation, ...] = ()
    problems: tuple[SchemeProblem, ...] = ()
    out_of_spec: bool = False
    spec_violations: tuple[Violation, ...] = ()
    outside_search: bool = False
    outside_search_quantities: tuple[str, ...] = ()
    search_range_not_checked: bool = False
    model_discontinuity: bool = False
    furniture_events: tuple[FurnitureEvent, ...] = ()
    total_cost: float | None = Field(default=None, ge=0)
    result_file: str | None = None
    fem_root: str | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.outcome is not None:
            self.as_outcome()
        elif self.total_cost is not None:
            raise ValueError("未交回的點沒有總代價")
        return self

    def as_outcome(self) -> PointOutcome:
        if self.outcome is None:
            raise ValueError("尚未交回的點沒有結局")
        return PointOutcome(self.outcome, self.total_cost, self.model_discontinuity,
                            self.out_of_spec, self.outside_search, self.search_range_not_checked)


class FinalistBoundaries(BaseModel):
    model_config = FROZEN
    trial_number: int | None
    distances: tuple[BoundaryDistance, ...]


class StabilitySummary(BaseModel):
    model_config = FROZEN
    schema_version: Literal["aosr.search_placement_stability.v1"] = "aosr.search_placement_stability.v1"
    search_id: str
    state: Literal["running", "done", "skipped", "failed", "stopped"] = "running"
    completed: bool = False
    reason_text: str = ""
    conclusion: OuterConclusion | None = None
    snapshot: OuterSnapshot = OuterSnapshot()
    rows: tuple[RowStamp, ...] = ()
    rows_fingerprint: str = ""
    crossover_stamp: str = ""
    identity: IdentityStamp
    observed_identity: IdentityStamp | None = None
    computed_points: int = Field(default=0, ge=0)
    total_points: int = Field(default=0, ge=0)
    selection: FinalistSelection = FinalistSelection((), ())
    points: tuple[PointRecord, ...] = ()
    arithmetic: StabilityArithmetic | None = None
    boundaries: tuple[FinalistBoundaries, ...] = ()

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.completed != (self.state in ("done", "skipped")):
            raise ValueError("只有完成或跳過的摘要帶完成記號")
        if self.computed_points > self.total_points:
            raise ValueError("已算點數不可超過共需計算點數")
        if self.state != "done" and (self.arithmetic is not None or self.boundaries):
            raise ValueError("只有完成的摘要可帶報表算術與基準點離邊界")
        keys = [(p.trial_number, p.shift_name) for p in self.points]
        if len(keys) != len(set(keys)):
            raise ValueError("每個入圍每個移位只能一列")
        return self


def scheme_hash(scheme: Scheme) -> str:
    document = scheme.model_dump(mode="json", exclude={"scheme_id"})
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def summary_path(folder: Path) -> Path:
    return folder / DIRECTORY / "summary.json"


def read_summary(folder: Path) -> StabilitySummary | None:
    path = summary_path(folder)
    return StabilitySummary.model_validate_json(path.read_bytes()) if path.exists() else None


def write_summary(folder: Path, summary: StabilitySummary) -> None:
    """每點先寫同資料夾暫存再換名，讀者永遠拿到完整的一代。"""
    summary = StabilitySummary.model_validate(summary.model_dump())
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


def is_stale(summary: StabilitySummary, current: StabilitySummary) -> bool:
    return (summary.search_id != current.search_id or summary.conclusion != current.conclusion
            or summary.snapshot != current.snapshot or summary.rows != current.rows
            or summary.rows_fingerprint != current.rows_fingerprint
            or summary.crossover_stamp != current.crossover_stamp or summary.identity != current.identity)


def attachment_path(folder: Path, relative: str) -> Path:
    root = summary_path(folder).parent.resolve()
    path = (folder / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("擺位穩定性檔案不能指向附件資料夾外")
    return path
