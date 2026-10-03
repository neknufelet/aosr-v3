"""細算帳的存放骨架，不執行計算、重排或搜尋回饋。

每個細算完成的候選寫一列並 flush（沖出緩衝）、fsync（同步到磁碟）。
只有一個主行程寫；只捨棄不完整末列，續寫時修復尾端。
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Final, Literal, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from aosr.search.ledger import LedgerRow
from aosr.search.store import FROZEN, REFINE_DIR


REFINE_LEDGER_VERSION: Final = "aosr.search_refine.v1"


class RefineHeader(BaseModel):
    """首列釘住搜尋與細算身分；實算入口下一支才核對搜尋快照。"""

    model_config = FROZEN
    ledger_version: Literal["aosr.search_refine.v1"]
    search_id: str = Field(min_length=1, strict=True)
    settings_fingerprint: str = Field(min_length=1, strict=True)
    physics_identity: str = Field(min_length=1, strict=True)
    program_fingerprint: str = Field(min_length=1, strict=True)


class RefineRow(BaseModel):
    """每個細算候選一列；原方案編號為 None，沒有總代價的區不填假數字。"""

    model_config = FROZEN
    round: int = Field(ge=1, strict=True)
    trial_number: int | None = Field(ge=0, strict=True)
    result_file: str = Field(strict=True)
    outcome: Literal["scored", "excluded", "not_evaluated", "not_comparable"]
    total_cost: float | None
    seconds: float = Field(ge=0)

    @field_validator("result_file")
    @classmethod
    def _refine_file(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (path.is_absolute() or len(path.parts) != 2 or path.parts[0] != REFINE_DIR
                or path.parts[1] in (".", "..") or path.as_posix() != value or "\\" in value):
            raise ValueError("result_file must name a file directly inside verification")
        return value

    @model_validator(mode="after")
    def _consistent_outcome(self) -> Self:
        if (self.outcome == "scored") != (self.total_cost is not None):
            raise ValueError("scored requires total_cost; other outcomes require no total_cost")
        return self


@dataclass(frozen=True)
class RefineRead:
    """完整列及位元組邊界；捨棄末列的旗標供狀態與續寫使用。"""

    header: RefineHeader
    rows: tuple[RefineRow, ...]
    dropped_last_line: bool
    valid_bytes: int


def _check_unique(rows: Sequence[RefineRow]) -> None:
    seen: set[tuple[int, int | None]] = set()
    for row in rows:
        key = (row.round, row.trial_number)
        if key in seen:
            raise ValueError("trial numbers must be unique within each refinement round")
        seen.add(key)


def _read_handle(handle: BinaryIO) -> RefineRead:
    lines = handle.readlines()
    if not lines or not lines[0].endswith(b"\n"):
        raise ValueError("missing or incomplete refinement header")
    header = RefineHeader.model_validate_json(lines[0])
    rows: list[RefineRow] = []
    valid_bytes = len(lines[0])
    dropped = False
    for index, line in enumerate(lines[1:], start=1):
        is_last = index == len(lines) - 1
        if is_last and not line.endswith(b"\n"):
            dropped = True
            break
        try:
            document: object = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            if not is_last:
                raise ValueError(f"corrupt refinement line {index + 1}") from error
            dropped = True
            break
        # JSON 有效但型別或欄位違規不代表截斷，末列也必須拒絕。
        rows.append(RefineRow.model_validate(document))
        valid_bytes += len(line)
    _check_unique(rows)
    return RefineRead(header, tuple(rows), dropped, valid_bytes)


def _write_line(handle: BinaryIO, value: RefineHeader | RefineRow) -> None:
    handle.write((json.dumps(value.model_dump(mode="json"), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())


class RefineLedger:
    """單一主行程逐列落地；重開與追加皆從磁碟讀帳。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    @classmethod
    def create(cls, path: Path, header: RefineHeader) -> RefineLedger:
        """獨佔建檔並同步表頭；已存在就報錯，不覆寫。"""
        header = RefineHeader.model_validate(header.model_dump())
        with path.open("xb") as handle:
            _write_line(handle, header)
        return cls(path)

    @classmethod
    def open(cls, path: Path) -> RefineLedger:
        """驗過才能接續；截斷末列留到下一次成功追加才修復。"""
        cls.read_status(path)
        return cls(path)

    def append(self, row: RefineRow) -> None:
        """排他鎖內核同輪唯一性，修復半列後追加並同步；拒絕時原檔不變。"""
        row = RefineRow.model_validate(row.model_dump())
        with self._path.open("r+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            status = _read_handle(handle)
            _check_unique((*status.rows, row))
            handle.seek(status.valid_bytes)
            if status.dropped_last_line:
                handle.truncate()
            _write_line(handle, row)

    @staticmethod
    def read(path: Path) -> tuple[RefineHeader, tuple[RefineRow, ...]]:
        """逐行驗型別，只回完整列；丟棄末列的旗標由 read_status（讀取狀態）提供。"""
        status = RefineLedger.read_status(path)
        return status.header, status.rows

    @staticmethod
    def read_status(path: Path) -> RefineRead:
        """只有缺換行或無法解碼的末列能捨棄；中間壞列與欄位違規報錯。"""
        with path.open("rb") as handle:
            return _read_handle(handle)


def refine_order(rows: Sequence[LedgerRow]) -> tuple[int, ...]:
    """只排有篩選分數的候選，分數小的先，同分按編號；原方案不在此序列。"""
    scored = [(row.score, row.trial_number) for row in rows if row.outcome == "scored" and row.score is not None]
    return tuple(trial_number for _, trial_number in sorted(scored))
