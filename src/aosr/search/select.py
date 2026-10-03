"""助理代按選中的細算結果：保留原檔、時間與出處，每份只選入一次。

32 位十六進位規則必須與 aosr.gui.app::RUN_ID 一致；搜尋層不拿網頁層的程式，考卷核兩份規則。
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import shutil
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Self

from pydantic import BaseModel, Field, model_validator

from aosr.reporting.result import SchemeResult
from aosr.search.ledger_io import read_rows, write_line
from aosr.search.refine import RefineLedger
from aosr.search.refine_run import header_for
from aosr.search.store import FROZEN, JSON_SUFFIX, JSONL_SUFFIX, SearchStore, refine_result_name, refine_scheme_id

RUN_ID = re.compile(r"[0-9a-f]{32}\Z")
SELECT_LEDGER_FILE = f"select{JSONL_SUFFIX}"


class SelectRow(BaseModel):
    """每份選入一列；原方案編號為 None，路徑與試算編號綁定。"""

    model_config = FROZEN
    trial_number: int | None = Field(ge=0, strict=True)
    result_file: str = Field(strict=True)
    data_dir: str = Field(min_length=1, strict=True)
    run_id: str = Field(strict=True)
    sha256: str = Field(strict=True, pattern=re.compile(r"^[0-9a-f]{64}\Z"))

    @model_validator(mode="after")
    def _result_matches(self) -> Self:
        if self.result_file != refine_result_name(self.trial_number):
            raise ValueError("result_file must be the refinement result of this trial number")
        if RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run_id must be 32 lowercase hexadecimal digits")
        if not Path(self.data_dir).is_absolute():
            raise ValueError("data_dir must be an absolute path")
        return self


@dataclass(frozen=True)
class SelectOutcome:
    """目標代號、完整路徑及本次是否新放入。"""

    run_id: str
    result_path: Path
    is_new: bool


@dataclass(frozen=True)
class SelectRead:
    """完整選取列、末列是否截斷、可安全追加的位元組位置。"""

    rows: tuple[SelectRow, ...]
    dropped_last_line: bool
    valid_bytes: int


def selection_ledger_path(store: SearchStore) -> Path:
    """選取帳在搜尋根層，與細算帳分開。"""
    return store.path / SELECT_LEDGER_FILE


def _check_unique(rows: Sequence[SelectRow]) -> None:
    """同一份細算結果放進同一個資料目錄只一次；目標代號全帳唯一。"""
    seen: set[tuple[int | None, str]] = set()
    targets: set[str] = set()
    for row in rows:
        if (row.trial_number, row.data_dir) in seen or row.run_id in targets:
            raise ValueError("each candidate is selected once per data directory and each target only once")
        seen.add((row.trial_number, row.data_dir))
        targets.add(row.run_id)


def _read_handle(handle: BinaryIO) -> SelectRead:
    handle.seek(0)
    rows, dropped, valid_bytes = read_rows(handle.readlines(), SelectRow, "selection")
    _check_unique(rows)
    return SelectRead(rows, dropped, valid_bytes)


def _append_handle(handle: BinaryIO, status: SelectRead, row: SelectRow) -> None:
    _check_unique((*status.rows, row))
    handle.seek(status.valid_bytes)
    if status.dropped_last_line:
        handle.truncate()
    write_line(handle, row)


class SelectLedger:
    """排他鎖內讀回、核唯一性並追加；拒絕不修復半列。"""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, row: SelectRow) -> None:
        row = SelectRow.model_validate(row.model_dump())
        with self._path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            _append_handle(handle, _read_handle(handle), row)

    @staticmethod
    def read(path: Path) -> tuple[SelectRow, ...]:
        return SelectLedger.read_status(path).rows

    @staticmethod
    def read_status(path: Path) -> SelectRead:
        with path.open("rb") as handle:
            return _read_handle(handle)


def _validated_source(store: SearchStore, number: int | None) -> tuple[Path, str]:
    """帳上完成且完整結果身分符合搜尋快照才可選取。"""
    refine_result_name(number)
    if not store.refine_ledger_path.is_file():
        raise ValueError(f"{number if number is not None else '原方案'} 沒細算過")
    header, rows = RefineLedger.read(store.refine_ledger_path)
    if not any(row.trial_number == number for row in rows):
        raise ValueError(f"{number if number is not None else '原方案'} 沒細算過")
    if header != header_for(store):
        raise ValueError("細算帳身分與搜尋快照不同")
    source = store.refine_result_path(number)
    content = source.read_bytes()
    result = SchemeResult.model_validate_json(content)
    identity = store.identity
    for name in ("physics_identity", "program_fingerprint", "purpose_settings"):
        if getattr(result, name) != getattr(identity, name):
            raise ValueError(f"細算結果 {name} 與搜尋快照不同")
    if result.scheme.scheme_id != refine_scheme_id(store.search_id, number):
        raise ValueError("細算結果方案代號與試算編號不同")
    return source, hashlib.sha256(content).hexdigest()


def _existing(row: SelectRow, digest: str, data_dir: Path) -> SelectOutcome:
    if row.sha256 != digest:
        raise ValueError("細算結果 SHA-256 與選取帳不同")
    target = data_dir / "results" / f"{row.run_id}{JSON_SUFFIX}"
    if not target.is_file():
        raise ValueError(f"已選入 {data_dir} 的目標 {row.run_id} 不存在，是否重放須由人決定")
    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError(f"已選入 {data_dir} 的目標 {row.run_id} 內容不同，拒絕重放")
    return SelectOutcome(row.run_id, target, False)


def _publish_new(source: Path, target: Path) -> None:
    """用硬連結發布：目的地已存在時 os.link 直接拒絕（FileExistsError），不會覆寫，也不留先查再改名的競態窗口。

    暫存檔與目的地在同一個資料夾；連結共用同一份內容與修改時間，暫存名由呼叫端刪掉。
    """
    os.link(source, target)


def _copy_new(source: Path, target: Path, digest: str) -> None:
    """先複製到同目錄暫存檔，再用硬連結發布；時間、位元組與不覆寫一起守。"""
    descriptor, name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.stem}-")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "rb"):
            shutil.copy2(source, temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise ValueError("細算結果在複製時改動，SHA-256 不同")
        _publish_new(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def select_refined(store: SearchStore, trial_number: int | None, data_dir: Path) -> SelectOutcome:
    """鎖住整次選入，避免兩個助理同時讀到未選而各自放一份。

    先寫帳再發布：發布失敗就把帳截回原位；萬一兩步之間整台當掉，帳上有列而目標不在，
    下次會走「目標不存在、由人決定」，不會默默放出第二份。
    """
    source, digest = _validated_source(store, trial_number)
    data_dir = data_dir.resolve()
    with selection_ledger_path(store).open("a+b") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        status = _read_handle(handle)
        existing = next((row for row in status.rows
                         if row.trial_number == trial_number and row.data_dir == str(data_dir)), None)
        if existing is not None:
            return _existing(existing, digest, data_dir)
        run_id = uuid.uuid4().hex
        row = SelectRow(trial_number=trial_number, result_file=refine_result_name(trial_number),
                        data_dir=str(data_dir), run_id=run_id, sha256=digest)
        target = data_dir / "results" / f"{run_id}{JSON_SUFFIX}"
        _append_handle(handle, status, row)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _copy_new(source, target, digest)
        except BaseException:
            handle.seek(status.valid_bytes)
            handle.truncate()
            os.fsync(handle.fileno())
            raise
        return SelectOutcome(run_id, target, True)
