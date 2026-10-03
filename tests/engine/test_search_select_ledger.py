"""選取帳逐行嚴格驗型別，尾列修復、唯一性與追加同步。"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aosr.search.store import refine_result_name


def _document(trial: int | None = 79) -> dict[str, object]:
    return {"trial_number": trial, "result_file": refine_result_name(trial),
            "run_id": "b" * 32, "sha256": "a" * 64}


@pytest.mark.parametrize("change", [
    {"trial_number": True}, {"trial_number": "79"}, {"trial_number": -1},
    {"result_file": refine_result_name(80)}, {"result_file": 1},
    {"run_id": "b" * 31}, {"run_id": "b" * 33}, {"run_id": "B" * 32},
    {"run_id": "b" * 32 + "\n"}, {"run_id": 1},
    {"sha256": "z" * 64}, {"sha256": "a" * 63}, {"sha256": "a" * 64 + "\n"}, {"sha256": None},
    {"extra": True},
])
def test_selection_ledger_rejects_invalid_complete_rows(tmp_path: Path, change: dict[str, object]) -> None:
    from aosr.search.select import SelectLedger

    path = tmp_path / "ledger"
    path.write_text(json.dumps(_document() | change) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SelectLedger.read(path)


@pytest.mark.parametrize("field", ["trial_number", "result_file", "run_id", "sha256"])
def test_selection_ledger_requires_each_field(tmp_path: Path, field: str) -> None:
    from aosr.search.select import SelectLedger

    document = _document()
    document.pop(field)
    path = tmp_path / "ledger"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SelectLedger.read(path)


@pytest.mark.parametrize("tail", [b'{"trial_number":', b'{broken}\n', json.dumps(_document(None)).encode()])
def test_selection_ledger_discards_only_incomplete_tail_and_repairs(tmp_path: Path, tail: bytes) -> None:
    from aosr.search.select import SelectLedger, SelectRow

    path = tmp_path / "ledger"
    first = SelectRow.model_validate(_document())
    book = SelectLedger(path)
    book.append(first)
    before = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(tail)
    assert SelectLedger.read(path) == (first,)
    assert path.read_bytes() == before + tail
    second = SelectRow.model_validate(_document(None) | {"run_id": "c" * 32})
    book.append(second)
    assert SelectLedger.read(path) == (first, second)
    assert path.read_bytes().endswith(b"\n")
    assert not SelectLedger.read_status(path).dropped_last_line


@pytest.mark.parametrize("trial", [None, 79])
def test_selection_ledger_duplicate_is_rejected_on_append_and_read(tmp_path: Path, trial: int | None) -> None:
    from aosr.search.select import SelectLedger, SelectRow

    path = tmp_path / "ledger"
    book = SelectLedger(path)
    row = SelectRow.model_validate(_document(trial))
    book.append(row)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        book.append(row.model_copy(update={"run_id": "c" * 32}))
    assert path.read_bytes() == before
    with path.open("ab") as handle:
        handle.write((row.model_dump_json() + "\n").encode())
    with pytest.raises(ValueError):
        SelectLedger.read(path)


@pytest.mark.parametrize("bad_line", [b'{broken}\n', b'{"trial_number":true}\n'])
def test_selection_ledger_rejects_bad_middle_line(tmp_path: Path, bad_line: bytes) -> None:
    from aosr.search.select import SelectLedger

    path = tmp_path / "ledger"
    path.write_bytes(bad_line + (json.dumps(_document()) + "\n").encode())
    with pytest.raises(ValueError):
        SelectLedger.read(path)


def test_selection_ledger_append_is_locked_and_synced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl
    from aosr.search.select import SelectLedger, SelectRow

    path = tmp_path / "ledger"
    locked: list[int] = []
    durable: list[bytes] = []
    real_lock, real_sync = fcntl.flock, os.fsync

    def lock(fd: int, operation: int) -> None:
        real_lock(fd, operation)
        if operation == fcntl.LOCK_EX:
            assert os.fstat(fd).st_ino == path.stat().st_ino
            locked.append(fd)

    def sync(fd: int) -> None:
        assert fd in locked
        durable.append(path.read_bytes())
        real_sync(fd)

    monkeypatch.setattr(fcntl, "flock", lock)
    monkeypatch.setattr(os, "fsync", sync)
    row = SelectRow.model_validate(_document())
    SelectLedger(path).append(row)
    assert durable and durable[-1] == path.read_bytes()
    assert SelectLedger.read(path) == (row,)
