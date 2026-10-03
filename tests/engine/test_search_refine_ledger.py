"""細算帳逐列驗型別、尾列修復與篩選順序；全部用合成資料。"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aosr.search.ledger import LedgerRow
from aosr.search.store import JSON_SUFFIX

if TYPE_CHECKING:
    from aosr.search.refine import RefineHeader, RefineRow


def test_refine_ledger_contract_is_available(tmp_path: Path) -> None:
    assert importlib.util.find_spec("aosr.search.refine") is not None
    assert tmp_path.is_dir()


@pytest.fixture
def header(tmp_path: Path) -> RefineHeader:
    from aosr.search.refine import RefineHeader

    assert tmp_path.is_dir()
    return RefineHeader(ledger_version="aosr.search_refine.v1", search_id="search-test",
                        settings_fingerprint="settings-test", project_fingerprint="project-test",
                        purpose_fingerprint="purpose-test", physics_identity="physics-test",
                        program_fingerprint="program-test")


def refine_document(trial: int | None, *, round_number: int = 1) -> dict[str, object]:
    name = "baseline" if trial is None else f"trial-{trial:06d}"
    return {"round": round_number, "trial_number": trial, "result_file": f"verification/{name}{JSON_SUFFIX}",
            "outcome": "scored", "total_cost": 0.75, "seconds": 0.25}


def test_refine_ledger_round_trip_and_each_row_is_durable(
    tmp_path: Path, header: RefineHeader, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.search.refine import RefineLedger, RefineRow

    path = tmp_path / "ledger"
    rows = tuple(RefineRow.model_validate(document) for document in (
        refine_document(None), refine_document(79),
        refine_document(4) | {"outcome": "not_evaluated", "total_cost": None},
    ))
    expected: list[RefineRow] = []
    durable: list[tuple[RefineHeader, tuple[RefineRow, ...]]] = []
    real_fsync = os.fsync

    def observed_fsync(fd: int) -> None:
        assert os.fstat(fd).st_ino == path.stat().st_ino
        durable.append(RefineLedger.read(path))
        real_fsync(fd)

    monkeypatch.setattr("aosr.search.refine.os.fsync", observed_fsync)
    ledger = RefineLedger.create(path, header)
    assert durable == [(header, ())]
    for row in rows:
        expected.append(row)
        ledger.append(row)
        assert durable[-1] == (header, tuple(expected))
        assert RefineLedger.read(path) == (header, tuple(expected))
    assert RefineLedger.open(path).read(path) == (header, rows)
    assert not RefineLedger.read_status(path).dropped_last_line


# 第三組：內容是完整合法的一列、只差換行——寫入沒做完（換行與同步沒落地），照樣捨棄，不准當成有效列。
@pytest.mark.parametrize("tail", [b'{"round":', b'{broken}\n', json.dumps(refine_document(80)).encode("utf-8")])
def test_refine_ledger_drops_only_incomplete_tail_and_repairs_on_append(
    tmp_path: Path, header: RefineHeader, tail: bytes,
) -> None:
    from aosr.search.refine import RefineLedger, RefineRow

    path = tmp_path / "ledger"
    ledger = RefineLedger.create(path, header)
    first = RefineRow.model_validate(refine_document(79))
    ledger.append(first)
    complete_bytes = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(tail)
    status = RefineLedger.read_status(path)
    assert status.header == header
    assert status.rows == (first,)
    assert status.dropped_last_line
    assert status.valid_bytes == len(complete_bytes)
    assert RefineLedger.read(path) == (header, (first,))
    assert path.read_bytes() == complete_bytes + tail
    second = RefineRow.model_validate(refine_document(None))
    RefineLedger.open(path).append(second)
    assert RefineLedger.read(path) == (header, (first, second))
    assert not RefineLedger.read_status(path).dropped_last_line


@pytest.mark.parametrize("trial", [None, 79])
def test_refine_ledger_refines_each_candidate_once(tmp_path: Path, header: RefineHeader, trial: int | None) -> None:
    """每個候選（含原方案）整本帳只細算一次：同輪、下一輪都不准再寫；被拒時原檔不變、讀回時也擋。"""
    from aosr.search.refine import RefineLedger, RefineRow

    path = tmp_path / "ledger"
    ledger = RefineLedger.create(path, header)
    row = RefineRow.model_validate(refine_document(trial))
    ledger.append(row)
    before = path.read_bytes()
    for round_number in (1, 2):
        with pytest.raises(ValueError):
            ledger.append(RefineRow.model_validate(refine_document(trial, round_number=round_number)))
        assert path.read_bytes() == before
    with path.open("ab") as handle:
        handle.write((RefineRow.model_validate(refine_document(trial, round_number=2)).model_dump_json() + "\n").encode())
    with pytest.raises(ValueError):
        RefineLedger.open(path)


def test_refine_ledger_rounds_do_not_decrease(tmp_path: Path, header: RefineHeader) -> None:
    from aosr.search.refine import RefineLedger, RefineRow

    path = tmp_path / "ledger"
    ledger = RefineLedger.create(path, header)
    ledger.append(RefineRow.model_validate(refine_document(None)))
    ledger.append(RefineRow.model_validate(refine_document(79, round_number=2)))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        ledger.append(RefineRow.model_validate(refine_document(80, round_number=1)))
    assert path.read_bytes() == before
    ledger.append(RefineRow.model_validate(refine_document(80, round_number=2)))
    assert [row.trial_number for row in RefineLedger.read(path)[1]] == [None, 79, 80]


@pytest.mark.parametrize("change", [
    {"ledger_version": "future"}, {"search_id": ""}, {"physics_identity": 1},
    {"program_fingerprint": ""}, {"settings_fingerprint": ""}, {"project_fingerprint": ""},
    {"purpose_fingerprint": ""}, {"extra": True},
])
def test_refine_header_validation(tmp_path: Path, header: RefineHeader, change: dict[str, object]) -> None:
    from aosr.search.refine import RefineLedger

    path = tmp_path / "ledger"
    path.write_text(json.dumps(header.model_dump() | change) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        RefineLedger.open(path)


def test_refine_ledger_create_refuses_existing_file(tmp_path: Path, header: RefineHeader) -> None:
    from aosr.search.refine import RefineLedger

    path = tmp_path / "ledger"
    path.write_bytes(b"previous contents")
    with pytest.raises(FileExistsError):
        RefineLedger.create(path, header)
    assert path.read_bytes() == b"previous contents"


@pytest.mark.parametrize("change", [
    {"result_file": f"candidates/trial-000079{JSON_SUFFIX}"},
    {"result_file": f"/verification/trial-000079{JSON_SUFFIX}"},
    {"result_file": f"verification/../trial-000079{JSON_SUFFIX}"},
    {"result_file": f"verification/sub/trial-000079{JSON_SUFFIX}"},
    {"result_file": "verification"}, {"result_file": None},
    # 檔名跟編號綁死：別的編號、原方案、工作方案檔都不准。
    {"result_file": f"verification/trial-000080{JSON_SUFFIX}"}, {"result_file": f"verification/baseline{JSON_SUFFIX}"},
    {"result_file": f"verification/trial-000079-scheme{JSON_SUFFIX}"},
    {"total_cost": None}, {"outcome": "excluded"}, {"outcome": "not_evaluated"},
    {"outcome": "not_comparable"}, {"outcome": "illegal", "total_cost": None},
    {"round": 0}, {"round": True}, {"trial_number": -1}, {"trial_number": True},
    {"seconds": -0.1}, {"seconds": float("inf")}, {"total_cost": float("nan")}, {"extra": True},
])
def test_refine_row_rejects_invalid_fields(tmp_path: Path, change: dict[str, object]) -> None:
    from aosr.search.refine import RefineRow

    with pytest.raises(ValueError):
        RefineRow.model_validate(refine_document(79) | change)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("outcome", ["excluded", "not_evaluated", "not_comparable"])
def test_refine_non_scored_rows_have_no_total_cost(tmp_path: Path, header: RefineHeader, outcome: str) -> None:
    from aosr.search.refine import RefineLedger, RefineRow

    row = RefineRow.model_validate(refine_document(None) | {"outcome": outcome, "total_cost": None})
    path = tmp_path / "ledger"
    path.write_text(header.model_dump_json() + "\n" + row.model_dump_json() + "\n", encoding="utf-8")
    assert RefineLedger.read(path)[1] == (row,)


@pytest.mark.parametrize("field", [
    "ledger_version", "search_id", "settings_fingerprint", "project_fingerprint", "purpose_fingerprint",
    "physics_identity", "program_fingerprint",
])
def test_refine_header_requires_all_fields(tmp_path: Path, header: RefineHeader, field: str) -> None:
    from aosr.search.refine import RefineLedger

    document = header.model_dump()
    document.pop(field)
    path = tmp_path / "ledger"
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        RefineLedger.read(path)


@pytest.mark.parametrize("contents", [b"", b'{"ledger_version":'])
def test_refine_ledger_rejects_missing_or_incomplete_header(tmp_path: Path, contents: bytes) -> None:
    from aosr.search.refine import RefineLedger

    path = tmp_path / "ledger"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        RefineLedger.open(path)


@pytest.mark.parametrize("bad_line", [b'{"round":0}\n', b'{broken}\n'])
def test_refine_ledger_rejects_bad_middle_lines(tmp_path: Path, header: RefineHeader, bad_line: bytes) -> None:
    from aosr.search.refine import RefineLedger, RefineRow

    path = tmp_path / "ledger"
    RefineLedger.create(path, header)
    row = RefineRow.model_validate(refine_document(79))
    with path.open("ab") as handle:
        handle.write(bad_line + (row.model_dump_json() + "\n").encode())
    with pytest.raises(ValueError):
        RefineLedger.read(path)


def test_refine_ledger_rejects_semantically_invalid_tail(tmp_path: Path, header: RefineHeader) -> None:
    from aosr.search.refine import RefineLedger

    path = tmp_path / "ledger"
    RefineLedger.create(path, header)
    with path.open("ab") as handle:
        handle.write((json.dumps(refine_document(79) | {"total_cost": None}) + "\n").encode())
    with pytest.raises(ValueError):
        RefineLedger.read_status(path)


def screening_row(trial: int, outcome: str, score: float | None) -> LedgerRow:
    from aosr.search.layout import UNIT_SPACE
    from aosr.search.store import candidate_name

    illegal = outcome == "illegal"
    return LedgerRow.model_validate({
        "batch_index": 0, "trial_number": trial, "unit_params_hex": {key: 0.5.hex() for key in UNIT_SPACE},
        "params_m": {key: 1.0 for key in UNIT_SPACE}, "outcome": outcome, "score": score,
        "reason": "wall_gap" if illegal else "eliminated" if outcome == "excluded" else None,
        "violation_m": 0.25 if illegal else None, "seconds": 0.1,
        "result_file": None if illegal else candidate_name(trial),
    })


def test_refine_order_only_scored_sorted_by_score_then_trial(tmp_path: Path) -> None:
    from aosr.search.refine import refine_order

    rows = (screening_row(79, "scored", 0.5), screening_row(2, "excluded", None),
            screening_row(4, "scored", 0.5), screening_row(1, "illegal", None),
            screening_row(8, "scored", 0.2), screening_row(0, "scored", 0.8))
    assert refine_order(rows) == (8, 4, 79, 0)
    assert refine_order((rows[1], rows[3])) == ()
    assert refine_order(()) == ()
    assert tmp_path.is_dir()
