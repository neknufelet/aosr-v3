"""帳本版本由表頭決定；第二版每列依自己的取樣界限與公尺量清單驗。"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.quality_targets import QualityTargets
from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import CandidateEvaluation
from aosr.scoring.ranking import ComparisonIdentity
from aosr.search.layout import SEARCH_QUANTITIES, LayoutParams, meter_quantities_for, params_to_meters
from aosr.search.ledger import (
    Ledger, LedgerHeader, LedgerHeaderV2, LedgerRow, LedgerRowV2, create_for, header_for, read_for, replay_history,
    row_outcome,
)
from aosr.search.sampler import Excluded, Outcome, Proposal, RankingZone, SamplerAdapter, Scored
from aosr.search.store import candidate_name
from tests.engine._search_ledger_v2_cases import METERS, SPACE, header_document, row_document, store_in
from tests.engine._search_run_cases import FakeCompute, run


def test_extended_space_selects_v2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = store_in(tmp_path)
    monkeypatch.setattr("aosr.search.ledger.space_for", lambda settings: SPACE)
    header = header_for(store)
    assert header.ledger_version == "aosr.search_ledger.v2"
    assert header.search_space == SPACE
    assert header.model_dump()["meter_quantities"] == SEARCH_QUANTITIES
    create_for(store)
    assert read_for(store).header == header


@pytest.mark.parametrize("locked", [False, True])
def test_today_search_stays_v1_bit_exact(tmp_path: Path, locked: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = store_in(tmp_path, locked=locked)
    header = header_for(store)
    project = store.project.model_dump(mode="json")
    if project["furniture"] is None:
        project.pop("furniture")
    expected = {"ledger_version": "aosr.search_ledger.v1", "search_id": store.search_id,
        "settings_fingerprint": store.settings.fingerprint,
        "project_fingerprint": hashlib.sha256(json.dumps(project, sort_keys=True,
            separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "sampler": {"seed": 0, "n_startup_trials": 2, "constant_liar": True},
        "batch_size": store.settings.batch_size,
        "search_space": {name: (0.0, 1.0) for name in
            (("front_distance", "spacing") if locked else ("front_distance", "spacing", "listening_distance"))},
        "physics_identity": store.identity.physics_identity,
        "program_fingerprint": store.identity.program_fingerprint,
        "purpose_fingerprint": store.identity.purpose_settings.fingerprint}
    assert header.model_dump() == expected
    assert header.model_dump().keys() == LedgerHeader.model_fields.keys()
    proposals: dict[int, dict[str, str]] = {}
    original = SamplerAdapter.ask_batch

    def observe(adapter: SamplerAdapter, size: int) -> tuple[Proposal, ...]:
        result = original(adapter, size)
        proposals.update({p.trial_number: {key: value.hex() for key, value in p.params.items()} for p in result})
        return result

    def screen(candidate: CandidateEvaluation, scheme: Scheme, *, registry: QualityTargets,
               run_date: date, engine_version: str, pinned: tuple[ComparisonIdentity, ...] | None,
               ) -> tuple[Outcome, tuple[ComparisonIdentity, ...]]:
        return Scored(0.5), ()

    # 本題只驗取樣與落帳；既有計算替身搭配固定評分，不讀工作區的品質登記簿。
    monkeypatch.setattr(SamplerAdapter, "ask_batch", observe)
    monkeypatch.setattr("aosr.search.run.screening_outcome", screen)
    status = run(store, registry, FakeCompute(store))
    assert status.state == "budget_exhausted"
    lines = store.ledger_path.read_bytes().splitlines(keepends=True)
    assert lines[0] == (json.dumps(header.model_dump(mode="json"), ensure_ascii=False, allow_nan=False) + "\n").encode()
    assert json.loads(lines[0])["ledger_version"] == "aosr.search_ledger.v1"
    rows = read_for(store).rows
    # 非空：預算內每一題都要題、都落帳，空跑不准綠。
    assert rows and len(rows) == len(proposals) == store.settings.budget
    assert {row.trial_number: row.unit_params_hex for row in rows} == proposals
    assert all(type(row) is LedgerRow for row in rows)


def bits(proposals: Sequence[Proposal]) -> tuple[tuple[int, dict[str, str]], ...]:
    return tuple((p.trial_number, {key: value.hex() for key, value in p.params.items()}) for p in proposals)


def test_v2_roundtrip_and_partial_replay(tmp_path: Path) -> None:
    header = LedgerHeaderV2.model_validate(header_document())
    book_path = tmp_path / "ledger"
    book = Ledger.create(book_path, header)
    adapter = SamplerAdapter(header.search_space, header.sampler)
    complete = adapter.ask_batch(header.batch_size)
    saved = tuple(LedgerRowV2.model_validate(row_document(p)) for p in reversed(complete))
    for row in saved:
        book.append(row)
    outcomes = {p.trial_number: row_outcome(row) for p, row in zip(reversed(complete), saved, strict=True)}
    adapter.tell_batch(outcomes)
    next_batch = adapter.ask_batch(header.batch_size)
    partial = LedgerRowV2.model_validate(row_document(next_batch[0], batch=1))
    Ledger.open(book_path).append(partial)
    status = Ledger.read_status(book_path)
    assert status.header == header
    assert status.rows == (*saved, partial)
    assert tuple(status.header.search_space) == tuple(SPACE)
    assert status.header.model_dump()["meter_quantities"] == METERS
    assert status.header.model_dump().keys() == LedgerHeader.model_fields.keys() | {"meter_quantities"}
    assert all(isinstance(row, LedgerRow) for row in status.rows)
    history, unfinished = replay_history(header, status.rows)
    assert bits(history[0][0]) == bits(complete)
    assert history[0][1] == outcomes
    assert unfinished == (partial,)
    replayed = SamplerAdapter(header.search_space, header.sampler)
    replayed.replay(history)
    assert bits(replayed.ask_batch(header.batch_size)) == bits(next_batch)
    for proposal in next_batch[1:]:
        book.append(LedgerRowV2.model_validate(row_document(proposal, batch=1)))
    history, unfinished = replay_history(header, Ledger.read(book_path)[1])
    assert not unfinished
    assert bits(history[-1][0]) == bits(next_batch)


INVALID_ROWS = (
    ("missing_unit", "unit_params_hex", {"front_distance": 0.25.hex(), "spacing": 0.5.hex(), "listening_distance": 0.75.hex()}),
    ("extra_unit", "unit_params_hex", {**dict.fromkeys(SPACE, 0.5.hex()), "extra": 0.5.hex()}),
    ("below_header", "unit_params_hex", {**dict.fromkeys(SPACE, 0.5.hex()), "room_width": (-2.5).hex()}),
    ("above_header", "unit_params_hex", {**dict.fromkeys(SPACE, 0.5.hex()), "room_width": 3.5.hex()}),
    ("noncanonical", "unit_params_hex", {**dict.fromkeys(SPACE, 0.5.hex()), "room_width": "0x1p+0"}),
    ("infinite_unit", "unit_params_hex", {**dict.fromkeys(SPACE, 0.5.hex()), "room_width": "inf"}),
    ("malformed_unit", "unit_params_hex", {**dict.fromkeys(SPACE, 0.5.hex()), "room_width": "invalid"}),
    ("overflow_unit", "unit_params_hex", {**dict.fromkeys(SPACE, 0.5.hex()), "room_width": "0x1p+999999"}),
    ("missing_meter", "params_m", {name: 1.0 for name in METERS[:-1]}),
    ("extra_meter", "params_m", {**dict.fromkeys(METERS, 1.0), "extra": 1.0}),
    ("zero_meter", "params_m", {**dict.fromkeys(METERS, 1.0), "derived_depth": 0.0}),
    ("negative_meter", "params_m", {**dict.fromkeys(METERS, 1.0), "derived_depth": -1.0}),
    ("infinite_meter", "params_m", {**dict.fromkeys(METERS, 1.0), "derived_depth": float("inf")}),
)


@pytest.mark.parametrize("operation", ["read", "append", "replay"])
@pytest.mark.parametrize("case,field,value", INVALID_ROWS, ids=[item[0] for item in INVALID_ROWS])
def test_v2_rejects_rows_against_header(tmp_path: Path, operation: str, case: str,
                                       field: str, value: object) -> None:
    header = LedgerHeaderV2.model_validate(header_document())
    path = tmp_path / "ledger"
    book = Ledger.create(path, header)
    before = path.read_bytes()
    document = row_document() | {field: value}
    with pytest.raises(ValueError):
        if operation == "read":
            with path.open("a") as stream:
                stream.write(json.dumps(document) + "\n")
            Ledger.read(path)
        else:
            # model_copy（複製時略過驗證）模擬被改過的列；操作入口仍得重驗。
            row = LedgerRowV2.model_validate(row_document()).model_copy(update={field: value})
            if operation == "append":
                book.append(row)
            else:
                replay_history(header, (row,))
    if operation == "append":
        assert path.read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("search_space", {}), ("search_space", {"x": (1.0, 1.0)}),
    ("search_space", {"x": (2.0, 1.0)}), ("search_space", {"x": (float("-inf"), 1.0)}),
    ("search_space", {"": (0.0, 1.0)}),
    ("meter_quantities", ()), ("meter_quantities", ("x", "x")), ("meter_quantities", ("",)),
    ("meter_quantities", {"x", "y"}),
])
def test_v2_header_rejects_invalid_quantity_contract(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        LedgerHeaderV2.model_validate(header_document() | {field: value})


@pytest.mark.parametrize("version", ["aosr.search_ledger.v2", "unexpected-ledger-version"])
def test_reader_dispatches_version_without_hiding_unknown(tmp_path: Path, version: str) -> None:
    path = tmp_path / "ledger"
    document = header_document() | {"ledger_version": version}
    path.write_text(json.dumps(document) + "\n")
    if version == "aosr.search_ledger.v2":
        assert Ledger.read(path)[0].ledger_version == version
    else:
        with pytest.raises(ValueError, match=version):
            Ledger.read(path)


def test_versions_do_not_accept_each_others_header_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LedgerHeaderV2.model_validate(header_document() | {"ledger_version": "aosr.search_ledger.v1"})
    store, _ = store_in(tmp_path)
    with pytest.raises(ValueError):
        LedgerHeader.model_validate(header_for(store).model_dump() | {"meter_quantities": METERS})
    path = tmp_path / "ledger"
    path.write_text(json.dumps(header_document() | {"ledger_version": "aosr.search_ledger.v1"}) + "\n")
    with pytest.raises(ValueError):
        Ledger.read(path)


def test_v2_preserves_other_row_rules(tmp_path: Path) -> None:
    header = LedgerHeaderV2.model_validate(header_document())
    path = tmp_path / "ledger"
    book = Ledger.create(path, header)
    for kind, updates in (
        ("scored", {"score": 0.5, "reason": None, "violation_m": None, "result_file": candidate_name(0)}),
        ("excluded", {"reason": "unassessed", "violation_m": None, "result_file": candidate_name(0)}),
    ):
        row = LedgerRowV2.model_validate(row_document() | {"outcome": kind} | updates)
        assert row_outcome(row) == (Scored(0.5) if kind == "scored" else Excluded(RankingZone.UNASSESSED))
    for update in ({"seconds": -1.0}, {"reason": "unknown"}, {"violation_m": 0.0},
                   {"score": 0.5}, {"result_file": candidate_name(1)}):
        with pytest.raises(ValueError):
            book.append(LedgerRowV2.model_validate(row_document()).model_copy(update=update))


@pytest.mark.parametrize("endpoint", [-2.0, 3.0])
def test_v2_accepts_header_endpoints_outside_unit_interval(tmp_path: Path, endpoint: float) -> None:
    header = LedgerHeaderV2.model_validate(header_document())
    document = row_document(Proposal(0, {"front_distance": 0.0, "spacing": 1.0,
                                        "listening_distance": 0.5, "room_width": endpoint}))
    row = LedgerRowV2.model_validate(document)
    path = tmp_path / "ledger"
    book = Ledger.create(path, header)
    book.append(row)
    history, partial = replay_history(header, Ledger.read(path)[1], batch_sizes={0: 1})
    assert not partial
    assert history[0][0][0].params["room_width"].hex() == endpoint.hex()


def test_read_for_compares_v2_meter_order_and_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _ = store_in(tmp_path)
    original = header_for(store)
    changed = LedgerHeaderV2.model_validate(original.model_dump() | {
        "ledger_version": "aosr.search_ledger.v2", "search_space": SPACE,
        "meter_quantities": SEARCH_QUANTITIES})
    Ledger.create(store.ledger_path, changed)
    with pytest.raises(ValueError):
        read_for(store)
    monkeypatch.setattr("aosr.search.ledger.space_for", lambda settings: SPACE)
    assert read_for(store).header == changed
    document = changed.model_dump(mode="json") | {"meter_quantities": list(reversed(SEARCH_QUANTITIES))}
    store.ledger_path.write_text(json.dumps(document) + "\n")
    with pytest.raises(ValueError, match="meter_quantities"):
        read_for(store)


def test_v1_validators_still_reject_extended_quantities(tmp_path: Path) -> None:
    store, _ = store_in(tmp_path)
    with pytest.raises(ValueError):
        LedgerHeader.model_validate(header_for(store).model_dump() | {"search_space": SPACE})
    document = row_document()
    with pytest.raises(ValueError):
        LedgerRow.model_validate(document)
    with pytest.raises(ValueError):
        LedgerRow.model_validate(document | {"unit_params_hex": {
            "front_distance": 0.25.hex(), "spacing": 0.5.hex(), "listening_distance": 0.75.hex()}})


@pytest.mark.parametrize("locked", [False, True])
def test_meter_source_preserves_names_order_and_bits(tmp_path: Path, locked: bool) -> None:
    store, _ = store_in(tmp_path, locked=locked)
    params = LayoutParams(0.1, 1.2345678901234567, 2.73125)
    expected = dict(zip(SEARCH_QUANTITIES,
                        (params.front_distance_m, params.spacing_m, params.listening_distance_m), strict=True))
    actual = params_to_meters(params)
    assert meter_quantities_for(store.settings.layout) == tuple(expected)
    assert tuple(actual) == tuple(expected)
    assert {key: value.hex() for key, value in actual.items()} == {key: value.hex() for key, value in expected.items()}


V1_UNIT = {"front_distance": 0.25.hex(), "spacing": 0.5.hex(), "listening_distance": 0.75.hex()}
V1_METERS = {"front_distance": 0.5, "spacing": 1.0, "listening_distance": 2.0}


@pytest.mark.parametrize("state", ["missing", "empty", "valid"])
def test_v1_bad_row_rejected_before_ledger_is_touched(tmp_path: Path, state: str) -> None:
    """改動前：壞的第一版列在開檔之前就被列驗證拒收（不論帳本缺檔、空檔或正常）。"""
    store, _ = store_in(tmp_path)
    path = tmp_path / "ledger"
    if state == "empty":
        path.write_bytes(b"")
    if state == "valid":
        Ledger.create(path, header_for(store))
    before = path.read_bytes() if path.exists() else None
    good = LedgerRow.model_validate(row_document() | {"unit_params_hex": V1_UNIT, "params_m": V1_METERS})
    with pytest.raises(ValidationError, match="seconds"):
        Ledger(path).append(good.model_copy(update={"seconds": -1.0}))
    assert (path.read_bytes() if path.exists() else None) == before


@pytest.mark.parametrize("first_line", [
    b'{"ledger_version": "aosr.search_ledger.v1",}\n', b"\xff\n", b"{}\n", b"[]\n",
    b'{"ledger_version": "unexpected-ledger-version"}\n',
    b'{"ledger_version": "aosr.search_ledger.v1", "extra": ' + b"[" * 20000 + b"0" + b"]" * 20000 + b"}\n",
], ids=["trailing_comma", "bad_utf8", "empty_object", "array", "unknown_version", "deep_nesting"])
def test_unreadable_header_keeps_v1_validation_error(tmp_path: Path, first_line: bytes) -> None:
    """改動前表頭一律走第一版的 JSON 驗證入口：壞 JSON、壞編碼、缺欄位、未知版號都是驗證錯誤。"""
    path = tmp_path / "ledger"
    path.write_bytes(first_line)
    # 錯誤由第一版表頭類別報（改動前的入口），不是看版號那一步自己的錯。
    with pytest.raises(ValidationError, match="for LedgerHeader\n"):
        Ledger.read(path)


def test_tampered_v1_header_keeps_v1_validation_error(tmp_path: Path) -> None:
    store, _ = store_in(tmp_path)
    tampered = header_for(store).model_copy(update={"ledger_version": "unexpected-ledger-version"})
    with pytest.raises(ValidationError, match="unexpected-ledger-version"):
        Ledger.create(tmp_path / "ledger", tampered)
    with pytest.raises(ValidationError, match="unexpected-ledger-version"):
        replay_history(tampered, ())
    assert not (tmp_path / "ledger").exists()
