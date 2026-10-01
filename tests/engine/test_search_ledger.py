"""搜尋帳本的考卷：每列落地、截斷辨識與逐位重播。"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aosr.search.layout import UNIT_SPACE, params_from_unit
from aosr.search.sampler import Excluded, Illegal, Outcome, Proposal, RankingZone, ReplayMismatch, SamplerAdapter, Scored
from tests.engine._search_store_cases import purpose_settings, settings_document

if TYPE_CHECKING:
    from aosr.search.ledger import Ledger, LedgerHeader, LedgerRow
    from aosr.search.settings import SearchSettings


def test_ledger_contract_is_available(tmp_path: Path) -> None:
    """缺少帳本介面時明確失敗。"""
    assert tmp_path.is_dir()
    assert importlib.util.find_spec("aosr.search.ledger") is not None


@pytest.fixture
def settings(tmp_path: Path) -> SearchSettings:
    from aosr.search.settings import SearchSettings

    assert tmp_path.is_dir()
    return SearchSettings.model_validate(settings_document())


@pytest.fixture
def header(tmp_path: Path, settings: SearchSettings) -> LedgerHeader:
    from aosr.search.ledger import LEDGER_VERSION, LedgerHeader

    assert tmp_path.is_dir()
    return LedgerHeader(
        ledger_version=LEDGER_VERSION, search_id="search-test", settings_fingerprint=settings.fingerprint,
        sampler=settings.sampler_settings(), batch_size=settings.batch_size, search_space=dict(UNIT_SPACE),
        physics_identity="physical-test", program_fingerprint="program-test",
        purpose_fingerprint=purpose_settings(settings.purpose).fingerprint,
    )


def make_row(settings: SearchSettings, batch_index: int, proposal: Proposal, outcome: Outcome) -> LedgerRow:
    from aosr.search.ledger import row_from_outcome
    from aosr.search.store import CANDIDATES_DIR, JSON_SUFFIX

    params = params_from_unit(proposal.params, settings.layout)
    result = None if isinstance(outcome, Illegal) else f"{CANDIDATES_DIR}/trial-{proposal.trial_number:06d}{JSON_SUFFIX}"
    return row_from_outcome(
        batch_index=batch_index, proposal=proposal, params_m={
            "front_distance": params.front_distance_m, "spacing": params.spacing_m,
            "listening_distance": params.listening_distance_m,
        }, outcome=outcome, seconds=0.25, result_file=result,
    )


def write_batches(ledger: Ledger, settings: SearchSettings, *, partial: bool = False) -> tuple[SamplerAdapter, tuple[LedgerRow, ...]]:
    adapter = SamplerAdapter(UNIT_SPACE, settings.sampler_settings())
    rows = []
    for batch_index in range(3):
        proposals = adapter.ask_batch(settings.batch_size)
        outcomes: dict[int, Outcome] = {
            proposals[0].trial_number: Scored(0.5 + batch_index),
            proposals[1].trial_number: Illegal("wall", 0.125),
            proposals[2].trial_number: Excluded(RankingZone.UNASSESSED),
        }
        chosen = proposals[:2] if partial and batch_index == 2 else proposals
        # 完成順序可以不同於要題順序；重播仍必須照試算編號。刻意打亂成「第 2、1、3 個」：
        # 單純倒序的話，「同批內倒過來」的錯法再倒一次就剛好變回照編號，抓不到。
        for proposal in (chosen[1], chosen[0], *chosen[2:]):
            row = make_row(settings, batch_index, proposal, outcomes[proposal.trial_number])
            ledger.append(row)
            rows.append(row)
        if partial and batch_index == 2:
            break
        adapter.tell_batch(outcomes)
    return adapter, tuple(rows)


def proposal_bits(proposals: Sequence[Proposal]) -> tuple[tuple[int, dict[str, str]], ...]:
    return tuple((p.trial_number, {key: value.hex() for key, value in p.params.items()}) for p in proposals)


def test_row_written_per_candidate_and_survives_crash(
    tmp_path: Path, header: LedgerHeader, settings: SearchSettings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.search.ledger import Ledger

    path = tmp_path / "ledger"
    ledger = Ledger.create(path, header)
    adapter = SamplerAdapter(UNIT_SPACE, settings.sampler_settings())
    expected: list[LedgerRow] = []
    synced: list[tuple[LedgerRow, ...]] = []
    real_fsync = os.fsync

    def observe_fsync(fd: int) -> None:
        synced.append(Ledger.read(path)[1])
        real_fsync(fd)

    monkeypatch.setattr("aosr.search.ledger.os.fsync", observe_fsync)
    outcomes: tuple[Outcome, ...] = (Scored(0.5), Illegal("wall", 0.125), Excluded(RankingZone.ELIMINATED))
    for proposal, outcome in zip(adapter.ask_batch(settings.batch_size), outcomes, strict=True):
        row = make_row(settings, 0, proposal, outcome)
        expected.append(row)
        previous = tuple(synced)
        ledger.append(row)
        assert tuple(synced) == (*previous, tuple(expected))
        assert Ledger.read(path) == (header, tuple(expected))
        assert row.unit_params_hex == {key: value.hex() for key, value in proposal.params.items()}
    # ledger 仍存在，沒有呼叫任何關檔動作；另一位讀者已能核對所有已落地列。
    assert synced[-1] == tuple(expected)


def test_truncated_last_line_is_dropped_but_middle_corruption_is_refused(
    tmp_path: Path, header: LedgerHeader, settings: SearchSettings,
) -> None:
    from aosr.search.ledger import Ledger

    path = tmp_path / "ledger"
    _, rows = write_batches(Ledger.create(path, header), settings)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-1]) + lines[-1][:len(lines[-1]) // 2])
    status = Ledger.read_status(path)
    assert status.rows == rows[:-1]
    assert status.header == header
    assert status.dropped_last_line
    assert Ledger.read(path) == (header, rows[:-1])
    # 有完整 JSON 但缺結尾換行，同樣不是已完成寫入的列。
    path.write_bytes(b"".join(lines).rstrip(b"\n"))
    assert Ledger.read_status(path).rows == rows[:-1]
    assert Ledger.read_status(path).dropped_last_line
    path.write_bytes(b"".join((*lines[:-1], b"{broken\n")))
    assert Ledger.read_status(path).rows == rows[:-1]
    assert Ledger.read_status(path).dropped_last_line
    path.write_bytes(b"".join((lines[0], b"{broken\n", *lines[2:])))
    with pytest.raises(ValueError):
        Ledger.read(path)


def test_replay_reproduces_next_batch_and_detects_changed_record(
    tmp_path: Path, header: LedgerHeader, settings: SearchSettings,
) -> None:
    from aosr.search.ledger import Ledger, replay_history

    path = tmp_path / "ledger"
    original, expected = write_batches(Ledger.create(path, header), settings)
    read_header, rows = Ledger.read(path)
    assert rows == expected
    history, partial = replay_history(read_header, rows)
    assert partial == ()
    assert tuple(p.trial_number for proposals, _ in history for p in proposals) == tuple(sorted(r.trial_number for r in rows))
    resumed = SamplerAdapter(UNIT_SPACE, settings.sampler_settings())
    resumed.replay(history)
    assert proposal_bits(resumed.ask_batch(settings.batch_size)) == proposal_bits(original.ask_batch(settings.batch_size))
    lines = path.read_text(encoding="utf-8").splitlines()
    changed = json.loads(lines[-1])
    name = next(iter(UNIT_SPACE))
    value = float.fromhex(changed["unit_params_hex"][name])
    changed["unit_params_hex"][name] = math.nextafter(value, math.inf).hex()
    lines[-1] = json.dumps(changed)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    changed_header, changed_rows = Ledger.read(path)
    changed_history, _ = replay_history(changed_header, changed_rows)
    fresh = SamplerAdapter(UNIT_SPACE, settings.sampler_settings())
    with pytest.raises(ReplayMismatch):
        fresh.replay(changed_history)


def test_partial_batch_is_returned_separately(tmp_path: Path, header: LedgerHeader, settings: SearchSettings) -> None:
    from aosr.search.ledger import Ledger, replay_history

    path = tmp_path / "ledger"
    _, rows = write_batches(Ledger.create(path, header), settings, partial=True)
    history, partial = replay_history(*Ledger.read(path))
    assert tuple(tuple(p.trial_number for p in proposals) for proposals, _ in history) == ((0, 1, 2), (3, 4, 5))
    assert partial == tuple(r for r in rows if r.batch_index == 2)
    assert {r.trial_number for r in partial} == {6, 7}
    resumed = SamplerAdapter(UNIT_SPACE, settings.sampler_settings())
    resumed.replay(history)
    proposed = {p.trial_number: p for p in resumed.ask_batch(settings.batch_size)}
    for row in partial:
        assert row.unit_params_hex == {key: value.hex() for key, value in proposed[row.trial_number].params.items()}


def test_budget_shortened_final_batch_needs_explicit_size(
    tmp_path: Path, header: LedgerHeader, settings: SearchSettings,
) -> None:
    from aosr.search.ledger import Ledger, replay_history

    path = tmp_path / "ledger"
    _, rows = write_batches(Ledger.create(path, header), settings, partial=True)
    default_history, partial = replay_history(header, rows)
    history, completed_partial = replay_history(header, rows, batch_sizes={2: 2})
    assert completed_partial == ()
    assert history[:-1] == default_history
    assert tuple(p.trial_number for p in history[-1][0]) == tuple(sorted(r.trial_number for r in partial))


@pytest.mark.parametrize("outcome", [Scored(0.5), Illegal("wall", 0.125), Excluded(RankingZone.INCOMPARABLE)])
def test_outcome_round_trip(tmp_path: Path, settings: SearchSettings, outcome: Outcome) -> None:
    from aosr.search.ledger import row_outcome

    proposal = Proposal(0, dict.fromkeys(UNIT_SPACE, 0.5))
    row = make_row(settings, 0, proposal, outcome)
    assert row_outcome(row) == outcome
    assert row.unit_params_hex == dict.fromkeys(UNIT_SPACE, 0.5.hex())
    assert row.params_m == {"front_distance": 1.125, "spacing": 1.25, "listening_distance": 1.75}
    assert tmp_path.is_dir()


@pytest.mark.parametrize("outcome,change", [
    (Scored(0.5), {"score": None}), (Scored(0.5), {"score": math.inf}),
    (Scored(0.5), {"reason": "wall"}), (Scored(0.5), {"violation_m": 0.1}),
    (Scored(0.5), {"result_file": None}), (Illegal("wall", 0.125), {"violation_m": None}),
    (Illegal("wall", 0.125), {"violation_m": 0.0}), (Illegal("wall", 0.125), {"violation_m": -0.1}),
    (Illegal("wall", 0.125), {"violation_m": math.inf}), (Illegal("wall", 0.125), {"score": 0.5}),
    (Illegal("wall", 0.125), {"reason": None}), (Illegal("wall", 0.125), {"reason": ""}),
    (Illegal("wall", 0.125), {"result_file": "candidates/result"}),
    (Excluded(RankingZone.UNASSESSED), {"score": 0.5}),
    (Excluded(RankingZone.UNASSESSED), {"reason": "wall"}),
    (Excluded(RankingZone.UNASSESSED), {"reason": None}),
    (Excluded(RankingZone.UNASSESSED), {"violation_m": 0.1}),
    (Excluded(RankingZone.UNASSESSED), {"result_file": None}),
    (Scored(0.5), {"result_file": "../outside"}), (Scored(0.5), {"result_file": "/candidates/result"}),
    (Scored(0.5), {"result_file": "candidates/../outside"}),
    (Scored(0.5), {"seconds": -0.1}), (Scored(0.5), {"seconds": math.nan}),
    (Scored(0.5), {"batch_index": -1}), (Scored(0.5), {"trial_number": -1}),
    (Scored(0.5), {"extra": True}), (Scored(0.5), {"unit_params_hex": {"front_distance": "0.5"}}),
    (Scored(0.5), {"params_m": dict.fromkeys(UNIT_SPACE, math.nan)}),
])
def test_row_consistency_rules(tmp_path: Path, settings: SearchSettings, outcome: Outcome, change: dict[str, object]) -> None:
    from aosr.search.ledger import LedgerRow

    row = make_row(settings, 0, Proposal(0, dict.fromkeys(UNIT_SPACE, 0.5)), outcome)
    with pytest.raises(ValueError):
        LedgerRow.model_validate(row.model_dump() | change)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("changed", [{"batch_index": 0, "trial_number": 1}, {"batch_index": 1, "trial_number": 0}])
def test_batch_order_and_duplicate_trials_are_refused(
    tmp_path: Path, header: LedgerHeader, settings: SearchSettings, changed: dict[str, int],
) -> None:
    from aosr.search.ledger import Ledger, LedgerRow

    path = tmp_path / "ledger"
    ledger = Ledger.create(path, header)
    first = make_row(settings, 1, Proposal(0, dict.fromkeys(UNIT_SPACE, 0.5)), Scored(0.5))
    ledger.append(first)
    bad = LedgerRow.model_validate(first.model_dump() | changed)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        ledger.append(bad)
    assert path.read_bytes() == before
    path.write_bytes(before + bad.model_dump_json().encode() + b"\n")
    with pytest.raises(ValueError):
        Ledger.read(path)


def test_existing_ledger_is_not_overwritten(tmp_path: Path, header: LedgerHeader) -> None:
    from aosr.search.ledger import Ledger

    path = tmp_path / "ledger"
    Ledger.create(path, header)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        Ledger.create(path, header)
    assert path.read_bytes() == before


def test_semantically_invalid_last_line_is_refused(tmp_path: Path, header: LedgerHeader) -> None:
    from aosr.search.ledger import Ledger

    path = tmp_path / "ledger"
    Ledger.create(path, header)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"batch_index": -1}) + "\n")
    with pytest.raises(ValueError):
        Ledger.read(path)


@pytest.mark.parametrize("sizes", [{2: 0}, {2: 4}, {-1: 2}])
def test_invalid_batch_sizes_are_refused(
    tmp_path: Path, header: LedgerHeader, settings: SearchSettings, sizes: dict[int, int],
) -> None:
    from aosr.search.ledger import Ledger, replay_history

    _, rows = write_batches(Ledger.create(tmp_path / "ledger", header), settings)
    with pytest.raises(ValueError):
        replay_history(header, rows, batch_sizes=sizes)


def test_incomplete_middle_batch_is_refused(tmp_path: Path, header: LedgerHeader, settings: SearchSettings) -> None:
    from aosr.search.ledger import Ledger, replay_history

    _, rows = write_batches(Ledger.create(tmp_path / "ledger", header), settings)
    with pytest.raises(ValueError):
        replay_history(header, rows[1:])


def test_reopened_ledger_repairs_only_the_interrupted_tail(
    tmp_path: Path, header: LedgerHeader, settings: SearchSettings,
) -> None:
    from aosr.search.ledger import Ledger

    path = tmp_path / "ledger"
    _, rows = write_batches(Ledger.create(path, header), settings)
    lines = path.read_bytes().splitlines(keepends=True)
    prefix = b"".join(lines[:-1])
    path.write_bytes(prefix + lines[-1][:len(lines[-1]) // 2])
    reopened = Ledger.open(path)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        reopened.append(rows[0])
    assert path.read_bytes() == before
    reopened.append(rows[-1])
    assert path.read_bytes().startswith(prefix)
    assert Ledger.read(path) == (header, rows)
    assert not Ledger.read_status(path).dropped_last_line


@pytest.mark.parametrize("change", [
    {"ledger_version": "future-version"}, {"extra": True}, {"batch_size": 0},
    {"search_space": {"x": (0.0, 1.0)}}, {"sampler": {"seed": True, "n_startup_trials": 2, "constant_liar": True}},
    {"sampler": {"seed": 0, "n_startup_trials": 2, "constant_liar": True, "extra": True}},
])
def test_header_validation(tmp_path: Path, header: LedgerHeader, change: dict[str, object]) -> None:
    from aosr.search.ledger import LedgerHeader

    with pytest.raises(ValueError):
        LedgerHeader.model_validate(header.model_dump() | change)
    with pytest.raises(ValueError):
        setattr(header, "batch_size", 4)
    assert tmp_path.is_dir()


def test_append_cuts_an_interrupted_tail_longer_than_the_new_row(
    tmp_path: Path, header: LedgerHeader, settings: SearchSettings,
) -> None:
    """截斷的尾巴比新的一列長時，只覆寫不裁掉會留下殘渣：下一次讀回又是一行不完整的尾巴。"""
    from aosr.search.ledger import Ledger

    path = tmp_path / "ledger"
    _, rows = write_batches(Ledger.create(path, header), settings)
    lines = path.read_bytes().splitlines(keepends=True)
    tail = lines[-1].rstrip(b"\n")
    assert len(tail) > 10, "樣本列太短，這一題沒在考"
    path.write_bytes(b"".join(lines[:-1]) + tail + tail)  # 寫到一半的垃圾比一整列還長、沒有結尾換行
    Ledger.open(path).append(rows[-1])
    assert path.read_bytes() == b"".join(lines)
    assert Ledger.read(path) == (header, rows)
    assert not Ledger.read_status(path).dropped_last_line
