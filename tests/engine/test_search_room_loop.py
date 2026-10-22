"""第一階段每題一間房：提案、建題、落帳與接續直接核房間。"""
from collections.abc import Iterator, Sequence
from pathlib import Path
from uuid import UUID
import uuid

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.reporting.scheme import Scheme
from aosr.search import ledger, timings
from aosr.search.layout import params_from_unit, room_from_unit, room_start, standard_start
from aosr.search.run import CandidateJob, SearchStatus, _adapter, _status_message
from aosr.search.sampler import Proposal, SamplerAdapter
from aosr.search.stage_one_run import resume_stage_one_search
from aosr.search.stage_one_worker import StageOneComputed
from aosr.search.store import SearchStore
from tests.engine import _search_stage_one_cases as search, _stage_one_cases as stage
from tests.engine._search_room_loop_cases import NO_START, RANGES, ROOM_KEYS, layout_for


@pytest.mark.parametrize("envelope,message", [
    ({"Lz": RANGES["Lz"], "min_floor_area_m2": 26.0}, "地板兩邊都不可調：專案地板面積 25.83 m² 小於最小地板面積 26.0 m²"),
    (RANGES | {"min_floor_area_m2": 41.0}, "外框最大地板面積 40.0 m² 小於最小地板面積 41.0 m²"),
    (RANGES, "型號適用聆聽距離"),
])
def test_room_creation_runs_prechecks(tmp_path: Path, envelope: dict[str, object], message: str) -> None:
    fields = {"listening_range_m": {"low": 20., "high": 21.}} if message == "型號適用聆聽距離" else {}
    with pytest.raises(ValueError, match=message):
        search.make_store(tmp_path, layout=layout_for(room_range=envelope, **fields))
    assert not (tmp_path / "searches").exists()


@pytest.mark.parametrize("edges", [("Lx",), ("Ly", "Lz"), ("Lx", "Ly", "Lz")])
def test_room_folder_writes_v2_header(tmp_path: Path, edges: tuple[str, ...]) -> None:
    store = search.make_store(tmp_path, layout=layout_for(room_range={edge: RANGES[edge] for edge in edges}))
    ledger.create_for(store)
    header, rows = ledger.Ledger.read(store.ledger_path)
    assert isinstance(header, ledger.LedgerHeaderV2)
    assert header.ledger_version == "aosr.search_ledger.v2"
    placement = ("front_distance", "spacing", "listening_distance")
    assert header.meter_quantities == (*placement, *ROOM_KEYS)
    assert tuple(header.search_space) == (*placement, *(f"room_{edge}" for edge in edges))
    assert not rows


@pytest.mark.parametrize("damage,code", [("bom", "json_invalid"), ("string", "model_type")])
def test_v2_bad_header_keeps_original_error(tmp_path: Path, damage: str, code: str) -> None:
    import json
    store = search.make_store(tmp_path, layout=layout_for())
    ledger.create_for(store)
    line = store.ledger_path.read_bytes().strip()
    store.ledger_path.write_bytes((b"\xef\xbb\xbf" + line if damage == "bom" else json.dumps(line.decode()).encode()) + b"\n")
    with pytest.raises(ValidationError) as caught:
        ledger.Ledger.read(store.ledger_path)
    assert str(caught.value).startswith("1 validation error for LedgerHeader\n")
    assert [error["type"] for error in caught.value.errors()] == [code]


def test_start_roundtrip_must_preserve_every_room_bit() -> None:
    # 6.3 位於外框內，但除回單位再乘回得到 6.300000000000001；不可冒充專案原房間。
    low, high, original = 1.6521842340173616, 12.472361488351497, 6.3
    assert low + (original - low) / (high - low) * (high - low) != original
    chosen = layout_for(room_range={"Lx": {"low": low, "high": high}})
    assert standard_start(stage.scheme(), chosen) is not None
    assert room_start(stage.scheme(), chosen) is None


@pytest.mark.parametrize("envelope", [
    {"Lx": {"low": 7., "high": 8.}}, RANGES | {"min_floor_area_m2": 26.},
])
def test_no_room_start_message_is_added_once(tmp_path: Path, envelope: dict[str, object]) -> None:
    store = search.make_store(tmp_path, layout=layout_for(room_range=envelope))
    adapter, enqueued = _adapter(store)
    assert not enqueued
    status = _status_message(store, SearchStatus(message="搜尋停止"))
    assert status.message == "搜尋停止；" + NO_START
    assert _status_message(store, status) == status
    assert all(proposal.params for proposal in adapter.ask_batch(store.settings.batch_size))


def test_first_proposal_keeps_project_room_and_standard_placement(tmp_path: Path) -> None:
    store = search.make_store(tmp_path, layout=layout_for())
    adapter, enqueued = _adapter(store)
    assert enqueued
    first = adapter.ask_batch(store.settings.batch_size)[0]
    room = room_from_unit(first.params, store.settings.layout, store.project)
    original = store.project.scene.room_m
    assert (room.Lx.hex(), room.Ly.hex(), room.Lz.hex()) == (original.Lx.hex(), original.Ly.hex(), original.Lz.hex())
    assert params_from_unit(first.params, store.settings.layout) == standard_start(store.project, store.settings.layout)
    assert _status_message(store, SearchStatus(message="搜尋停止")).message == "搜尋停止"


def test_candidate_rooms_and_meters_follow_each_proposal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = search.make_store(tmp_path, layout=layout_for())
    fake = search.FakeCompute(store, monkeypatch)
    seen: dict[int, Scheme] = {}
    proposed: dict[int, Proposal] = {}
    ask = SamplerAdapter.ask_batch
    def capture(self: SamplerAdapter, size: int) -> tuple[Proposal, ...]:
        batch = ask(self, size)
        proposed.update((proposal.trial_number, proposal) for proposal in batch)
        return batch
    monkeypatch.setattr(SamplerAdapter, "ask_batch", capture)
    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[StageOneComputed]:
        for job in jobs:
            assert job.trial_number is not None
            expected = room_from_unit(proposed[job.trial_number].params, store.settings.layout, store.project)
            assert job.scheme.scene.room_m == expected
            assert job.scheme.receiver_set.primary.position_m[1] == expected.Ly / 2.
            seen[job.trial_number] = job.scheme
        yield from fake(jobs, workers)
    from aosr.search.stage_one_run import start_stage_one_search
    status = start_stage_one_search(store, compute=compute, probe=lambda: store.identity,
        registry_path=config_path("quality_targets.toml"), run_date=stage.run_kwargs()["run_date"],
        engine_version=store.identity.program_fingerprint)
    assert status.state == "budget_exhausted" and seen
    for row in ledger.read_for(store).rows:
        unit = proposed[row.trial_number].params
        assert row.unit_params_hex == {name: value.hex() for name, value in unit.items()}
        expected = room_from_unit(unit, store.settings.layout, store.project)
        edges = (expected.Lx, expected.Ly, expected.Lz)
        assert tuple(row.params_m[name].hex() for name in ROOM_KEYS) == tuple(value.hex() for value in edges)
        assert set(ROOM_KEYS) <= unit.keys()
        if row.trial_number in seen:
            actual = seen[row.trial_number].scene.room_m
            assert (actual.Lx.hex(), actual.Ly.hex(), actual.Lz.hex()) == tuple(value.hex() for value in edges)
    assert len({scheme.scene.room_m for scheme in seen.values()}) > 1


def test_floor_rejection_never_reaches_physics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = layout_for(room_range={"Lx": {"low": 5., "high": 9.}, "min_floor_area_m2": 28.},
        front_distance_m={"low": .5, "high": 1.}, spacing_m={"low": .5, "high": 1.},
        listening_distance_m={"low": .5, "high": 1.})
    store = search.make_store(tmp_path, layout=chosen)
    fake = search.FakeCompute(store, monkeypatch)
    search.run(store, fake)
    rows = ledger.read_for(store).rows
    below = [row for row in rows if row.params_m["room_Lx"] * row.params_m["room_Ly"] < 28.]
    assert below and fake.calls
    for row in below:
        assert row.outcome == "illegal" and row.reason == "floor_area_below_minimum"
        assert row.result_file is None and row.trial_number not in fake.calls


@pytest.mark.parametrize("kill_after", [1, 4])
def test_room_search_resume_is_bit_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kill_after: int) -> None:
    monkeypatch.setattr(uuid, "uuid4", lambda: UUID(int=123))
    monkeypatch.setattr(timings, "_now", lambda: 0.)
    control = search.make_store(tmp_path / "control", workers=3, layout=layout_for())
    interrupted = search.make_store(tmp_path / "interrupted", workers=3, layout=layout_for())
    search.run(control, search.FakeCompute(control, monkeypatch))
    with pytest.raises(search.Killed):
        search.run(interrupted, search.FakeCompute(interrupted, monkeypatch, kill_after=kill_after))
    reopened = type(interrupted).open(interrupted.path)
    resume_stage_one_search(reopened, compute=search.FakeCompute(reopened, monkeypatch), probe=lambda: reopened.identity,
        registry_path=config_path("quality_targets.toml"), run_date=stage.run_kwargs()["run_date"],
        engine_version=reopened.identity.program_fingerprint)
    def stable(store: SearchStore) -> list[str]:
        return sorted(row.model_dump_json() for row in ledger.read_for(store).rows)
    assert stable(control) == stable(interrupted)
    assert control.ledger_path.read_bytes() == interrupted.ledger_path.read_bytes()
    assert control.status_path.read_bytes() == interrupted.status_path.read_bytes()
    for row in ledger.read_for(control).rows:
        if row.result_file is not None:
            assert control.candidate_path(row.trial_number).read_bytes() == interrupted.candidate_path(row.trial_number).read_bytes()
