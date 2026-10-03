"""選中的細算結果才進清單：位元、時間、身分與選取歷史都要保住。"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.reporting.result import SchemeResult
from aosr.search.refine import RefineLedger
from aosr.search.store import JSON_SUFFIX, SearchStore, refine_result_name, refine_scheme_id
from tests.engine._search_select_cases import refined_store


def test_selection_contract_available(tmp_path: Path) -> None:
    assert importlib.util.find_spec("aosr.search.select") is not None
    assert tmp_path.is_dir()


def _trial(store: SearchStore) -> int:
    return next(row.trial_number for row in RefineLedger.read(store.refine_ledger_path)[1]
                if row.trial_number is not None)


@pytest.mark.parametrize("baseline", [False, True])
def test_selected_refinement_is_byte_exact_and_visible_in_gui(tmp_path: Path, baseline: bool) -> None:
    from aosr.gui.app import RUN_ID, _result_paths
    from aosr.gui.jobs import JobManager
    from aosr.gui.result_list import summarize_result
    from aosr.search.select import SelectLedger, select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    number = None if baseline else _trial(store)
    source = store.refine_result_path(number)
    os.utime(source, ns=(1_600_000_000_000_000_000, 1_600_000_000_123_456_789))
    before = store.status_path.read_bytes()
    outcome = select_refined(store, number, tmp_path / "data")
    assert outcome.is_new
    assert RUN_ID.fullmatch(outcome.run_id)
    assert uuid.UUID(hex=outcome.run_id).version == 4
    assert outcome.result_path == tmp_path / "data" / "results" / f"{outcome.run_id}{JSON_SUFFIX}"
    assert outcome.result_path.read_bytes() == source.read_bytes()
    assert outcome.result_path.stat().st_mtime_ns == source.stat().st_mtime_ns
    result = SchemeResult.model_validate_json(outcome.result_path.read_bytes())
    assert result.scheme.scheme_id == refine_scheme_id(store.search_id, number)
    assert result.scheme.scene.low_frequency_axis == LowFrequencyAxis.VERIFICATION
    assert result.origin.search_id == store.search_id and result.origin.trial_number == number
    assert result.origin.kind == ("search_baseline" if baseline else "search_candidate")
    summary = summarize_result(outcome.result_path, store.identity.physics_identity,
                               {store.project.purpose: store.identity.purpose_settings.fingerprint})
    assert summary.scheme_id.endswith("-verification")
    assert summary.duration_text == "0.250 秒" and summary.calculation_text == "跟現在相同"
    assert summary.finished_text != "讀不出" and summary.registry_text == "跟現在相同"
    assert _result_paths(tmp_path / "data") == [outcome.result_path]
    assert not (tmp_path / "data" / "runs").exists()
    manager = JobManager(tmp_path / "data", (), "test", tmp_path / "capabilities")
    assert manager.result_status(outcome.run_id).status == "none"
    assert not list((tmp_path / "data" / "runs").iterdir())
    assert store.status_path.read_bytes() == before
    row, = SelectLedger.read(selection_ledger_path(store))
    assert row.trial_number == number and row.result_file == refine_result_name(number)
    assert row.run_id == outcome.run_id and row.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()


def test_unreffined_or_absent_trials_leave_files_unchanged(tmp_path: Path) -> None:
    from aosr.search.select import select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    select_refined(store, None, tmp_path / "data")
    ledger = selection_ledger_path(store)
    before = ledger.read_bytes()
    files = set((tmp_path / "data").rglob("*"))
    refined = {row.trial_number for row in RefineLedger.read(store.refine_ledger_path)[1]}
    existing = next(path for path in (store.path / "candidates").glob(f"*{JSON_SUFFIX}")
                    if "scheme" not in path.stem and int(path.stem.split("-")[-1]) not in refined)
    for number in (int(existing.stem.split("-")[-1]), 999999):
        with pytest.raises(ValueError, match="沒細算過"):
            select_refined(store, number, tmp_path / "data")
        assert ledger.read_bytes() == before
        assert set((tmp_path / "data").rglob("*")) == files


def test_missing_refinement_ledger_creates_nothing(tmp_path: Path) -> None:
    from aosr.search.select import select_refined, selection_ledger_path
    from tests.engine._search_refine_cases import stopped_store

    store, _ = stopped_store(tmp_path)
    with pytest.raises(ValueError, match="沒細算過"):
        select_refined(store, None, tmp_path / "data")
    assert not (tmp_path / "data").exists()
    assert not selection_ledger_path(store).exists()


def test_repeat_selection_is_read_only(tmp_path: Path) -> None:
    from aosr.search.select import select_refined

    store = refined_store(tmp_path)
    first = select_refined(store, _trial(store), tmp_path / "data")
    paths = [path for root in (store.path, tmp_path / "data") for path in root.rglob("*") if path.is_file()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    repeat = select_refined(store, _trial(store), tmp_path / "data")
    assert not repeat.is_new and repeat.run_id == first.run_id and repeat.result_path == first.result_path
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths} == before
    assert set((tmp_path / "data" / "results").iterdir()) == {first.result_path}


@pytest.mark.parametrize("damage", ["deleted", "changed"])
def test_damaged_selected_target_is_rejected(tmp_path: Path, damage: str) -> None:
    from aosr.search.select import select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    first = select_refined(store, None, tmp_path / "data")
    if damage == "deleted":
        first.result_path.unlink()
    else:
        first.result_path.write_bytes(b"changed")
    before = selection_ledger_path(store).read_bytes()
    paths = set((tmp_path / "data" / "results").iterdir())
    with pytest.raises(ValueError, match="目標.*(不存在|不同)"):
        select_refined(store, None, tmp_path / "data")
    assert selection_ledger_path(store).read_bytes() == before
    assert set((tmp_path / "data" / "results").iterdir()) == paths


@pytest.mark.parametrize("field", ["physics_identity", "program_fingerprint", "purpose_settings", "scheme_id", "invalid"])
def test_invalid_refined_result_is_rejected_before_writing(tmp_path: Path, field: str) -> None:
    from aosr.search.select import select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    path = store.refine_result_path(None)
    document = json.loads(path.read_bytes())
    if field == "scheme_id":
        # 整份把細算代號換成搜尋那一份的代號（檔內一致、只少了 -verification）：要被方案代號那一關擋下。
        document = json.loads(path.read_text(encoding="utf-8").replace(
            f"{store.search_id}-baseline-verification", f"{store.search_id}-baseline"))
    elif field == "purpose_settings":
        content = document[field]["content"] | {"changed": True}
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        document[field] |= {"content": content, "fingerprint": digest}
    elif field == "invalid":
        document.pop("pairs")
    else:
        document[field] = ("phys-v1:" if field == "physics_identity" else "calc-v1:") + "c" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="方案代號" if field == "scheme_id" else ""):
        select_refined(store, None, tmp_path / "data")
    assert not (tmp_path / "data").exists() and not selection_ledger_path(store).exists()


def test_changed_source_digest_is_rejected(tmp_path: Path) -> None:
    from aosr.search.select import select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    first = select_refined(store, None, tmp_path / "data")
    ledger_before = selection_ledger_path(store).read_bytes()
    source = store.refine_result_path(None)
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        select_refined(store, None, tmp_path / "data")
    assert selection_ledger_path(store).read_bytes() == ledger_before
    assert set((tmp_path / "data" / "results").iterdir()) == {first.result_path}


@pytest.mark.parametrize("timing", ["before", "during_copy"])
def test_uuid_collision_never_overwrites_existing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timing: str,
) -> None:
    from aosr.search.select import select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    first = select_refined(store, None, tmp_path / "data")
    before = first.result_path.read_bytes(), selection_ledger_path(store).read_bytes()
    occupied_id = uuid.UUID("eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee")
    occupied = first.result_path.with_name(f"{occupied_id.hex}{JSON_SUFFIX}")
    real_copy = shutil.copy2

    def racing_copy(source: Path, target: Path) -> str:
        occupied.write_bytes(b"original")
        return real_copy(str(source), str(target))

    if timing == "before":
        occupied.write_bytes(b"original")
    else:
        monkeypatch.setattr("aosr.search.select.shutil.copy2", racing_copy)
    monkeypatch.setattr("aosr.search.select.uuid.uuid4", lambda: occupied_id)
    with pytest.raises(FileExistsError):
        select_refined(store, _trial(store), tmp_path / "data")
    assert (first.result_path.read_bytes(), selection_ledger_path(store).read_bytes()) == before
    assert occupied.read_bytes() == b"original"
    assert set((tmp_path / "data" / "results").iterdir()) == {first.result_path, occupied}


def test_source_changed_during_copy_is_not_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.search.select import SelectLedger, select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    real_copy = shutil.copy2

    def changing_copy(source: Path, target: Path) -> str:
        source.write_bytes(source.read_bytes() + b"\n")
        return real_copy(str(source), str(target))

    monkeypatch.setattr("aosr.search.select.shutil.copy2", changing_copy)
    with pytest.raises(ValueError, match="SHA-256"):
        select_refined(store, None, tmp_path / "data")
    assert not list((tmp_path / "data" / "results").iterdir())
    assert not SelectLedger.read(selection_ledger_path(store))


def test_simultaneous_selection_publishes_once(tmp_path: Path) -> None:
    from aosr.search.select import SelectLedger, select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(select_refined, store, None, tmp_path / "data") for _ in ("first", "second")]
        outcomes = [future.result() for future in futures]
    assert {outcome.is_new for outcome in outcomes} == {True, False}
    assert {outcome.run_id for outcome in outcomes} == {outcomes[0].run_id}
    assert set((tmp_path / "data" / "results").iterdir()) == {outcomes[0].result_path}
    row, = SelectLedger.read(selection_ledger_path(store))
    assert row.run_id == outcomes[0].run_id


def test_refinement_header_must_match_snapshot(tmp_path: Path) -> None:
    from aosr.search.select import select_refined, selection_ledger_path

    store = refined_store(tmp_path)
    lines = store.refine_ledger_path.read_bytes().splitlines(keepends=True)
    lines[0] = (json.dumps(json.loads(lines[0]) | {"physics_identity": "other"}) + "\n").encode()
    store.refine_ledger_path.write_bytes(b"".join(lines))
    with pytest.raises(ValueError, match="細算帳身分"):
        select_refined(store, None, tmp_path / "data")
    assert not (tmp_path / "data").exists() and not selection_ledger_path(store).exists()


@pytest.mark.parametrize("token", ["a" * 32, "0" * 32, "A" * 32, "f" * 31, "f" * 33, "g" * 32, "a" * 32 + "\n"])
def test_search_run_id_rule_matches_gui(tmp_path: Path, token: str) -> None:
    from aosr.gui.app import RUN_ID as GUI_RUN_ID
    from aosr.search.select import RUN_ID

    assert RUN_ID.pattern == GUI_RUN_ID.pattern
    assert bool(RUN_ID.fullmatch(token)) == bool(GUI_RUN_ID.fullmatch(token))
    assert tmp_path.is_dir()
