"""審查 A/B 的接續補題：快照、原方案快取與停止邊界。"""

import json
from pathlib import Path

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis
from aosr.search import run as search_run
from aosr.search.layout_settings import Span
from aosr.search.run import SearchStatus, resume_search
from aosr.search.sampler import Outcome, SamplerAdapter
from aosr.search.store import PROJECT_FILE, SearchStore
from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, Killed, make_store, next_params, rows, run


def resume(store: SearchStore, registry: Path, compute: FakeCompute) -> SearchStatus:
    return resume_search(store, compute=compute, probe=lambda: store.identity,
                         registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)


@pytest.mark.parametrize("field", ("source_model", "seat", "low_frequency_axis"))
def test_changed_project_snapshot_interrupts_freshly_opened_store(tmp_path: Path, field: str) -> None:
    """防接續拿新磁碟快照自比而混帳；既有題只改表頭，沒有重開已改過的專案方案。"""
    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=5, kill=True, persist_baseline=True))
    project = store.project
    if field == "seat":
        seats = list(project.receiver_set.points)
        seat = seats[-1]
        seats[-1] = seat.model_copy(update={"position_m": (seat.position_m[0], seat.position_m[1] + 0.3, seat.position_m[2])})
        project = project.model_copy(update={"receiver_set": project.receiver_set.model_copy(update={"points": tuple(seats)})})
    elif field == "source_model":
        project = project.model_copy(update={"source_model": "omnidirectional"})
    else:
        project = project.model_copy(update={"scene": project.scene.model_copy(update={"low_frequency_axis": LowFrequencyAxis.VERIFICATION})})
    (store.path / PROJECT_FILE).write_text(project.model_dump_json(), encoding="utf-8")
    reopened = SearchStore.open(store.path)
    before = store.ledger_path.read_bytes()
    compute = FakeCompute(reopened)
    status = resume(reopened, registry, compute)
    assert status.state == "interrupted" and "project_fingerprint" in status.message
    assert store.ledger_path.read_bytes() == before
    assert not compute.calls


def test_project_fingerprint_is_canonical_sha256(tmp_path: Path) -> None:
    """防專案指紋未排序、未壓緊 JSON 或不是 SHA-256；改快照題只查變動，抓不到摘要格式偏離。"""
    import hashlib
    from aosr.search.ledger import header_for

    store, _ = make_store(tmp_path)
    canonical = json.dumps(store.project.model_dump(mode="json"), sort_keys=True,
                           separators=(",", ":"), allow_nan=False)
    assert header_for(store).project_fingerprint == hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@pytest.mark.parametrize("point", ("tell", "batch_status", "terminal_status"))
def test_last_batch_convergence_survives_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, point: str) -> None:
    """防末批同時收斂與用完預算時接續回錯原因；既有接續題沒有在這三個邊界砍掉。"""
    whole, registry = make_store(tmp_path / "whole", budget=6, batch=3, convergence=5)
    split, other = make_store(tmp_path / "split", budget=6, batch=3, convergence=5)
    expected = run(whole, registry, FakeCompute(whole, flat=True))
    original_tell, original_write = SamplerAdapter.tell_batch, search_run._write_status

    def tell(adapter: SamplerAdapter, outcomes: dict[int, Outcome]) -> None:
        if rows(split) and max(row.batch_index for row in rows(split)) > 0:
            raise Killed()
        original_tell(adapter, outcomes)

    def write(store: SearchStore, status: SearchStatus) -> SearchStatus:
        if store == split and status.asked == split.settings.budget:
            if point == "batch_status" and status.computed == status.asked and status.state == "running":
                raise Killed()
            if point == "terminal_status" and status.state == "converged":
                raise Killed()
        return original_write(store, status)

    with monkeypatch.context() as patch:
        patch.setattr(SamplerAdapter, "tell_batch", tell if point == "tell" else original_tell)
        patch.setattr(search_run, "_write_status", write)
        with pytest.raises(Killed):
            run(split, other, FakeCompute(split, flat=True, persist_baseline=True))
    assert resume(split, other, FakeCompute(split, flat=True)) == expected
    assert rows(split) == rows(whole)
    assert next_params(split) == next_params(whole)


@pytest.mark.parametrize("damage", ("half", "empty", "invalid", "wrong_id", "missing_package", "missing_identity"))
def test_unreadable_baseline_is_recomputed(tmp_path: Path, damage: str) -> None:
    """防原方案快取損壞永久判失敗或接受錯代號；既有接續只考檔案存在與完全不在。"""
    whole, registry = make_store(tmp_path / "whole")
    split, other = make_store(tmp_path / "split")
    expected = run(whole, registry, FakeCompute(whole))
    with pytest.raises(Killed):
        run(split, other, FakeCompute(split, fail_after=5, kill=True, persist_baseline=True))
    text = split.baseline_path.read_text(encoding="utf-8")
    if damage == "wrong_id":
        text = text.replace(f"{split.search_id}-baseline", "wrong-baseline")
    elif damage == "missing_identity":
        text = json.dumps({"candidate": json.loads(text)["candidate"]})
    else:
        text = {"half": text[:len(text) // 2], "empty": "", "invalid": "not json", "missing_package": "{}"}[damage]
    split.baseline_path.write_text(text, encoding="utf-8")
    compute = FakeCompute(split, persist_baseline=True)
    assert resume(split, other, compute) == expected
    assert None in compute.calls
    document = json.loads(split.baseline_path.read_bytes())
    assert document["candidate"]["candidate_id"] == f"{split.search_id}-baseline"
    assert rows(split) == rows(whole)


@pytest.mark.parametrize("outside", (False, True))
def test_snapshot_failure_preserves_previous_status(tmp_path: Path, outside: bool) -> None:
    """防讀快照失敗把計數、最佳、起點歸零或漏訊息；既有題只斷言中斷標籤。"""
    # 前牆原距離在範圍外才確實沒有起點；間距現在會夾到可行區間。
    changes: dict[str, object] | None = {"front_distance_m": Span(low=1.5, high=2.0)} if outside else None
    store, registry = make_store(tmp_path, layout_changes=changes)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=5, kill=True, persist_baseline=True))
    previous = SearchStatus.model_validate_json(store.status_path.read_bytes())
    (store.path / PROJECT_FILE).unlink()
    status = resume(store, registry, FakeCompute(store))
    assert status.state == "interrupted"
    assert status.model_dump(exclude={"state", "message"}) == previous.model_dump(exclude={"state", "message"})
    note = "起點不在搜尋範圍或夾角、耳距限制內，沒有排入"
    assert (note in status.message) == outside
    assert status == SearchStatus.model_validate_json(store.status_path.read_bytes())


def test_partial_batch_must_finish_even_when_saved_rows_reach_convergence(tmp_path: Path) -> None:
    """防 M1 移除未完成批保護而提前收斂；既有接續題的部分列尚未達收斂數。"""
    whole, registry = make_store(tmp_path / "whole", batch=3, convergence=4)
    split, other = make_store(tmp_path / "split", batch=3, convergence=4)
    expected = run(whole, registry, FakeCompute(whole, flat=True))
    with pytest.raises(Killed):
        run(split, other, FakeCompute(split, flat=True, fail_after=5, kill=True, persist_baseline=True))
    saved = rows(split)
    compute = FakeCompute(split, flat=True)
    assert resume(split, other, compute) == expected
    assert rows(split) == rows(whole)
    assert set(compute.calls) - {None} == {row.trial_number for row in rows(whole)} - {row.trial_number for row in saved}


def test_completed_batch_convergence_is_respected_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """防接續迴圈入口 >= 改 > 多要一批；新跑邊界題走批尾，末批題走預算尾，都考不到此入口。"""
    whole, registry = make_store(tmp_path / "whole", batch=3, convergence=2)
    split, other = make_store(tmp_path / "split", batch=3, convergence=2)
    expected = run(whole, registry, FakeCompute(whole, flat=True))
    original = search_run._write_status

    def write(store: SearchStore, status: SearchStatus) -> SearchStatus:
        if status.state == "converged":
            raise Killed()
        return original(store, status)

    with monkeypatch.context() as patch:
        patch.setattr(search_run, "_write_status", write)
        with pytest.raises(Killed):
            run(split, other, FakeCompute(split, flat=True, persist_baseline=True))
    compute = FakeCompute(split, flat=True)
    assert resume(split, other, compute) == expected
    assert not compute.calls
    assert rows(split) == rows(whole)
    assert next_params(split) == next_params(whole)


@pytest.mark.parametrize("state", ("converged", "budget_exhausted", "user_stopped", "failed", "interrupted"))
def test_every_terminal_state_refuses_resume(tmp_path: Path, state: search_run.State) -> None:
    """防 M4 只擋兩種正常停止卻重開失敗或中斷；既有題只考預算停止。"""
    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=1, kill=True, persist_baseline=True))
    previous = SearchStatus.model_validate_json(store.status_path.read_bytes())
    store.status_path.write_text(previous.model_copy(update={"state": state}).model_dump_json(), encoding="utf-8")
    compute = FakeCompute(store)
    with pytest.raises(ValueError, match="停止"):
        resume(store, registry, compute)
    assert not compute.calls


@pytest.mark.parametrize("damage", ["broken-json", "unknown-state"])
def test_damaged_status_file_refuses_resume(tmp_path: Path, damage: str) -> None:
    """狀態檔壞掉或寫著認不得的狀態，不准落回「進行中」：已失敗的搜尋會被悄悄重試（複查抓到的退步）。"""
    from aosr.search.run import resume_search

    store, registry = make_store(tmp_path)
    with pytest.raises(RuntimeError):
        run(store, registry, FakeCompute(store, fail_after=4))
    text = store.status_path.read_text(encoding="utf-8")
    store.status_path.write_text(text[: len(text) // 2] if damage == "broken-json"
                                 else text.replace('"failed"', '"paused"'), encoding="utf-8")
    compute = FakeCompute(store)
    with pytest.raises(ValueError, match="狀態檔"):
        resume_search(store, compute=compute, probe=lambda: store.identity,
                      registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert not compute.calls


@pytest.mark.parametrize("field", ("physics_identity", "program_fingerprint", "purpose_settings"))
def test_saved_baseline_from_other_program_interrupts_resume(tmp_path: Path, field: str) -> None:
    """接續讀回的原方案也核身分：別版程式算的原方案不能被釘成比較基準（第 4b 步修補審查）。"""
    from tests.engine.test_search_result_identity import changed_identity

    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=5, kill=True, persist_baseline=True))
    document = json.loads(store.baseline_path.read_bytes())
    changed = changed_identity(store.identity, field)
    document[field] = (changed.purpose_settings.model_dump(mode="json") if field == "purpose_settings"
                       else getattr(changed, field))
    store.baseline_path.write_text(json.dumps(document), encoding="utf-8")
    before = SearchStatus.model_validate_json(store.status_path.read_bytes())
    compute = FakeCompute(store, persist_baseline=True)
    status = resume(store, registry, compute)
    assert status.state == "interrupted"
    assert "原方案" in status.message and field in status.message
    assert None not in compute.calls
    # 中斷在重播之前：上一份狀態的進度照留，不歸零。
    assert (status.asked, status.start_enqueued, status.baseline_outcome) == (
        before.asked, before.start_enqueued, before.baseline_outcome)
    assert before.asked and before.start_enqueued


@pytest.mark.parametrize("value", (None, 7))
def test_saved_baseline_with_broken_identity_type_is_recomputed(tmp_path: Path, value: object) -> None:
    """身分欄型別壞掉＝快取讀不回，重算原方案，不把搜尋判成中斷（第 4b 步修補複查）。"""
    store, registry = make_store(tmp_path)
    with pytest.raises(Killed):
        run(store, registry, FakeCompute(store, fail_after=5, kill=True, persist_baseline=True))
    document = json.loads(store.baseline_path.read_bytes())
    document["physics_identity"] = value
    store.baseline_path.write_text(json.dumps(document), encoding="utf-8")
    compute = FakeCompute(store, persist_baseline=True)
    status = resume(store, registry, compute)
    assert status.state != "interrupted"
    assert None in compute.calls
