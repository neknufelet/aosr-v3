"""審查補題：每道計算契約與排名區都必須被考到。"""

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.search import run as search_run
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus, start_search
from aosr.search.sampler import Illegal, Outcome, Proposal, RankingZone, Scored
from aosr.search.store import SearchIdentity, SearchStore, candidate_name
from tests.engine._search_review_cases import with_matching
from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, make_store, rows, run


@pytest.mark.parametrize("baseline", (False, True))
def test_eliminated_candidate_keeps_its_zone_and_baseline_can_search(tmp_path: Path, baseline: bool) -> None:
    """防淘汰改報未評估或不能同表，也防淘汰原方案拒跑；舊替身從未交回踩底線的候選。"""
    store, registry = make_store(tmp_path)
    fake = FakeCompute(store)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in fake(jobs, workers):
            breached = result.job.trial_number is None if baseline else result.job.trial_number == 0
            yield replace(result, candidate=with_matching(result.candidate, breached=breached))

    status = start_search(store, compute=compute, probe=lambda: store.identity,
                          registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "budget_exhausted"
    if baseline:
        assert status.baseline_outcome == "eliminated"
        assert status.baseline_reason_codes == ("channel_matching_level_worst_beyond_limit",)
        assert any(row.outcome == "scored" for row in rows(store))
    else:
        eliminated = [row for row in rows(store) if row.reason == "eliminated"]
        assert eliminated and eliminated[0].trial_number == 0
        assert status.excluded["eliminated"] == len(eliminated)
        assert all(row.outcome == "excluded" and row.score is None for row in eliminated)


def test_missing_baseline_reports_category_and_reason(tmp_path: Path) -> None:
    """防缺類訊息只說排不進表而無法診斷；舊題只接受籠統失敗句。"""
    store, registry = make_store(tmp_path)
    status = run(store, registry, FakeCompute(store, missing=frozenset({None})))
    assert status.state == "failed"
    assert "timbre_balance" in status.message and "mandatory_category_missing" in status.message
    assert "定不出比較身分" in status.message
    assert not rows(store)


def test_convergence_at_exact_batch_boundary(tmp_path: Path) -> None:
    """防 >= 改 > 少算一次收斂；舊 K=3、收斂4 兩種寫法都在同批停止。"""
    store, registry = make_store(tmp_path, batch=3, convergence=2)
    status = run(store, registry, FakeCompute(store, flat=True))
    assert status.state == "converged"
    assert status.asked == store.settings.batch_size
    assert status.streak == store.settings.convergence_run


@pytest.mark.parametrize("field", ("physics_identity", "purpose_settings"))
def test_probe_checks_each_identity_component(tmp_path: Path, field: str) -> None:
    """防探針只比程式或忽略評分設定 M5；舊題只單獨變動程式指紋。"""
    from tests.engine._search_store_cases import purpose_settings

    store, registry = make_store(tmp_path)
    changed = (replace(store.identity, physics_identity="changed") if field == "physics_identity"
               else replace(store.identity, purpose_settings=purpose_settings(store.project.purpose)))

    def probe() -> SearchIdentity:
        return changed if rows(store) else store.identity

    status = start_search(store, compute=FakeCompute(store), probe=probe,
                          registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "interrupted"
    assert status.asked == store.settings.batch_size
    assert {row.batch_index for row in rows(store)} == {0}


@pytest.mark.parametrize("zone", (RankingZone.INCOMPARABLE, RankingZone.UNASSESSED))
def test_excluded_rankings_count_towards_streak(tmp_path: Path, zone: RankingZone) -> None:
    """防不能同表與缺類不算連續數；舊題只查排除區，沒有讓它們決定停止批次。"""
    store, registry = make_store(tmp_path, batch=3, convergence=2)
    numbers = frozenset({1, 2})
    compute = FakeCompute(store, flat=True, different=numbers if zone == RankingZone.INCOMPARABLE else frozenset(),
                          missing=numbers if zone == RankingZone.UNASSESSED else frozenset())
    status = run(store, registry, compute)
    assert status.state == "converged" and status.asked == store.settings.batch_size
    assert status.streak == store.settings.convergence_run
    assert {row.reason for row in rows(store) if row.outcome == "excluded"} == {zone.value}


@pytest.mark.parametrize("target", ("baseline", "trial"))
@pytest.mark.parametrize("violation", ("path", "scheme", "candidate_id", "missing_file"))
def test_compute_contract_rejects_changed_work_and_missing_results(tmp_path: Path, target: str, violation: str) -> None:
    """防 M2/M3 放過換工作、方案、錯代號與未存檔；舊計算失敗題只丟例外，沒違反交回契約。"""
    store, registry = make_store(tmp_path)
    fake = FakeCompute(store)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in fake(jobs, workers):
            if (result.job.trial_number is None) == (target == "baseline"):
                if violation == "missing_file":
                    result.job.result_path.unlink()
                elif violation == "path":
                    result = replace(result, job=replace(result.job, result_path=tmp_path / "wrong"))
                elif violation == "scheme":
                    scheme = result.job.scheme.model_copy(update={"source_model": "omnidirectional"})
                    result = replace(result, job=replace(result.job, scheme=scheme))
                else:
                    result = replace(result, candidate=result.candidate.model_copy(update={"candidate_id": "wrong"}))
            yield result

    match = "沒有保存" if violation == "missing_file" else "工作或候選代號"
    with pytest.raises(ValueError, match=match):
        start_search(store, compute=compute, probe=lambda: store.identity,
                     registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).state == "failed"


def test_duplicate_baseline_is_rejected(tmp_path: Path) -> None:
    """防 M6 原方案重複交回被忽略；舊重複試算檢查不經原方案路徑。"""
    store, registry = make_store(tmp_path)
    fake = FakeCompute(store)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for result in fake(jobs, workers):
            yield result
            yield result

    with pytest.raises(ValueError, match="重複原方案"):
        start_search(store, compute=compute, probe=lambda: store.identity,
                     registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()).state == "failed"


def test_each_completed_batch_is_saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """防刪掉每批完成後寫狀態而只在停止時寫；舊題只讀最後狀態看不出中途缺檔。"""
    store, registry = make_store(tmp_path, budget=7)
    original = search_run._write_status
    captured: list[SearchStatus] = []

    def write(current: SearchStore, status: SearchStatus) -> SearchStatus:
        saved = original(current, status)
        assert SearchStatus.model_validate_json(current.status_path.read_bytes()) == status
        captured.append(status)
        return saved

    monkeypatch.setattr(search_run, "_write_status", write)
    run(store, registry, FakeCompute(store))
    ends = {min((row.batch_index + 1) * store.settings.batch_size, store.settings.budget) for row in rows(store)}
    completed = {status.asked for status in captured if status.state == "running" and status.computed == status.asked}
    assert ends <= completed


def test_illegal_reason_codes_are_counted_separately(tmp_path: Path) -> None:
    """防多原因整串被當一個代碼；舊題只查原因字典非空，不能驗拆分。"""
    from aosr.search.ledger import row_from_outcome

    assert tmp_path.is_dir()
    meters = {"front_distance": 1.0, "spacing": 1.2, "listening_distance": 2.0}
    codes = ("cabinet_outside_room", "wall_gap")
    outcomes: tuple[Outcome, ...] = (Illegal("+".join(codes), 0.1), Illegal(codes[0], 0.2), Scored(1.0))
    built = tuple(row_from_outcome(batch_index=0, proposal=Proposal(n, dict.fromkeys(meters, 0.5)),
                                  params_m=meters, outcome=outcome, seconds=0.0,
                                  result_file=candidate_name(n) if isinstance(outcome, Scored) else None)
                  for n, outcome in enumerate(outcomes))
    status = search_run._progress(SearchStatus(), built)
    assert status.illegal_reasons == {code: sum(isinstance(outcome, Illegal) and code in outcome.reason.split("+")
                                               for outcome in outcomes) for code in codes}
