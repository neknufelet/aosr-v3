"""第二階段搜尋的時間預算（老闆 2026-10-24，#558 房 109 受控試跑：搜尋三小時）；只用假計算。

到預算在批界停、標因預算停止，自動外圈接著細算與附件；回饋那一輪的搜尋時間也算進同一份預算。
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.search import ledger, timings
from aosr.search.placement_stability_record import read_summary
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus, _write_status
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchStore
from tests.engine._search_outer_cases import CompleteOuterCompute
from tests.engine._search_run_cases import make_store

HOUR = 3600.


class TimedOuterCompute(CompleteOuterCompute):
    """每一批搜尋算完，替身時鐘走一小時；細算不動時鐘，搜尋牆鐘只跟搜尋批數有關。"""

    def __init__(self, store: SearchStore, clock: list[float]) -> None:
        super().__init__(store)
        self.clock = clock
        self.search_batches = 0

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        yield from super().__call__(jobs, workers)
        if jobs[0].result_path.parent != self.refine.store.refine_dir:
            self.clock[0] += HOUR
            self.search_batches += 1


def _seed(tmp_path: Path, budget_s: float | None) -> tuple[SearchStore, Path]:
    """照命令列 start 的寫法開好資料夾：狀態檔從開搜尋起就有計時。"""
    base, registry = make_store(tmp_path / "base", batch=2, budget=40, convergence=7,
                                refine={"budget": 20, "convergence_run": 2}, feedback={"offset": 0.125})
    extra = {} if budget_s is None else {"search_time_budget_s": budget_s}
    settings = SearchSettings.model_validate(base.settings.model_dump(mode="json") | extra)
    # 細算替身寫的是全向聲源的結果（照 test_search_outer_auto.py 的完整附件那題）；方案跟著改，擺位穩定性才讀得回。
    project = base.project.model_copy(update={"source_model": "omnidirectional"})
    store = SearchStore.create(tmp_path / "searches", project=project, settings=settings, identity=base.identity,
                               versions={"python": "test", "numpy": "test", "optuna": "test"})
    ledger.create_for(store)
    _write_status(store, SearchStatus(timed_from_start=True))
    return store, registry


def _invoke(store: SearchStore, registry: Path, monkeypatch: pytest.MonkeyPatch, compute: TimedOuterCompute) -> int:
    """照 test_search_outer_auto.py 完整附件那題走命令列 auto：能力表用產品預設，附件才真的跑。"""
    from aosr.config.paths import config_path
    from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState
    from aosr.search import cli
    from tests.engine._modal_cases import runner
    from tests.engine._stability_attach_cases import ShiftCompute
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    return cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(store.path.parent / "cache")],
        compute_factory=lambda *a: compute, stability_compute_factory=lambda root: ShiftCompute(store),
        modal_runner=runner(store.path.parent / "modal-runner", ModalDiagnosis(state=ModalDiagnosisState.NOT_COMPUTED)))


def _auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget_s: float | None) -> tuple[SearchStore, SearchStatus, TimedOuterCompute]:
    clock = [0.]
    monkeypatch.setattr(timings, "_now", lambda: clock[0])
    store, registry = _seed(tmp_path, budget_s)
    compute = TimedOuterCompute(store, clock)
    code = _invoke(store, registry, monkeypatch, compute)
    assert code == 0
    return store, SearchStatus.model_validate_json(store.status_path.read_bytes()), compute


def test_unbudgeted_control_feeds_back_once_and_completes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """控制組：沒設預算時這組假計算第一輪收斂、回饋一輪、外圈完成；下面兩題的預算照這裡的搜尋秒數取。"""
    _, status, compute = _auto(tmp_path, monkeypatch, None)
    assert status.outer.conclusion == "complete" and status.round == 2
    assert status.search_seconds[1] > 2 * HOUR and status.search_seconds[2] >= HOUR
    assert sum(status.search_seconds.values()) == compute.search_batches * HOUR


# 第一輪中途用完：細算第一名不是篩選第一名，本該回饋卻被預算擋下；第一輪收斂、回饋那一輪才用完：細算第一名沒換，標穩定但搜尋預算用完。
@pytest.mark.parametrize("budget_round,conclusion", [(1, "feedback_blocked_by_budget"), (2, "stable_but_search_budget")])
def test_time_budget_stops_search_then_auto_refines_and_attaches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 budget_round: int, conclusion: str) -> None:
    """兩種都停在批界、標因預算停止，細算照跑、附件照補、不再回饋；外圈不寫完成。"""
    _, control, _ = _auto(tmp_path / "control", monkeypatch, None)
    budget_s = control.search_seconds[1] / 2 if budget_round == 1 else control.search_seconds[1] + HOUR / 2
    store, status, compute = _auto(tmp_path / "budgeted", monkeypatch, budget_s)
    used = sum(status.search_seconds.values())
    assert status.state == "budget_exhausted" and status.round == budget_round
    assert budget_s <= used < budget_s + HOUR and used == compute.search_batches * HOUR
    assert status.message.startswith(f"因預算停止：搜尋時間預算 {budget_s / HOUR:g} 小時用完")
    assert status.refine.state == "stopped" and status.refine.round == budget_round
    assert status.outer.conclusion == conclusion
    summary = read_summary(store.path)
    assert summary is not None and summary.state == "done" and summary.completed


REFUSED = "先前用掉的搜尋時間追不回來，拒絕接續"


def _files(store: SearchStore) -> dict[Path, bytes]:
    return {path.relative_to(store.path): path.read_bytes() for path in store.path.rglob("*") if path.is_file()}


def test_resume_refuses_budgeted_search_whose_status_was_rebuilt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture[str]) -> None:
    """狀態檔遺失後重建（不是從開搜尋起就計時）：先前用掉的搜尋時間追不回來，拒絕接續、不寫任何檔。"""
    store, registry = _seed(tmp_path, 2 * HOUR)
    _write_status(store, SearchStatus())
    before = _files(store)
    code = _invoke(store, registry, monkeypatch, TimedOuterCompute(store, [0.]))
    assert code == 1 and REFUSED in capsys.readouterr().err
    assert _files(store) == before


def test_resume_refuses_when_only_the_baseline_was_saved_before_the_status_was_lost(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """原方案單獨存、不佔帳本列：原方案算完、第一個候選還沒落帳就被砍，狀態檔又遺失，帳本沒有列也不准當成從頭計時。"""
    from aosr.search import cli
    from aosr.search.run import resume_search
    from tests.engine._search_run_cases import ENGINE, RUN_DATE, Killed
    clock = [0.]
    monkeypatch.setattr(timings, "_now", lambda: clock[0])
    store, registry = _seed(tmp_path, 2 * HOUR)
    killed = TimedOuterCompute(store, clock)
    killed.kill_after = 1
    with pytest.raises(Killed):
        resume_search(store, compute=killed, probe=lambda: store.identity, registry_path=registry,
                      run_date=RUN_DATE, engine_version=ENGINE)
    assert store.baseline_path.is_file() and not ledger.read_for(store).rows
    store.status_path.unlink()
    before = _files(store)
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    code = cli.main(["resume", str(store.path), "--engine-commit", "test"], compute_factory=lambda *a: TimedOuterCompute(store, clock))
    assert code == 1 and REFUSED in capsys.readouterr().err
    assert _files(store) == before


def test_baseline_that_already_spent_the_budget_asks_no_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """原方案那一題算完就超過預算：批界先記上原方案的時間，一批都不派，標因預算停止，細算照接。"""
    _, status, compute = _auto(tmp_path, monkeypatch, HOUR / 2)
    assert status.state == "budget_exhausted" and status.asked == 0
    assert compute.search_batches == 1 and status.search_seconds[1] == HOUR
    assert status.refine.state == "stopped" and status.outer.conclusion != "complete"


def test_stage_two_settings_take_the_time_budget_but_not_the_candidate_cap(tmp_path: Path) -> None:
    store, _ = make_store(tmp_path)
    document = store.settings.model_dump(mode="json")
    assert "stage" not in document and "search_time_budget_s" not in document
    budgeted = SearchSettings.model_validate(document | {"search_time_budget_s": 3 * HOUR})
    assert budgeted.fingerprint != store.settings.fingerprint
    assert SearchSettings.model_validate(budgeted.model_dump(mode="json")) == budgeted
    with pytest.raises(ValueError, match="單題搜尋成本上限只准第一階段"):
        SearchSettings.model_validate(document | {"stage_one_candidate_cap_s": 60.})
