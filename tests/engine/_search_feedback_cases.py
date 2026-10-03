"""回饋考卷用真搜尋與細算，僅計算使用合成候選。"""

from pathlib import Path

from aosr.search.ledger import read_for
from aosr.search.run import SearchStatus, resume_search
from aosr.search.store import SearchStore
from tests.engine._search_refine_cases import RefineCompute, SearchCompute, refine
from tests.engine._search_run_cases import ENGINE, RUN_DATE, make_store, run


def prepared(tmp_path: Path, *, batch: int = 2, workers: int = 1,
             feedback: bool = True, offset: float = 0.125,
             refine_budget: int = 30, refine_convergence: int = 30,
             anchor_number: int | None = None) -> tuple[SearchStore, Path, SearchStatus]:
    store, registry = make_store(tmp_path, batch=batch, workers=workers, budget=40, convergence=7,
                                refine={"budget": refine_budget, "convergence_run": refine_convergence},
                                feedback={"offset": offset} if feedback else None)
    stopped = run(store, registry, SearchCompute(store, flat=True, persist_baseline=True))
    assert stopped.state == "converged"
    anchor = next(row.trial_number for row in reversed(read_for(store).rows)
                  if row.outcome == "scored" and row.trial_number != stopped.best_trial
                  and (anchor_number is None or row.trial_number == anchor_number))
    final = refine(store, registry, RefineCompute(store, {None: 4.0, anchor: 0.1}))
    assert final.refine.state == "stopped" and final.refine.best == anchor
    return store, registry, final


def snapshot(store: SearchStore) -> dict[str, bytes]:
    return {str(path.relative_to(store.path)): path.read_bytes()
            for path in store.path.rglob("*") if path.is_file()}


def resume(store: SearchStore, registry: Path, compute: SearchCompute) -> SearchStatus:
    return resume_search(store, compute=compute, probe=lambda: store.identity,
                         registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
