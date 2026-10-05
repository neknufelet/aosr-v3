"""目前最佳圖的搜尋樣本：只有控制組結果，全部寫在考卷暫存目錄。"""
from pathlib import Path
from typing import Literal

import pytest

from aosr.reporting.result import SchemeResult, save_result
from aosr.search import ledger
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.refine_run import header_for
from aosr.search.run import RefineStatus, SearchStatus
from aosr.search.store import SearchStore, candidate_name, refine_result_name
from tests.engine._search_run_cases import make_store
from tests.engine.test_scheme_pipeline import shared_control_result


@pytest.fixture(scope="module")
def best_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def best_store(tmp_path: Path, result: SchemeResult, refined: int | Literal["baseline"] | None = None) -> SearchStore:
    store, _ = make_store(tmp_path)
    book = ledger.create_for(store)
    params = {"front_distance": 1.0, "spacing": 2.0, "listening_distance": 3.0}
    book.append(ledger.LedgerRow(batch_index=0, trial_number=0,
                               unit_params_hex={key: (0.5).hex() for key in params}, params_m=params,
                               outcome="scored", score=0.5, reason=None, violation_m=None,
                               seconds=1.0, result_file=candidate_name(0)))
    save_result(result, store.candidate_path(0))
    refine = RefineStatus()
    if refined is not None:
        trial = None if refined == "baseline" else int(refined)
        store.ensure_refine_dir()
        save_result(result, store.refine_result_path(trial))
        RefineLedger.create(store.refine_ledger_path, header_for(store)).append(
            RefineRow(round=1, trial_number=trial, result_file=refine_result_name(trial),
                      outcome="scored", total_cost=0.25, seconds=1.0))
        refine = RefineStatus(state="running", best=refined, best_total_cost=0.25)
    store.status_path.write_text(SearchStatus(state="budget_exhausted", best_trial=0,
                                             best_score=0.5, refine=refine).model_dump_json())
    return store
