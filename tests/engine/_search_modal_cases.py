"""搜尋附件考卷：真帳本與明列擺位，僅模態求解使用替身。"""
from __future__ import annotations

import json
from pathlib import Path

from aosr.reporting.scheme import Scheme, load_scheme
from aosr.search.outer import conclude
from aosr.search.outer_status import OuterConclusion
from aosr.search.run import SearchStatus
from aosr.search.store import SearchStore
from tests.engine._search_refine_cases import RefineCompute, SearchCompute, refine
from tests.engine._search_run_cases import make_store, run


def prepared(folder: Path, *, refined: bool = True,
             conclusion: OuterConclusion = "refine_budget") -> tuple[SearchStore, Path, SearchStatus]:
    store, registry = make_store(folder, budget=3, refine={"budget": 3, "convergence_run": 30})
    run(store, registry, SearchCompute(store, persist_baseline=True, values={0: 0.1, 1: 0.2, 2: 0.3}))
    if refined:
        compute = RefineCompute(store, {None: 4.0, 0: 0.3, 1: 0.1})
        refine(store, registry, compute)
        for job in compute.jobs:
            store.scheme_path_for(job.result_path).write_text(job.scheme.model_dump_json())
    for path in (store.baseline_path, *(store.path / "candidates").glob("trial-*.json"),
                 *store.refine_dir.glob("*.json")):
        if path.name.endswith("-scheme.json"):
            continue
        value = json.loads(path.read_bytes()) if path.read_bytes() else {}
        path.write_text(json.dumps(value | {"scope": "stage_two_subset"}))
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    return store, registry, conclude(store, status, conclusion)


def scheme_for(store: SearchStore, role: str) -> Scheme:
    if role == "baseline":
        return store.project
    result = store.candidate_path(0) if role == "search_best" else store.refine_result_path(1)
    return load_scheme(store.scheme_path_for(result))


def protected(store: SearchStore) -> dict[str, bytes]:
    return {str(path.relative_to(store.path)): path.read_bytes() for path in store.path.rglob("*")
            if path.is_file() and "modal-diagnosis" not in path.parts and "crossover-sensitivity" not in path.parts
            and "placement-stability" not in path.parts}


def change_first(store: SearchStore, role: str, number: int = 2) -> None:
    """只改考卷隔離帳本；模擬補完後另一列變成第一名。"""
    from aosr.search.ledger import Ledger
    from aosr.search.refine import RefineLedger
    if role == "search_best":
        header, rows = Ledger.read(store.ledger_path)
        store.ledger_path.unlink()
        book = Ledger.create(store.ledger_path, header)
        for row in rows:
            book.append(row.model_copy(update={"score": 0.01}) if row.trial_number == number else row)
    else:
        refine_header, refine_rows = RefineLedger.read(store.refine_ledger_path)
        store.refine_ledger_path.unlink()
        refined = RefineLedger.create(store.refine_ledger_path, refine_header)
        for refined_row in refine_rows:
            refined.append(refined_row.model_copy(update={"total_cost": 0.01}) if refined_row.trial_number == number else refined_row)
