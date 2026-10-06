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
            if path.is_file() and "modal-diagnosis" not in path.parts}
