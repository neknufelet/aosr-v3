"""被擋原方案的手算幾何；搜尋關仍由沒家具快照建檔後注入。"""
from __future__ import annotations

import json
import multiprocessing
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from aosr.reporting.scheme import Scheme
from aosr.scoring.contract import CandidateEvaluation
from aosr.search.run import CandidateJob, ComputedCandidate
from aosr.search.store import SearchStore
from tests.engine._search_furniture_cases import FurnitureCompute
from tests.engine._search_run_cases import make_store


def blocked_store(tmp_path: Path, *, workers: int = 1, batch: int = 3,
                  budget: int = 12, blocked: bool = True) -> tuple[SearchStore, Path]:
    store, registry = make_store(tmp_path, workers=workers, batch=batch, budget=budget,
                                 convergence=2, refine={"budget": 50, "convergence_run": 50},
                                 layout_changes={"front_distance_m": {"low": 0.25, "high": 1.3},
                                                 "listening_distance_m": {"low": 0.5, "high": 1.3}})
    # 原方案左喇叭 (1,1.3,1.2) 到主位 (3.2,1.9,1.2)；
    # 懸空方塊 x=[2.7,2.9]、y=[1.7,2.0]、z=[1.1,1.4] 擋住直達。
    project = Scheme.model_validate(store.project.model_dump() | {"furniture": [{
        "furniture_id": "blocker", "kind": "ceiling_cloud", "material": "wood",
        "width_m": 0.2, "depth_m": 0.3, "height_m": 0.3,
        "placement": {"bottom_center_m": [2.8, 1.85, 1.1 if blocked else 2.0], "yaw_deg": 0},
    }]})
    (store.path / "project.json").write_text(project.model_dump_json(), encoding="utf-8")
    return SearchStore.open(store.path), registry


class SavedFurnitureCompute(FurnitureCompute):
    """保存寫入端候選包與方案，供接續與細算；完成順序沿用工作行程變體。"""

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        assert all(job.trial_number is not None for job in jobs), "被擋原方案不得送計算"
        for result in super().__call__(jobs, workers):
            identity = result.identity
            result.job.result_path.write_text(json.dumps({
                "candidate": result.candidate.model_dump(mode="json"),
                "physics_identity": identity.physics_identity,
                "program_fingerprint": identity.program_fingerprint,
                "purpose_settings": identity.purpose_settings.model_dump(mode="json"),
            }), encoding="utf-8")
            SearchStore.scheme_path_for(result.job.result_path).write_text(result.job.scheme.model_dump_json())
            yield result


def delayed_result(number: int, scheme_json: str, result_path: Path, store_path: Path,
                   first: int, second: int, signal_path: Path) -> tuple[int, str]:
    """真的子行程交回替身結果；編號小的等待第二個已保存結果後才完成。"""
    if number == first:
        deadline = time.monotonic() + 10.0
        while not signal_path.exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("第二個候選沒有完成")
            time.sleep(0.01)
        time.sleep(0.1)
    store = SearchStore.open(store_path)
    job = CandidateJob(number, Scheme.model_validate_json(scheme_json), result_path)
    result = next(SavedFurnitureCompute(store, different=frozenset({second}))((job,), 1))
    if number == second:
        signal_path.write_text("第二個完成")
    return number, result.candidate.model_dump_json()


class DelayedFurnitureCompute(SavedFurnitureCompute):
    """多行程各算各存，由真的完成通知逐個交回。"""

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        numbers = sorted(job.trial_number for job in jobs if job.trial_number is not None)
        if workers == 1 or not numbers or numbers[0] == numbers[-1]:
            yield from super().__call__(jobs, workers)
            return
        first, second = numbers[:2]
        signal_path = self.store.path / f"completed-{second}"
        expected = {job.trial_number: job for job in jobs}
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(delayed_result, job.trial_number, job.scheme.model_dump_json(),
                                       job.result_path, self.store.path, first, second, signal_path)
                       for job in jobs if job.trial_number is not None]
            for future in as_completed(futures):
                number, candidate_json = future.result()
                self.calls.append(number)
                yield ComputedCandidate(expected[number], CandidateEvaluation.model_validate_json(candidate_json),
                                        0.25, self.store.identity)
