"""以真正建檔入口建立家具搜尋，另提供手定候選與驗家具的計算替身。"""
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.reporting.validation import furniture_problems
from aosr.search import layout, ledger, run as search_run
from aosr.search.run import CandidateJob, ComputedCandidate
from aosr.search.sampler import SamplerAdapter
from aosr.search.store import SearchStore
from tests.engine._furniture_cases import relative_item
from tests.engine._search_refine_cases import RefineCompute
from tests.engine._search_run_cases import FakeCompute, make_store


def furnished_store(tmp_path: Path, *, workers: int = 1, budget: int = 3, batch: int = 3,
                    scene_changes: dict[str, object] | None = None) -> tuple[SearchStore, Path]:
    furniture = [
        relative_item(furniture_id="rear-seat", placement={
            "forward_m": -1.0, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0}),
        relative_item(furniture_id="desk", kind="desk", material="wood",
                      width_m=0.3, depth_m=0.3, height_m=0.5, placement={
                          "forward_m": 0.25, "left_m": 0.4, "bottom_height_m": 1.0, "yaw_deg": 0}),
    ]
    return make_store(tmp_path, workers=workers, budget=budget, batch=batch, convergence=1, furniture=furniture,
                      scene_changes=scene_changes)


def enqueue_hand_placements(monkeypatch: pytest.MonkeyPatch, *, extra_legal: bool = False,
                            placements: Sequence[tuple[float, float]] | None = None) -> None:
    """手定的（前距, 聆聽距離）照順序入列，間距一律 1.2 m；不給 placements 就用下面三個手算點。"""
    original = search_run._adapter
    monkeypatch.setattr(layout, "standard_start", lambda project, settings: None)

    def adapter(store: SearchStore) -> tuple[SamplerAdapter, bool]:
        result, enqueued = original(store)
        # 左聲源 (1,1.4,1.25)、主位 (1.5,2,1.25)：桌板擋住；
        # 主位移到 x=5 時後方座椅最大 x=6.25，超出 6 m 房間 0.25 m；
        # 主位 x=2.5 時桌板離開直達，合法。
        chosen = [(1.0, 0.5), (2.0, 3.0), (1.0, 1.5)] if placements is None else list(placements)
        if extra_legal:
            chosen.extend(((1.0, 1.75), (1.25, 1.5), (1.5, 1.75)))
        for front, listening in chosen:
            result.enqueue(layout.unit_from_params(layout.LayoutParams(front, 1.2, listening), store.settings.layout))
        return result, enqueued

    monkeypatch.setattr(search_run, "_adapter", adapter)


def furnished_next_params(store: SearchStore) -> list[dict[str, str]]:
    """重播手定入列點後再要一批，核兩種工作行程數的取樣器下一步逐位相同。"""
    header, recorded = ledger.Ledger.read(store.ledger_path)
    settings = store.settings
    sizes = {index: min(settings.batch_size, settings.budget - index * settings.batch_size)
             for index in range((settings.budget + settings.batch_size - 1) // settings.batch_size)}
    history, partial = ledger.replay_history(header, recorded, batch_sizes=sizes)
    assert not partial
    adapter, _ = search_run._adapter(store)
    adapter.replay(history)
    return [{key: value.hex() for key, value in item.params.items()} for item in adapter.ask_batch(settings.batch_size)]


class FurnitureCompute(FakeCompute):
    """候選若漏掉家具預篩，計算替身立刻失敗。"""

    def _check_legal(self, job: search_run.CandidateJob) -> None:
        from aosr.reporting.validation import furniture_problems

        super()._check_legal(job)
        assert not furniture_problems(job.scheme)


class _FurnitureSearchCompute(FakeCompute):
    def _check_legal(self, job: CandidateJob) -> None:
        # FakeCompute 的四牆核對假設前牆是 x0，換牆的情境用不上（四牆限制另有 test_search_run 的考卷守）；
        # 這裡只獨立守家具預篩。
        if furniture_problems(job.scheme):
            raise AssertionError("家具預篩漏收")


class FurnitureFlowCompute:
    """搜尋、細算皆驗家具並存方案；替身結果不冒充完整物理報表。"""

    def __init__(self, store: SearchStore) -> None:
        self.store = store
        self.search = _FurnitureSearchCompute(store, persist_baseline=True)
        self.refine = RefineCompute(store, {})
        self.jobs: list[CandidateJob] = []

    def __call__(self, jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        for job in jobs:
            problems = furniture_problems(job.scheme)
            if problems:
                raise AssertionError(f"計算不收家具擺放錯或直達被擋：{problems}")
            SearchStore.scheme_path_for(job.result_path).write_text(job.scheme.model_dump_json())
        compute = self.refine if jobs[0].result_path.parent == self.store.refine_dir else self.search
        for result in compute(jobs, workers):
            self.jobs.append(result.job)
            identity = result.identity
            result.job.result_path.write_text(json.dumps({
                "candidate": result.candidate.model_dump(mode="json"),
                "physics_identity": identity.physics_identity,
                "program_fingerprint": identity.program_fingerprint,
                "purpose_settings": identity.purpose_settings.model_dump(mode="json"),
            }), encoding="utf-8")
            yield result
