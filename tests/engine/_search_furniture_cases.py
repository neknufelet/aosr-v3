"""搜尋關仍關閉時，僅在暫存搜尋快照注入家具與手定候選。"""
from pathlib import Path

import pytest

from aosr.search import layout, ledger, run as search_run
from aosr.search.sampler import SamplerAdapter
from aosr.search.store import SearchStore
from tests.engine._furniture_cases import relative_item
from tests.engine._search_run_cases import FakeCompute, make_store


def furnished_store(tmp_path: Path, *, workers: int = 1, budget: int = 3,
                    batch: int = 3) -> tuple[SearchStore, Path]:
    store, registry = make_store(tmp_path, workers=workers, budget=budget, batch=batch, convergence=1)
    furniture = [
        relative_item(furniture_id="rear-seat", placement={
            "forward_m": -1.0, "left_m": 0.0, "bottom_height_m": 0.0, "yaw_deg": 0}),
        relative_item(furniture_id="desk", kind="desk", material="wood",
                      width_m=0.3, depth_m=0.3, height_m=0.5, placement={
                          "forward_m": 0.25, "left_m": 0.4, "bottom_height_m": 1.0, "yaw_deg": 0}),
    ]
    project = type(store.project).model_validate(store.project.model_dump() | {"furniture": furniture})
    (store.path / "project.json").write_text(project.model_dump_json(), encoding="utf-8")
    return SearchStore.open(store.path), registry


def enqueue_hand_placements(monkeypatch: pytest.MonkeyPatch, *, extra_legal: bool = False) -> None:
    original = search_run._adapter
    monkeypatch.setattr(layout, "standard_start", lambda project, settings: None)

    def adapter(store: SearchStore) -> tuple[SamplerAdapter, bool]:
        result, enqueued = original(store)
        # 左聲源 (1,1.4,1.25)、主位 (1.5,2,1.25)：桌板擋住；
        # 主位移到 x=5 時後方座椅最大 x=6.25，超出 6 m 房間 0.25 m；
        # 主位 x=2.5 時桌板離開直達，合法。
        placements = [(1.0, 0.5), (2.0, 3.0), (1.0, 1.5)]
        if extra_legal:
            placements.extend(((1.0, 1.75), (1.25, 1.5), (1.5, 1.75)))
        for front, listening in placements:
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
    """候選若漏掉家具預篩，計算替身立刻失敗；原方案留待第二步。"""

    def _check_legal(self, job: search_run.CandidateJob) -> None:
        from aosr.reporting.validation import furniture_problems

        super()._check_legal(job)
        assert not furniture_problems(job.scheme)
