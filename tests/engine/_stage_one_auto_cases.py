"""自動外圈替身：搜尋與細算照真入口，只換計算及診斷子行程。"""
from pathlib import Path

import pytest

from aosr.search import cli
from aosr.search.run import SearchStatus
from aosr.search.layout_settings import LayoutSettings
from aosr.search.settings import SearchSettings
from aosr.search.stage_one_worker import StageOneComputeFunction
from aosr.search.store import SearchStore
from tests.engine import _search_stage_one_cases as search


def running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, layout: LayoutSettings | None = None,
            area: dict[str, float] | None = None) -> SearchStore:
    source = search.make_store(tmp_path / "source", budget=4, layout=layout)
    settings = SearchSettings.model_validate(source.settings.model_dump(mode="json") | {
        "stage_one_refine": {"top_n": 2, "far_control_min_front_distance_m": .8, **(area or {})}})
    store = SearchStore.create(tmp_path / "searches", project=source.project, settings=settings,
        identity=source.identity, versions={"python": "test", "numpy": "test", "optuna": "test"})
    with pytest.raises(search.Killed):
        search.run(store, search.FakeCompute(store, monkeypatch, kill_after=1))
    assert read(store).state == "running"
    return store


def read(store: SearchStore) -> SearchStatus:
    return SearchStatus.model_validate_json(store.status_path.read_bytes())


def invoke(store: SearchStore, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *,
           compute: StageOneComputeFunction | None = None, runner: tuple[str, ...] = ("must-not-run",)) -> int:
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    for symbol in ("auto_search", "attach_modal", "attach_crossover", "attach_stability", "_finish_attachments"):
        monkeypatch.setattr(cli, symbol, lambda *a, **k: pytest.fail("第一階段不可走第二階段"))
    return cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
        compute_factory=lambda *a: compute if compute is not None else search.FakeCompute(store, monkeypatch),
        modal_runner=runner)


def no_compute(*args: object) -> StageOneComputeFunction:
    return lambda *a: pytest.fail("不可開計算")
