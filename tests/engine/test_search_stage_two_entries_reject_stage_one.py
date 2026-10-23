"""#884：第二階段的函式庫入口直接呼叫，也在寫任何檔之前拒收第一階段資料夾（命令列早已分流去第一階段或拒收）。"""
import re
from collections.abc import Callable
from pathlib import Path
from typing import NoReturn

import pytest

from aosr.config.paths import config_path
from aosr.search.crossover_sensitivity import attach_crossover
from aosr.search.feedback import feedback_search
from aosr.search.modal_attach import attach_modal
from aosr.search.outer import auto_search
from aosr.search.placement_stability_attach import attach_stability
from aosr.search.refine_run import refine_search
from aosr.search.run import SearchStatus, resume_search, start_search
from aosr.search.select import select_refined
from aosr.search.store import SearchStore
from tests.engine import _search_stage_one_cases as search, _stage_one_cases as stage

REGISTRY = config_path("quality_targets.toml")


def _never(*args: object, **kwargs: object) -> NoReturn:
    pytest.fail("第一階段資料夾不可開第二階段計算")


def _status(store: SearchStore) -> SearchStatus:
    return SearchStatus.model_validate_json(store.status_path.read_bytes())


def _start(store: SearchStore, tmp_path: Path) -> object:
    return start_search(store, compute=_never, probe=lambda: store.identity, registry_path=REGISTRY,
        run_date=stage.run_kwargs()["run_date"], engine_version=store.identity.program_fingerprint)


def _resume(store: SearchStore, tmp_path: Path) -> object:
    return resume_search(store, compute=_never, probe=lambda: store.identity, registry_path=REGISTRY,
        run_date=stage.run_kwargs()["run_date"], engine_version=store.identity.program_fingerprint)


def _auto(store: SearchStore, tmp_path: Path) -> object:
    return auto_search(store, compute=_never, probe=lambda: store.identity, registry_path=REGISTRY,
        run_date=stage.run_kwargs()["run_date"], engine_version=store.identity.program_fingerprint)


def _refine(store: SearchStore, tmp_path: Path) -> object:
    return refine_search(store, compute=_never, probe=lambda: store.identity, registry_path=REGISTRY,
        run_date=stage.run_kwargs()["run_date"], engine_version=store.identity.program_fingerprint)


def _select(store: SearchStore, tmp_path: Path) -> object:
    return select_refined(store, 0, tmp_path / "data")


def _feedback(store: SearchStore, tmp_path: Path) -> object:
    return feedback_search(store)


def _modal(store: SearchStore, tmp_path: Path) -> object:
    return attach_modal(store, status=_status(store), cache_dir=tmp_path / "cache", runner=("must-not-run",))


def _crossover(store: SearchStore, tmp_path: Path) -> object:
    return attach_crossover(store, status=_status(store), quality_targets_path=REGISTRY)


def _stability(store: SearchStore, tmp_path: Path) -> object:
    return attach_stability(store, status=_status(store), quality_targets_path=REGISTRY,
        run_date=stage.run_kwargs()["run_date"], probe=lambda: store.identity, compute_factory=_never)


ENTRIES: dict[str, tuple[str, Callable[[SearchStore, Path], object]]] = {
    "start": ("搜尋", _start), "resume": ("搜尋", _resume), "auto": ("自動外圈", _auto), "refine": ("細算", _refine),
    "select": ("入選", _select), "feedback": ("回饋", _feedback), "modal": ("模態附件", _modal),
    "crossover": ("300 Hz 交接敏感度附件", _crossover), "stability": ("擺位穩定性附件", _stability)}


def _tree(tmp_path: Path) -> dict[Path, bytes | None]:
    return {path: path.read_bytes() if path.is_file() else None for path in tmp_path.rglob("*")}


@pytest.mark.parametrize("entry", sorted(ENTRIES))
def test_stage_two_entry_refuses_stage_one_folder_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 entry: str) -> None:
    # 固定房間的第一階段資料夾：帶房間範圍的那種另有考卷（test_search_room_loop）；開始用剛建好的，其餘用搜尋跑完的。
    store = search.make_store(tmp_path)
    if entry != "start":
        search.run(store, search.FakeCompute(store, monkeypatch))
    label, call = ENTRIES[entry]
    before = _tree(tmp_path)
    with pytest.raises(ValueError, match=re.escape(f"這是第一階段搜尋資料夾，第二階段的{label}不接第一階段")):
        call(store, tmp_path)
    assert _tree(tmp_path) == before
