"""自動細算保留停止記號；手動細算繼續由原有考卷守預設行為。"""
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.search.stage_one_refine import refine_stage_one_search
from aosr.search.stage_one_refine_stop import StageOneRefineStopped
from aosr.search.store import SearchIdentity
from tests.engine import _stage_one_cases as stage
from tests.engine._stage_one_auto_cases import no_compute
from tests.engine._stage_one_post_check_cases import prepared
from tests.engine._stage_one_refine_cases import pool, snapshot
from tests.engine.test_search_stage_one_refine import ROWS


@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("when", ["entry", "probe"])
def test_keep_stop_marker_never_computes_or_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  saved: bool, when: str) -> None:
    store = prepared(tmp_path, monkeypatch) if saved else pool(tmp_path, monkeypatch, ROWS)
    if when == "entry":
        store.refine_stop_path.touch()
    before = snapshot(store)
    def probe() -> SearchIdentity:
        store.refine_stop_path.touch()
        return store.identity
    with pytest.raises(StageOneRefineStopped):
        refine_stage_one_search(store, compute=no_compute(), probe=probe, keep_stop_marker=True,
            registry_path=config_path("quality_targets.toml"), run_date=stage.run_kwargs()["run_date"],
            engine_version=store.identity.program_fingerprint)
    assert store.refine_stop_path.exists()
    after = snapshot(store)
    if when == "probe":
        after.pop(store.refine_stop_path.relative_to(store.path).as_posix())
    assert after == before
