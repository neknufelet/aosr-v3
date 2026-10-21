"""自動細算保留停止記號；手動細算繼續由原有考卷守預設行為。"""
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.search.run import IdentityChanged
from aosr.search.stage_one_refine import refine_stage_one_search
from aosr.search.stage_one_refine_stop import StageOneRefineStopped
from aosr.search.store import SearchIdentity
from tests.engine import _stage_one_cases as stage
from tests.engine._stage_one_auto_cases import no_compute
from tests.engine._stage_one_post_check_cases import prepared
from tests.engine._stage_one_refine_cases import pool, snapshot
from tests.engine.test_search_stage_one_refine import ROWS


@pytest.mark.parametrize("marker", ["search", "refine"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize("when", ["entry", "probe"])
def test_keep_stop_marker_never_computes_or_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                  saved: bool, when: str, marker: str) -> None:
    store = prepared(tmp_path, monkeypatch) if saved else pool(tmp_path, monkeypatch, ROWS)
    stop = store.stop_path if marker == "search" else store.refine_stop_path
    if when == "entry":
        stop.touch()
    before = snapshot(store)
    def probe() -> SearchIdentity:
        if when == "entry":
            # 記號已在：入口就要停，走到核身分再中斷會寫一份「中斷」摘要。
            raise IdentityChanged("不該走到核身分")
        stop.touch()
        return store.identity
    with pytest.raises(StageOneRefineStopped):
        refine_stage_one_search(store, compute=no_compute(), probe=probe, keep_stop_marker=True,
            registry_path=config_path("quality_targets.toml"), run_date=stage.run_kwargs()["run_date"],
            engine_version=store.identity.program_fingerprint)
    assert stop.exists()
    after = snapshot(store)
    if when == "probe":
        after.pop(stop.relative_to(store.path).as_posix())
    assert after == before
