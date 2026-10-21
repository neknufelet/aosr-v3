"""第一階段細算入口的快取參數與附件獨立出口。"""
from pathlib import Path
from collections.abc import Iterator

import pytest

from aosr.search import cli
from aosr.search.stage_one_worker import StageOneComputed
from tests.engine import _search_stage_one_cases as search
from tests.engine._stage_one_post_check_cases import prepared, runner
from tests.engine._stage_one_refine_cases import pool, snapshot
from tests.engine.test_search_stage_one_refine import ROWS


@pytest.mark.parametrize("enabled", [True, False])
def test_missing_cache_rejected_before_lock_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                      capsys: pytest.CaptureFixture[str], enabled: bool) -> None:
    store = pool(tmp_path, monkeypatch, ROWS, enabled=enabled); before = snapshot(store)
    monkeypatch.setattr(cli, "_search_lock", lambda *a: pytest.fail("must reject before lock"))
    code = cli.main(["refine", str(store.path), "--engine-commit", "test"])
    assert code == 1
    assert "第一階段細算要給 --modal-cache-dir（入選後檢查用的低頻模態快取資料夾，跟網頁共用）" in capsys.readouterr().err
    assert snapshot(store) == before


def test_second_stage_cache_rejected_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    from tests.engine._search_refine_cases import stopped_store
    store, _ = stopped_store(tmp_path); before = snapshot(store)
    monkeypatch.setattr(cli, "_search_lock", lambda *a: pytest.fail("must reject before lock"))
    code = cli.main(["refine", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")])
    assert code == 1
    assert "第二階段細算不收 --modal-cache-dir" in capsys.readouterr().err
    assert snapshot(store) == before


@pytest.mark.parametrize("state", ["diagnosed_not_scored", "failed", "not_computed", "out_of_scope"])
def test_completed_refine_exit_zero_independent_of_diagnosis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                            capsys: pytest.CaptureFixture[str], state: str) -> None:
    # 先製造替身所需的方案，再讓真 CLI 重讀並接續細算。
    store = prepared(tmp_path, monkeypatch); root = tmp_path / "child"
    command = runner(root, store, state=state)
    for path in store.refine_dir.iterdir():
        path.unlink()
    (store.path / "stage_one_refine.json").unlink()
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    assert not cli.main(["refine", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(root / "cache")],
        compute_factory=lambda *a: search.FakeCompute(store, monkeypatch), modal_runner=command)
    import json
    assert json.loads((store.path / "stage_one_refine.json").read_bytes())["state"] == "done"
    assert json.loads((store.path / "stage-one-check" / "summary.json").read_bytes())["seats"][0]["state"] == state
    if state == "failed":
        assert "入選後檢查失敗" in capsys.readouterr().err


def test_post_check_entrance_exception_cannot_change_refine_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               capsys: pytest.CaptureFixture[str]) -> None:
    from aosr.search import stage_one_post_check
    store = prepared(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    def fail(*args: object, **kwargs: object) -> None:
        raise OSError("disk failure")
    monkeypatch.setattr(stage_one_post_check, "run_stage_one_post_check", fail)
    assert not cli.main(["refine", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
        compute_factory=lambda *a: search.FakeCompute(store, monkeypatch))
    assert "入選後檢查失敗：disk failure" in capsys.readouterr().err


@pytest.mark.parametrize("stop", [True, False])
def test_unfinished_refine_never_runs_post_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stop: bool) -> None:
    from aosr.search.run import ComputeFailed
    from aosr.search import stage_one_post_check
    store = pool(tmp_path, monkeypatch, ROWS)
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    monkeypatch.setattr(stage_one_post_check, "run_stage_one_post_check", lambda *a, **k: pytest.fail("no post check"))
    def fail(*args: object) -> Iterator[StageOneComputed]:
        raise KeyboardInterrupt if stop else ComputeFailed("bad compute")
        yield
    code = cli.main(["refine", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
                    compute_factory=lambda *a: fail)
    expected = 3 if stop else 1
    assert code == expected
    assert not (store.path / "stage-one-check").exists()


def test_stopped_post_check_preserves_zero_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                               capsys: pytest.CaptureFixture[str]) -> None:
    store = prepared(tmp_path, monkeypatch); root = tmp_path / "child"
    monkeypatch.setattr(cli, "_identity", lambda *a: store.identity)
    assert not cli.main(["refine", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(root / "cache")],
        compute_factory=lambda *a: search.FakeCompute(store, monkeypatch), modal_runner=runner(root, store, stop_after=True))
    assert "入選後檢查已停止" in capsys.readouterr().err
    assert store.refine_stop_path.exists()
