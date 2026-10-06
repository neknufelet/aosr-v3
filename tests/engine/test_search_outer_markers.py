"""停止記號落在步與步之間的窄縫、或上一輪留下的使用者停止：外圈照實收，不刪記號（複查重現）。"""

from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.search import cli
from aosr.search import outer as outer_module
from aosr.search.feedback import comparison_trial, feedback_search
from aosr.search.run import CandidateJob, ComputedCandidate, SearchStatus, _write_status
from aosr.search.store import SearchStore
from tests.engine._search_feedback_cases import prepared
from tests.engine._search_outer_cases import OuterCompute, invoke
from tests.engine.test_search_outer_auto import seed


def _read(store: SearchStore) -> SearchStatus:
    return SearchStatus.model_validate_json(store.status_path.read_bytes())


def test_previous_round_refine_user_stop_does_not_block_next_round(tmp_path: Path,
                                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, status = prepared(tmp_path, refine_budget=4, refine_convergence=2, anchor_number=1)
    stopped = status.refine.model_copy(update={"stop_reason": "user_stopped"})
    _write_status(store, status.model_copy(update={"refine": stopped}))
    after_feedback = feedback_search(store)
    assert (after_feedback.round, after_feedback.state, after_feedback.refine.round) == (2, "running", 1)
    exit_code = invoke(store, registry, monkeypatch)
    final = _read(store)
    assert exit_code == 0
    assert final.state != "running"
    assert final.refine.round == final.round == 2
    assert final.outer.conclusion not in (None, "user_stopped")


def test_refine_marker_placed_after_auto_check_is_kept_and_honoured(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = seed(tmp_path)
    placed: list[bool] = []

    def identity(purpose: str, capabilities: Path) -> object:
        current = _read(store)
        if not placed and current.state != "running" and current.refine.state == "not_started":
            store.refine_stop_path.touch()  # auto 查完記號、細算開頭之前，使用者剛好放記號
            placed.append(True)
        return store.identity

    monkeypatch.setattr(cli, "config_path", lambda name: registry if name.startswith("quality_targets") else config_path(name))
    monkeypatch.setattr(cli, "_identity", identity)
    exit_code = cli.main(["auto", str(store.path), "--engine-commit", "requested", "--capabilities", str(registry),
                          "--modal-cache-dir", str(tmp_path / "modal-cache")],
                         compute_factory=lambda opened, capabilities, commit: OuterCompute(opened))
    final = _read(store)
    assert placed
    assert exit_code == 0
    assert store.refine_stop_path.exists()
    assert final.outer.conclusion == "user_stopped"
    assert final.round == 1
    assert final.refine.stop_reason == "user_stopped"


def test_search_marker_placed_before_feedback_concludes_user_stopped(tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry, _ = prepared(tmp_path, refine_budget=4, refine_convergence=2, anchor_number=1)

    def compare(opened: SearchStore, current: SearchStatus) -> int | None:
        opened.stop_path.touch()  # auto 查完記號之後、回饋之前放搜尋停止記號
        return comparison_trial(opened, current)

    monkeypatch.setattr(outer_module, "comparison_trial", compare)
    exit_code = invoke(store, registry, monkeypatch)
    final = _read(store)
    assert exit_code == 0
    assert final.outer.conclusion == "user_stopped"
    assert store.stop_path.exists()
    assert final.round == 1


def test_refine_marker_during_search_step_stops_before_refine(tmp_path: Path,
                                                              monkeypatch: pytest.MonkeyPatch) -> None:
    store, registry = seed(tmp_path)
    inner = OuterCompute(store)

    def compute(jobs: Sequence[CandidateJob], workers: int) -> Iterator[ComputedCandidate]:
        if jobs[0].result_path.parent != store.refine_dir:
            store.refine_stop_path.touch()  # 只在搜尋那一步裡放細算記號；搜尋本身不看它
        yield from inner(jobs, workers)

    exit_code = invoke(store, registry, monkeypatch, compute)
    final = _read(store)
    assert exit_code == 0
    assert final.outer.conclusion == "user_stopped"
    assert store.refine_stop_path.exists()
    assert final.refine.state == "not_started"
