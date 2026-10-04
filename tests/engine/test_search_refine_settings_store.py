"""細算骨架：舊快照相容、設定身分與分開存放，不求解。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from aosr.search.run import SearchStatus
from aosr.search.settings import SearchSettings
from aosr.search.store import SETTINGS_FILE, SearchStore
from tests.engine._search_run_cases import make_store
from tests.engine._search_store_cases import settings_document


def test_legacy_settings_keep_fingerprint_and_omit_refine(tmp_path: Path) -> None:
    document = settings_document()
    expected = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = tmp_path / SETTINGS_FILE
    path.write_text(json.dumps(document), encoding="utf-8")
    settings = SearchSettings.model_validate_json(path.read_bytes())
    assert settings.fingerprint == expected
    assert settings.canonical() == document
    assert "refine" not in json.loads(settings.model_dump_json())
    assert settings.refine is None
    store, _ = make_store(tmp_path)
    snapshot = json.loads((store.path / SETTINGS_FILE).read_bytes())
    assert "refine" not in snapshot["settings"]
    legacy_document = document | {"purpose": store.project.purpose}
    legacy_fingerprint = hashlib.sha256(json.dumps(
        legacy_document, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    (store.path / SETTINGS_FILE).write_text(json.dumps({
        "settings": legacy_document, "fingerprint": legacy_fingerprint,
    }), encoding="utf-8")
    assert SearchStore.open(store.path).settings.refine is None


def test_refine_settings_are_part_of_fingerprint(tmp_path: Path) -> None:
    document = settings_document() | {"refine": {"convergence_run": 3, "budget": 12}}
    settings = SearchSettings.model_validate(document)
    assert settings.canonical() == document
    assert json.loads(settings.model_dump_json())["refine"] == document["refine"]
    assert SearchSettings.model_validate_json(settings.model_dump_json()) == settings
    fingerprints = [settings.fingerprint, SearchSettings.model_validate(settings_document()).fingerprint]
    for chosen in ({"convergence_run": 4, "budget": 12}, {"convergence_run": 3, "budget": 13}):
        changed = SearchSettings.model_validate(document | {"refine": chosen})
        assert changed.fingerprint not in fingerprints
        fingerprints.append(changed.fingerprint)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("chosen", [
    {}, {"budget": 2}, {"convergence_run": 2},
    {"budget": 0, "convergence_run": 2}, {"budget": 2, "convergence_run": 0},
    {"budget": -1, "convergence_run": 2}, {"budget": 2, "convergence_run": -1},
    {"budget": True, "convergence_run": 2}, {"budget": 2, "convergence_run": "2"},
    {"budget": 2.0, "convergence_run": 2}, {"budget": 2, "convergence_run": 2, "extra": 1},
])
def test_refine_settings_require_strict_positive_fields(tmp_path: Path, chosen: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SearchSettings.model_validate(settings_document() | {"refine": chosen})
    assert tmp_path.is_dir()


def test_legacy_status_defaults_and_new_status_round_trip(tmp_path: Path) -> None:
    from aosr.search.run import RefineStatus

    store, _ = make_store(tmp_path)
    document = {"state": "budget_exhausted", "message": "舊搜尋停止", "best_trial": 7}
    store.status_path.write_text(json.dumps(document), encoding="utf-8")
    status = SearchStatus.model_validate_json(store.status_path.read_bytes())
    assert status.refine == RefineStatus()
    assert status.refine.model_dump() == {
        "state": "not_started", "stop_reason": None, "round": 1, "refined": 0,
        "best": None, "best_total_cost": None, "streak": 0,
        "message": "細算未開始：還沒有任何候選用驗證軸細算",
        # 2026-10-04 細算子物件加各輪牆鐘秒數（細算只改細算子物件，所以住這裡）；舊檔讀回是空的。
        "seconds": {},
    }
    refined = RefineStatus(state="stopped", stop_reason="refine_budget", round=2, refined=4,
                          best="baseline", best_total_cost=0.25, streak=3, message="原方案仍是第一名")
    changed = SearchStatus.model_validate(status.model_dump() | {"refine": refined.model_dump()})
    store.status_path.write_text(changed.model_dump_json(), encoding="utf-8")
    assert SearchStatus.model_validate_json(store.status_path.read_bytes()) == changed
    assert changed.state == status.state
    assert changed.message == status.message


@pytest.mark.parametrize("change", [
    {"state": "converged"}, {"round": 0}, {"refined": -1}, {"streak": -1},
    {"best": "candidate"}, {"best_total_cost": float("nan")}, {"extra": True},
    # 停止原因只認固定代碼，自由文字不准（外圈與報告靠代碼分辨）。
    {"state": "stopped", "stop_reason": "用完細算預算"},
    # 停止原因只在已停時出現、而且已停一定要有。
    {"state": "running", "stop_reason": "stable"}, {"state": "not_started", "stop_reason": "stable"},
    {"state": "stopped"},
])
def test_refine_status_rejects_invalid_fields(tmp_path: Path, change: dict[str, object]) -> None:
    from aosr.search.run import RefineStatus

    with pytest.raises(ValueError):
        RefineStatus.model_validate(change)
    assert tmp_path.is_dir()


def test_refine_models_are_frozen(tmp_path: Path) -> None:
    from aosr.search.run import RefineStatus
    from aosr.search.settings import RefineSettings

    settings = RefineSettings(convergence_run=3, budget=12)
    status = RefineStatus()
    with pytest.raises(ValueError):
        settings.budget = 13
    with pytest.raises(ValueError):
        status.message = "不能改凍結物件"
    assert tmp_path.is_dir()


@pytest.mark.parametrize("trial", [None, 0, 79, 1_234_567])
def test_refine_paths_do_not_collide_and_directory_is_lazy(tmp_path: Path, trial: int | None) -> None:
    from aosr.search.store import JSONL_SUFFIX, JSON_SUFFIX

    store, _ = make_store(tmp_path)
    path = store.refine_result_path(trial)
    screening = store.baseline_path if trial is None else store.candidate_path(trial)
    name = "baseline" if trial is None else f"trial-{trial:06d}"
    assert path == store.path / "verification" / f"{name}{JSON_SUFFIX}"
    assert path.parent == store.refine_dir
    assert path != screening
    assert path != store.baseline_path
    assert path != store.candidate_path(79)
    for related in (SearchStore.scheme_path_for, SearchStore.stderr_path_for):
        assert related(path).parent == store.refine_dir
        assert related(path) != related(screening)
    assert store.refine_ledger_path == store.path / f"refine{JSONL_SUFFIX}"
    assert not store.refine_dir.exists()
    assert not store.refine_ledger_path.exists()
    assert SearchStore.open(store.path).search_id == store.search_id
    store.ensure_refine_dir()
    marker = store.refine_dir / "preserved"
    marker.write_text("已保存的細算", encoding="utf-8")
    store.ensure_refine_dir()
    assert store.refine_dir.is_dir()
    assert marker.read_text(encoding="utf-8") == "已保存的細算"


@pytest.mark.parametrize("trial,expected", [
    (None, "search-test-baseline-verification"), (79, "search-test-trial-000079-verification"),
])
def test_refine_scheme_id_has_verification_suffix(tmp_path: Path, trial: int | None, expected: str) -> None:
    from aosr.search.store import refine_scheme_id

    assert refine_scheme_id("search-test", trial) == expected
    assert tmp_path.is_dir()


@pytest.mark.parametrize("trial", [-1, True, 1.5, "79"])
def test_refine_paths_and_ids_reject_invalid_numbers(tmp_path: Path, trial: object) -> None:
    from aosr.search.store import refine_scheme_id

    store, _ = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.refine_result_path(cast(int, trial))
    with pytest.raises(ValueError):
        refine_scheme_id(store.search_id, cast(int, trial))
