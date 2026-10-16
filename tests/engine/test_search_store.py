"""搜尋資料夾的考卷：快照、身分核對與拒絕覆寫。"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import UUID
from typing import TYPE_CHECKING

import pytest

from aosr.config.frequency_axis import LowFrequencyAxis

from tests.engine._search_store_cases import purpose_settings, reference_project, settings_document

if TYPE_CHECKING:
    from aosr.search.store import SearchStore


def test_store_contract_is_available(tmp_path: Path) -> None:
    """缺少搜尋存放介面時明確失敗。"""
    assert tmp_path.is_dir()
    assert importlib.util.find_spec("aosr.search.store") is not None


@pytest.fixture
def store(tmp_path: Path) -> SearchStore:
    from aosr.search.settings import SearchSettings
    from aosr.search.store import SearchIdentity, SearchStore

    project = reference_project(tmp_path)
    settings = SearchSettings.model_validate(settings_document() | {"purpose": project.purpose})
    identity = SearchIdentity("physical-test", "program-test", purpose_settings(project.purpose))
    return SearchStore.create(tmp_path / "searches", project=project, settings=settings, identity=identity,
                              versions={"python": "test-python", "optuna": "test-optuna", "numpy": "test-numpy"})


def test_snapshots_round_trip(tmp_path: Path, store: SearchStore) -> None:
    from aosr.search.store import IDENTITY_FILE, PROJECT_FILE, PURPOSE_FILE, SETTINGS_FILE, SearchStore

    opened = SearchStore.open(store.path)
    assert opened.search_id == store.search_id == store.path.name
    assert opened.path == store.path
    assert opened.project == store.project
    assert opened.settings.canonical() == store.settings.canonical()
    assert opened.settings.fingerprint == store.settings.fingerprint
    assert opened.identity == store.identity
    assert opened.versions == store.versions
    expected = {
        PROJECT_FILE: store.project.model_dump(mode="json"),
        SETTINGS_FILE: {"settings": store.settings.canonical(), "fingerprint": store.settings.fingerprint},
        PURPOSE_FILE: store.identity.purpose_settings.model_dump(mode="json"),
    }
    for name, document in expected.items():
        text = (store.path / name).read_text(encoding="utf-8")
        assert json.loads(text) == document
        assert text == json.dumps(document, ensure_ascii=False, indent=1, allow_nan=False) + "\n"
    identity = json.loads((store.path / IDENTITY_FILE).read_text(encoding="utf-8"))
    assert identity["physics_identity"] == store.identity.physics_identity
    assert identity["program_fingerprint"] == store.identity.program_fingerprint
    assert identity["versions"] == dict(store.versions)
    assert "當次評分" in (store.path / PURPOSE_FILE).read_text(encoding="utf-8")
    assert tmp_path.is_dir()


def test_each_create_uses_a_new_id(tmp_path: Path, store: SearchStore) -> None:
    from aosr.search.store import SearchStore

    before = {p.name: p.read_bytes() for p in store.path.iterdir() if p.is_file()}
    second = SearchStore.create(store.path.parent, project=store.project, settings=store.settings,
                                identity=store.identity, versions=store.versions)
    assert second.search_id != store.search_id
    assert second.path != store.path
    assert {p.name: p.read_bytes() for p in store.path.iterdir() if p.is_file()} == before
    assert SearchStore.open(second.path).project == store.project
    assert tmp_path.is_dir()


def test_existing_directory_is_refused_without_overwrite(
    tmp_path: Path, store: SearchStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aosr.search.store import SearchStore

    fixed = UUID(int=42)
    monkeypatch.setattr("aosr.search.store.uuid.uuid4", lambda: fixed)
    existing = store.path.parent / fixed.hex
    existing.mkdir()
    marker = existing / "untouched"
    marker.write_text("老闆原有內容", encoding="utf-8")
    before = tuple(existing.iterdir())
    with pytest.raises(FileExistsError):
        SearchStore.create(store.path.parent, project=store.project, settings=store.settings,
                           identity=store.identity, versions=store.versions)
    assert tuple(existing.iterdir()) == before
    assert marker.read_text(encoding="utf-8") == "老闆原有內容"
    assert tmp_path.is_dir()


def test_purpose_mismatch_is_refused(tmp_path: Path, store: SearchStore) -> None:
    from aosr.search.settings import SearchSettings
    from aosr.search.store import SearchIdentity, SearchStore

    changed = SearchSettings.model_validate(store.settings.canonical() | {"purpose": "other-purpose"})
    with pytest.raises(ValueError):
        SearchStore.create(tmp_path / "wrong", project=store.project, settings=changed,
                           identity=store.identity, versions=store.versions)
    identity = SearchIdentity("physical-test", "program-test", purpose_settings("other-purpose"))
    with pytest.raises(ValueError):
        SearchStore.create(tmp_path / "wrong", project=store.project, settings=store.settings,
                           identity=identity, versions=store.versions)
    assert not (tmp_path / "wrong").exists()


def test_verification_axis_project_is_refused(tmp_path: Path, store: SearchStore) -> None:
    """第六節第 6 條：驗證軸只給細算；寫了驗證軸的專案不准開搜尋，也不留半個資料夾。"""
    from aosr.reporting.scheme import Scheme
    from aosr.search.store import SearchStore

    scene = store.project.scene.model_dump(mode="json") | {"low_frequency_axis": LowFrequencyAxis.VERIFICATION}
    project = Scheme.model_validate(store.project.model_dump(mode="json") | {"scene": scene})
    with pytest.raises(ValueError, match="search axis"):
        SearchStore.create(tmp_path / "verification", project=project, settings=store.settings,
                           identity=store.identity, versions=store.versions)
    assert not (tmp_path / "verification").exists()


def test_explicit_search_axis_project_is_accepted(tmp_path: Path, store: SearchStore) -> None:
    """明寫舊搜尋軸仍可重現舊搜尋，不被改成 48 點。"""
    from aosr.reporting.scheme import Scheme
    from aosr.search.store import SearchStore

    scene = store.project.scene.model_dump(mode="json") | {"low_frequency_axis": LowFrequencyAxis.SEARCH}
    project = Scheme.model_validate(store.project.model_dump(mode="json") | {"scene": scene})
    created = SearchStore.create(tmp_path / "explicit", project=project, settings=store.settings,
                                 identity=store.identity, versions=store.versions)
    assert created.project.scene.low_frequency_axis is LowFrequencyAxis.SEARCH
    assert SearchStore.open(created.path).project.scene.low_frequency_axis is LowFrequencyAxis.SEARCH


@pytest.mark.parametrize("missing", ["python", "optuna", "numpy"])
def test_required_versions_are_checked(tmp_path: Path, store: SearchStore, missing: str) -> None:
    from aosr.search.store import SearchStore

    versions = dict(store.versions)
    del versions[missing]
    with pytest.raises(ValueError):
        SearchStore.create(tmp_path / "wrong", project=store.project, settings=store.settings,
                           identity=store.identity, versions=versions)
    assert not (tmp_path / "wrong").exists()


def test_candidate_paths_are_inside_candidates(tmp_path: Path, store: SearchStore) -> None:
    from aosr.search.store import CANDIDATES_DIR, JSON_SUFFIX, LEDGER_FILE

    with pytest.raises(ValueError):
        store.candidate_path(-1)
    candidate = store.candidate_path(42)
    assert candidate.parent == store.path / CANDIDATES_DIR
    assert candidate.name == f"trial-000042{JSON_SUFFIX}"
    assert candidate.parent.is_dir()
    assert not candidate.exists()
    assert store.ledger_path == store.path / LEDGER_FILE
    assert tmp_path.is_dir()


@pytest.mark.parametrize("name,change", [
    ("identity", {"store_version": "future-version"}), ("identity", {"versions": {"python": "test"}}),
    ("identity", {"physics_identity": 1}), ("identity", {"extra": True}),
    ("identity", {"search_id": "changed"}), ("settings", {"fingerprint": "changed"}),
    ("project", {"purpose": "changed"}), ("purpose", {"purpose": "changed"}),
])
def test_open_validates_snapshots(
    tmp_path: Path, store: SearchStore, name: str, change: dict[str, object],
) -> None:
    from aosr.search.store import IDENTITY_FILE, PROJECT_FILE, PURPOSE_FILE, SETTINGS_FILE, SearchStore

    path = store.path / {"identity": IDENTITY_FILE, "settings": SETTINGS_FILE,
                         "project": PROJECT_FILE, "purpose": PURPOSE_FILE}[name]
    data = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(json.dumps(data | change) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        SearchStore.open(store.path)
    assert tmp_path.is_dir()


def test_store_exposes_read_only_snapshots(tmp_path: Path, store: SearchStore) -> None:
    with pytest.raises(AttributeError):
        setattr(store, "search_id", "changed")
    with pytest.raises(FrozenInstanceError):
        setattr(store.identity, "physics_identity", "changed")
    project = store.project
    project.speakers.clear()
    identity = store.identity
    identity.purpose_settings.content.clear()
    assert store.project.speakers
    assert store.identity.purpose_settings.content
    assert tmp_path.is_dir()


def test_implicit_axis_is_persisted_for_new_search(tmp_path: Path, store: SearchStore) -> None:
    """開新搜尋必須把解析度寫進快照，不能落回單方案的預設軸。"""
    from aosr.search.store import PROJECT_FILE, SearchStore

    assert store.project.scene.low_frequency_axis is LowFrequencyAxis.SEARCH_48
    saved = SearchStore.open(store.path)
    assert saved.project.scene.low_frequency_axis is store.project.scene.low_frequency_axis
    assert json.loads((store.path / PROJECT_FILE).read_text())["scene"]["low_frequency_axis"] == (
        store.project.scene.low_frequency_axis.value)


@pytest.mark.parametrize("axis", (LowFrequencyAxis.SEARCH, LowFrequencyAxis.SEARCH_48, LowFrequencyAxis.VERIFICATION))
def test_resume_checks_both_search_axes_and_refuses_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, axis: LowFrequencyAxis,
) -> None:
    """舊版建過的驗證軸搜尋照擋；24 與 48 點均可重播並繼續。"""
    from aosr.search import store as store_module
    from aosr.search.run import resume_search
    from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, Killed, make_store, run

    with monkeypatch.context() as patch:
        patch.setattr(store_module, "check_search_axis", lambda project: None)
        created, registry = make_store(tmp_path, budget=6, batch=1,
                                      scene_changes={"low_frequency_axis": axis})
        with pytest.raises(Killed):
            run(created, registry, FakeCompute(created, fail_after=1, kill=True, persist_baseline=True))
    from aosr.search.store import SearchStore

    reopened = SearchStore.open(created.path)
    before = reopened.ledger_path.read_bytes()
    compute = FakeCompute(reopened)
    status = resume_search(reopened, compute=compute, probe=lambda: reopened.identity,
                           registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert reopened.project.scene.low_frequency_axis is axis
    if axis is LowFrequencyAxis.VERIFICATION:
        assert status.state == "interrupted" and status.message.startswith("拒絕接續：")
        assert "search axis" in status.message
        assert not compute.calls and reopened.ledger_path.read_bytes() == before
    else:
        assert status.state == "budget_exhausted"
        assert compute.calls


def test_old_snapshot_without_axis_resumes_unchanged(tmp_path: Path) -> None:
    """#435 之前建的搜尋快照那一格是 null：接續照准，讀回不補 48，快照與帳本表頭都不改寫。"""
    from aosr.search.ledger import header_for
    from aosr.search.run import resume_search
    from aosr.search.store import PROJECT_FILE, SearchStore
    from tests.engine._search_run_cases import ENGINE, RUN_DATE, FakeCompute, Killed, make_store, run

    created, registry = make_store(tmp_path, budget=6, batch=1)
    snapshot = json.loads((created.path / PROJECT_FILE).read_text())
    snapshot["scene"]["low_frequency_axis"] = None
    (created.path / PROJECT_FILE).write_text(json.dumps(snapshot))
    project_bytes = (created.path / PROJECT_FILE).read_bytes()
    old = SearchStore.open(created.path)
    assert old.project.scene.low_frequency_axis is None
    with pytest.raises(Killed):
        run(old, registry, FakeCompute(old, fail_after=1, kill=True, persist_baseline=True))
    assert (created.path / PROJECT_FILE).read_bytes() == project_bytes
    header_line = old.ledger_path.read_bytes().splitlines()[0]
    reopened = SearchStore.open(created.path)
    assert header_for(reopened) == header_for(old)
    compute = FakeCompute(reopened)
    status = resume_search(reopened, compute=compute, probe=lambda: reopened.identity,
                           registry_path=registry, run_date=RUN_DATE, engine_version=ENGINE)
    assert status.state == "budget_exhausted" and compute.calls
    assert reopened.project.scene.low_frequency_axis is None
    assert (created.path / PROJECT_FILE).read_bytes() == project_bytes
    assert reopened.ledger_path.read_bytes().splitlines()[0] == header_line
