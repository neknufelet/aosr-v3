"""搜尋資料夾的考卷：快照、身分核對與拒絕覆寫。"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from uuid import UUID
from typing import TYPE_CHECKING

import pytest

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
