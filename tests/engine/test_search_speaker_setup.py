"""施工單第 5 條：任何擺法暫擋建檔，省略與 null 維持主線指紋及快照。"""
import json
from pathlib import Path

import pytest

from aosr.reporting.scheme import Scheme
from aosr.search.ledger import header_for
from tests.engine import _speaker_setup_cases as cases
from tests.engine._furniture_cases import reference_document
from tests.engine.test_scheme_furniture import _store


@pytest.mark.parametrize("mount", ["stand", "desk", "floor"])
def test_search_setup_is_rejected_before_directory_creation(tmp_path: Path, mount: str) -> None:
    root = tmp_path / "new-search"
    with pytest.raises(ValueError, match=r"搜尋尚未支援喇叭類型與擺法（#559 第七支第五步施工中）"):
        _store(root, Scheme.model_validate(cases.document(mount)))
    assert not root.exists()


@pytest.mark.parametrize("explicit_null", [False, True])
def test_no_setup_project_snapshot_and_fingerprint_are_unchanged(tmp_path: Path, explicit_null: bool) -> None:
    document = reference_document()
    if explicit_null:
        document["speaker_setup"] = None
    store = _store(tmp_path / "search", Scheme.model_validate(document))
    assert "speaker_setup" not in json.loads((store.path / "project.json").read_text())
    # 出處：主線 8f9493a1 的 test_no_furniture_reference_fingerprints_are_unchanged。
    assert header_for(store).project_fingerprint == "0cd0e215f3fa0fe6d0de2798098fbc57b283caa35e946c68160690406c2b4d1f"
