"""搜尋總表的考卷：必填停止條件、取樣設定與內容指紋。"""

import importlib.util
import hashlib
import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.engine._search_store_cases import settings_document


def test_settings_contract_is_available(tmp_path: Path) -> None:
    """缺少搜尋總表時明確失敗，不把收集失敗當成有效紅燈。"""
    assert tmp_path.is_dir()
    assert importlib.util.find_spec("aosr.search.settings") is not None


@pytest.mark.parametrize("field", ["budget", "convergence_run"])
def test_stop_conditions_are_required(tmp_path: Path, field: str) -> None:
    from aosr.search.settings import SearchSettings

    document = settings_document()
    del document[field]
    with pytest.raises(ValidationError):
        SearchSettings.model_validate(document)
    assert tmp_path.is_dir()


@pytest.mark.parametrize("field", ["batch_size", "max_workers", "n_startup_trials", "budget", "convergence_run"])
@pytest.mark.parametrize("value", [0, -1])
def test_positive_counts_are_required(tmp_path: Path, field: str, value: int) -> None:
    from aosr.search.settings import SearchSettings

    with pytest.raises(ValidationError):
        SearchSettings.model_validate(settings_document() | {field: value})
    assert tmp_path.is_dir()


def test_sampler_settings_match(tmp_path: Path) -> None:
    from aosr.search.sampler import SamplerSettings
    from aosr.search.settings import SearchSettings

    for liar in (True, False):
        settings = SearchSettings.model_validate(settings_document() | {"constant_liar": liar})
        assert settings.sampler_settings() == SamplerSettings(0, 2, liar)
    document = settings_document()
    del document["constant_liar"]
    assert SearchSettings.model_validate(document).sampler_settings().constant_liar is True
    assert tmp_path.is_dir()


@pytest.mark.parametrize("field,value", [
    ("purpose", "production"), ("seed", 1), ("n_startup_trials", 4), ("constant_liar", False),
    ("batch_size", 4), ("max_workers", 2), ("budget", 12), ("convergence_run", 6),
    ("layout", {"front_wall": "y0"}),
])
def test_fingerprint_tracks_every_setting(tmp_path: Path, field: str, value: object) -> None:
    from aosr.search.settings import SearchSettings

    document = settings_document()
    original = SearchSettings.model_validate(document)
    if field == "layout":
        value = original.layout.model_dump(mode="json") | {"front_wall": "y0"}
    changed = SearchSettings.model_validate(document | {field: value})
    assert changed.fingerprint != original.fingerprint
    assert SearchSettings.model_validate(dict(reversed(tuple(document.items())))).fingerprint == original.fingerprint
    assert original.canonical() == document
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert original.fingerprint == hashlib.sha256(canonical.encode()).hexdigest()
    assert tmp_path.is_dir()


@pytest.mark.parametrize("field,value", [
    ("purpose", ""), ("purpose", "   "), ("unexpected", True), ("seed", True),
    ("n_startup_trials", 1.5), ("constant_liar", 1), ("max_workers", "2"),
])
def test_settings_reject_invalid_fields(tmp_path: Path, field: str, value: object) -> None:
    from aosr.search.settings import SearchSettings

    with pytest.raises(ValidationError):
        SearchSettings.model_validate(settings_document() | {field: value})
    assert tmp_path.is_dir()


def test_settings_are_frozen_and_finite(tmp_path: Path) -> None:
    from aosr.search.settings import SearchSettings

    settings = SearchSettings.model_validate(settings_document())
    with pytest.raises(ValidationError):
        setattr(settings, "budget", 1)
    for value in (math.inf, math.nan):
        with pytest.raises(ValidationError):
            SearchSettings.model_validate(settings.canonical() | {
                "layout": settings.layout.model_dump() | {"speaker_height_m": value},
            })
    assert tmp_path.is_dir()
