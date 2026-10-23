"""細算設定與登記簿控制組；控制值在更動登記簿之前從主線輸入實算。"""
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.scoring.stage_one_score import stage_one_score_settings
from aosr.search.settings import SearchSettings
from tests.engine._search_store_cases import settings_document

REFINE = {"top_n": 2, "far_control_min_front_distance_m": .8}


def test_absent_refine_preserves_baseline_bytes() -> None:
    # 更動前：SearchSettings.model_validate(settings_document()).fingerprint。
    settings = SearchSettings.model_validate(settings_document())
    assert settings.fingerprint == "211ef3f9969fd541d729c9ea08ef09efc65ae2ec3acbfd40f430190fe3b5c203"
    assert json.loads(settings.model_dump_json()) == settings_document() == settings.canonical()


@pytest.mark.parametrize("missing", list(REFINE))
def test_both_selection_settings_are_required(missing: str) -> None:
    values = dict(REFINE)
    del values[missing]
    with pytest.raises(ValidationError):
        SearchSettings.model_validate(settings_document() | {"stage": "stage_one_fem_only", "stage_one_refine": values})


def test_refine_requires_stage_and_is_frozen() -> None:
    with pytest.raises(ValidationError, match="stage"):
        SearchSettings.model_validate(settings_document() | {"stage_one_refine": REFINE})
    settings = SearchSettings.model_validate(settings_document() | {"stage": "stage_one_fem_only", "stage_one_refine": REFINE})
    assert settings.stage_one_refine is not None
    with pytest.raises(ValidationError):
        settings.stage_one_refine.top_n = 3
    for field, value in (("top_n", True), ("top_n", 0), ("far_control_min_front_distance_m", 0),
                         ("far_control_min_front_distance_m", float("inf"))):
        with pytest.raises(ValidationError):
            type(settings.stage_one_refine).model_validate(REFINE | {field: value})
    for field, value in (("top_n", 3), ("far_control_min_front_distance_m", .9)):
        changed = SearchSettings.model_validate(settings.canonical() | {"stage_one_refine": REFINE | {field: value}})
        assert changed.fingerprint != settings.fingerprint


def test_selection_entry_does_not_change_score_or_registry_identity() -> None:
    # 更動前：load_quality_targets(config_path(...)).fingerprint 與 stage_one_score_settings 的 fingerprint。
    registry = load_quality_targets(config_path("quality_targets.toml"))
    assert registry.fingerprint == "fffdea5cd8b942248a046dc2ae8539c1ac08d46ea103ee19ab5448ad02a9575e"
    assert stage_one_score_settings(registry, "dedicated_two_channel_listening_room").fingerprint == (
        "2d9c47e3be7e3ffcb29f4c6f57fc17fb55da1c0fb77f7bb25a449cc2673f98ea")
    entry = registry.stage_one("dedicated_two_channel_listening_room").entry(
        "stage_one_selection.big_peak_control_margin_db")
    assert isinstance(entry, SettingEntry) and entry.value == 1.


def test_selection_entry_keeps_comparison_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.reporting.stage_one_result import run_stage_one, stage_one_comparison_identity
    from tests.engine import _stage_one_cases as stage
    kwargs = stage.run_kwargs()
    registry = kwargs["registry"]
    section = registry.stage_one("dedicated_two_channel_listening_room")
    old = registry.model_copy(update={"stage_one_sections": (section.model_copy(update={"setting": tuple(
        e for e in section.setting if e.key != "stage_one_selection.big_peak_control_margin_db")}),)})
    scheme = stage.scheme()
    parts = stage.shards(monkeypatch, scheme)
    kwargs["registry"] = old
    before = run_stage_one(scheme, shards=parts, **kwargs)
    kwargs["registry"] = registry
    after = run_stage_one(scheme, shards=parts, **kwargs)
    assert stage_one_comparison_identity(before) == stage_one_comparison_identity(after)
    assert tmp_path.is_dir()


AREA = {"area_control_max_spread": .1, "area_control_min_difference": .15}


def test_area_thresholds_are_optional_and_paired() -> None:
    # 更動前（主線 64c6e4ff 的 settings.py）：同一份 stage 加 REFINE 的指紋。沒設面積門檻的舊設定，指紋與快照都不變。
    document = settings_document() | {"stage": "stage_one_fem_only", "stage_one_refine": REFINE}
    old = SearchSettings.model_validate(document)
    assert old.fingerprint == "313a9bd555f6e1c7cfad432aade73095a5976ab16a25ca8c3af1c3e531f728ed"
    assert json.loads(old.model_dump_json())["stage_one_refine"] == REFINE == old.canonical()["stage_one_refine"]
    new = SearchSettings.model_validate(document | {"stage_one_refine": REFINE | AREA})
    assert json.loads(new.model_dump_json())["stage_one_refine"] == REFINE | AREA == new.canonical()["stage_one_refine"]
    assert new.fingerprint != old.fingerprint
    for field in AREA:
        with pytest.raises(ValidationError, match="一起設定"):
            SearchSettings.model_validate(document | {"stage_one_refine": REFINE | {field: AREA[field]}})
    for field, value in (("area_control_max_spread", 0), ("area_control_min_difference", 0),
                         ("area_control_min_difference", 1), ("area_control_max_spread", float("nan"))):
        with pytest.raises(ValidationError):
            SearchSettings.model_validate(document | {"stage_one_refine": REFINE | AREA | {field: value}})
