"""#558 讀回與搜尋身分：實際存檔、真重評，不替換 reevaluate（重評）。"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.calculation_fingerprint import calculation_fingerprint
from aosr.reporting.evaluation import ResultStanding, load_result, purpose_settings, reevaluate
from aosr.reporting.result import SchemeResult
from aosr.scoring.contract import QualityCategory
from aosr.scoring.ranking import RankingContext, rank_candidates
from aosr.search.run import IdentityChanged, _check_identity
from aosr.search.store import SearchIdentity
from tests.engine._directivity import DIRECTIVITY
from tests.engine._stage_one_cases import scheme
from tests.engine.test_scheme_pipeline import _run_control
from tests.engine.test_stage_one_score import _rewrite

TARGETS = config_path("quality_targets.toml")
CAPABILITIES = load_capabilities(config_path("capabilities.toml"))


@pytest.fixture(scope="module")
def saved() -> SchemeResult:
    return _run_control(scheme())


def _load(path: Path, registry: Path, original: SchemeResult) -> object:
    return load_result(path, capabilities=CAPABILITIES, directivity=DIRECTIVITY,
        quality_targets_path=registry, physics_identity=original.physics_identity)


def _context(result: SchemeResult) -> RankingContext:
    return RankingContext(purpose=result.scheme.purpose,
        receiver_set_fingerprint=result.scheme.receiver_set.fingerprint,
        channel_group_fingerprint=result.scheme.channel_group.fingerprint,
        run_date=date(2026, 10, 25), engine_version="control")


def test_old_file_without_new_category_is_remeasured_and_ranked_with_it(saved: SchemeResult, tmp_path: Path) -> None:
    document = saved.model_dump(mode="json")
    document["candidate"]["evaluations"] = [e for e in document["candidate"]["evaluations"]
        if e["category"] != QualityCategory.LOW_FREQUENCY_RESPONSE.value]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    loaded = load_result(path, capabilities=CAPABILITIES, directivity=DIRECTIVITY,
        quality_targets_path=TARGETS, physics_identity=saved.physics_identity)
    assert loaded.standing is ResultStanding.REMEASURED
    assert loaded.stored_candidate is not None
    assert QualityCategory.LOW_FREQUENCY_RESPONSE not in {e.category for e in loaded.stored_candidate.evaluations}
    row, = rank_candidates([loaded.result.candidate], load_quality_targets(TARGETS), _context(saved)).rankable
    line, = (c for c in row.categories if c.identity.category is QualityCategory.LOW_FREQUENCY_RESPONSE)
    assert line.weighted_cost > 0
    assert row.total_cost == pytest.approx(sum(c.weighted_cost for c in row.categories))


def test_changed_stage_one_scored_range_remeasures_saved_numbers(saved: SchemeResult, tmp_path: Path) -> None:
    changed = _rewrite(tmp_path, 'key = "stage_one_low_frequency.scored_range_hz"', "value", [40.0, 250.0])
    path = tmp_path / "saved.json"
    path.write_text(saved.model_dump_json(), encoding="utf-8")
    loaded = load_result(path, capabilities=CAPABILITIES, directivity=DIRECTIVITY,
        quality_targets_path=changed, physics_identity=saved.physics_identity)
    assert loaded.standing is ResultStanding.REMEASURED
    before, = (e for e in saved.candidate.evaluations if e.category is QualityCategory.LOW_FREQUENCY_RESPONSE)
    after, = (e for e in loaded.result.candidate.evaluations if e.category is QualityCategory.LOW_FREQUENCY_RESPONSE)
    assert before.raw_quantities != after.raw_quantities


def test_stage_one_cell_changes_both_new_category_fingerprints(saved: SchemeResult, tmp_path: Path) -> None:
    changed = _rewrite(tmp_path, 'key = "stage_one_low_frequency.scored_range_hz"', "value", [40.0, 250.0])
    new_registry = load_quality_targets(changed)
    old_registry = load_quality_targets(TARGETS)
    assert old_registry.fingerprint == new_registry.fingerprint
    after = reevaluate(saved, quality_targets_path=changed, capabilities=CAPABILITIES, directivity=DIRECTIVITY)
    before_rank, = rank_candidates([saved.candidate], old_registry, _context(saved)).rankable
    after_rank, = rank_candidates([after], new_registry, _context(saved)).rankable
    first, = (c for c in before_rank.categories if c.identity.category is QualityCategory.LOW_FREQUENCY_RESPONSE)
    second, = (c for c in after_rank.categories if c.identity.category is QualityCategory.LOW_FREQUENCY_RESPONSE)
    assert first.identity.settings_fingerprint != second.identity.settings_fingerprint
    assert first.identity.cost_settings_fingerprint != second.identity.cost_settings_fingerprint


def test_stage_one_registry_bytes_change_program_identity_and_reject_resume(tmp_path: Path) -> None:
    # 真正隔離套件位元組，不改正在跑的套件；計算指紋吃包內的 TOML，不靠測試假指紋。
    package = tmp_path / "aosr"
    shutil.copytree(TARGETS.parents[2], package, ignore=shutil.ignore_patterns("__pycache__"))
    copied = package / "config" / "data" / TARGETS.name
    cap = config_path("capabilities.toml")
    before = calculation_fingerprint(capabilities_path=cap, package_root=package)
    text = copied.read_text(encoding="utf-8")
    marker = 'key = "stage_one_low_frequency.scored_range_hz"'
    head, mark, rest = text.partition(marker)
    assert mark
    block, boundary, tail = rest.partition("[[")
    block = block.replace("value = [20.0, 300.0]", "value = [40.0, 250.0]")
    copied.write_text(head + mark + block + boundary + tail, encoding="utf-8")
    after = calculation_fingerprint(capabilities_path=cap, package_root=package)
    assert before != after
    purpose = purpose_settings(TARGETS, scheme().purpose)
    with pytest.raises(IdentityChanged, match="program_fingerprint"):
        _check_identity(SearchIdentity("same-physics", after, purpose),
                        SearchIdentity("same-physics", before, purpose), "接續")


@pytest.mark.parametrize("scene_kind", ("directivity", "furniture", "zones", "desk", "unequal_walls"))
def test_second_stage_scene_conditions_do_not_block_measurement(saved: SchemeResult, scene_kind: str) -> None:
    from aosr.reporting.low_frequency_response import evaluate_result_low_frequency
    from aosr.reporting.scheme import FurnitureSpec, Scheme, SpeakerSetup, SpeakerCabinet
    from aosr.scoring.contract import EvaluationState
    from aosr.scoring.contract_base import Flag
    from tests.engine._furniture_cases import relative_item
    document = saved.scheme.model_dump()
    document["source_model"] = "product_default"
    if scene_kind == "furniture":
        document["furniture"] = (FurnitureSpec.model_validate(relative_item()),)
    elif scene_kind == "zones":
        from aosr.physics.report_surfaces import WholeWallZone
        from tests.engine.test_measured_impedance import measured
        zone = WholeWallZone.model_validate({"area": "whole_wall", "material": measured()})
        document["scene"].pop("impedance_pa_s_per_m_by_wall")
        document["scene"]["surface_zones_by_wall"] = {wall: (zone,) for wall in ("floor", "ceiling", "x0", "xL", "y0", "yL")}
    elif scene_kind == "desk":
        document["furniture"] = (FurnitureSpec.model_validate(relative_item(kind="desk", material="wood")),)
        document["speaker_setup"] = SpeakerSetup(kind="bookshelf", mount="desk", representative=False,
            cabinet=SpeakerCabinet(width_m=.2, depth_m=.3, height_m=.4,
                acoustic_center_behind_front_m=.1, acoustic_center_above_bottom_m=.2))
    elif scene_kind == "unequal_walls":
        document["scene"]["impedance_pa_s_per_m_by_wall"]["floor"] *= 2
    changed = Scheme.model_validate(document)
    # 接線只讀既存有限元素欄；不呼叫第一階段場景檢查，也不硬改回全向。
    result = evaluate_result_low_frequency(saved.model_copy(update={"scheme": changed}), TARGETS,
        saved.candidate.scene_fingerprint)
    original, = (e for e in saved.candidate.evaluations if e.category is QualityCategory.LOW_FREQUENCY_RESPONSE)
    assert result.state is EvaluationState.MEASURED
    assert result.raw_quantities == original.raw_quantities
    assert result.source_model_fingerprint is None
    assert Flag.FURNITURE_MODEL_APPROXIMATE not in result.flags
    assert changed.source_model == "product_default"


def test_new_category_version_or_category_weight_only_keeps_existing_reload_policy(
    saved: SchemeResult, tmp_path: Path,
) -> None:
    document = saved.model_dump_json().replace("aosr.scoring.low_frequency_response.v1",
                                              "aosr.scoring.low_frequency_response.v999")
    path = tmp_path / "version.json"
    path.write_text(document, encoding="utf-8")
    loaded = load_result(path, capabilities=CAPABILITIES, directivity=DIRECTIVITY,
        quality_targets_path=TARGETS, physics_identity=saved.physics_identity)
    assert loaded.standing is ResultStanding.CURRENT
    # 只改外層類權重：量法不變，照既有機制 reranked（重新排名）。
    text = TARGETS.read_text(encoding="utf-8")
    head, marker, rest = text.partition('name = "low_frequency_response"')
    assert marker
    changed = tmp_path / "weights.toml"
    changed.write_text(head + marker + rest.replace("value = 0.25", "value = 0.5", 1), encoding="utf-8")
    loaded = load_result(path, capabilities=CAPABILITIES, directivity=DIRECTIVITY,
        quality_targets_path=changed, physics_identity=saved.physics_identity)
    assert loaded.standing is ResultStanding.RERANKED


def test_settings_fingerprint_only_change_does_not_remeasure_identical_numbers(
    saved: SchemeResult, tmp_path: Path,
) -> None:
    # less_is_better（越小越好）不讀目標的 value，但設定身分仍收它；只變身分不能冒充重量測。
    changed = _rewrite(tmp_path, 'key = "stage_one_low_frequency.own_line_ripple_rms_db"', "value", 0.2)
    path = tmp_path / "same-numbers.json"
    path.write_text(saved.model_dump_json(), encoding="utf-8")
    loaded = load_result(path, capabilities=CAPABILITIES, directivity=DIRECTIVITY,
        quality_targets_path=changed, physics_identity=saved.physics_identity)
    before, = (e for e in saved.candidate.evaluations if e.category is QualityCategory.LOW_FREQUENCY_RESPONSE)
    after, = (e for e in loaded.result.candidate.evaluations if e.category is QualityCategory.LOW_FREQUENCY_RESPONSE)
    assert before.settings_fingerprint != after.settings_fingerprint
    assert before.raw_quantities == after.raw_quantities
    assert loaded.standing is ResultStanding.CURRENT
