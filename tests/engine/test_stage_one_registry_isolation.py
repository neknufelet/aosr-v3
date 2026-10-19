"""第一階段條目不連帶動第二階段（票 #558，審查 M8；老闆 2026-10-20「不連帶重做第二階段」）。

控制組：同一份第二階段控制組結果，在「拿掉第一階段那一段」與「正式登記簿（有那一段）」兩份登記簿上
各重評一次、各排一次、各讀回一次，量出的內容、比較身分、用途快照、整份登記簿指紋與讀回分級逐字相同。
拿掉那一段的登記簿就是加條目以前的形狀；兩份都寫在 ``tmp_path``，不寫進真的 repo。
"""
from __future__ import annotations

import json
import tomllib
from datetime import date
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.reporting.evaluation import ResultStanding, load_result, purpose_settings, reevaluate
from aosr.reporting.result import SchemeResult
from aosr.search.scoring import screening_outcome
from tests.engine import _scoring_source_model_control as control
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_scheme_pipeline import shared_control_result

_SECTION_HEADER = "\n[[stage_one]]\n"
_PURPOSE_HEADER = "\n[[purpose]]\n"


@pytest.fixture(scope="module")
def wall_1(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


@pytest.fixture
def registries(tmp_path: Path) -> tuple[Path, Path]:
    """（沒有第一階段那一段、正式登記簿原樣）兩份；那一段在第一個用途之前，從第一個 [[stage_one]] 切到第一個用途表頭。"""
    text = control.TARGETS.read_text(encoding="utf-8")
    head, marker, rest = text.partition(_SECTION_HEADER)
    assert marker, "正式登記簿沒有第一階段那一段"
    _, purposes, tail = rest.partition(_PURPOSE_HEADER)
    assert purposes, "第一階段那一段後面找不到用途表頭，形狀變了"
    head = head + purposes + tail
    full, stripped = tomllib.loads(text), tomllib.loads(head)
    # 切掉的只有第一階段那一段：其餘頂層內容逐格相同。
    assert "stage_one" in full and "stage_one" not in stripped
    assert {key: value for key, value in full.items() if key != "stage_one"} == stripped
    without = tmp_path / "without_stage_one.toml"
    with_section = tmp_path / "with_stage_one.toml"
    without.write_text(head, encoding="utf-8")
    with_section.write_text(text, encoding="utf-8")
    return without, with_section


def _reevaluated(result: SchemeResult, registry: Path) -> str:
    candidate = reevaluate(result, quality_targets_path=registry,
                           capabilities=load_capabilities(config_path("capabilities.toml")),
                           directivity=DIRECTIVITY)
    return json.dumps(candidate.model_dump(mode="json"), sort_keys=True)


def test_stage_one_section_leaves_registry_and_purpose_identity_unchanged(registries: tuple[Path, Path]) -> None:
    """整份登記簿指紋（音色設定指紋、排名表頭、結果檔都抄它）與第二階段用途快照，有沒有那一段都一樣。"""
    without, with_section = registries
    assert load_quality_targets(without).fingerprint == load_quality_targets(with_section).fingerprint
    assert purpose_settings(without, control.PURPOSE) == purpose_settings(with_section, control.PURPOSE)
    # 那一段真的讀得到：不是因為讀不進來才相等。
    assert load_quality_targets(with_section).stage_one(control.PURPOSE).entries


def test_stage_two_reevaluation_and_ranking_are_byte_identical(
        wall_1: SchemeResult, registries: tuple[Path, Path]) -> None:
    """同一份第二階段結果在兩份登記簿上重評：候選包逐字相同；排名的總代價與比較身分也相同。"""
    without, with_section = registries
    assert _reevaluated(wall_1, without) == _reevaluated(wall_1, with_section)
    candidate = reevaluate(wall_1, quality_targets_path=without,
                           capabilities=load_capabilities(config_path("capabilities.toml")),
                           directivity=DIRECTIVITY)
    outcomes = [
        screening_outcome(candidate, wall_1.scheme, registry=load_quality_targets(path),
                          run_date=date(2026, 10, 20), engine_version="control", pinned=None)
        for path in (without, with_section)
    ]
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][1] is not None, "控制組要排得進表，比較身分才有東西比"


def test_saved_stage_two_result_reload_standing_is_unchanged(
        wall_1: SchemeResult, registries: tuple[Path, Path], tmp_path: Path) -> None:
    """第二階段的存檔在兩份登記簿上讀回都是「目前」（current），不是「只重新排名」。"""
    saved = tmp_path / "saved.json"
    saved.write_text(wall_1.model_dump_json(), encoding="utf-8")
    standings = {
        path.name: load_result(saved, capabilities=load_capabilities(config_path("capabilities.toml")),
                               directivity=DIRECTIVITY, quality_targets_path=path,
                               physics_identity=wall_1.physics_identity).standing
        for path in registries
    }
    assert set(standings.values()) == {ResultStanding.CURRENT}
    assert wall_1.quality_targets_fingerprint == load_quality_targets(registries[0]).fingerprint
