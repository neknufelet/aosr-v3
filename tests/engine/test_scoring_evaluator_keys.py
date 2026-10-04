"""各類宣告的「評估器那一層讀的尺」要齊（#633）：報告判某一類用到的尺全部校準完，靠的就是這份宣告。

兩道：①重評一份控制組結果時記下每一處 QualityPurpose.entry 讀到的鍵，跟五類宣告的聯集比——評估器多讀一條、
宣告沒跟上就紅；②逐條把登記簿的值改成另一個合法值再重評，哪一類的評估輸出變了，那一類就必須宣告那一條——吃上游評估
（聆聽區、聲道匹配吃逐座位音色）或編排層代讀交進去的，都由這一道抓。照規則整份登記簿的指紋不算「用到」，
所以那兩處指紋（整份登記簿、整個用途）在這裡固定住；只從自己那幾條值算出來的設定指紋（例如反射的）照比——
它被抄進別類的輸出，就是別類用到了那幾條（複查）。音色的設定指紋與聲道匹配的 registry_fingerprint 本身就是整份
登記簿的指紋，跟著固定，那兩類靠真的數字被抓。只改狀態、不改值的流向這一道看不到，由報告考卷逐條留一守住。
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityEntry, QualityPurpose, QualityTargets
from aosr.reporting.evaluation import reevaluate
from aosr.reporting.result import SchemeResult
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import CandidateEvaluation, QualityCategory
from tests.engine import _scoring_source_model_control as control
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_scheme_pipeline import shared_control_result

# 每一條改成另一個合法值（照現在登記簿的值挑，改了要確實跟原值不同）；評估器多讀一條，這裡要補一種改法。
CHANGED: dict[str, object] = {
    "timbre_balance.coverage_range_hz": [40.0, 8000.0],
    "timbre_balance.tilt_fit_range_hz": [100.0, 4000.0],
    "timbre_balance.ripple_range_hz": [50.0, 8000.0],
    "timbre_balance.smoothing_width_octave_tilt": 0.5,
    "timbre_balance.smoothing_width_octave_ripple": 0.1666,
    "timbre_balance.feature_min_width_octave": 0.5,
    "timbre_balance.min_points": 100000,
    "timbre_balance.target_tilt_db_per_octave": -1.0,
    "channel_matching.broadband_range_hz": [40.0, 8000.0],
    "channel_matching.direct_time_cost_enabled": 1,
    "reflections_and_echo.window_upper_ms": 20.0,
    "reflections_and_echo.frequency_range_hz": [400.0, 8000.0],
    "reflections_and_echo.flutter_decay_db": 40.0,
    "reflections_and_echo.flutter_alert_band_centers_hz": [400.0, 500.0, 630.0],
    # 方向分區要改到真的有反射換區、聲道匹配讀的那一格最強路徑跟著變（30→20、135→120 在控制組裡換不到，複查）。
    "direction_zones.vertical_min_abs_elevation_deg": 60.0,
    "direction_zones.front_max_abs_azimuth_deg": 30.0,
    "direction_zones.rear_min_abs_azimuth_deg": 170.0,
    "reverberation.adjacent_t20_logarithm_base": 10.0,
}


@pytest.fixture(scope="module")
def wall_1(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


def _fixed(label: str) -> property:
    def fingerprint(self: object) -> str:
        return label
    return property(fingerprint)


def _reevaluate(result: SchemeResult, registry: Path) -> CandidateEvaluation:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(QualityTargets, "fingerprint", _fixed("整份登記簿"))
        patch.setattr(QualityPurpose, "fingerprint", _fixed("整個用途"))
        return reevaluate(result, quality_targets_path=registry,
                          capabilities=load_capabilities(config_path("capabilities.toml")), directivity=DIRECTIVITY)


def _outputs(candidate: CandidateEvaluation) -> dict[QualityCategory, str]:
    return {item.category: json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in candidate.evaluations}


def _with_value(text: str, key: str, value: object) -> str:
    """登記簿 TOML 裡含這個鍵的那一個條目區塊，把 value 那一行換掉（每個用途各一塊，全部換）。"""
    blocks = text.split("\n[[")
    hits = [index for index, block in enumerate(blocks) if f'key = "{key}"\n' in block]
    assert hits, key
    for index in hits:
        blocks[index], count = re.subn(r"(?m)^value = .*$", f"value = {json.dumps(value)}", blocks[index])
        assert count == 1, key
    return "\n[[".join(blocks)


def _declared(category: QualityCategory) -> tuple[str, ...]:
    return CATEGORY_REGISTRY[category].evaluator_keys


def test_every_key_the_evaluators_read_is_declared(wall_1: SchemeResult, monkeypatch: pytest.MonkeyPatch) -> None:
    read: set[str] = set()
    original: Callable[[QualityPurpose, str], QualityEntry] = QualityPurpose.entry

    def recording(self: QualityPurpose, key: str) -> QualityEntry:
        read.add(key)
        return original(self, key)

    monkeypatch.setattr(QualityPurpose, "entry", recording)
    _reevaluate(wall_1, control.TARGETS)
    declared = {key for category in CATEGORY_REGISTRY for key in _declared(category)}
    assert read == declared
    # 改法清單也要跟實際讀到的一致：多讀一條就得補一種改法，第二道才蓋得到它。
    assert set(CHANGED) == read


@pytest.fixture(scope="module")
def baseline(wall_1: SchemeResult) -> dict[QualityCategory, str]:
    return _outputs(_reevaluate(wall_1, control.TARGETS))


@pytest.mark.parametrize("key", sorted(CHANGED))
def test_every_category_a_key_changes_declares_it(
        key: str, wall_1: SchemeResult, baseline: dict[QualityCategory, str], tmp_path: Path) -> None:
    registry = tmp_path / "quality_targets.toml"
    registry.write_text(_with_value(control.TARGETS.read_text(encoding="utf-8"), key, CHANGED[key]), encoding="utf-8")
    changed = {category for category, text in _outputs(_reevaluate(wall_1, registry)).items()
               if text != baseline[category]}
    # 改了沒有任何一類變，這一條的改法就證明不了任何依賴，要換一種改法。
    assert changed, key
    assert {category.value for category in changed if key not in _declared(category)} == set()
