"""評估器交出的原始量單位要等於登記簿那一條的單位（#405 第 2 條）。

代價端不讀原始量，讀的是評估結果本體的欄位；它自己的「鍵→預期單位」表每次讀登記簿都會核。
原始量上的單位是各評估器另外寫的字面值，報表與存檔照抄，以前沒有人拿它對登記簿：
評估器哪天改了宣告的單位、登記簿沒跟（或反過來），這幾題會紅。
登記簿沒有對應鍵的原始量逐名列在下面；評估器多交一個沒列、也對不到鍵的原始量也紅，要寫的人自己決定它對哪一條。
"""
from __future__ import annotations

from collections.abc import Callable

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import TargetEntry, load_quality_targets
from aosr.scoring.channel_matching_cost import _MEAN_KEYS, _WORST_KEYS
from aosr.scoring.contract import CategoryEvaluation, EvaluationState
from aosr.scoring.listening_area_cost import _LISTENING_AREA_TARGET_KEYS
from aosr.scoring.timbre_cost import TIMBRE_TARGET_KEYS
from tests.engine.test_channel_matching import _channel_points, _group, _receivers
from tests.engine.test_channel_matching import _evaluate as _channel_matching_evaluate
from tests.engine.test_listening_area import _evaluate as _listening_area_evaluate
from tests.engine.test_listening_area import _receiver_set, _results
from tests.engine.test_scoring_timbre import _evaluate as _timbre_evaluate
from tests.engine.test_scoring_timbre import _flat_input

_TIMBRE_KEYS = {
    "tilt_db_per_octave": TIMBRE_TARGET_KEYS["tilt"][0],
    "residual_rms_db": TIMBRE_TARGET_KEYS["residual_rms"][0],
    "target_deviation_rms_db": TIMBRE_TARGET_KEYS["target_deviation"][0],
}
_CHANNEL_STATS = {"weighted_mean_absolute_difference": _MEAN_KEYS, "worst_absolute_difference": _WORST_KEYS}


def _timbre_key(name: str) -> str | None:
    return _TIMBRE_KEYS.get(name)


def _listening_area_key(name: str) -> str | None:
    metric, _, stat = name.split(".")
    return _LISTENING_AREA_TARGET_KEYS.get(f"{metric}_{stat}")


def _channel_matching_key(name: str) -> str | None:
    if name == "assessed_point_count":
        return None
    _, _, metric, stat = name.split(".")
    return _CHANNEL_STATS[stat][metric]


def _timbre() -> CategoryEvaluation:
    return _timbre_evaluate(_flat_input())


def _listening_area() -> CategoryEvaluation:
    receivers = _receiver_set()
    return _listening_area_evaluate(receivers, _results(receivers))


def _channel_matching() -> CategoryEvaluation:
    receivers, group = _receivers(), _group()
    return _channel_matching_evaluate(receivers, group, _channel_points(receivers, group))


# 每一類：怎麼造一份真的量到的評估、原始量名字怎麼對到登記簿鍵、該蓋到哪些鍵、哪些名字本來就沒有鍵。
_CASES: dict[str, tuple[Callable[[], CategoryEvaluation], Callable[[str], str | None], frozenset[str], frozenset[str]]] = {
    "timbre": (_timbre, _timbre_key, frozenset(_TIMBRE_KEYS.values()), frozenset({"feature_count"})),
    "listening_area": (
        _listening_area, _listening_area_key, frozenset(_LISTENING_AREA_TARGET_KEYS.values()),
        frozenset(f"peak_dip_consistency.{group}.{stat}"
                  for group in ("primary_to_surrounding", "surrounding_to_surrounding")
                  for stat in ("weighted_mean_deviation", "worst_deviation")),
    ),
    "channel_matching": (
        _channel_matching, _channel_matching_key,
        frozenset(_MEAN_KEYS.values()) | frozenset(_WORST_KEYS.values()), frozenset({"assessed_point_count"}),
    ),
}


@pytest.mark.parametrize("category", sorted(_CASES))
def test_raw_quantity_units_match_the_registry(category: str) -> None:
    build, key_of, expected_keys, keyless = _CASES[category]
    evaluation = build()
    assert evaluation.state is EvaluationState.MEASURED
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose("dedicated_two_channel_listening_room")
    mismatched: list[tuple[str, str, str]] = []
    covered: set[str] = set()
    unmapped: set[str] = set()
    for quantity in evaluation.raw_quantities:
        key = key_of(quantity.name)
        if key is None:
            unmapped.add(quantity.name)
            continue
        entry = purpose.entry(key)
        assert isinstance(entry, TargetEntry), key
        covered.add(key)
        if quantity.unit != entry.unit:
            mismatched.append((quantity.name, quantity.unit, entry.unit))
    assert mismatched == []
    # 每一條有對應原始量的門檻鍵都要真的被這份評估蓋到（只蓋到一部分的話，沒蓋到的那幾條等於沒核）。
    # 音色的峰谷深度兩條鍵（peak_depth_db、dip_depth_db）讀的是每個峰谷自己的深度，沒有對應的原始量，不在這裡核。
    assert covered == expected_keys
    assert unmapped <= keyless
