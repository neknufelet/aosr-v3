"""排名層考卷的共用夾具（原樣從 test_scoring_ranking.py 搬出，騰行數；#505 第三刀）。

test_scoring_ranking.py 用同名轉出，其他考卷照舊 ``from tests.engine import test_scoring_ranking as ranking_fixtures`` 取用。
"""
from __future__ import annotations

import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Final


from aosr.config.frequency_axis import GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
from aosr.config.paths import config_path
from aosr.config.quality_targets import (
    QualityTargets,
)
from aosr.scoring import ranking
from aosr.scoring.channel_matching import ChannelDefinition, ChannelGroup
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    InputProvenance,
)
from aosr.scoring.ranking import (
    ExternalFloors,
    RankingContext,
    RankingResult,
)
from aosr.scoring.timbre_channels import (
    evaluate_timbre_channels,
)
from tests.engine._placement import EMPTY_PLACEMENT, POINT_PLACEMENT


_PURPOSE_NAME: Final[str] = "dedicated_two_channel_listening_room"
_EVALUATOR: Final[str] = "timbre-fixture-v1"
_SETTINGS: Final[str] = "timbre-settings-a"
_SCENE_FINGERPRINT: Final[str] = "a" * 64
_TIMBRE_GROUP: Final[ChannelGroup] = ChannelGroup(
    channels=(ChannelDefinition(role="left", speaker_id="left"),),
    comparisons=(),
    feature_match_tolerance_hz=0.0,
)
_CONTEXT: Final[RankingContext] = RankingContext(
    purpose=_PURPOSE_NAME,
    receiver_set_fingerprint="receivers-fixture",
    channel_group_fingerprint=_TIMBRE_GROUP.fingerprint,
    run_date=date(2026, 9, 19),
    engine_version="engine-fixture",
)
# 警戒以下的峰谷：可排名的候選都帶這一組，外加一個太窄、深度遠超警戒的峰——
# 它保留在清單上、帶標記，但不進代價、也不掛警戒。
_QUIET_FEATURES: Final[list[dict[str, object]]] = [
    {"kind": "peak", "center_frequency_hz": 100.0, "depth_db": 2.0, "width_octave": 0.5, "flags": []},
    {
        "kind": "dip",
        "center_frequency_hz": 200.0,
        "depth_db": -2.5,
        "width_octave": None,
        "flags": ["feature_boundary_incomplete"],
    },
    {
        "kind": "peak",
        "center_frequency_hz": 300.0,
        "depth_db": 20.0,
        "width_octave": 0.05,
        "flags": ["feature_too_narrow"],
    },
]


_TILT_KEY: Final[str] = "timbre_balance.target_tilt_db_per_octave"
# 考卷自己的尺：驗算法用，不跟著產品數字走（理由見檔頭）。
_PINNED_TILT_TOLERANCE: Final[str] = "0.5"
_PINNED_TILT_WORSE_REFERENCE: Final[str] = "2.0"


def _registry() -> QualityTargets:
    """正式登記簿，但傾斜那一條的容許帶釘成這支考卷自己的值（理由見檔頭）。"""
    text = config_path("quality_targets.toml").read_text(encoding="utf-8")
    head, marker, rest = text.partition(f'key = "{_TILT_KEY}"')
    assert marker, f"正式登記簿裡找不到 {_TILT_KEY}，形狀變了"
    block, next_table, tail = rest.partition("[[purpose.")
    pinned, tolerance_hits = re.subn(
        r"^tolerance = .*$", f"tolerance = {_PINNED_TILT_TOLERANCE}", block, count=1, flags=re.M
    )
    pinned, reference_hits = re.subn(
        r"^worse_reference = .*$", f"worse_reference = {_PINNED_TILT_WORSE_REFERENCE}", pinned, count=1, flags=re.M
    )
    assert tolerance_hits == 1, f"{_TILT_KEY} 沒有 tolerance 可以釘，形狀變了"
    assert reference_hits == 1, f"{_TILT_KEY} 沒有 worse_reference 可以釘，形狀變了"
    return QualityTargets.model_validate(tomllib.loads(head + marker + pinned + next_table + tail))


def _registry_with_unit(
    tmp_path: Path, *, key: str, expected: str, changed: str
) -> QualityTargets:
    text = config_path("quality_targets.toml").read_text(encoding="utf-8")
    marker = f'key = "{key}"'
    head, found, tail = text.partition(marker)
    assert found
    block, next_table, rest = tail.partition("[[purpose.")
    old = f'unit = "{expected}"'
    assert old in block
    path = tmp_path / "quality_targets.toml"
    path.write_text(
        head + found + block.replace(old, f'unit = "{changed}"', 1) + next_table + rest,
        encoding="utf-8",
    )
    return QualityTargets.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def _provenance(candidate_id: str) -> InputProvenance:
    return InputProvenance(
        report_id=f"report-{candidate_id}",
        engine_commit="fixture-engine",
        speaker_id="left",
        receiver_id="main-seat",
    )


def _single_timbre(
    candidate_id: str,
    *,
    tilt: float = 1.0,
    residual: float = 2.0,
    target_deviation: float = 3.0,
    features: list[dict[str, object]] | None = None,
    evaluator_version: str = _EVALUATOR,
    settings_fingerprint: str = _SETTINGS,
    flags: tuple[str, ...] = ("unvalidated",),
) -> CategoryEvaluation:
    chosen = _QUIET_FEATURES if features is None else features
    peaks = [index for index, item in enumerate(chosen) if item["kind"] == "peak"]
    dips = [index for index, item in enumerate(chosen) if item["kind"] == "dip"]
    return CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "scene_fingerprint": _SCENE_FINGERPRINT,
            "placement": POINT_PLACEMENT,
            "category": "timbre_balance",
            "state": "measured",
            "payload": {
                "category": "timbre_balance",
                "tilt_db_per_octave": tilt,
                "tilt_fit_range_hz": [80.0, 4000.0],
                "tilt_dependency_range_hz": [71.0, 4490.0],
                "target_tilt_db_per_octave": 0.0,
                "target_deviation_rms_db": target_deviation,
                "deviation_curve": [[100.0, -1.0], [200.0, 1.0]],
                "residual_rms_db": residual,
                "ripple_range_hz": [40.0, 4000.0],
                "ripple_dependency_range_hz": [37.0, 4240.0],
                "features": chosen,
                "strongest_peak_index": peaks[0] if peaks else None,
                "deepest_dip_index": dips[0] if dips else None,
                "data_range_hz": [20.0, 8000.0],
                "coverage_range_hz": [20.0, 8000.0],
                "model_validation_status": "validated",
                "model_validation_frequency_range_hz": [20.0, 8000.0],
            },
            "raw_quantities": [
                {"name": "tilt", "value": tilt, "unit": "dB/oct"},
                {"name": "residual_rms", "value": residual, "unit": "dB"},
                {"name": "target_deviation", "value": target_deviation, "unit": "dB"},
            ],
            "category_cost": None,
            "flags": list(flags),
            "reason_codes": [],
            "evaluator_version": evaluator_version,
            "settings_fingerprint": settings_fingerprint,
            "provenance": _provenance(candidate_id),
        }
    )


def _timbre(
    candidate_id: str,
    *,
    tilt: float = 1.0,
    residual: float = 2.0,
    target_deviation: float = 3.0,
    features: list[dict[str, object]] | None = None,
    evaluator_version: str = _EVALUATOR,
    settings_fingerprint: str = _SETTINGS,
    flags: tuple[str, ...] = ("unvalidated",),
) -> CategoryEvaluation:
    """排名候選只走主位聲道彙總；這組舊考卷用單聲道維持原本手算值。"""
    single = _single_timbre(
        candidate_id,
        tilt=tilt,
        residual=residual,
        target_deviation=target_deviation,
        features=features,
        evaluator_version=evaluator_version,
        settings_fingerprint=settings_fingerprint,
        flags=flags,
    )
    return evaluate_timbre_channels(
        _TIMBRE_GROUP,
        "main-seat",
        {"left": single},
        candidate_id=candidate_id,
        scene_fingerprint=_SCENE_FINGERPRINT,
        timbre_settings_fingerprint=single.settings_fingerprint,
    )


def _unavailable(candidate_id: str, category: str, reasons: tuple[str, ...]) -> CategoryEvaluation:
    return CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "scene_fingerprint": _SCENE_FINGERPRINT,
            "placement": EMPTY_PLACEMENT,
            "category": category,
            "state": "unavailable",
            "payload": None,
            "raw_quantities": [],
            "category_cost": None,
            "flags": ["data_coverage_short"],
            "reason_codes": list(reasons),
            "evaluator_version": _EVALUATOR,
            "settings_fingerprint": _SETTINGS,
            "provenance": _provenance(candidate_id),
        }
    )


# 殘響那一類的假 payload：這支考卷驗的是排名層怎麼處理「選評類」，不是殘響怎麼量；
# 正式報表帶都有 T20，讓這份共用樣本不會另踩本票新增的三條資料資格。
_REVERBERATION_METRIC: Final[dict[str, object]] = {
    "value": 1.0,
    "unit": "s",
    "state": "measured",
    "reason_codes": [],
    "reason": None,
}
_REVERBERATION_CENTERS: Final[tuple[float, ...]] = (
    GEOMETRIC_REPORT_OCTAVE_CENTERS_HZ
)
_REVERBERATION_PAYLOAD: Final[dict[str, object]] = {
    "category": "reverberation",
    "bands": [
        {
            "center_frequency_hz": center,
            "band_range_hz": [center / 2**0.5, center * 2**0.5],
            "schroeder_position": "above",
            "model_validation_status": "experimental",
            "t20": _REVERBERATION_METRIC,
            "t30": {**_REVERBERATION_METRIC, "value": 1.1},
            "fitting_difference": {**_REVERBERATION_METRIC, "unit": "1", "value": 1.1},
        }
        for center in _REVERBERATION_CENTERS
    ],
    "adjacent_band_changes": [
        {
            "lower_center_frequency_hz": lower,
            "upper_center_frequency_hz": upper,
            "signed_log_ratio": 0.0,
            "state": "measured",
            "reason_codes": [],
            "reason": None,
        }
        for lower, upper in zip(
            _REVERBERATION_CENTERS[:-1],
            _REVERBERATION_CENTERS[1:],
            strict=True,
        )
    ],
    "logarithm_base": 2.0,
}


def _reverberation(candidate_id: str) -> CategoryEvaluation:
    """殘響（選評類）的原始量；代價一律留給排名層依這一跑的登記簿算。"""
    return CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "scene_fingerprint": _SCENE_FINGERPRINT,
            "placement": POINT_PLACEMENT,
            "category": "reverberation",
            "state": "measured",
            "payload": _REVERBERATION_PAYLOAD,
            "raw_quantities": [{"name": "t30_spread", "value": 0.2, "unit": "1"}],
            "category_cost": None,
            "flags": [],
            "reason_codes": [],
            "evaluator_version": "reverb-fixture-v1",
            "settings_fingerprint": "reverb-settings-a",
            "provenance": _provenance(candidate_id),
        }
    )


def _uncosted_category(candidate_id: str) -> CategoryEvaluation:
    """尚未註冊第二層代價的選評類，刻意帶著不可信的上游代價。

    聲道匹配從 #350 起有自己的代價，所以這裡改用還沒有代價的低頻拖尾；
    要守的事沒變：沒有代價器的類，排名層不准自己生一個代價出來。
    """
    return CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": candidate_id,
            "scene_fingerprint": _SCENE_FINGERPRINT,
            "placement": POINT_PLACEMENT,
            "category": "low_frequency_decay",
            "state": "costed",
            "payload": {"category": "low_frequency_decay"},
            "raw_quantities": [{"name": "fixture", "value": 1.0, "unit": "1"}],
            "category_cost": {
                "value": 0.0,
                "components": {},
                "cost_settings_fingerprint": "unregistered-upstream-cost",
            },
            "flags": [],
            "reason_codes": [],
            "evaluator_version": "channel-matching-fixture-v1",
            "settings_fingerprint": "channel-matching-settings-a",
            "provenance": _provenance(candidate_id),
        }
    )


def _candidate(*evaluations: CategoryEvaluation) -> CandidateEvaluation:
    first = evaluations[0]
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=first.candidate_id,
        scene_fingerprint=first.scene_fingerprint,
        evaluations=evaluations,
    )


def _rank(
    *candidates: CandidateEvaluation,
    registry: QualityTargets | None = None,
    external: ExternalFloors | None = None,
) -> RankingResult:
    return ranking.rank_candidates(candidates, registry or _registry(), _CONTEXT, external)


def _order(result: RankingResult) -> list[str]:
    return [row.candidate_id for row in result.rankable]


def _three() -> list[CandidateEvaluation]:
    """固定三候選。類代價＝傾斜 max(0,|t|−0.5)/2 ＋ 殘差 r/4（基線等權）：
    A 0.05+0.25=0.30、B 0.15+0.375=0.525、C 0.25+0.5=0.75。"""
    return [
        _candidate(_timbre("candidate-a", tilt=0.6, residual=1.0)),
        _candidate(_timbre("candidate-b", tilt=0.8, residual=1.5)),
        _candidate(_timbre("candidate-c", tilt=1.0, residual=2.0)),
    ]


def _calibrated_registry(keep_baseline: frozenset[str] = frozenset()) -> QualityTargets:
    """正式登記簿的每一條升 calibrated、補齊結構化出處；``keep_baseline`` 列的鍵留在 baseline。

    權重列用「表鍵.列名」指名（例如 ``ranking.category_weights.timbre_balance``）。
    """
    structured = {
        "source_id": "fixture-source",
        "source_version": "1",
        "locator": "p. 1",
        "conditions": "fixture",
        "frequency_range_hz": [20.0, 8000.0],
        "context": "fixture",
        "verification_digest": "sha256:" + "0" * 64,
    }
    document = _registry().model_dump(mode="json", by_alias=True)
    for purpose in document["purpose"]:
        rows = purpose["setting"] + purpose["target"] + purpose["qualification"]
        named = [(row["key"], row) for row in rows]
        for table in purpose["weight"]:
            named += [(f"{table['key']}.{row['name']}", row) for row in table["item"]]
        for key, row in named:
            if key not in keep_baseline:
                row.update(structured, status="calibrated")
    return QualityTargets.model_validate(document)
