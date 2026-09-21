"""排名層最小版的假樣本考卷（票 #357）：第二層代價、第三層狀態與分表、五塊排名資料。

假樣本全部用凍結契約的模型造（契約的不變條件在建構時就會擋），登記簿用正式那一份；
要「全部已校準」的登記簿時，在記憶體裡把正式那一份的每一條升成 calibrated 並補齊結構化出處，
不寫檔。預期的代價數字都由登記簿的值手算，算式寫在每一題旁邊。

**傾斜那一條的容許帶在這支考卷裡是釘死的**（`_PINNED_TILT_TOLERANCE`）：這裡驗的是代價的
算法，不是今天的產品數字。產品數字是一題一題拍出來的（#368 拍目標值、#376 拍容許帶與較差
參考，後面還有六題），每拍一次就讓這幾題的算式跟著改，等於考卷跟著被測的東西一起漂。
"""
from __future__ import annotations

import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Final

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import (
    QualificationEntry,
    QualityTargets,
    load_quality_targets,
)
from aosr.scoring import ranking
from aosr.scoring.category_registry import CATEGORY_REGISTRY
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    EvaluationState,
    Flag,
    InputProvenance,
    QualityCategory,
    ReasonCode,
    TimbrePayload,
)
from aosr.scoring.ranking import (
    CandidateStatus,
    EligibilityApplication,
    EliminationReason,
    ExternalAcceptance,
    ExternalFloors,
    NotEvaluatedReason,
    RankingContext,
    RankingResult,
)
from tests.engine._placement import EMPTY_PLACEMENT, POINT_PLACEMENT


_PURPOSE_NAME: Final[str] = "dedicated_two_channel_listening_room"
_EVALUATOR: Final[str] = "timbre-fixture-v1"
_SETTINGS: Final[str] = "timbre-settings-a"
_SCENE_FINGERPRINT: Final[str] = "a" * 64
_CONTEXT: Final[RankingContext] = RankingContext(
    purpose=_PURPOSE_NAME,
    receiver_set_fingerprint="receivers-fixture",
    channel_group_fingerprint="channels-fixture",
    run_date=date(2026, 9, 19),
    engine_version="engine-fixture",
)
# 門檻以下的峰谷（正式登記簿基線門檻是 3 dB）：可排名的候選都帶這一組，外加一個太窄、
# 深度遠超門檻的峰——它保留在清單上、帶標記，但不進代價、也不淘汰。
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
# 六帶都有 T20，讓這份共用樣本不會另踩本票新增的三條資料資格。
_REVERBERATION_METRIC: Final[dict[str, object]] = {
    "value": 1.0,
    "unit": "s",
    "state": "measured",
    "reason_codes": [],
    "reason": None,
}
_REVERBERATION_CENTERS: Final[tuple[float, ...]] = (
    125.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
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


# ── 第二層：代價 ────────────────────────────────────────────────────────────


def test_timbre_costing_uses_three_shapes_without_mutating_input() -> None:
    """改錯任一公式、把太窄的峰算進去、讓對照或保護進類代價、或原地改輸入，都會紅。

    傾斜 (|1−0|−0.5)/2=0.25；殘差 2/4=0.5；對照 3/6=0.5；峰谷只算 5 dB 峰與 −4 dB 谷
    （寬度未知照算）：(5−3)/9+(4−3)/9=1/3，20 dB 的太窄峰不進。類代價只加主要分項：0.75。
    """
    registry = _registry()
    features: list[dict[str, object]] = [
        {"kind": "peak", "center_frequency_hz": 100.0, "depth_db": 5.0, "width_octave": 0.5, "flags": []},
        {
            "kind": "dip",
            "center_frequency_hz": 200.0,
            "depth_db": -4.0,
            "width_octave": None,
            "flags": ["feature_boundary_incomplete"],
        },
        _QUIET_FEATURES[2],
    ]
    measured = _timbre("candidate-a", features=features)

    costed = ranking.cost_timbre_evaluation(measured, registry.purpose(_PURPOSE_NAME), registry.fingerprint)

    assert measured.state is EvaluationState.MEASURED
    assert measured.category_cost is None
    assert costed.state is EvaluationState.COSTED
    assert costed.category_cost is not None
    assert costed.category_cost.value == pytest.approx(0.75)
    assert costed.category_cost.components == pytest.approx(
        {"tilt": 0.25, "residual_rms": 0.5, "target_deviation": 0.5, "peaks_dips": 1.0 / 3.0}
    )
    assert costed.category_cost.cost_settings_fingerprint == registry.fingerprint
    assert costed.payload == measured.payload
    assert costed.raw_quantities == measured.raw_quantities
    assert costed.flags == measured.flags


def test_residual_rms_rejects_registry_unit_mismatch(tmp_path: Path) -> None:
    """若代價接線不驗 dB，殘差數值不變但報表可被登記簿錯標成秒。"""
    key = "timbre_balance.residual_rms_db"
    registry = _registry_with_unit(tmp_path, key=key, expected="dB", changed="s")

    with pytest.raises(
        ValueError,
        match=f"{key} 單位應為 dB，登記簿寫 s",
    ):
        _rank(_candidate(_timbre("candidate-a")), registry=registry)


def test_tilt_inside_tolerance_costs_nothing() -> None:
    """範圍內最好：傾斜落在容許帶（基線 ±0.5 dB/oct）內代價是零，帶外才開始長。"""
    registry = _registry()
    purpose = registry.purpose(_PURPOSE_NAME)
    inside = ranking.cost_timbre_evaluation(_timbre("inside", tilt=-0.4), purpose, registry.fingerprint)
    outside = ranking.cost_timbre_evaluation(_timbre("outside", tilt=-0.9), purpose, registry.fingerprint)

    assert inside.category_cost is not None
    assert outside.category_cost is not None
    assert inside.category_cost.components["tilt"] == 0.0
    assert outside.category_cost.components["tilt"] == pytest.approx(0.2)


# ── 排序 ────────────────────────────────────────────────────────────────────


def test_fixed_candidates_rank_in_expected_order() -> None:
    """三個固定候選照手算的 J 由小到大排；J 等於逐類加權代價的和。"""
    result = _rank(*_three())

    assert _order(result) == ["candidate-a", "candidate-b", "candidate-c"]
    assert [row.rank for row in result.rankable] == [1, 2, 3]
    assert [row.total_cost for row in result.rankable] == pytest.approx([0.30, 0.525, 0.75])
    for row in result.rankable:
        assert row.total_cost == pytest.approx(sum(line.weighted_cost for line in row.categories))


def test_worse_tilt_moves_second_place_to_third() -> None:
    """只把第二名的傾斜從 0.8 改成 2.0：它的 J 變 0.75+0.375=1.125，掉到 C 後面。"""
    a, _, c = _three()
    worse_b = _candidate(_timbre("candidate-b", tilt=2.0, residual=1.5))

    result = _rank(a, worse_b, c)

    assert _order(result) == ["candidate-a", "candidate-c", "candidate-b"]


def test_reference_and_protection_components_do_not_move_the_score() -> None:
    """對目標偏差改到很差、太窄的峰改到很深：J 與名次都不動（對照只印、保護只擋）。"""
    a, b, c = _three()
    noisy_a = _candidate(
        _timbre(
            "candidate-a",
            tilt=0.6,
            residual=1.0,
            target_deviation=60.0,
            features=[*_QUIET_FEATURES[:2], {**_QUIET_FEATURES[2], "depth_db": 90.0}],
        )
    )

    baseline = _rank(a, b, c)
    noisy = _rank(noisy_a, b, c)

    assert _order(noisy) == _order(baseline)
    assert noisy.rankable[0].total_cost == pytest.approx(baseline.rankable[0].total_cost)
    roles = {line.name: line.role for line in noisy.rankable[0].categories[0].components}
    assert roles == {
        "tilt": "principal",
        "residual_rms": "principal",
        "target_deviation": "reference",
        "peaks_dips": "protection",
    }


# ── 淘汰 ────────────────────────────────────────────────────────────────────


def test_floor_breach_stays_eliminated_even_with_best_score() -> None:
    """傾斜與起伏都完美、但峰谷各踩一條、外部底線也沒過：三條原因全部列出，不回到榜上。"""
    breaching = _candidate(
        _timbre(
            "candidate-d",
            tilt=0.0,
            residual=0.0,
            features=[
                {"kind": "peak", "center_frequency_hz": 80.0, "depth_db": 6.0, "width_octave": 0.4, "flags": []},
                {"kind": "dip", "center_frequency_hz": 160.0, "depth_db": -5.0, "width_octave": 0.3, "flags": []},
            ],
        )
    )
    external = ExternalFloors(
        declared_by="fixture-project",
        verdicts={
            "candidate-a": ExternalAcceptance.PASSED,
            "candidate-b": ExternalAcceptance.PASSED,
            "candidate-c": ExternalAcceptance.PASSED,
            "candidate-d": ExternalAcceptance.FAILED,
        },
    )

    result = _rank(*_three(), breaching, external=external)

    assert "candidate-d" not in _order(result)
    assert result.status_of("candidate-d") is CandidateStatus.ELIMINATED
    (row,) = [row for row in result.eliminated if row.candidate_id == "candidate-d"]
    assert set(row.reasons) == {
        EliminationReason.TIMBRE_PEAK_BEYOND_LIMIT,
        EliminationReason.TIMBRE_DIP_BEYOND_LIMIT,
        EliminationReason.EXTERNAL_FLOOR_FAILED,
    }
    assert "total_cost" not in row.model_dump()
    assert result.header.overall_acceptance is ExternalAcceptance.PASSED


def test_unknown_width_feature_still_counts_toward_the_floor() -> None:
    """寬度未知（邊界不完整）的谷照深度算：踩線就淘汰，標記原樣留在輸出裡。"""
    features: list[dict[str, object]] = [
        {
            "kind": "dip",
            "center_frequency_hz": 60.0,
            "depth_db": -7.0,
            "width_octave": None,
            "flags": ["feature_boundary_incomplete"],
        }
    ]
    result = _rank(_candidate(_timbre("candidate-e", tilt=0.0, residual=0.0, features=features)))

    (row,) = result.eliminated
    assert row.reasons == (EliminationReason.TIMBRE_DIP_BEYOND_LIMIT,)
    (evaluation,) = row.evaluations
    assert isinstance(evaluation.payload, TimbrePayload)
    assert evaluation.payload.features[0].flags == (Flag.FEATURE_BOUNDARY_INCOMPLETE,)


# ── 未評估 ──────────────────────────────────────────────────────────────────


def test_timbre_scoring_gap_is_not_evaluated_and_never_ranked() -> None:
    """必評音色在計分範圍缺段：列成未評估、原因原樣帶著，而且不進可排名區。"""
    reason = ReasonCode.TIMBRE_SCORING_RANGE_GAP
    missing = _candidate(_unavailable("candidate-u", "timbre_balance", (reason.value,)))

    result = _rank(*_three(), missing)

    assert result.status_of("candidate-u") is CandidateStatus.NOT_EVALUATED
    assert "candidate-u" not in _order(result)
    (row,) = result.not_evaluated
    (gap,) = row.missing
    assert gap.category is QualityCategory.TIMBRE_BALANCE
    assert gap.reason is NotEvaluatedReason.MANDATORY_CATEGORY_UNAVAILABLE
    assert gap.evaluator_reason_codes == (reason,)
    (evaluation,) = row.evaluations
    assert evaluation.category_cost is None
    assert evaluation.payload is None
    assert "total_cost" not in row.model_dump()


def test_absent_mandatory_category_is_not_evaluated() -> None:
    """必評類整條沒送來（只送了選評類）：未評估，原因是「缺」不是「不可估」。"""
    only_optional = _candidate(_reverberation("candidate-r"))

    result = _rank(only_optional)

    (row,) = result.not_evaluated
    assert [(gap.category, gap.reason) for gap in row.missing] == [
        (QualityCategory.TIMBRE_BALANCE, NotEvaluatedReason.MANDATORY_CATEGORY_MISSING)
    ]


def test_costed_category_without_registered_coster_refuses_ranking() -> None:
    """選評類沒有註冊代價器時，上游代價不算數，原因是 cost_not_computed。"""
    half_done = _candidate(
        _timbre("candidate-m", tilt=0.0, residual=0.0),
        _uncosted_category("candidate-m"),
    )

    result = _rank(*_three(), half_done)

    assert result.status_of("candidate-m") is CandidateStatus.NOT_EVALUATED
    (row,) = result.not_evaluated
    assert [(gap.category, gap.reason) for gap in row.missing] == [
        (QualityCategory.LOW_FREQUENCY_DECAY, NotEvaluatedReason.COST_NOT_COMPUTED)
    ]


# ── 同表比較 ────────────────────────────────────────────────────────────────


def test_different_settings_fingerprint_is_not_comparable() -> None:
    """設定指紋不同：分到不可同表區，只列代號與指紋、不列分數；主表是人數多的那組。"""
    odd = _candidate(_timbre("candidate-f", tilt=0.0, residual=0.0, settings_fingerprint="timbre-settings-b"))

    result = _rank(*_three(), odd)

    assert result.status_of("candidate-f") is CandidateStatus.NOT_COMPARABLE
    assert _order(result) == ["candidate-a", "candidate-b", "candidate-c"]
    (row,) = result.not_comparable.rows
    assert set(row.model_dump()) == {"status", "candidate_id", "identity"}
    assert [item.settings_fingerprint for item in row.identity] == ["timbre-settings-b"]
    assert [item.settings_fingerprint for item in result.header.main_table_identity] == [_SETTINGS]


def test_different_evaluator_version_is_not_comparable() -> None:
    """評估器版本不同：不可同表，不是淘汰。"""
    odd = _candidate(_timbre("candidate-v", tilt=0.0, residual=0.0, evaluator_version="timbre-fixture-v2"))

    result = _rank(*_three(), odd)

    assert result.status_of("candidate-v") is CandidateStatus.NOT_COMPARABLE
    assert "candidate-v" not in [row.candidate_id for row in result.eliminated]


def test_different_evaluated_category_set_goes_to_another_table() -> None:
    """多評了一類（殘響有代價）的候選跟只評音色的不同表：缺類不補 0 也不補平均。"""
    richer = _candidate(
        _timbre("candidate-g", tilt=0.0, residual=0.0), _reverberation("candidate-g")
    )

    result = _rank(*_three(), richer)

    assert result.status_of("candidate-g") is CandidateStatus.NOT_COMPARABLE
    (row,) = result.not_comparable.rows
    assert [item.category for item in row.identity] == [
        QualityCategory.REVERBERATION,
        QualityCategory.TIMBRE_BALANCE,
    ]


def test_missing_optional_category_still_ranks_and_header_says_uncovered() -> None:
    """選評類沒送來或不可估：仍可排名，表頭列未蓋到，J 只有音色那一類。"""
    with_gap = _candidate(
        _timbre("candidate-a", tilt=0.6, residual=1.0),
        _unavailable("candidate-a", "reverberation", ("evaluator_not_implemented",)),
    )

    result = _rank(with_gap, *_three()[1:])

    assert _order(result) == ["candidate-a", "candidate-b", "candidate-c"]
    header = result.header
    assert header.mandatory_covered == (QualityCategory.TIMBRE_BALANCE,)
    assert header.optional_covered == ()
    registry_optional = _registry().purpose(_PURPOSE_NAME).entry("ranking.optional_categories")
    assert isinstance(registry_optional, QualificationEntry)
    assert isinstance(registry_optional.value, tuple)
    assert {item.value for item in header.optional_uncovered} == {str(name) for name in registry_optional.value}
    row = result.rankable[0]
    assert row.total_cost == pytest.approx(0.30)
    uncovered = {item.category: item.reason_codes for item in row.uncovered}
    assert [code.value for code in uncovered[QualityCategory.REVERBERATION]] == ["evaluator_not_implemented"]


# ── 校準污染 ────────────────────────────────────────────────────────────────


def test_official_registry_marks_result_baseline() -> None:
    """正式登記簿還有 baseline 條目（拍完的只有傾斜那一條）：整份結果就是 baseline。"""
    result = _rank(*_three())

    assert result.header.calibration == "baseline"
    assert result.header.calibration_note == ranking.BASELINE_NOTE


def test_fully_calibrated_registry_marks_result_calibrated() -> None:
    """全部條目 calibrated、評估器也沒自報基線：才寫 calibrated。"""
    candidates = [_candidate(_timbre(c.candidate_id, tilt=0.6, residual=1.0, flags=())) for c in _three()]

    result = _rank(*candidates, registry=_calibrated_registry())

    assert result.header.calibration == "calibrated"


@pytest.mark.parametrize(
    "key",
    [
        "timbre_balance.residual_rms_db",
        "timbre_balance.peak_depth_db",
        "timbre_balance.within_category_weights.tilt",
        "ranking.category_weights.timbre_balance",
        "ranking.mandatory_categories",
    ],
)
def test_one_participating_baseline_entry_contaminates_the_result(key: str) -> None:
    """其餘全部 calibrated、只留一條參與排名的條目 baseline：整份照樣 baseline，而且指得出是哪一條。"""
    candidates = [_candidate(_timbre(c.candidate_id, tilt=0.6, residual=1.0, flags=())) for c in _three()]

    result = _rank(*candidates, registry=_calibrated_registry(frozenset({key})))

    assert result.header.calibration == "baseline"
    baseline_keys = [item.key for item in result.header.calibration_sources if item.status == "baseline"]
    assert baseline_keys == [key]


def test_baseline_entry_that_did_not_take_part_does_not_contaminate() -> None:
    """沒參與排名的條目（沒評的類的權重、這一刀沒套上的資格規則）留 baseline：結果仍是 calibrated。"""
    untouched = frozenset(
        {"ranking.category_weights.reverberation", "ranking.eligibility.min_valid_bands"}
    )
    candidates = [_candidate(_timbre(c.candidate_id, tilt=0.6, residual=1.0, flags=())) for c in _three()]

    result = _rank(*candidates, registry=_calibrated_registry(untouched))

    assert result.header.calibration == "calibrated"


@pytest.mark.parametrize(
    ("category", "category_sources"),
    (
        (
            QualityCategory.TIMBRE_BALANCE,
            (
                "timbre_balance.target_tilt_db_per_octave",
                "timbre_balance.residual_rms_db",
                "timbre_balance.target_deviation_rms_db",
                "timbre_balance.peak_depth_db",
                "timbre_balance.dip_depth_db",
                "timbre_balance.within_category_weights.tilt",
                "timbre_balance.within_category_weights.residual_rms",
            ),
        ),
        (
            QualityCategory.LISTENING_AREA_STABILITY,
            (
                "listening_area_stability.tilt_weighted_mean_deviation",
                "listening_area_stability.ripple_rms_weighted_mean_deviation",
                "listening_area_stability.overall_level_weighted_mean_deviation",
                "listening_area_stability.tilt_worst_deviation",
                "listening_area_stability.ripple_rms_worst_deviation",
                "listening_area_stability.overall_level_worst_deviation",
                "listening_area_stability.within_category_weights.tilt_weighted_mean_deviation",
                "listening_area_stability.within_category_weights.ripple_rms_weighted_mean_deviation",
                "listening_area_stability.within_category_weights.overall_level_weighted_mean_deviation",
            ),
        ),
    ),
)
def test_registry_move_preserves_both_affected_category_header_source_orders(
    category: QualityCategory,
    category_sources: tuple[str, ...],
) -> None:
    """註冊表搬家不得改掉主線原本送進表頭的條目內容或順序。"""
    registry = _registry()
    rules = ranking._read_rules(registry, _PURPOSE_NAME)

    sources = ranking._registry_sources({category}, rules)

    assert tuple(item.key for item in sources) == (
        "ranking.mandatory_categories",
        "ranking.optional_categories",
        f"ranking.category_weights.{category.value}",
        *category_sources,
    )


def test_baseline_flag_on_an_unavailable_evaluation_still_contaminates() -> None:
    """同一批裡有一條評估明說用了基線設定（即使它不可估、進不了表），表頭不准寫已校準。"""
    ranked = _candidate(_timbre("candidate-a", tilt=0.6, residual=1.0, flags=()))
    document = _unavailable("candidate-u", "timbre_balance", ("missing_points",)).model_dump(mode="python")
    document["flags"] = ["baseline_settings"]
    flagged = _candidate(CategoryEvaluation.model_validate(document))

    result = _rank(ranked, flagged, registry=_calibrated_registry())

    assert result.header.calibration == "baseline"
    assert "candidate-u/timbre_balance" in [
        item.key for item in result.header.calibration_sources if item.kind == "evaluator_flag"
    ]


def test_peak_and_dip_limits_must_share_one_unit() -> None:
    """峰與谷兩條界線若單位不同，保護分項只印其中一條的單位就是錯的；要報錯、不靜靜取第一條。"""
    document = _registry().model_dump(mode="json", by_alias=True)
    for purpose in document["purpose"]:
        for row in purpose["target"]:
            if row["key"] == "timbre_balance.dip_depth_db":
                row["unit"] = "oct"
    mismatched = QualityTargets.model_validate(document)

    with pytest.raises(
        ValueError,
        match="timbre_balance.dip_depth_db 單位應為 dB，登記簿寫 oct",
    ):
        _rank(*_three(), registry=mismatched)


def test_one_component_cannot_declare_two_expected_units() -> None:
    """程式這一側替共用同一個分項的兩條鍵宣告了不同單位：報錯，不靜靜取第一條。"""
    purpose = _registry().purpose(_CONTEXT.purpose)
    keys = ("timbre_balance.peak_depth_db", "timbre_balance.residual_rms_db")

    assert ranking._shared_unit(purpose, keys, {keys[0]: "dB", keys[1]: "dB"}) == "dB"
    document = _registry().model_dump(mode="json", by_alias=True)
    for row in document["purpose"][0]["target"]:
        if row["key"] == keys[1]:
            row["unit"] = "oct"
    mixed = QualityTargets.model_validate(document).purpose(_CONTEXT.purpose)
    with pytest.raises(ValueError, match="預期單位卻不一致"):
        ranking._shared_unit(mixed, keys, {keys[0]: "dB", keys[1]: "oct"})


def test_official_registry_passes_every_category_unit_contract() -> None:
    """控制組：正式登記簿原樣，四類會讀的每一條都對得上各自宣告的預期單位。"""
    purpose = load_quality_targets(config_path("quality_targets.toml")).purpose(
        _CONTEXT.purpose
    )
    for category, registration in CATEGORY_REGISTRY.items():
        assert registration.registry_sources(purpose), category


def test_evaluator_baseline_flag_contaminates_the_result() -> None:
    """登記簿全 calibrated、但評估器自報用了基線設定：整份 baseline。"""
    flagged = _candidate(_timbre("candidate-a", tilt=0.6, residual=1.0, flags=("baseline_settings",)))

    result = _rank(flagged, registry=_calibrated_registry())

    assert result.header.calibration == "baseline"
    assert [item.key for item in result.header.calibration_sources if item.kind == "evaluator_flag"] == [
        "candidate-a/timbre_balance"
    ]


# ── 資料不丟、出身、表頭 ────────────────────────────────────────────────────


def test_inputs_survive_the_hand_off() -> None:
    """輸入的原始量、類標記、特徵標記與原因代碼，在輸出裡逐條找得回來。"""
    candidates = _three()
    gap = _unavailable("candidate-u", "timbre_balance", ("missing_points", "solver_unavailable"))

    result = _rank(*candidates, _candidate(gap))

    for candidate, row in zip(candidates, result.rankable, strict=True):
        (source,) = candidate.evaluations
        (line,) = row.categories
        assert line.evaluation.raw_quantities == source.raw_quantities
        assert line.evaluation.flags == source.flags
        assert line.evaluation.payload == source.payload
        assert row.scene_fingerprint == candidate.scene_fingerprint
        assert set(source.flags) <= set(row.flags)
        assert Flag.FEATURE_TOO_NARROW in row.flags
    (skipped,) = result.not_evaluated
    assert skipped.evaluations == (gap,)
    assert skipped.missing[0].evaluator_reason_codes == gap.reason_codes


def test_every_number_names_its_evaluator_and_settings() -> None:
    """每一類的代價旁邊都帶評估器版本、評估設定指紋與代價設定（登記簿）指紋；分項帶原始值。"""
    registry = _registry()

    result = _rank(*_three(), registry=registry)

    assert result.header.registry_fingerprint == registry.fingerprint
    for row in result.rankable:
        for line in row.categories:
            assert line.identity.evaluator_version == _EVALUATOR
            assert line.identity.settings_fingerprint == _SETTINGS
            assert line.identity.cost_settings_fingerprint == registry.fingerprint
            raw = {item.name: item.raw_value for item in line.components}
            assert raw["tilt"] == line.evaluation.raw_quantities[0].value
            assert raw["residual_rms"] == line.evaluation.raw_quantities[1].value


def test_undeclared_external_floors_leave_acceptance_unchecked() -> None:
    """外部底線沒宣告：候選照樣可排名，每列與整體驗收都是「未檢查」，不是合格。"""
    result = _rank(*_three())

    assert _order(result) == ["candidate-a", "candidate-b", "candidate-c"]
    assert result.header.external_floors_declared_by is None
    assert result.header.overall_acceptance is ExternalAcceptance.NOT_CHECKED
    assert result.header.acceptance_note == ranking.NOT_CHECKED_NOTE
    assert {row.external_acceptance for row in result.rankable} == {ExternalAcceptance.NOT_CHECKED}


def test_header_carries_context_and_eligibility_not_applicable() -> None:
    """表頭帶呼叫端給的指紋、日期、引擎版本；三條資料資格照讀、對音色記未套用；過濾計數結構在。"""
    registry = _registry()

    result = _rank(*_three(), registry=registry)

    header = result.header
    assert header.purpose == _PURPOSE_NAME
    assert header.receiver_set_fingerprint == _CONTEXT.receiver_set_fingerprint
    assert header.channel_group_fingerprint == _CONTEXT.channel_group_fingerprint
    assert header.run_date == _CONTEXT.run_date
    assert header.engine_version == _CONTEXT.engine_version
    purpose = registry.purpose(_PURPOSE_NAME)
    for rule in header.eligibility_rules:
        entry = purpose.entry(rule.key)
        assert isinstance(entry, QualificationEntry)
        assert rule.value == entry.value
        assert rule.per_category == {QualityCategory.TIMBRE_BALANCE: EligibilityApplication.NOT_APPLICABLE}
    assert {rule.key for rule in header.eligibility_rules} == {
        key for key in (entry.key for entry in purpose.qualification) if key.startswith("ranking.eligibility.")
    }
    assert result.not_comparable.candidate_filter.illegal_count == 0
    assert result.not_comparable.candidate_filter.illegal_reasons == {}
    assert CandidateStatus.ILLEGAL.value == "illegal"


def test_external_verdict_for_unknown_candidate_is_rejected() -> None:
    """外部底線列了不在這一批的代號：報錯，不讓拼錯的代號把真的候選靜靜變成未檢查。"""
    external = ExternalFloors(declared_by="fixture-project", verdicts={"candidate-z": ExternalAcceptance.PASSED})

    with pytest.raises(ValueError, match="candidate-z"):
        _rank(*_three(), external=external)


def test_declared_floors_with_empty_table_are_not_a_pass() -> None:
    """外部底線宣告了、但榜上一個候選都沒有：整體驗收是「未檢查」，不是空洞的通過。"""
    external = ExternalFloors(declared_by="fixture-project", verdicts={"candidate-u": ExternalAcceptance.PASSED})
    missing = _candidate(_unavailable("candidate-u", "timbre_balance", ("insufficient_coverage",)))

    result = _rank(missing, external=external)

    assert result.rankable == ()
    assert result.header.overall_acceptance is ExternalAcceptance.NOT_CHECKED
