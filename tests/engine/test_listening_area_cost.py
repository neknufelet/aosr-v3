"""聆聽區穩定性的容許帶代價、底線保護與排名接線考卷（票 #349 第二段）。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring import listening_area_cost, ranking
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    EvaluationState,
    Flag,
    InputProvenance,
    ListeningAreaStabilityPayload,
    QualityCategory,
    ReasonCode,
)
from aosr.scoring.ranking import CandidateStatus, EliminationReason, RankingContext


_PURPOSE = "dedicated_two_channel_listening_room"
_COST_FINGERPRINT = "cost-registry-fixture"
_SCENE_FINGERPRINT = "a" * 64
_PROVENANCE = InputProvenance(
    report_id="listening-area-report",
    engine_commit="engine-fixture",
    speaker_id="left",
    receiver_id="main",
)


def _registry() -> QualityTargets:
    return load_quality_targets(config_path("quality_targets.toml"))


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
    return load_quality_targets(path)


def _aggregate(mean: float, worst: float, receiver_id: str) -> dict[str, object]:
    return {
        "weighted_mean_deviation": mean,
        "worst_deviation": {
            "value": worst,
            "receiver": {
                "receiver_id": receiver_id,
                "direction_relative_to_primary": "front",
                "importance": 1.0,
            },
            "reference": {
                "receiver_id": "main",
                "direction_relative_to_primary": None,
                "importance": 1.0,
            },
        },
    }


def _comparison(
    mean: float,
    worst: float,
    *,
    peer_mean: float | None = None,
    peer_worst: float | None = None,
) -> dict[str, object]:
    peer = (
        None
        if peer_mean is None
        else _aggregate(peer_mean, peer_worst if peer_worst is not None else peer_mean, "back")
    )
    return {
        "primary_to_surrounding": _aggregate(mean, worst, "front"),
        "surrounding_to_surrounding": peer,
        "surrounding_to_surrounding_reason": (
            "no_surrounding_pairs" if peer is None else None
        ),
    }


def _point_provenance() -> list[dict[str, object]]:
    return [
        {
            "receiver_id": "main",
            "report_id": "report-main",
            "evaluator_version": "timbre-v1",
            "settings_fingerprint": "timbre-settings-a",
        },
        {
            "receiver_id": "front",
            "report_id": "report-front",
            "evaluator_version": "timbre-v1",
            "settings_fingerprint": "timbre-settings-a",
        },
    ]


def _stability_payload(
    *,
    tilt_mean: float,
    ripple_mean: float,
    level_mean: float,
    tilt_worst: float | None,
    ripple_worst: float | None,
    level_worst: float | None,
    peer_means: tuple[float, float, float] | None,
    peer_worsts: tuple[float, float, float] | None,
    peak_dip_mean: float,
    peak_dip_worst: float,
) -> dict[str, object]:
    peer_tilt, peer_ripple, peer_level = peer_means or (None, None, None)
    peer_tilt_worst, peer_ripple_worst, peer_level_worst = peer_worsts or (
        None,
        None,
        None,
    )
    return {
        "category": "listening_area_stability",
        "candidate_id": "candidate-a",
        "speaker_id": "left",
        "receiver_set_fingerprint": "receiver-set-a",
        "timbre_settings_fingerprint": "timbre-settings-a",
        "settings_fingerprint": "listening-area-settings-a",
        "point_provenance": _point_provenance(),
        "tilt_stability": _comparison(
            tilt_mean,
            tilt_mean if tilt_worst is None else tilt_worst,
            peer_mean=peer_tilt,
            peer_worst=peer_tilt_worst,
        ),
        "ripple_rms_stability": _comparison(
            ripple_mean,
            ripple_mean if ripple_worst is None else ripple_worst,
            peer_mean=peer_ripple,
            peer_worst=peer_ripple_worst,
        ),
        "overall_level_stability": _comparison(
            level_mean,
            level_mean if level_worst is None else level_worst,
            peer_mean=peer_level,
            peer_worst=peer_level_worst,
        ),
        "peak_dip_consistency": _comparison(peak_dip_mean, peak_dip_worst),
        "peak_dip_occurrences": [],
        "target_deviation_position_spread_curve_db": [],
        "target_deviation_common_frequency_count": 0,
        "target_deviation_discarded_frequency_value_count": 0,
    }


def _measured(
    *,
    tilt_mean: float = 0.0,
    ripple_mean: float = 0.0,
    level_mean: float = 0.0,
    tilt_worst: float | None = None,
    ripple_worst: float | None = None,
    level_worst: float | None = None,
    peer_means: tuple[float, float, float] | None = None,
    peer_worsts: tuple[float, float, float] | None = None,
    peak_dip_mean: float = 0.0,
    peak_dip_worst: float = 0.0,
) -> CategoryEvaluation:
    payload = _stability_payload(
        tilt_mean=tilt_mean,
        ripple_mean=ripple_mean,
        level_mean=level_mean,
        tilt_worst=tilt_worst,
        ripple_worst=ripple_worst,
        level_worst=level_worst,
        peer_means=peer_means,
        peer_worsts=peer_worsts,
        peak_dip_mean=peak_dip_mean,
        peak_dip_worst=peak_dip_worst,
    )
    return CategoryEvaluation.model_validate(
        {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "candidate_id": "candidate-a",
            "scene_fingerprint": _SCENE_FINGERPRINT,
            "category": "listening_area_stability",
            "state": "measured",
            "payload": payload,
            "raw_quantities": [
                {
                    "name": "tilt.primary_to_surrounding.weighted_mean_deviation",
                    "value": tilt_mean,
                    "unit": "dB/oct",
                }
            ],
            "category_cost": None,
            "flags": [],
            "reason_codes": [],
            "evaluator_version": "listening-area-v1",
            "settings_fingerprint": "listening-area-settings-a",
            "provenance": _PROVENANCE,
        }
    )


def _cost(measured: CategoryEvaluation, registry: QualityTargets | None = None) -> CategoryEvaluation:
    chosen = registry or _registry()
    # #350 把「哪一類用哪支代價」收進註冊表之後，這一支住在它自己的檔。
    return listening_area_cost.cost_listening_area_evaluation(
        measured,
        chosen.purpose(_PURPOSE),
        _COST_FINGERPRINT,
    )


def _target_value(key: str, field: str) -> float:
    target = _registry().purpose(_PURPOSE).entry(key)
    value = getattr(target, field)
    assert isinstance(value, float)
    return value


def test_tilt_weighted_mean_dead_band_is_zero_inside_and_grows_outside() -> None:
    """拿掉傾斜容許帶、或帶外仍回零時會紅。"""
    tolerance = _target_value(
        "listening_area_stability.tilt_weighted_mean_deviation", "tolerance"
    )

    inside = _cost(_measured(tilt_mean=tolerance)).category_cost
    outside = _cost(_measured(tilt_mean=tolerance + 0.5)).category_cost

    assert inside is not None
    assert outside is not None
    assert inside.components["tilt_weighted_mean_deviation"] == 0.0
    assert outside.components["tilt_weighted_mean_deviation"] > 0.0


def test_listening_area_tilt_rejects_registry_unit_mismatch(tmp_path: Path) -> None:
    """若聆聽區代價沒聲明 dB/oct，傾斜偏差可被錯標成 dB 後照算。"""
    key = "listening_area_stability.tilt_weighted_mean_deviation"
    registry = _registry_with_unit(
        tmp_path, key=key, expected="dB/oct", changed="dB"
    )

    with pytest.raises(
        ValueError,
        match=f"{key} 單位應為 dB/oct，登記簿寫 dB",
    ):
        _cost(_measured(), registry)


def test_ripple_weighted_mean_dead_band_is_zero_inside_and_grows_outside() -> None:
    """拿掉起伏容許帶、或帶外仍回零時會紅。"""
    tolerance = _target_value(
        "listening_area_stability.ripple_rms_weighted_mean_deviation", "tolerance"
    )

    inside = _cost(_measured(ripple_mean=tolerance)).category_cost
    outside = _cost(_measured(ripple_mean=tolerance + 0.5)).category_cost

    assert inside is not None
    assert outside is not None
    assert inside.components["ripple_rms_weighted_mean_deviation"] == 0.0
    assert outside.components["ripple_rms_weighted_mean_deviation"] > 0.0


def test_level_weighted_mean_dead_band_is_zero_inside_and_grows_outside() -> None:
    """拿掉整體音量容許帶、或帶外仍回零時會紅。"""
    tolerance = _target_value(
        "listening_area_stability.overall_level_weighted_mean_deviation", "tolerance"
    )

    inside = _cost(_measured(level_mean=tolerance)).category_cost
    outside = _cost(_measured(level_mean=tolerance + 0.5)).category_cost

    assert inside is not None
    assert outside is not None
    assert inside.components["overall_level_weighted_mean_deviation"] == 0.0
    assert outside.components["overall_level_weighted_mean_deviation"] > 0.0


def test_only_three_weighted_mean_components_form_the_principal_cost() -> None:
    """最差偏差或峰谷一致性被重複加進主要代價時會紅；原 payload 標記仍須留下。"""
    registry = _registry()
    purpose = registry.purpose(_PURPOSE)
    tolerances = (
        _target_value("listening_area_stability.tilt_weighted_mean_deviation", "tolerance"),
        _target_value("listening_area_stability.ripple_rms_weighted_mean_deviation", "tolerance"),
        _target_value("listening_area_stability.overall_level_weighted_mean_deviation", "tolerance"),
    )
    base = _measured(
        tilt_mean=tolerances[0] + 0.5,
        ripple_mean=tolerances[1] + 0.5,
        level_mean=tolerances[2] + 0.5,
    )
    worst_changed = _measured(
        tilt_mean=tolerances[0] + 0.5,
        ripple_mean=tolerances[1] + 0.5,
        level_mean=tolerances[2] + 0.5,
        tilt_worst=50.0,
        ripple_worst=50.0,
        level_worst=50.0,
    )
    peak_changed = _measured(
        tilt_mean=tolerances[0] + 0.5,
        ripple_mean=tolerances[1] + 0.5,
        level_mean=tolerances[2] + 0.5,
        peak_dip_mean=20.0,
        peak_dip_worst=40.0,
    )

    base_costed = listening_area_cost.cost_listening_area_evaluation(
        base, purpose, registry.fingerprint
    )
    worst_costed = listening_area_cost.cost_listening_area_evaluation(
        worst_changed, purpose, registry.fingerprint
    )
    peak_costed = listening_area_cost.cost_listening_area_evaluation(
        peak_changed, purpose, registry.fingerprint
    )

    assert base_costed.category_cost is not None
    assert worst_costed.category_cost is not None
    assert peak_costed.category_cost is not None
    assert worst_costed.category_cost.value == pytest.approx(base_costed.category_cost.value)
    assert peak_costed.category_cost.value == pytest.approx(base_costed.category_cost.value)
    assert (
        worst_costed.category_cost.components["tilt_worst_deviation"]
        > base_costed.category_cost.components["tilt_worst_deviation"]
    )
    assert "peak_dip_consistency" not in peak_costed.category_cost.components
    assert isinstance(peak_costed.payload, ListeningAreaStabilityPayload)
    assert isinstance(base_costed.payload, ListeningAreaStabilityPayload)
    assert peak_costed.payload.peak_dip_consistency != base_costed.payload.peak_dip_consistency


@pytest.mark.parametrize("mutation", ["extra", "missing"])
def test_principal_weight_names_must_be_exact(mutation: str) -> None:
    """類內權重表多一項或少一項都要報錯，不能靜靜忽略或補預設值。"""
    registry = _registry()
    document = registry.model_dump(mode="json", by_alias=True)
    tables = document["purpose"][0]["weight"]
    table = next(
        item
        for item in tables
        if item["key"] == "listening_area_stability.within_category_weights"
    )
    if mutation == "extra":
        table["item"].append({**table["item"][0], "name": "unexpected_component"})
    else:
        table["item"] = table["item"][1:]
    changed = QualityTargets.model_validate(document)

    with pytest.raises(ValueError, match="名稱必須剛好"):
        _cost(_measured(), changed)


def test_missing_peer_group_keeps_primary_cost_and_reason_trace() -> None:
    """少一組若補零或平均、完整兩組若平均，答案都會與較嚴重者不同。"""
    tolerance = _target_value(
        "listening_area_stability.tilt_weighted_mean_deviation", "tolerance"
    )
    worse_reference = _target_value(
        "listening_area_stability.tilt_weighted_mean_deviation",
        "worse_reference",
    )
    primary_value = tolerance + worse_reference
    peer_value = tolerance + 3.0 * worse_reference
    without_peer = _cost(_measured(tilt_mean=primary_value))
    with_peer = _cost(
        _measured(tilt_mean=primary_value, peer_means=(peer_value, 0.0, 0.0))
    )

    assert without_peer.category_cost is not None
    assert with_peer.category_cost is not None
    assert without_peer.category_cost.components[
        "tilt_weighted_mean_deviation"
    ] == pytest.approx(1.0)
    assert with_peer.category_cost.components[
        "tilt_weighted_mean_deviation"
    ] == pytest.approx(3.0)
    group_suffixes = (
        ".primary_to_surrounding",
        ".surrounding_to_surrounding",
    )
    without_peer_groups = {
        name: value
        for name, value in without_peer.category_cost.components.items()
        if name.startswith("tilt_weighted_mean_deviation.")
        and name.endswith(group_suffixes)
    }
    assert set(without_peer_groups) == {
        "tilt_weighted_mean_deviation.primary_to_surrounding"
    }
    assert without_peer_groups[
        "tilt_weighted_mean_deviation.primary_to_surrounding"
    ] == pytest.approx(1.0)
    assert with_peer.category_cost.components[
        "tilt_weighted_mean_deviation.primary_to_surrounding"
    ] == pytest.approx(1.0)
    assert with_peer.category_cost.components[
        "tilt_weighted_mean_deviation.surrounding_to_surrounding"
    ] == pytest.approx(3.0)
    assert Flag.LISTENING_AREA_PEER_GROUP_MISSING in without_peer.flags
    assert Flag.LISTENING_AREA_PEER_GROUP_MISSING not in with_peer.flags
    assert isinstance(without_peer.payload, ListeningAreaStabilityPayload)
    assert (
        without_peer.payload.tilt_stability.surrounding_to_surrounding_reason
        is ReasonCode.NO_SURROUNDING_PAIRS
    )


def _listening_only_registry() -> QualityTargets:
    registry = _registry()
    document = registry.model_dump(mode="json", by_alias=True)
    qualifications = document["purpose"][0]["qualification"]
    for row in qualifications:
        if row["key"] == "ranking.mandatory_categories":
            row["value"] = ["listening_area_stability"]
        elif row["key"] == "ranking.optional_categories":
            row["value"] = [
                name for name in row["value"] if name != "listening_area_stability"
            ]
    return QualityTargets.model_validate(document)


def _candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id="candidate-a",
        scene_fingerprint=evaluation.scene_fingerprint,
        evaluations=(evaluation,),
    )


def _context() -> RankingContext:
    return RankingContext(
        purpose=_PURPOSE,
        receiver_set_fingerprint="receiver-set-a",
        channel_group_fingerprint="channel-group-a",
        run_date=date(2026, 9, 20),
        engine_version="engine-fixture",
    )


def test_ranking_dispatches_measured_listening_area_to_the_registered_coster() -> None:
    """忘記註冊 coster 時，已量的聆聽區會錯落到 cost_not_computed。"""
    result = ranking.rank_candidates(
        [
            _candidate(
                _measured(
                    peer_means=(0.0, 0.0, 0.0),
                    peer_worsts=(0.0, 0.0, 0.0),
                )
            )
        ],
        _listening_only_registry(),
        _context(),
    )

    assert result.status_of("candidate-a") is CandidateStatus.RANKABLE
    (row,) = result.rankable
    (line,) = row.categories
    assert line.evaluation.state is EvaluationState.COSTED
    assert line.identity.settings_fingerprint == "listening-area-settings-a"
    assert line.identity.cost_settings_fingerprint == result.header.registry_fingerprint
    assert {component.name: component.role for component in line.components} == {
        "tilt_weighted_mean_deviation": "principal",
        "ripple_rms_weighted_mean_deviation": "principal",
        "overall_level_weighted_mean_deviation": "principal",
        "tilt_worst_deviation": "protection",
        "ripple_rms_worst_deviation": "protection",
        "overall_level_worst_deviation": "protection",
        "tilt_weighted_mean_deviation.primary_to_surrounding": "reported",
        "tilt_weighted_mean_deviation.surrounding_to_surrounding": "reported",
        "ripple_rms_weighted_mean_deviation.primary_to_surrounding": "reported",
        "ripple_rms_weighted_mean_deviation.surrounding_to_surrounding": "reported",
        "overall_level_weighted_mean_deviation.primary_to_surrounding": "reported",
        "overall_level_weighted_mean_deviation.surrounding_to_surrounding": "reported",
        "tilt_worst_deviation.primary_to_surrounding": "protection",
        "tilt_worst_deviation.surrounding_to_surrounding": "protection",
        "ripple_rms_worst_deviation.primary_to_surrounding": "protection",
        "ripple_rms_worst_deviation.surrounding_to_surrounding": "protection",
        "overall_level_worst_deviation.primary_to_surrounding": "protection",
        "overall_level_worst_deviation.surrounding_to_surrounding": "protection",
    }


def test_each_worst_deviation_is_a_floor_protection_not_a_principal_cost() -> None:
    """三種最差偏差超過各自門檻時都要淘汰，不能只印分項後仍留在榜上。"""
    purpose = _registry().purpose(_PURPOSE)
    thresholds = {
        name: getattr(
            purpose.entry(f"listening_area_stability.{name}_worst_deviation"),
            "value",
        )
        for name in ("tilt", "ripple_rms", "overall_level")
    }
    result = ranking.rank_candidates(
        [
            _candidate(
                _measured(
                    tilt_worst=float(thresholds["tilt"]) + 0.5,
                    ripple_worst=float(thresholds["ripple_rms"]) + 0.5,
                    level_worst=float(thresholds["overall_level"]) + 0.5,
                )
            )
        ],
        _listening_only_registry(),
        _context(),
    )

    assert result.status_of("candidate-a") is CandidateStatus.ELIMINATED
    (row,) = result.eliminated
    assert {reason.value for reason in row.reasons} == {
        "listening_area_tilt_primary_to_surrounding_worst_beyond_limit",
        "listening_area_ripple_primary_to_surrounding_worst_beyond_limit",
        "listening_area_level_primary_to_surrounding_worst_beyond_limit",
    }


def test_peer_only_floor_breach_names_the_peer_group() -> None:
    """只有周圍彼此踩線時，淘汰理由與保護分項都不可誤指主位那組。"""
    threshold = _target_value(
        "listening_area_stability.tilt_worst_deviation", "value"
    )
    result = ranking.rank_candidates(
        [
            _candidate(
                _measured(
                    tilt_worst=0.0,
                    peer_means=(0.0, 0.0, 0.0),
                    peer_worsts=(threshold + 0.5, 0.0, 0.0),
                )
            )
        ],
        _listening_only_registry(),
        _context(),
    )

    assert result.status_of("candidate-a") is CandidateStatus.ELIMINATED
    (row,) = result.eliminated
    assert row.reasons == (
        EliminationReason.LISTENING_AREA_TILT_SURROUNDING_TO_SURROUNDING_WORST_BEYOND_LIMIT,
    )
    (evaluation,) = row.evaluations
    assert evaluation.category_cost is not None
    assert evaluation.category_cost.components[
        "tilt_worst_deviation.primary_to_surrounding"
    ] == 0.0
    assert evaluation.category_cost.components[
        "tilt_worst_deviation.surrounding_to_surrounding"
    ] > 0.0


def test_unavailable_listening_area_stays_uncosted_and_not_evaluated() -> None:
    """不可彙總若被當成零代價，候選會錯進榜。"""
    unavailable = CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id="candidate-a",
        scene_fingerprint=_SCENE_FINGERPRINT,
        category=QualityCategory.LISTENING_AREA_STABILITY,
        state=EvaluationState.UNAVAILABLE,
        payload=None,
        raw_quantities=(),
        category_cost=None,
        flags=(),
        reason_codes=(ReasonCode.MISSING_POINTS,),
        evaluator_version="listening-area-v1",
        settings_fingerprint="listening-area-settings-a",
        provenance=_PROVENANCE,
    )

    result = ranking.rank_candidates(
        [_candidate(unavailable)], _listening_only_registry(), _context()
    )

    assert result.status_of("candidate-a") is CandidateStatus.NOT_EVALUATED
    (row,) = result.not_evaluated
    (preserved,) = row.evaluations
    assert preserved == unavailable
    assert preserved.category_cost is None


def test_costing_returns_a_new_fully_validated_object_with_both_fingerprints() -> None:
    """原地改輸入或把評估設定指紋覆成登記簿指紋時會紅。"""
    measured = _measured()

    costed = _cost(measured)

    assert costed is not measured
    assert measured.state is EvaluationState.MEASURED
    assert measured.category_cost is None
    assert costed.state is EvaluationState.COSTED
    assert costed.settings_fingerprint == measured.settings_fingerprint
    assert costed.category_cost is not None
    assert costed.category_cost.cost_settings_fingerprint == _COST_FINGERPRINT
