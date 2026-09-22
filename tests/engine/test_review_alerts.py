"""峰谷複核警戒從音色原始特徵一路接到排名列的考卷（票 #445）。"""

from __future__ import annotations

from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring.contract import CategoryCost, Feature
from aosr.scoring.ranking import CandidateStatus, RankingResult, rank_candidates
from aosr.scoring.review_alert import ReviewAlert
from tests.engine import test_scoring_ranking as ranking_fixtures
from tests.engine import test_timbre_channels as channel_fixtures


def _feature(
    kind: str,
    depth_db: float,
    *,
    frequency_hz: float,
    width_octave: float | None = 0.4,
    narrower_than_axis: bool = False,
) -> dict[str, object]:
    return {
        "kind": kind,
        "center_frequency_hz": frequency_hz,
        "depth_db": depth_db,
        "width_octave": width_octave,
        "flags": ["feature_narrower_than_axis"] if narrower_than_axis else [],
    }


def _rank_features(*features: dict[str, object]) -> RankingResult:
    evaluation = ranking_fixtures._timbre(
        "review-alert-candidate",
        tilt=0.0,
        residual=0.0,
        features=list(features),
    )
    return ranking_fixtures._rank(ranking_fixtures._candidate(evaluation))


def _alert_by_kind(result: RankingResult, kind: str) -> ReviewAlert:
    row = next(
        row
        for row in result.rankable
        if row.candidate_id == "review-alert-candidate"
    )
    return next(alert for alert in row.review_alerts if alert.kind == kind)


@pytest.mark.parametrize(
    ("kind", "depth_db", "frequency_hz", "limit_db"),
    (("peak", 7.2, 80.0, 6.0), ("dip", -16.2, 75.5, 15.0)),
)
def test_one_breaching_feature_alerts_without_elimination(
    kind: str, depth_db: float, frequency_hz: float, limit_db: float
) -> None:
    """拔掉任一種峰谷警戒或把它接回淘汰，對應參數題會紅。"""
    result = _rank_features(
        _feature(kind, depth_db, frequency_hz=frequency_hz)
    )

    assert result.status_of("review-alert-candidate") is CandidateStatus.RANKABLE
    assert not any(
        row.candidate_id == "review-alert-candidate" for row in result.eliminated
    )
    alert = _alert_by_kind(result, kind)
    assert (alert.depth_db, alert.center_frequency_hz, alert.limit_db) == (
        depth_db,
        frequency_hz,
        limit_db,
    )
    assert "待複核" in alert.note


def test_peak_and_dip_alert_without_eliminating_candidate() -> None:
    """拔掉警戒接線、把警戒誤接回淘汰，或漏掉任一種類，這題會紅。"""
    result = _rank_features(
        _feature("peak", 7.2, frequency_hz=80.0),
        _feature("dip", -16.2, frequency_hz=75.5, width_octave=0.44),
    )

    assert result.status_of("review-alert-candidate") is CandidateStatus.RANKABLE
    assert not any(
        row.candidate_id == "review-alert-candidate" for row in result.eliminated
    )
    peak = _alert_by_kind(result, "peak")
    dip = _alert_by_kind(result, "dip")
    assert (
        peak.center_frequency_hz,
        peak.depth_db,
        peak.width_octave,
        peak.limit_db,
    ) == (80.0, 7.2, 0.4, 6.0)
    assert (
        dip.center_frequency_hz,
        dip.depth_db,
        dip.width_octave,
        dip.limit_db,
    ) == (75.5, -16.2, 0.44, 15.0)
    assert "待複核" in peak.note
    assert "待複核" in dip.note
    row = next(row for row in result.rankable if row.candidate_id == "review-alert-candidate")
    assert row.review_alerts.index(dip) < row.review_alerts.index(peak)


def test_narrow_peak_alert_keeps_peak_component_cost() -> None:
    """拔掉窄峰計分、解析度布林值或加密提示，這題會紅。"""
    result = _rank_features(
        _feature(
            "peak",
            7.0,
            frequency_hz=63.0,
            width_octave=1.0 / 60.0,
            narrower_than_axis=True,
        )
    )

    alert = _alert_by_kind(result, "peak")
    assert alert.narrower_than_axis is True
    assert "待加密確認" in alert.note
    row = next(row for row in result.rankable if row.candidate_id == "review-alert-candidate")
    cost = cast(CategoryCost, row.categories[0].evaluation.category_cost)
    assert cost.components["left.peaks_dips"] > 0.0


def test_below_alert_limits_has_no_review_alerts() -> None:
    """把等於或低於警戒的特徵也掛上警戒，這題會紅。"""
    result = _rank_features(
        _feature("peak", 6.0, frequency_hz=80.0),
        _feature("dip", -14.9, frequency_hz=160.0),
    )

    row = next(row for row in result.rankable if row.candidate_id == "review-alert-candidate")
    assert row.review_alerts == ()


def test_channel_alert_names_only_the_breaching_speaker() -> None:
    """把彙總警戒錯綁第一支聲道或丟掉喇叭身分，這題會紅。"""
    group = channel_fixtures._group()
    aggregate = channel_fixtures._aggregate(
        group,
        {
            "left": channel_fixtures._single("left"),
            "right": channel_fixtures._single(
                "right",
                payload=channel_fixtures._payload(
                    features=(
                        Feature(
                            kind="peak",
                            center_frequency_hz=125.0,
                            depth_db=7.0,
                            width_octave=0.25,
                            flags=(),
                        ),
                    )
                ),
            ),
        },
    )

    result = rank_candidates(
        (channel_fixtures._candidate(aggregate),),
        load_quality_targets(config_path("quality_targets.toml")),
        channel_fixtures._context(group),
    )

    row = next(row for row in result.rankable if row.candidate_id == aggregate.candidate_id)
    alert = next(alert for alert in row.review_alerts if alert.kind == "peak")
    assert alert.speaker_id == "speaker-right"
    assert alert.receiver_id == "main"


def test_three_db_registry_would_alert_a_twelve_point_eight_db_dip() -> None:
    """把正式 15 dB 警戒退回 3 dB，或排名不讀這一跑的登記簿，這題會紅。"""
    feature = _feature("dip", -12.8, frequency_hz=75.5)
    official = _rank_features(feature)
    document = load_quality_targets(config_path("quality_targets.toml")).model_dump(
        mode="json", by_alias=True
    )
    for purpose in document["purpose"]:
        for target in purpose["target"]:
            if target["key"] in {
                "timbre_balance.peak_depth_db",
                "timbre_balance.dip_depth_db",
            }:
                target["value"] = 3.0
    old_registry = QualityTargets.model_validate(document)
    evaluation = ranking_fixtures._timbre(
        "review-alert-candidate",
        tilt=0.0,
        residual=0.0,
        features=[feature],
    )
    old = ranking_fixtures._rank(
        ranking_fixtures._candidate(evaluation), registry=old_registry
    )

    official_row = next(row for row in official.rankable if row.candidate_id == "review-alert-candidate")
    old_row = next(row for row in old.rankable if row.candidate_id == "review-alert-candidate")
    assert official_row.review_alerts == ()
    assert any(alert.kind == "dip" for alert in old_row.review_alerts)
