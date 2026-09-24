"""峰谷複核警戒從音色原始特徵一路接到排名列的考卷（票 #445）。"""

from __future__ import annotations

from typing import cast

import pytest

from aosr.config.paths import config_path
from aosr.config.quality_targets import QualityTargets, load_quality_targets
from aosr.scoring.contract import CategoryCost, Feature
from aosr.scoring.ranking import (
    CandidateStatus,
    ExternalAcceptance,
    ExternalFloors,
    RankingResult,
    rank_candidates,
)
from aosr.scoring.review_alert import PeakDipReviewAlert
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


def _alert_by_kind(result: RankingResult, kind: str) -> PeakDipReviewAlert:
    row = next(
        row
        for row in result.rankable
        if row.candidate_id == "review-alert-candidate"
    )
    return next(alert for alert in row.review_alerts
                if isinstance(alert, PeakDipReviewAlert) and alert.kind == kind)


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


def test_peak_dip_alerts_keep_frequency_order_after_union_change() -> None:
    """警戒型別加入顫動後，峰谷的同一身分仍按中心頻率排序。"""
    result = _rank_features(
        _feature("peak", 7.0, frequency_hz=120.0),
        _feature("peak", 7.0, frequency_hz=80.0),
        _feature("dip", -16.0, frequency_hz=75.5),
    )
    row = next(row for row in result.rankable if row.candidate_id == "review-alert-candidate")
    assert [(alert.kind, alert.center_frequency_hz) for alert in row.review_alerts] == [
        ("dip", 75.5), ("peak", 80.0), ("peak", 120.0)]


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


def test_too_narrow_feature_is_listed_but_never_alerts() -> None:
    """評估器標「太窄」（feature_too_narrow）的峰谷只留在清單、不進代價，也不掛警戒；
    #432 之後最小寬度是 0 所以正式跑不會出現，但規則還在、要有考卷守：把排除那一行拿掉，這題會紅。"""
    too_narrow = _feature("peak", 20.0, frequency_hz=90.0, width_octave=0.01)
    too_narrow["flags"] = ["feature_too_narrow"]

    result = _rank_features(too_narrow)

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


def test_externally_eliminated_candidate_still_shows_its_review_alerts() -> None:
    """被外部底線（可施工、預算之類）淘汰的候選同時踩了峰谷警戒，淘汰列上要查得出來——
    把淘汰列的警戒寫成空的，這題會紅。"""
    evaluation = ranking_fixtures._timbre(
        "review-alert-candidate",
        tilt=0.0,
        residual=0.0,
        features=[_feature("peak", 20.0, frequency_hz=90.0)],
    )
    external = ExternalFloors(
        declared_by="fixture-project",
        verdicts={"review-alert-candidate": ExternalAcceptance.FAILED},
    )

    result = ranking_fixtures._rank(ranking_fixtures._candidate(evaluation), external=external)

    assert result.status_of("review-alert-candidate") is CandidateStatus.ELIMINATED
    row = next(row for row in result.eliminated if row.candidate_id == "review-alert-candidate")
    assert [alert.kind for alert in row.review_alerts] == ["peak"]
    assert "待複核" in row.review_alerts[0].note


def test_alert_order_is_category_then_identity_then_frequency() -> None:
    """排序鍵逐格釘住：峰谷依（類別、喇叭、接收點、中心頻率），顫動依（類別、牆對、中心頻率）。

    只用同一支喇叭、同一個點的警戒看不出喇叭與接收點兩格對調（#351 PR-D 找碴）。
    """
    from aosr.scoring.contract import QualityCategory
    from aosr.scoring.ranking_alerts import _alert_key
    from aosr.scoring.review_alert import FlutterReviewAlert, PeakDipReviewAlert

    def peak(speaker: str, receiver: str, center: float) -> PeakDipReviewAlert:
        return PeakDipReviewAlert(
            category=QualityCategory.TIMBRE_BALANCE, speaker_id=speaker, receiver_id=receiver,
            kind="peak", center_frequency_hz=center, depth_db=7.0, width_octave=0.2,
            limit_db=6.0, narrower_than_axis=False, note="待複核",
        )

    def flutter(walls: tuple[str, str], center: float) -> FlutterReviewAlert:
        return FlutterReviewAlert(
            category=QualityCategory.REFLECTIONS_AND_ECHO, walls=walls,
            nominal_center_hz=int(center), center_frequency_hz=center,
            lower_hz=center / 1.1, upper_hz=center * 1.1, decay_duration_s=2.0,
            room_t20_s=1.0, decay_db=60.0, note="待複核",
        )

    alerts = (
        flutter(("x0", "xL"), 1000.0), peak("L", "r2", 100.0), flutter(("floor", "ceiling"), 4000.0),
        peak("R", "r1", 50.0), flutter(("x0", "xL"), 500.0), peak("L", "r1", 300.0),
    )
    ordered = sorted(alerts, key=_alert_key)
    peaks = [(a.speaker_id, a.receiver_id, a.center_frequency_hz)
             for a in ordered if isinstance(a, PeakDipReviewAlert)]
    flutters = [(a.walls, a.center_frequency_hz)
                for a in ordered if isinstance(a, FlutterReviewAlert)]
    assert peaks == [("L", "r1", 300.0), ("L", "r2", 100.0), ("R", "r1", 50.0)]
    assert flutters == [(("floor", "ceiling"), 4000.0), (("x0", "xL"), 500.0), (("x0", "xL"), 1000.0)]
    categories = [a.category for a in ordered]
    assert categories == sorted(categories, key=lambda category: category.value)
