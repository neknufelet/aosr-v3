"""排名第幾、外部驗收、複核狀態、推薦狀態四格分開的考卷（票 #449）。"""

from __future__ import annotations

from aosr.scoring.ranking import (
    ExternalAcceptance,
    ExternalFloors,
    RankableRow,
    RankingResult,
)
from aosr.scoring.recommendation import (
    NotFinalReason,
    RecommendationStatus,
    ReviewStatus,
)
from tests.engine import test_scoring_ranking as ranking_fixtures

_PEAK_42_HZ: dict[str, object] = {
    "kind": "peak",
    "center_frequency_hz": 42.0,
    "depth_db": 12.0,
    "width_octave": 0.08,
    "flags": [],
}


def _two_candidates(
    *, external: ExternalFloors | None, flags: tuple[str, ...] = ("unvalidated",), calibrated: bool = False
) -> RankingResult:
    """A 帶一根 42 Hz +12 dB 的峰、B 沒有；峰谷不進總代價，兩者同分照代號排，A 排第一。"""
    alerted = ranking_fixtures._timbre(
        "candidate-a", tilt=0.6, residual=1.0, features=[_PEAK_42_HZ], flags=flags
    )
    clean = ranking_fixtures._timbre("candidate-b", tilt=0.6, residual=1.0, flags=flags)
    return ranking_fixtures._rank(
        ranking_fixtures._candidate(alerted),
        ranking_fixtures._candidate(clean),
        registry=ranking_fixtures._calibrated_registry() if calibrated else None,
        external=external,
    )


def _row(result: RankingResult, candidate_id: str) -> RankableRow:
    return next(row for row in result.rankable if row.candidate_id == candidate_id)


def _all_passed() -> ExternalFloors:
    return ExternalFloors(
        declared_by="fixture-project",
        verdicts={
            "candidate-a": ExternalAcceptance.PASSED,
            "candidate-b": ExternalAcceptance.PASSED,
        },
    )


def test_top_candidate_with_open_alert_keeps_external_pass_but_is_not_final() -> None:
    """老闆的例子：第一名外部驗收通過、帶一根待加密的 +12 dB 峰。
    把警戒接進外部驗收（整體或單列變不通過）、把複核狀態寫死 clear、或漏掉待複核原因，這題會紅。"""
    result = _two_candidates(external=_all_passed())

    top = result.rankable[0]
    assert (top.candidate_id, top.rank) == ("candidate-a", 1)
    assert result.header.overall_acceptance is ExternalAcceptance.PASSED
    assert top.external_acceptance is ExternalAcceptance.PASSED
    assert top.review_status is ReviewStatus.PENDING
    assert top.recommendation_status is RecommendationStatus.NOT_FINAL
    assert NotFinalReason.REVIEW_PENDING in top.not_final_reasons
    assert [(alert.center_frequency_hz, alert.depth_db) for alert in top.review_alerts] == [(42.0, 12.0)]


def test_candidate_without_alerts_is_clear_but_still_not_final() -> None:
    """沒有警戒的那一列是 clear、原因裡沒有待複核；把狀態寫死 pending 或原因寫死，這題會紅。
    今天沒有升成最終推薦的機制，所以它仍然是非最終、而且說得出是這個原因。"""
    row = _row(_two_candidates(external=_all_passed()), "candidate-b")

    assert row.review_status is ReviewStatus.CLEAR
    assert row.recommendation_status is RecommendationStatus.NOT_FINAL
    assert NotFinalReason.REVIEW_PENDING not in row.not_final_reasons
    assert NotFinalReason.EXTERNAL_NOT_CHECKED not in row.not_final_reasons
    assert row.not_final_reasons[-1] is NotFinalReason.NO_FINALIZING_PROCESS


def test_external_not_checked_is_its_own_reason() -> None:
    """沒宣告外部底線時每一列都多一條「外部驗收未檢查」；把這條拿掉或反過來判，這題會紅。"""
    result = _two_candidates(external=None)

    assert result.header.overall_acceptance is ExternalAcceptance.NOT_CHECKED
    for row in result.rankable:
        assert row.external_acceptance is ExternalAcceptance.NOT_CHECKED
        assert NotFinalReason.EXTERNAL_NOT_CHECKED in row.not_final_reasons


def test_dip_alert_alone_also_makes_the_row_pending() -> None:
    """只有谷的警戒也算待複核；把狀態改成只看峰（谷被當成沒事），這題會紅（找碴席抓到的盲點）。"""
    dip = {"kind": "dip", "center_frequency_hz": 75.5, "depth_db": -16.2, "width_octave": 0.44, "flags": []}
    evaluation = ranking_fixtures._timbre("candidate-a", tilt=0.6, residual=1.0, features=[dip])

    row = _row(ranking_fixtures._rank(ranking_fixtures._candidate(evaluation)), "candidate-a")

    assert [alert.kind for alert in row.review_alerts] == ["dip"]
    assert row.review_status is ReviewStatus.PENDING
    assert NotFinalReason.REVIEW_PENDING in row.not_final_reasons


def test_external_reason_reads_each_rows_own_acceptance() -> None:
    """A 外部通過、B 沒列進外部底線：只有 B 多「外部未檢查」，A 不能被整張榜拖累；
    表頭整體驗收照舊因為 B 未檢查而是未檢查。把原因改成看整張榜，這題會紅（找碴席抓到的盲點）。"""
    partial = ExternalFloors(declared_by="fixture-project", verdicts={"candidate-a": ExternalAcceptance.PASSED})

    result = _two_candidates(external=partial)

    assert result.header.overall_acceptance is ExternalAcceptance.NOT_CHECKED
    assert _row(result, "candidate-a").external_acceptance is ExternalAcceptance.PASSED
    assert NotFinalReason.EXTERNAL_NOT_CHECKED not in _row(result, "candidate-a").not_final_reasons
    assert NotFinalReason.EXTERNAL_NOT_CHECKED in _row(result, "candidate-b").not_final_reasons


def test_reasons_follow_calibration_and_keep_fixed_order() -> None:
    """登記簿與評估器全部 calibrated 時不列「未校準」；基線時列。原因順序固定、可重現。
    把校準那一條寫死、或讀錯表頭的校準狀態，這題會紅。"""
    calibrated = _two_candidates(external=_all_passed(), flags=(), calibrated=True)
    baseline = _two_candidates(external=None)

    assert calibrated.header.calibration == "calibrated"
    assert _row(calibrated, "candidate-a").not_final_reasons == (
        NotFinalReason.REVIEW_PENDING,
        NotFinalReason.NO_FINALIZING_PROCESS,
    )
    assert baseline.header.calibration == "baseline"
    assert _row(baseline, "candidate-a").not_final_reasons == (
        NotFinalReason.REVIEW_PENDING,
        NotFinalReason.EXTERNAL_NOT_CHECKED,
        NotFinalReason.CALIBRATION_BASELINE,
        NotFinalReason.NO_FINALIZING_PROCESS,
    )


def test_status_fields_are_controlled_strings_in_json() -> None:
    """機器讀的輸出裡兩個狀態是受控字串、不是布林值（以後加「已解除」不必重做契約）；改成布林值這題會紅。"""
    dumped = _row(_two_candidates(external=_all_passed()), "candidate-a").model_dump(mode="json")

    assert dumped["review_status"] == "pending"
    assert dumped["recommendation_status"] == "not_final"
    assert dumped["not_final_reasons"] == ["review_pending", "calibration_baseline", "no_finalizing_process"]
