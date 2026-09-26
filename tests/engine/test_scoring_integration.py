"""音色評估器到排名結果的整合考卷（票 #359）。

有限元素（FEM）那一路是假能量；這一支守的是交接，不是物理。幾何路、晚期混響、
報表收成、音色評估與排名都走正式產品入口，不拿假的 ``CategoryEvaluation`` 代替音色輸出。

峰谷從票 #445 起是複核警戒，不是淘汰線；三個假能量候選因此直接使用正式登記簿，
讓這支考卷同時守住真評估輸出可排名、警戒可追查與代價仍從原始量重算。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import pytest

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import (
    QualityTargets,
    SettingEntry,
    load_quality_targets,
)
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, three_lane_report
from aosr.physics.report_source import SourceModelKind, SourceModelSpec
from aosr.physics.report_io import ReportOutput
from aosr.physics.report_output import output_from_report
from aosr.scoring.channel_matching import ChannelDefinition, ChannelGroup
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    EvaluationState,
    Flag,
    InputProvenance,
    ModelValidationStatus,
    QualityCategory,
    RawQuantity,
    ReasonCode,
    ReverberationPayload,
    SpatialImpressionPayload,
    TimbreChannelsPayload,
    TimbrePayload,
)
from aosr.scoring.ranking import (
    BASELINE_NOTE,
    CandidateStatus,
    NotEvaluatedReason,
    RankingContext,
    RankingResult,
    rank_candidates,
)
from aosr.scoring.timbre import (
    TIMBRE_EVALUATOR_VERSION,
    evaluate_timbre,
    timbre_input_from_report,
)
from aosr.scoring.timbre_channels import (
    TIMBRE_CHANNELS_EVALUATOR_VERSION,
    evaluate_timbre_channels,
)
from tests.engine._placement import POINT_PLACEMENT


_PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
_REGISTRY_PATH: Final[Path] = config_path("quality_targets.toml")
_ROOM: Final[Room] = Room(6.0, 4.0, 3.0)
_SOURCE: Final[Point] = Point(1.2, 1.3, 1.1)
_RECEIVER: Final[Point] = Point(4.7, 2.8, 1.4)
_SOUND_SPEED_M_S: Final[float] = 343.0
_DENSITY_KG_M3: Final[float] = 1.2
_RHO_C_PA_S_PER_M: Final[float] = _SOUND_SPEED_M_S * _DENSITY_KG_M3
_SCENE_FINGERPRINT: Final[str] = "a" * 64
_IMPEDANCE_MULTIPLES: Final[tuple[float, ...]] = (4.0, 7.0, 10.0)
_TIMBRE_GROUP: Final[ChannelGroup] = ChannelGroup(
    channels=(
        ChannelDefinition(role="reference", speaker_id="reference-speaker"),
    ),
    comparisons=(),
    feature_match_tolerance_hz=0.0,
)
_CONTEXT: Final[RankingContext] = RankingContext(
    purpose=_PURPOSE,
    receiver_set_fingerprint="reference-seat",
    channel_group_fingerprint=_TIMBRE_GROUP.fingerprint,
    run_date=date(2026, 9, 19),
    engine_version="three-lane-integration-fixture",
)


@dataclass(frozen=True)
class _IntegratedCandidate:
    """一份快報表與由它真的量出的音色評估。"""

    report: ReportOutput
    evaluation: CategoryEvaluation


def _fake_fem_energy(
    *,
    room: Room,
    source: Point,
    receiver: Point,
    wall_impedances: Mapping[Wall, float],
    frequencies_hz: tuple[float, ...],
    density_kg_m3: float,
    sound_speed_m_s: float,
) -> tuple[float, ...]:
    """只替掉昂貴的有限元素（FEM）求解；輸出形狀與正式入口相同。"""
    del room, source, receiver, wall_impedances, density_kg_m3, sound_speed_m_s
    return tuple(frequency_hz * 3.0 + 7.0 for frequency_hz in frequencies_hz)


def _solve_report(monkeypatch: pytest.MonkeyPatch, impedance_multiple: float) -> ReportOutput:
    """用不同牆面阻抗跑真幾何路與真晚期混響，再收成帶細軸的報表。"""
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    inputs = report_io.load_input_document(
        {
            "room_m": {"Lx": _ROOM.Lx, "Ly": _ROOM.Ly, "Lz": _ROOM.Lz},
            "source_model": {"kind": "omnidirectional"},
            "source_m": {"x": _SOURCE.x, "y": _SOURCE.y, "z": _SOURCE.z},
            "receiver_m": {
                "x": _RECEIVER.x,
                "y": _RECEIVER.y,
                "z": _RECEIVER.z,
            },
            "sound_speed_m_s": _SOUND_SPEED_M_S,
            "density_kg_m3": _DENSITY_KG_M3,
            "impedance_pa_s_per_m_by_wall": {
                wall.wall_name(): impedance_multiple * _RHO_C_PA_S_PER_M
                for wall in Wall.all()
            },
        },
        load_capabilities(config_path("capabilities.toml")),
    )
    solved = report_io.solver_inputs(inputs)
    report = three_lane_report.solve_three_lane_report(
        source_model=SourceModelSpec(kind=SourceModelKind.OMNIDIRECTIONAL),
        room=solved.room,
        source=solved.source,
        receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        density_kg_m3=solved.density_kg_m3,
        impedance_by_wall=solved.impedance_by_wall,
        scattering_by_wall=solved.scattering_by_wall,
        reflection_order_k=solved.reflection_order_k,
    )
    return output_from_report(report, inputs=inputs, with_points=True)


def _provenance(candidate_id: str) -> InputProvenance:
    return InputProvenance(
        report_id=f"three-lane-{candidate_id}",
        engine_commit="integration-fixture",
        speaker_id="reference-speaker",
        receiver_id="reference-seat",
    )


def _evaluate_report(
    report: ReportOutput, candidate_id: str, registry_path: Path = _REGISTRY_PATH
) -> CategoryEvaluation:
    """報表細軸經正式收成器進正式音色評估器，不改曲線。"""
    data = timbre_input_from_report(
        report,
        candidate_id=candidate_id,
        speaker_id="reference-speaker",
        receiver_id="reference-seat",
        source_reference="三路接合報表共同能量基準",
        provenance=_provenance(candidate_id),
    )
    return evaluate_timbre(
        data,
        purpose=_PURPOSE,
        quality_targets_path=registry_path,
    )


def _candidate(evaluation: CategoryEvaluation) -> CandidateEvaluation:
    if (
        evaluation.category is QualityCategory.TIMBRE_BALANCE
        and evaluation.state is not EvaluationState.UNAVAILABLE
    ):
        evaluation = evaluate_timbre_channels(
            _TIMBRE_GROUP,
            "reference-seat",
            {"reference": evaluation},
            candidate_id=evaluation.candidate_id,
            scene_fingerprint=evaluation.scene_fingerprint,
            timbre_settings_fingerprint=evaluation.settings_fingerprint,
        )
    return CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=evaluation.candidate_id,
        scene_fingerprint=evaluation.scene_fingerprint,
        evaluations=(evaluation,),
    )


def _registry_with_limits(directory: Path, name: str, limit_db: str) -> Path:
    """複製正式登記簿、只改峰與谷兩條界線；兩條都要真的改到，改不到就紅。"""
    text = _REGISTRY_PATH.read_text(encoding="utf-8")
    current = {
        "timbre_balance.peak_depth_db": "6.0",
        "timbre_balance.dip_depth_db": "15.0",
    }
    for key, value in current.items():
        old = f'key = "{key}"\nvalue = {value}'
        assert old in text
        text = text.replace(old, f'key = "{key}"\nvalue = {limit_db}', 1)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def integrated_candidates() -> tuple[_IntegratedCandidate, ...]:
    """三個材料候選只算一次；每一題仍各自重跑評估與排名入口以外的判斷。"""
    monkeypatch = pytest.MonkeyPatch()
    try:
        return tuple(
            _IntegratedCandidate(
                report=(report := _solve_report(monkeypatch, multiple)),
                evaluation=_evaluate_report(report, f"wall-{multiple:g}"),
            )
            for multiple in _IMPEDANCE_MULTIPLES
        )
    finally:
        monkeypatch.undo()


def _registry() -> QualityTargets:
    return load_quality_targets(_REGISTRY_PATH)


def _rank(
    *evaluations: CategoryEvaluation, registry: QualityTargets | None = None
) -> RankingResult:
    return rank_candidates(
        tuple(_candidate(evaluation) for evaluation in evaluations),
        registry or _registry(),
        _CONTEXT,
    )


def _ranked_evaluation(result: RankingResult, candidate_id: str) -> CategoryEvaluation:
    """從候選所在區塊取回排名層保存的完整評估，不預設它落哪一區。"""
    for rankable_row in result.rankable:
        if rankable_row.candidate_id == candidate_id:
            return next(line.evaluation for line in rankable_row.categories)
    for eliminated_row in result.eliminated:
        if eliminated_row.candidate_id == candidate_id:
            return next(iter(eliminated_row.evaluations))
    for unevaluated_row in result.not_evaluated:
        if unevaluated_row.candidate_id == candidate_id:
            return next(iter(unevaluated_row.evaluations))
    raise AssertionError(f"候選 {candidate_id} 的完整評估沒有留在排名結果")


def _registry_with_changed_timbre_source(tmp_path: Path, base: Path) -> Path:
    """只改一條音色設定的來源欄，數值不動；內容指紋必須跟著改。"""
    marker = 'key = "timbre_balance.feature_min_width_octave"'
    original_source = 'source = "測試基線，未查證；正式值等 #358"'
    changed_source = 'source = "整合考卷的第二份音色設定"'
    before, after = base.read_text(encoding="utf-8").split(marker, maxsplit=1)
    changed = after.replace(original_source, changed_source, 1)
    path = tmp_path / "quality-targets-changed.toml"
    path.write_text(before + marker + changed, encoding="utf-8")
    return path


def test_three_real_timbre_outputs_are_ranked_by_their_computed_total_cost(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
) -> None:
    """若真音色輸出接不上排名層，候選就不會全進可排名區或總代價 J 會對不起來。"""
    evaluations = tuple(item.evaluation for item in integrated_candidates)
    registry = load_quality_targets(_REGISTRY_PATH)
    assert all(item.state is EvaluationState.MEASURED for item in evaluations)
    # 這幾條是評估器自己算的，不是考卷捏的：版本是評估器的、指紋是它讀的那份登記簿的。
    assert {item.evaluator_version for item in evaluations} == {TIMBRE_EVALUATOR_VERSION}
    assert {item.settings_fingerprint for item in evaluations} == {registry.fingerprint}

    result = _rank(*evaluations, registry=registry)

    assert {
        line.identity.evaluator_version for row in result.rankable for line in row.categories
    } == {TIMBRE_CHANNELS_EVALUATOR_VERSION}
    assert {result.status_of(item.candidate_id) for item in evaluations} == {
        CandidateStatus.RANKABLE
    }
    costs = tuple(row.total_cost for row in result.rankable)
    assert costs == tuple(sorted(costs))
    assert all(
        row.total_cost == pytest.approx(sum(line.weighted_cost for line in row.categories))
        for row in result.rankable
    )


def test_formal_registry_real_curves_alert_but_stay_rankable(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
) -> None:
    """正式登記簿（峰 6／谷 15 dB 警戒）下，假能量的真曲線（峰約 +11、谷約 −19 dB）要掛出警戒、
    而不是被淘汰——#445 之前這三個候選全被 3 dB 底線淘汰、考卷得用放寬 200 dB 的副本才排得起來。"""
    evaluations = tuple(item.evaluation for item in integrated_candidates)

    result = _rank(*evaluations)

    assert result.eliminated == ()
    assert all(
        result.status_of(item.candidate_id) is CandidateStatus.RANKABLE for item in evaluations
    )
    kinds = {alert.kind for row in result.rankable for alert in row.review_alerts}
    assert kinds == {"peak", "dip"}
    assert all("待複核" in alert.note for row in result.rankable for alert in row.review_alerts)


def test_report_capability_flag_reaches_ranking_without_changing_status_or_cost(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
) -> None:
    """若能力標記被排名當成資格或代價，只有狀態不同的同曲線候選會分表或不同分。"""
    original_report = integrated_candidates[0].report
    experimental_report = original_report.model_copy(
        update={
            "capability": original_report.capability.model_copy(
                update={"status": "experimental", "frequency_hz": (20.0, 8000.0)}
            )
        }
    )
    validated_report = experimental_report.model_copy(
        update={
            "capability": experimental_report.capability.model_copy(
                update={"status": "validated"}
            )
        }
    )
    experimental = _evaluate_report(experimental_report, "capability-experimental")
    validated = _evaluate_report(validated_report, "capability-validated")

    assert isinstance(experimental.payload, TimbrePayload)
    assert (
        experimental.payload.model_validation_status
        is ModelValidationStatus.EXPERIMENTAL
    )
    assert Flag.UNVALIDATED in experimental.flags
    assert isinstance(validated.payload, TimbrePayload)
    assert validated.payload.model_validation_status is ModelValidationStatus.VALIDATED
    assert Flag.UNVALIDATED not in validated.flags

    result = _rank(
        experimental,
        validated,
        registry=load_quality_targets(_REGISTRY_PATH),
    )
    experimental_row = next(
        row for row in result.rankable if row.candidate_id == experimental.candidate_id
    )
    validated_row = next(
        row for row in result.rankable if row.candidate_id == validated.candidate_id
    )

    assert result.status_of(experimental.candidate_id) is CandidateStatus.RANKABLE
    assert result.status_of(validated.candidate_id) is CandidateStatus.RANKABLE
    assert experimental_row.total_cost == validated_row.total_cost
    assert Flag.UNVALIDATED in experimental_row.flags
    assert Flag.UNVALIDATED not in validated_row.flags


def test_unavailable_real_evaluation_stays_numeric_value_free(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
) -> None:
    """若不可估被補零，未評估區會出現捏造的 payload、原始量或類代價。"""
    original = integrated_candidates[0].report
    assert original.points is not None
    broken_point = original.points[0].model_copy(update={"total_energy": 0.0})
    broken_report = original.model_copy(
        update={"points": (broken_point, *original.points[1:])}
    )
    evaluation = _evaluate_report(broken_report, "non-positive-energy")
    result = _rank(evaluation)

    assert evaluation.state is EvaluationState.UNAVAILABLE
    assert ReasonCode.NON_POSITIVE_ENERGY in evaluation.reason_codes
    assert result.status_of(evaluation.candidate_id) is CandidateStatus.NOT_EVALUATED
    kept = _ranked_evaluation(result, evaluation.candidate_id)
    assert kept.payload is None
    assert not kept.raw_quantities
    assert kept.category_cost is None
    row = next(item for item in result.not_evaluated if item.candidate_id == evaluation.candidate_id)
    assert any(
        ReasonCode.NON_POSITIVE_ENERGY in missing.evaluator_reason_codes for missing in row.missing
    )
    assert "total_cost" not in row.model_dump(mode="python")


def test_real_report_with_a_band_removed_is_not_evaluated_instead_of_ranked(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
) -> None:
    """同一份真報表拿掉計分範圍中間一段：完整的可排名，缺段的只能是未評估（票 #394）。"""
    original = integrated_candidates[0].report
    assert original.points is not None
    kept_points = tuple(
        point
        for point in original.points
        if not 400.0 <= point.frequency_hz <= 1200.0
    )
    assert len(kept_points) < len(original.points)
    holed_report = original.model_copy(update={"points": kept_points})
    registry = load_quality_targets(_REGISTRY_PATH)

    complete = _evaluate_report(original, "complete-axis")
    holed = _evaluate_report(holed_report, "band-removed")
    result = _rank(complete, holed, registry=registry)

    assert result.status_of("complete-axis") is CandidateStatus.RANKABLE
    assert holed.state is EvaluationState.UNAVAILABLE
    assert holed.reason_codes == (ReasonCode.TIMBRE_SCORING_RANGE_GAP,)
    assert result.status_of("band-removed") is CandidateStatus.NOT_EVALUATED


def test_measured_optional_category_without_a_coster_is_not_ranked() -> None:
    """若排名層替不會算的類補代價，measured（已量）會被誤當成 costed（已算代價）。

    殘響這一類從 #348 第二段起有自己的代價，所以這一題改用還沒有代價的空間感：
    要守的事沒變——沒有代價的類，排名層不准自己生一個出來。
    """
    candidate_id = "spatial-impression-without-coster"
    evaluation = CategoryEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=candidate_id,
        scene_fingerprint=_SCENE_FINGERPRINT,
        placement=POINT_PLACEMENT,
        category=QualityCategory.SPATIAL_IMPRESSION,
        state=EvaluationState.MEASURED,
        payload=SpatialImpressionPayload(category="spatial_impression"),
        raw_quantities=(RawQuantity(name="envelopment", value=0.2, unit="1"),),
        category_cost=None,
        flags=(),
        reason_codes=(),
        evaluator_version="spatial-impression-integration-fixture",
        settings_fingerprint="spatial-impression-settings-fixture",
        provenance=_provenance(candidate_id),
    )
    result = _rank(evaluation)
    row = next(item for item in result.not_evaluated if item.candidate_id == candidate_id)

    assert result.status_of(candidate_id) is CandidateStatus.NOT_EVALUATED
    assert any(
        missing.category is QualityCategory.SPATIAL_IMPRESSION
        and missing.reason is NotEvaluatedReason.COST_NOT_COMPUTED
        for missing in row.missing
    )
    assert row.evaluations[0].category_cost is None


def test_same_curve_with_different_settings_fingerprints_is_split_without_scores(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
    tmp_path: Path,
) -> None:
    """若設定指紋不同仍同表，兩種量法算出的數字會被當成同一把尺。"""
    report = integrated_candidates[0].report
    changed_path = _registry_with_changed_timbre_source(tmp_path, _REGISTRY_PATH)
    official = _evaluate_report(report, "fingerprint-official")
    changed = _evaluate_report(report, "fingerprint-changed", changed_path)
    result = _rank(official, changed, registry=load_quality_targets(changed_path))

    assert official.settings_fingerprint != changed.settings_fingerprint
    assert result.status_of(official.candidate_id) is not result.status_of(changed.candidate_id)
    separated = next(iter(result.not_comparable.rows))
    dumped = separated.model_dump(mode="python")
    assert separated.candidate_id in {official.candidate_id, changed.candidate_id}
    assert {identity.settings_fingerprint for identity in separated.identity} <= {
        official.settings_fingerprint,
        changed.settings_fingerprint,
    }
    assert "total_cost" not in dumped
    assert "categories" not in dumped


def test_ranking_preserves_every_real_measurement_and_adds_a_traceable_cost(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
) -> None:
    """若轉接漏欄，真評估的原始量、標記、原因或完整單支 payload 會在結果中消失。"""
    original = integrated_candidates[0].evaluation
    result = _rank(original, registry=load_quality_targets(_REGISTRY_PATH))
    kept = _ranked_evaluation(result, original.candidate_id)

    assert result.status_of(original.candidate_id) is CandidateStatus.RANKABLE

    assert tuple(
        (item.name.removeprefix("reference."), item.value, item.unit)
        for item in kept.raw_quantities
    ) == tuple((item.name, item.value, item.unit) for item in original.raw_quantities)
    assert set(kept.flags) == set(original.flags)
    assert kept.reason_codes == original.reason_codes
    assert kept.provenance.receiver_id == original.provenance.receiver_id
    assert isinstance(kept.payload, TimbreChannelsPayload)
    assert isinstance(original.payload, TimbrePayload)
    assert kept.payload.channels[0].payload == original.payload
    assert kept.category_cost is not None
    assert kept.category_cost.cost_settings_fingerprint == result.header.registry_fingerprint
    assert kept.category_cost.components


def test_official_baseline_registry_marks_the_whole_real_result_uncalibrated(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
) -> None:
    """若任何參與設定是 baseline（基線）卻標已校準，表頭會把試驗數字冒充品質判決。"""
    official = tuple(
        _evaluate_report(item.report, item.evaluation.candidate_id)
        for item in integrated_candidates
    )
    result = _rank(*official)

    assert result.header.calibration == "baseline"
    assert result.header.calibration_note == BASELINE_NOTE
    assert result.header.calibration_note


def test_report_axis_short_of_registry_ceiling_keeps_coverage_flag_on_rankable_row(
    integrated_candidates: tuple[_IntegratedCandidate, ...],
) -> None:
    """報表細軸沒蓋滿診斷覆蓋範圍時，覆蓋不足標記必須從評估一路留在可排名列。

    正式細軸現在蓋到起伏計分上限（票 #347），高端不再短；只有診斷下限（20 Hz）到計分下限之間
    那一段可以短而仍可估，所以拿掉報表低端的點來造例子。
    """
    registry = load_quality_targets(_REGISTRY_PATH)
    purpose = registry.purpose(_PURPOSE)
    coverage = purpose.entry("timbre_balance.coverage_range_hz")
    ripple = purpose.entry("timbre_balance.ripple_range_hz")
    assert isinstance(coverage, SettingEntry)
    assert isinstance(ripple, SettingEntry)
    assert isinstance(coverage.value, tuple)
    assert isinstance(ripple.value, tuple)
    short_lower_hz = (float(coverage.value[0]) + float(ripple.value[0])) / 2.0
    report = integrated_candidates[0].report
    data = timbre_input_from_report(
        report,
        candidate_id="axis-short-low",
        speaker_id="reference-speaker",
        receiver_id="reference-seat",
        source_reference="三路接合報表共同能量基準",
        provenance=_provenance("axis-short-low"),
    )
    kept = tuple(
        index
        for index, frequency in enumerate(data.frequencies_hz)
        if frequency >= short_lower_hz
    )
    assert 0 < len(kept) < len(data.frequencies_hz)
    truncated = data.model_copy(
        update={
            "frequencies_hz": tuple(data.frequencies_hz[index] for index in kept),
            "total_energy": tuple(data.total_energy[index] for index in kept),
        }
    )
    evaluation = evaluate_timbre(
        truncated, purpose=_PURPOSE, quality_targets_path=_REGISTRY_PATH
    )
    assert isinstance(evaluation.payload, TimbrePayload)
    assert evaluation.payload.data_range_hz[0] > evaluation.payload.coverage_range_hz[0]
    assert Flag.DATA_COVERAGE_SHORT in evaluation.flags

    result = _rank(evaluation, registry=load_quality_targets(_REGISTRY_PATH))

    assert result.status_of(evaluation.candidate_id) is CandidateStatus.RANKABLE
    row = next(item for item in result.rankable if item.candidate_id == evaluation.candidate_id)
    assert Flag.DATA_COVERAGE_SHORT in row.flags
    assert Flag.DATA_COVERAGE_SHORT in row.categories[0].evaluation.flags


def test_tight_alert_limits_on_a_real_curve_keep_candidate_and_evaluation(
    integrated_candidates: tuple[_IntegratedCandidate, ...], tmp_path: Path
) -> None:
    """若真曲線的峰谷警戒漏類、誤淘汰，或排名後丟掉原始評估，這題會紅。"""
    tight_path = _registry_with_limits(tmp_path, "tight.toml", "0.001")
    report = integrated_candidates[0].report
    evaluation = _evaluate_report(report, "tight-alert", tight_path)

    result = _rank(evaluation, registry=load_quality_targets(tight_path))

    assert result.status_of(evaluation.candidate_id) is CandidateStatus.RANKABLE
    row = next(item for item in result.rankable if item.candidate_id == evaluation.candidate_id)
    assert {"peak", "dip"} <= {item.kind for item in row.review_alerts}
    kept = _ranked_evaluation(result, evaluation.candidate_id)
    assert tuple(
        (item.name.removeprefix("reference."), item.value, item.unit)
        for item in kept.raw_quantities
    ) == tuple((item.name, item.value, item.unit) for item in evaluation.raw_quantities)
