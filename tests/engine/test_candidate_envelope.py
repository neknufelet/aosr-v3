"""候選包正式四類入口整合考卷（票 #408）。

有限元素能量跟既有整合考卷一樣用快假值；幾何、晚期混響、報表收成與四類評估仍走正式入口。
指揮者量過這一題兩次真求解約 7 秒，所以同一場景只真的求解左聲道主位一份，
右聲道與周圍點三份以 ``model_copy`` 換 ``scene`` 座標；不同材料的拒收樣本另真求解一份，
因此兩個場景指紋都來自正式報表輸入。
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Final

import pytest
from pydantic import ValidationError

from aosr.config.capabilities import load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import report_io, three_lane_report
from aosr.physics.report_io import ReportOutput, SceneSection
from aosr.physics.report_output import output_from_report
from aosr.scoring.channel_matching import (
    ChannelComparison,
    ChannelDefinition,
    ChannelGroup,
    ChannelPointInput,
    ChannelResponse,
    evaluate_channel_matching,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION,
    CandidateEvaluation,
    CategoryEvaluation,
    EvaluationState,
    InputProvenance,
    QualityCategory,
)
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.ranking import (
    CandidateStatus,
    RankingContext,
    RankingResult,
    rank_candidates,
)
from aosr.scoring.receiver_set import ReceiverPoint, ReceiverRole, ReceiverSet
from aosr.scoring.reverberation import evaluate_reverberation
from aosr.scoring.timbre import evaluate_timbre, timbre_input_from_report
from aosr.scoring.timbre_channels import evaluate_timbre_channels


_CANDIDATE: Final[str] = "candidate-scene-envelope"
_PURPOSE: Final[str] = "dedicated_two_channel_listening_room"
_TARGETS = config_path("quality_targets.toml")
_ROOM: Final[Room] = Room(6.0, 4.0, 3.0)
_LEFT: Final[Point] = Point(1.2, 1.3, 1.1)
_RIGHT: Final[Point] = Point(1.2, 2.7, 1.1)
_MAIN: Final[Point] = Point(4.7, 2.0, 1.4)
_FRONT: Final[Point] = Point(4.5, 2.0, 1.4)
_SOUND_SPEED: Final[float] = 343.0
_DENSITY: Final[float] = 1.2


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
    del room, source, receiver, wall_impedances, density_kg_m3, sound_speed_m_s
    return tuple(frequency_hz * 3.0 + 7.0 for frequency_hz in frequencies_hz)


def _solve_report(
    monkeypatch: pytest.MonkeyPatch, impedance_multiple: float
) -> ReportOutput:
    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _fake_fem_energy)
    rho_c = _SOUND_SPEED * _DENSITY
    inputs = report_io.load_input_document(
        {
            "room_m": {"Lx": _ROOM.Lx, "Ly": _ROOM.Ly, "Lz": _ROOM.Lz},
            "source_m": {"x": _LEFT.x, "y": _LEFT.y, "z": _LEFT.z},
            "receiver_m": {"x": _MAIN.x, "y": _MAIN.y, "z": _MAIN.z},
            "sound_speed_m_s": _SOUND_SPEED,
            "density_kg_m3": _DENSITY,
            "impedance_pa_s_per_m_by_wall": {
                wall.wall_name(): impedance_multiple * rho_c for wall in Wall.all()
            },
        },
        load_capabilities(config_path("capabilities.toml")),
    )
    solved = report_io.solver_inputs(inputs)
    solved_report = three_lane_report.solve_three_lane_report(
        room=solved.room,
        source=solved.source,
        receiver=solved.receiver,
        sound_speed_m_s=solved.sound_speed_m_s,
        density_kg_m3=solved.density_kg_m3,
        impedance_by_wall=solved.impedance_by_wall,
        scattering_by_wall=solved.scattering_by_wall,
        reflection_order_k=solved.reflection_order_k,
    )
    return output_from_report(solved_report, inputs=inputs, with_points=True)


def _at(report: ReportOutput, source: Point, receiver: Point) -> ReportOutput:
    return report.model_copy(
        update={
            "scene": SceneSection(
                scene_fingerprint=report.scene.scene_fingerprint,
                source_m=source,
                receiver_m=receiver,
            )
        }
    )


def _provenance(speaker: str, receiver: str) -> InputProvenance:
    return InputProvenance(
        report_id=f"report-{speaker}-{receiver}",
        engine_commit="integration-fixture",
        speaker_id=speaker,
        receiver_id=receiver,
    )


def _timbre(report: ReportOutput, speaker: str, receiver: str) -> CategoryEvaluation:
    collected = timbre_input_from_report(
        report,
        candidate_id=_CANDIDATE,
        speaker_id=speaker,
        receiver_id=receiver,
        source_reference="三路接合報表共同能量基準",
        provenance=_provenance(speaker, receiver),
    )
    return evaluate_timbre(
        collected, purpose=_PURPOSE, quality_targets_path=_TARGETS
    )


def _mean_level_db(report: ReportOutput) -> float:
    assert report.points is not None
    energies = tuple(point.total_energy for point in report.points)
    return 10.0 * math.log10(sum(energies) / len(energies))


def _distance(source: Point, receiver: Point) -> float:
    return math.sqrt(
        (source.x - receiver.x) ** 2
        + (source.y - receiver.y) ** 2
        + (source.z - receiver.z) ** 2
    )


def _channel_response(
    role: str,
    evaluation: CategoryEvaluation,
    report: ReportOutput,
) -> ChannelResponse:
    assert report.points is not None
    return ChannelResponse(
        role=role,
        timbre_evaluation=evaluation,
        frequencies_hz=tuple(point.frequency_hz for point in report.points),
        total_energy=tuple(point.total_energy for point in report.points),
        direct_distance_m=_distance(report.scene.source_m, report.scene.receiver_m),
    )


def _reports_and_timbres(
    base: ReportOutput,
) -> tuple[
    dict[tuple[str, str], ReportOutput],
    dict[tuple[str, str], CategoryEvaluation],
]:
    reports = {
        ("left", "main"): _at(base, _LEFT, _MAIN),
        ("left", "front"): _at(base, _LEFT, _FRONT),
        ("right", "main"): _at(base, _RIGHT, _MAIN),
        ("right", "front"): _at(base, _RIGHT, _FRONT),
    }
    timbres = {key: _timbre(report, *key) for key, report in reports.items()}
    return reports, timbres


def _receivers() -> ReceiverSet:
    return ReceiverSet(
        points=(
            ReceiverPoint(
                receiver_id="main",
                position_m=(_MAIN.x, _MAIN.y, _MAIN.z),
                role=ReceiverRole.PRIMARY,
                importance=1.0,
            ),
            ReceiverPoint(
                receiver_id="front",
                position_m=(_FRONT.x, _FRONT.y, _FRONT.z),
                role=ReceiverRole.SURROUNDING,
                importance=1.0,
                direction_relative_to_primary="front",
            ),
        )
    )


def _listening(
    receivers: ReceiverSet,
    reports: dict[tuple[str, str], ReportOutput],
    timbres: dict[tuple[str, str], CategoryEvaluation],
) -> CategoryEvaluation:
    points = tuple(
        ReceiverPointResult(
            receiver_id=receiver,
            receiver_set_fingerprint=receivers.fingerprint,
            timbre_evaluation=timbres[("left", receiver)],
            broadband_mean_total_energy_db=_mean_level_db(
                reports[("left", receiver)]
            ),
        )
        for receiver in ("main", "front")
    )
    return evaluate_listening_area(
        receivers,
        points,
        candidate_id=_CANDIDATE,
        speaker_id="left",
        timbre_settings_fingerprint=timbres[("left", "main")].settings_fingerprint,
        scene_fingerprint=timbres[("left", "main")].scene_fingerprint,
        feature_match_tolerance_hz=10.0,
    )


def _group() -> ChannelGroup:
    return ChannelGroup(
        channels=(
            ChannelDefinition(role="left", speaker_id="left"),
            ChannelDefinition(role="right", speaker_id="right"),
        ),
        comparisons=(ChannelComparison(left_role="left", right_role="right"),),
        feature_match_tolerance_hz=10.0,
    )


def _channel(
    receivers: ReceiverSet,
    group: ChannelGroup,
    listening: CategoryEvaluation,
    reports: dict[tuple[str, str], ReportOutput],
    timbres: dict[tuple[str, str], CategoryEvaluation],
) -> CategoryEvaluation:
    points = tuple(
        ChannelPointInput(
            receiver_id=receiver,
            receiver_set_fingerprint=receivers.fingerprint,
            timbre_settings_fingerprint=timbres[("left", receiver)].settings_fingerprint,
            listening_area_settings_fingerprint=listening.settings_fingerprint,
            channel_group_fingerprint=group.fingerprint,
            responses=tuple(
                _channel_response(
                    role, timbres[(role, receiver)], reports[(role, receiver)]
                )
                for role in ("left", "right")
            ),
        )
        for receiver in ("main", "front")
    )
    return evaluate_channel_matching(
        receivers,
        points,
        candidate_id=_CANDIDATE,
        scene_fingerprint=timbres[("left", "main")].scene_fingerprint,
        timbre_settings_fingerprint=timbres[("left", "main")].settings_fingerprint,
        listening_area_settings_fingerprint=listening.settings_fingerprint,
        channel_group=group,
        purpose=_PURPOSE,
        quality_targets_path=_TARGETS,
        sound_speed_m_s=_SOUND_SPEED,
    )


def _four_evaluations(
    base: ReportOutput,
) -> tuple[ReceiverSet, ChannelGroup, tuple[CategoryEvaluation, ...]]:
    reports, timbres = _reports_and_timbres(base)
    receivers = _receivers()
    listening = _listening(receivers, reports, timbres)
    group = _group()
    timbre_channels = evaluate_timbre_channels(
        group,
        "main",
        {role: timbres[(role, "main")] for role in ("left", "right")},
        candidate_id=_CANDIDATE,
        scene_fingerprint=timbres[("left", "main")].scene_fingerprint,
        timbre_settings_fingerprint=timbres[("left", "main")].settings_fingerprint,
    )
    channel = _channel(receivers, group, listening, reports, timbres)
    reverberation = evaluate_reverberation(
        base,
        candidate_id=_CANDIDATE,
        provenance=_provenance("left", "main"),
        logarithm_base=2.0,
    )
    return receivers, group, (
        timbre_channels,
        listening,
        channel,
        reverberation,
    )


def _ranked_evaluations(
    result: RankingResult, status: CandidateStatus
) -> tuple[CategoryEvaluation, ...]:
    """從排名輸出拿回這個候選每一類的評估；可排名與被淘汰兩塊的形狀不同。"""
    if status is CandidateStatus.RANKABLE:
        return tuple(
            line.evaluation
            for row in result.rankable
            if row.candidate_id == _CANDIDATE
            for line in row.categories
        )
    return next(
        row.evaluations for row in result.eliminated if row.candidate_id == _CANDIDATE
    )


def _expected_placements() -> dict[QualityCategory, dict[str, object]]:
    return {
        QualityCategory.TIMBRE_BALANCE: {
            "speaker_positions_m": (
                ("left", (_LEFT.x, _LEFT.y, _LEFT.z)),
                ("right", (_RIGHT.x, _RIGHT.y, _RIGHT.z)),
            ),
            "receiver_positions_m": (("main", (_MAIN.x, _MAIN.y, _MAIN.z)),),
        },
        QualityCategory.LISTENING_AREA_STABILITY: {
            "speaker_positions_m": (("left", (_LEFT.x, _LEFT.y, _LEFT.z)),),
            "receiver_positions_m": (
                ("front", (_FRONT.x, _FRONT.y, _FRONT.z)),
                ("main", (_MAIN.x, _MAIN.y, _MAIN.z)),
            ),
        },
        QualityCategory.CHANNEL_MATCHING: {
            "speaker_positions_m": (
                ("left", (_LEFT.x, _LEFT.y, _LEFT.z)),
                ("right", (_RIGHT.x, _RIGHT.y, _RIGHT.z)),
            ),
            "receiver_positions_m": (
                ("front", (_FRONT.x, _FRONT.y, _FRONT.z)),
                ("main", (_MAIN.x, _MAIN.y, _MAIN.z)),
            ),
        },
        QualityCategory.REVERBERATION: {
            "speaker_positions_m": (("left", (_LEFT.x, _LEFT.y, _LEFT.z)),),
            "receiver_positions_m": (("main", (_MAIN.x, _MAIN.y, _MAIN.z)),),
        },
    }


def _assert_ranking_does_not_copy_placement(
    result: RankingResult, status: CandidateStatus
) -> None:
    if status is CandidateStatus.RANKABLE:
        row_fields = type(result.rankable[0]).model_fields
    else:
        row_fields = type(result.eliminated[0]).model_fields
    assert "placement" not in row_fields


def test_four_formal_categories_share_scene_without_rewriting_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """四類真入口可原樣裝包；換入不同材料的真評估時，包在排名前就指名場景拒收。"""
    base = _solve_report(monkeypatch, 4.0)
    receivers, group, evaluations = _four_evaluations(base)
    provenance_by_category = {
        evaluation.category: evaluation.provenance for evaluation in evaluations
    }
    placement_by_category = {
        evaluation.category: evaluation.model_dump(mode="python")["placement"]
        for evaluation in evaluations
    }
    assert placement_by_category == _expected_placements()
    candidate = CandidateEvaluation(
        schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=_CANDIDATE,
        scene_fingerprint=base.scene.scene_fingerprint,
        evaluations=evaluations,
    )

    result = rank_candidates(
        (candidate,),
        load_quality_targets(_TARGETS),
        RankingContext(
            purpose=_PURPOSE,
            receiver_set_fingerprint=receivers.fingerprint,
            channel_group_fingerprint=group.fingerprint,
            run_date=date(2026, 9, 21),
            engine_version="candidate-envelope-integration-fixture",
        ),
    )

    # 四類都真的量到了（不是靠「不可估」混過裝包），而且四類都在包裡。
    assert {evaluation.state for evaluation in evaluations} == {EvaluationState.MEASURED}
    assert {evaluation.category for evaluation in evaluations} == {
        QualityCategory.TIMBRE_BALANCE,
        QualityCategory.LISTENING_AREA_STABILITY,
        QualityCategory.CHANNEL_MATCHING,
        QualityCategory.REVERBERATION,
    }
    status = result.status_of(_CANDIDATE)
    assert status in {
        CandidateStatus.RANKABLE,
        CandidateStatus.ELIMINATED,
    }
    assert {
        evaluation.category: evaluation.provenance
        for evaluation in _ranked_evaluations(result, status)
    } == provenance_by_category
    assert {
        evaluation.category: evaluation.model_dump(mode="python")["placement"]
        for evaluation in _ranked_evaluations(result, status)
    } == placement_by_category
    _assert_ranking_does_not_copy_placement(result, status)

    other_scene = evaluate_reverberation(
        _solve_report(monkeypatch, 7.0),
        candidate_id=_CANDIDATE,
        provenance=_provenance("left", "main"),
        logarithm_base=2.0,
    )
    with pytest.raises(ValidationError, match="scene_fingerprint"):
        CandidateEvaluation(
            schema_version=CONTRACT_SCHEMA_VERSION,
            candidate_id=_CANDIDATE,
            scene_fingerprint=base.scene.scene_fingerprint,
            evaluations=(*evaluations[:-1], other_scene),
        )


@pytest.mark.parametrize(
    ("table", "shared_id"),
    (
        ("speaker_positions_m", "left"),
        ("receiver_positions_m", "main"),
    ),
)
def test_timbre_and_listening_area_cannot_disagree_on_one_id_position(
    monkeypatch: pytest.MonkeyPatch, table: str, shared_id: str
) -> None:
    """音色與聆聽區的同一真實喇叭或接收點代號若換座標，候選包須指名代號拒收。"""
    base = _solve_report(monkeypatch, 4.0)
    _, _, evaluations = _four_evaluations(base)
    timbre, listening = evaluations[:2]
    rows = tuple(
        (identifier, (9.0, 8.0, 7.0) if identifier == shared_id else coordinate)
        for identifier, coordinate in getattr(listening.placement, table)
    )
    changed = listening.model_copy(
        update={
            "placement": listening.placement.model_copy(update={table: rows})
        }
    )

    with pytest.raises(ValueError, match=shared_id):
        CandidateEvaluation(
            schema_version=CONTRACT_SCHEMA_VERSION,
            candidate_id=_CANDIDATE,
            scene_fingerprint=base.scene.scene_fingerprint,
            evaluations=(timbre, changed),
        )
