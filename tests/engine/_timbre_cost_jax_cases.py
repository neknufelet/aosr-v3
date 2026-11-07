"""自足輸入與既有 numpy 評估器／代價裁判；不載入 JAX。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.channel_group import ChannelComparison, ChannelDefinition, ChannelGroup
from aosr.scoring.channel_matching import ChannelPointInput, ChannelResponse, evaluate_channel_matching
from aosr.scoring.channel_matching_cost import cost_channel_matching_evaluation
from aosr.scoring.contract import CategoryEvaluation
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.listening_area_channels import evaluate_listening_area_channels
from aosr.scoring.listening_area_cost import cost_listening_area_evaluation
from aosr.scoring.receiver_set import ReceiverRole, ReceiverSet
from aosr.scoring.timbre import evaluate_timbre
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.scoring.timbre_cost import cost_timbre_evaluation
from tests.engine.test_listening_area import _receiver_set
from tests.engine.test_scoring_timbre import _curve_input

TARGETS = config_path("quality_targets.toml")
PURPOSE = "dedicated_two_channel_listening_room"
CATEGORIES = ("timbre_balance", "listening_area_stability", "channel_matching")


@dataclass(frozen=True)
class Case:
    frequencies: NDArray[np.float64]
    energies: NDArray[np.float64]
    distances: NDArray[np.float64]
    receivers: ReceiverSet
    group: ChannelGroup
    sound_speed: float
    targets: Path = TARGETS


def curves() -> Case:
    """既有小房座位與音色曲線夾具；不同傾斜、起伏、音量、非等權及零權最差座位。"""
    data = _curve_input(lambda f: np.zeros_like(f), lower_hz=20.0, upper_hz=20000.0, point_count=81)
    frequencies = np.asarray(data.frequencies_hz)
    x = np.log2(frequencies / 200.0)
    energies = np.asarray([[10.0 ** ((-slope * x + ripple * np.sin(x * 2.7) + level) / 10.0)
                            for slope, ripple, level in rows]
                           for rows in (((0.4, 1.2, 0.0), (2.6, 3.0, 6.0), (4.5, 5.0, 12.0)),
                                        ((1.7, 2.1, 3.0), (4.1, 5.3, 14.0), (6.8, 8.0, 22.0)))])
    group = ChannelGroup(channels=(ChannelDefinition(role="left", speaker_id="left"),
                                   ChannelDefinition(role="right", speaker_id="right")),
                         comparisons=(ChannelComparison(left_role="left", right_role="right"),),
                         feature_match_tolerance_hz=10.0)
    return Case(frequencies, energies, np.asarray([[2.0, 2.4, 2.8], [3.0, 4.0, 6.0]]),
                _receiver_set(front_importance=2.0, back_importance=0.5), group, 343.0)


def small_room() -> Case:
    """既有 6.3×4.1×2.9 m 管線夾具；真鏡像路，FEM／晚期沿用考卷的替身。"""
    import math
    from tests.engine.test_scheme_pipeline import _run_control, _scheme
    result = _run_control(_scheme("wall-1"))
    pairs = {(pair.role, pair.receiver_id): pair.report for pair in result.pairs}
    first = result.pairs[0].report.points
    assert first is not None
    energies, distances = [], []
    for channel in result.scheme.channel_group.channels:
        row, distance_row = [], []
        for point in result.scheme.receiver_set.points:
            report = pairs[channel.role, point.receiver_id]
            assert report.points is not None
            row.append([item.total_energy for item in report.points])
            distance_row.append(math.dist(report.scene.source_m.as_tuple(), report.scene.receiver_m.as_tuple()))
        energies.append(row)
        distances.append(distance_row)
    return Case(np.asarray([point.frequency_hz for point in first]), np.asarray(energies), np.asarray(distances),
                result.scheme.receiver_set, result.scheme.channel_group, result.scheme.scene.sound_speed_m_s)


def numpy_evaluations(case: Case, energies: NDArray[np.float64]) -> tuple[CategoryEvaluation, ...]:
    """真的逐點評估、逐聲道彙總，再套現有三支代價函式。"""
    points = case.receivers.points
    base = _curve_input(lambda f: np.zeros_like(f))
    timbres = {}
    for c, channel in enumerate(case.group.channels):
        for s, point in enumerate(points):
            data = base.model_copy(update={
                "speaker_id": channel.speaker_id, "receiver_id": point.receiver_id,
                "source_position_m": (float(c), 0.0, 1.2), "receiver_position_m": point.position_m,
                "frequencies_hz": tuple(case.frequencies), "total_energy": tuple(energies[c, s]),
                "provenance": base.provenance.model_copy(update={
                    "speaker_id": channel.speaker_id, "receiver_id": point.receiver_id}),
            })
            timbres[channel.role, point.receiver_id] = evaluate_timbre(
                data, purpose=PURPOSE, quality_targets_path=case.targets)
    settings = next(iter(timbres.values())).settings_fingerprint
    singles = {channel.role: evaluate_listening_area(case.receivers, tuple(ReceiverPointResult(
        receiver_id=point.receiver_id, receiver_set_fingerprint=case.receivers.fingerprint,
        timbre_evaluation=timbres[channel.role, point.receiver_id],
        frequencies_hz=tuple(case.frequencies), total_energy=tuple(energies[c, s]),
    ) for s, point in enumerate(points)), candidate_id=base.candidate_id,
        speaker_id=channel.speaker_id, timbre_settings_fingerprint=settings,
        scene_fingerprint=base.scene_fingerprint, feature_match_tolerance_hz=case.group.feature_match_tolerance_hz,
        broadband_range_hz=_broadband(case.targets)) for c, channel in enumerate(case.group.channels)}
    listening = evaluate_listening_area_channels(case.group, case.receivers, singles,
        candidate_id=base.candidate_id, scene_fingerprint=base.scene_fingerprint,
        listening_area_settings_fingerprint=next(iter(singles.values())).settings_fingerprint)
    timbre = evaluate_timbre_channels(case.group, case.receivers.primary.receiver_id,
        {channel.role: timbres[channel.role, case.receivers.primary.receiver_id] for channel in case.group.channels},
        candidate_id=base.candidate_id, scene_fingerprint=base.scene_fingerprint, timbre_settings_fingerprint=settings)
    matching = _matching(case, energies, timbres, listening, settings, base.candidate_id, base.scene_fingerprint)
    purpose = load_quality_targets(case.targets).purpose(PURPOSE)
    return tuple(cost(item, purpose, settings) for cost, item in zip(
        (cost_timbre_evaluation, cost_listening_area_evaluation, cost_channel_matching_evaluation),
        (timbre, listening, matching), strict=True))


def _broadband(targets: Path) -> tuple[float, float]:
    from aosr.scoring.channel_matching_settings import _load_settings
    return _load_settings(targets, PURPOSE).broadband_range_hz


def _matching(case: Case, energies: NDArray[np.float64], timbres: dict[tuple[str, str], CategoryEvaluation],
              listening: CategoryEvaluation, settings: str, candidate: str, scene: str) -> CategoryEvaluation:
    inputs = tuple(ChannelPointInput(receiver_id=point.receiver_id,
        receiver_set_fingerprint=case.receivers.fingerprint, timbre_settings_fingerprint=settings,
        listening_area_settings_fingerprint=listening.settings_fingerprint, channel_group_fingerprint=case.group.fingerprint,
        responses=tuple(ChannelResponse(role=channel.role, timbre_evaluation=timbres[channel.role, point.receiver_id],
            frequencies_hz=tuple(case.frequencies), total_energy=tuple(energies[c, s]),
            direct_distance_m=float(case.distances[c, s])) for c, channel in enumerate(case.group.channels)),
    ) for s, point in enumerate(case.receivers.points) if point.role != ReceiverRole.OTHER_SEAT)
    return evaluate_channel_matching(case.receivers, inputs, candidate_id=candidate, scene_fingerprint=scene,
        timbre_settings_fingerprint=settings, listening_area_settings_fingerprint=listening.settings_fingerprint,
        channel_group=case.group, purpose=PURPOSE, quality_targets_path=case.targets,
        sound_speed_m_s=case.sound_speed, reflections=None)


def numpy_costs(case: Case, energies: NDArray[np.float64]) -> NDArray[np.float64]:
    results = numpy_evaluations(case, energies)
    assert all(item.category_cost is not None for item in results), results
    return np.asarray([item.category_cost.value for item in results if item.category_cost is not None])


def numpy_metrics(case: Case, energies: NDArray[np.float64]) -> NDArray[np.float64]:
    from aosr.scoring.channel_matching_settings import _load_settings
    from aosr.scoring.timbre import _octave_mean_level_db, load_timbre_settings, measure_timbre_ripple
    settings = load_timbre_settings(case.targets, PURPOSE)
    matching = _load_settings(case.targets, PURPOSE)
    rows = []
    for channel in energies:
        row = []
        for energy in channel:
            measured = measure_timbre_ripple(case.frequencies, energy, settings.ripple_range_hz, settings)
            other = measure_timbre_ripple(case.frequencies, energy, matching.ripple_range_hz, settings)
            row.append((measured.tilt_db_per_octave, measured.residual_rms_db,
                        other.residual_rms_db, _octave_mean_level_db(case.frequencies, energy, matching.broadband_range_hz)))
        rows.append(row)
    return np.asarray(rows)
