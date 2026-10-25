"""#558 控制曲線：有斜率、低頻與中高頻峰谷；不跑物理。"""
from pathlib import Path

import numpy as np

from aosr.config.frequency_axis import GEOMETRIC_LANE_FREQUENCIES_HZ
from aosr.config.paths import config_path
from aosr.config.quality_targets import SettingEntry, load_quality_targets
from aosr.scoring.channel_matching import ChannelPointInput, ChannelResponse, evaluate_channel_matching
from aosr.scoring.channel_matching_cost import cost_channel_matching_evaluation
from aosr.scoring.contract import CategoryEvaluation, EvaluationState, InputProvenance, ModelValidationStatus
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.timbre import TimbreInput, evaluate_timbre
from tests.engine._placement import channel_point_placement
from tests.engine._source_model import OMNI_SOURCE_MODEL
from tests.engine.test_channel_matching import _group, _receivers

REGISTRY = config_path("quality_targets.toml")
PURPOSE = "dedicated_two_channel_listening_room"


def curve_input(receiver: str = "main", role: str = "left") -> TimbreInput:
    """低頻峰的高度隨座位、聲道變；中高頻有不同峰谷，不能退化成零控制組。"""
    f = np.asarray(GEOMETRIC_LANE_FREQUENCIES_HZ)
    x = np.log2(f)
    amplitude = 10.0 if receiver == "main" else 17.0
    amplitude *= 1.0 if role == "left" else 0.65
    db = (-0.8 * x + amplitude * np.exp(-((x - np.log2(48.0)) / 0.25) ** 2)
          - 8.0 * np.exp(-((x - np.log2(180.0)) / 0.2) ** 2)
          + 5.0 * np.exp(-((x - np.log2(1200.0)) / 0.3) ** 2)
          - 6.0 * np.exp(-((x - np.log2(3200.0)) / 0.2) ** 2))
    placement = channel_point_placement(receiver, role)
    return TimbreInput(
        candidate_id="split-control", scene_fingerprint="a" * 64,
        source_model=OMNI_SOURCE_MODEL, speaker_id=f"speaker-{role}", receiver_id=receiver,
        source_position_m=placement.speaker_positions_m[0][1],
        receiver_position_m=placement.receiver_positions_m[0][1],
        frequencies_hz=GEOMETRIC_LANE_FREQUENCIES_HZ,
        total_energy=tuple(float(v) for v in 10.0 ** (db / 10.0)),
        source_reference="共同功率", report_flags=(),
        model_validation_status=ModelValidationStatus.VALIDATED,
        model_validation_frequency_range_hz=(float(f[0]), float(f[-1])),
        provenance=InputProvenance(report_id=f"split-{receiver}-{role}", engine_commit="control",
                                   speaker_id=f"speaker-{role}", receiver_id=receiver),
    )


def measured(data: TimbreInput, registry: Path = REGISTRY) -> CategoryEvaluation:
    return evaluate_timbre(data, purpose=PURPOSE, quality_targets_path=registry)


def channel_points(registry: Path = REGISTRY) -> tuple[ChannelPointInput, ...]:
    receivers, group = _receivers(), _group()
    fingerprint = load_quality_targets(registry).fingerprint
    points = []
    for receiver in receivers.points:
        responses = []
        for role in ("left", "right"):
            data = curve_input(receiver.receiver_id, role)
            responses.append(ChannelResponse(
                role=role, timbre_evaluation=measured(data, registry),
                frequencies_hz=data.frequencies_hz, total_energy=data.total_energy,
                direct_distance_m=2.7 if role == "left" else 2.9,
            ))
        points.append(ChannelPointInput(
            receiver_id=receiver.receiver_id, receiver_set_fingerprint=receivers.fingerprint,
            timbre_settings_fingerprint=fingerprint, listening_area_settings_fingerprint="listening",
            channel_group_fingerprint=group.fingerprint, responses=tuple(responses),
        ))
    return tuple(points)


def channel(registry: Path = REGISTRY, points: tuple[ChannelPointInput, ...] | None = None) -> CategoryEvaluation:
    targets = load_quality_targets(registry)
    evaluation = evaluate_channel_matching(
        _receivers(), channel_points(registry) if points is None else points,
        candidate_id="split-control", scene_fingerprint="a" * 64,
        timbre_settings_fingerprint=targets.fingerprint,
        listening_area_settings_fingerprint="listening", channel_group=_group(), purpose=PURPOSE,
        quality_targets_path=registry, sound_speed_m_s=343.0, reflections=None,
    )
    if evaluation.state is EvaluationState.UNAVAILABLE:
        return evaluation
    return cost_channel_matching_evaluation(evaluation, targets.purpose(PURPOSE), targets.fingerprint)


def listening(registry: Path = REGISTRY) -> CategoryEvaluation:
    receivers = _receivers()
    results = []
    for receiver in receivers.points:
        data = curve_input(receiver.receiver_id)
        results.append(ReceiverPointResult(
            receiver_id=receiver.receiver_id, receiver_set_fingerprint=receivers.fingerprint,
            timbre_evaluation=measured(data, registry), frequencies_hz=data.frequencies_hz,
            total_energy=data.total_energy,
        ))
    entry = load_quality_targets(registry).purpose(PURPOSE).entry("timbre_balance.coverage_range_hz")
    assert isinstance(entry, SettingEntry) and isinstance(entry.value, tuple)
    bounds = float(entry.value[0]), float(entry.value[1])
    return evaluate_listening_area(
        receivers, results, candidate_id="split-control", speaker_id="speaker-left",
        timbre_settings_fingerprint=load_quality_targets(registry).fingerprint,
        scene_fingerprint="a" * 64, feature_match_tolerance_hz=_group().feature_match_tolerance_hz,
        broadband_range_hz=bounds,
    )
