"""正式管線必須與手拼控制組逐位相同，並經過批次入口。"""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from pathlib import Path
import math
import re

import pytest

from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.physics import report_io
from aosr.physics.report_io import ReportOutput
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_output import output_from_report
from aosr.physics.report_source import default_source_model
from aosr.physics.third_octave_decay import build_third_octave_decay
from aosr.geometry.shoebox import Point, Room, Wall
from aosr.physics import three_lane_report
from aosr.scoring.channel_matching import (
    ChannelComparison, ChannelDefinition, ChannelGroup, ChannelPointInput,
    ChannelResponse, evaluate_channel_matching,
)
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CandidateEvaluation, CategoryEvaluation,
    InputProvenance, ListeningAreaChannelsPayload, QualityCategory,
)
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.listening_area_channels import evaluate_listening_area_channels
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections
from aosr.scoring.reverberation import evaluate_reverberation
from aosr.scoring.timbre import evaluate_timbre, timbre_input_from_report
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.reporting import pipeline
from aosr.reporting.compare import compare_results
from aosr.reporting.result import SchemeResult, load_result, reevaluate, save_result
from aosr.reporting.scheme import scheme_from_document
from aosr.reporting.scheme import Scheme
from tests.engine import _scoring_source_model_control as control
from tests.engine import _source_model_control as source_control
from tests.engine._directivity import DIRECTIVITY
from tests.engine._scheme_cache import shared_json


def _many_fem(
    *, room: Room, sources: Mapping[str, Point], receivers: Mapping[str, Point],
    wall_impedances: Mapping[Wall, float], frequencies_hz: tuple[float, ...],
    density_kg_m3: float, sound_speed_m_s: float,
) -> dict[tuple[str, str], tuple[float, ...]]:
    return {(speaker, receiver): source_control.fake_fem_energy(
        room=room, source=source_point, receiver=receiver_point,
        wall_impedances=wall_impedances, frequencies_hz=frequencies_hz,
        density_kg_m3=density_kg_m3, sound_speed_m_s=sound_speed_m_s,
    ) for speaker, source_point in sources.items()
        for receiver, receiver_point in receivers.items()}


def _scheme(candidate: str) -> Scheme:
    scene = {key: value for key, value in source_control.SCENE_WITHOUT_SOURCE_MODEL.items()
             if key not in {"source_m", "receiver_m"}}
    walls = scene["impedance_pa_s_per_m_by_wall"]
    assert isinstance(walls, dict)
    scene["impedance_pa_s_per_m_by_wall"] = {
        wall: control.IMPEDANCE_MULTIPLES[candidate] * value
        for wall, value in walls.items()}
    return scheme_from_document({
        "schema_version": "aosr.scheme.v1", "scheme_id": candidate,
        "purpose": control.PURPOSE, "scene": scene,
        "source_model": "omnidirectional", "speakers": control.SPEAKERS,
        "channel_group": control.group().model_dump(mode="json"),
        "receiver_set": control.receivers().model_dump(mode="json"),
    })


def _control_result(candidate: str) -> SchemeResult:
    def _forbidden(**kwargs: object) -> tuple[float, ...]:
        raise AssertionError("管線走了單對 FEM")

    with pytest.MonkeyPatch.context() as patch:
        for module, name, fake in control.STAND_INS:
            patch.setattr(module, name, fake)
        patch.setattr(three_lane_report, "_solve_fem_energy", _forbidden)
        patch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
        patch.setattr(pipeline, "report_capability",
                      lambda table: three_lane_report._unchecked_capability())
        return pipeline.run_scheme(_scheme(candidate),
            capabilities=load_capabilities(config_path("capabilities.toml")),
            directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
            engine_commit="control", run_date=date(2026, 9, 27))


def shared_control_result(tmp_path_factory: pytest.TempPathFactory, worker_id: str,
                          candidate: str) -> SchemeResult:
    """管線跑控制組那一個候選（禁走單對入口）；同一次 pytest 只跑一次。"""
    def produce() -> str:
        # 共用的是存成 JSON 的那一份；先確認剛算出來的物件跟它讀回來的逐欄相同，
        # 存讀一致這件事才不會因為共用而沒人考（掉任何一格這裡就紅）。
        calculated = _control_result(candidate)
        text = calculated.model_dump_json()
        assert SchemeResult.model_validate_json(text) == calculated
        return text

    return SchemeResult.model_validate_json(shared_json(tmp_path_factory, worker_id,
                                                        f"scheme-{candidate}", produce))


def shared_control_candidate(tmp_path_factory: pytest.TempPathFactory, worker_id: str,
                             candidate: str) -> CandidateEvaluation:
    """考卷手拼的控制組候選包（走單對入口與控制組替身）；同一次 pytest 只拼一次。"""
    def produce() -> str:
        with pytest.MonkeyPatch.context() as patch:
            for module, name, fake in control.STAND_INS:
                patch.setattr(module, name, fake)
            return control.candidate(candidate).model_dump_json()

    text = shared_json(tmp_path_factory, worker_id, f"control-{candidate}", produce)
    return CandidateEvaluation.model_validate_json(text)


@pytest.fixture(scope="module")
def wall_1(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-1")


@pytest.fixture(scope="module")
def wall_2(tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> SchemeResult:
    return shared_control_result(tmp_path_factory, worker_id, "wall-2")


def _assert_control(result: SchemeResult, expected: CandidateEvaluation, tmp_path: Path) -> None:
    assert result.candidate == expected
    assert result.candidate.model_dump_json() == expected.model_dump_json()
    shared = result.pairs[0].report
    assert all(pair.report.top.f_s_hz == shared.top.f_s_hz
               and pair.report.top.eyring_t60_by_band_s == shared.top.eyring_t60_by_band_s
               and tuple((band.t20_s, band.t30_s) for band in pair.report.bands)
               == tuple((band.t20_s, band.t30_s) for band in shared.bands)
               for pair in result.pairs)
    assert any(item.category is QualityCategory.REFLECTIONS_AND_ECHO
               and item.state.value == "measured" for item in result.candidate.evaluations)
    path = tmp_path / f"{result.scheme.scheme_id}.json"
    save_result(result, path)
    loaded = load_result(path, capabilities=load_capabilities(config_path("capabilities.toml")),
                         directivity=DIRECTIVITY, quality_targets_path=control.TARGETS)
    assert loaded == result
    assert reevaluate(loaded, quality_targets_path=control.TARGETS) == result.candidate


def test_pipeline_control_wall_1(wall_1: SchemeResult, tmp_path: Path,
                                 tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    _assert_control(wall_1, shared_control_candidate(tmp_path_factory, worker_id, "wall-1"), tmp_path)


def test_pipeline_control_wall_2(wall_2: SchemeResult, tmp_path: Path,
                                 tmp_path_factory: pytest.TempPathFactory, worker_id: str) -> None:
    _assert_control(wall_2, shared_control_candidate(tmp_path_factory, worker_id, "wall-2"), tmp_path)


def test_pipeline_control_results_compare(wall_1: SchemeResult, wall_2: SchemeResult) -> None:
    results = [wall_1, wall_2]
    ranking = compare_results(results, quality_targets=load_quality_targets(control.TARGETS),
                              run_date=date(2026, 9, 27))
    assert {row.candidate_id for row in ranking.rankable} == set(control.IMPEDANCE_MULTIPLES)
    _compare_variants(results)


def _compare_variants(results: list[SchemeResult]) -> None:
    first, second = results
    assert isinstance(first, SchemeResult) and isinstance(second, SchemeResult)
    targets = load_quality_targets(control.TARGETS)
    run_date = date(2026, 9, 27)
    variants = (
        (second.model_copy(update={"scheme": second.scheme.model_copy(update={"purpose": "different"})}), "purpose"),
        (second.model_copy(update={"scheme": second.scheme.model_copy(update={
            "channel_group": second.scheme.channel_group.model_copy(update={
                "feature_match_tolerance_hz": 9.0})})}), "聲道組指紋"),
        (second.model_copy(update={"engine_commit": "other"}), "engine_commit"),
        (first, "候選代號重複"),
    )
    for variant, message in variants:
        with pytest.raises(ValueError, match=message):
            compare_results([first, variant], quality_targets=targets, run_date=run_date)
    reverse = second.scheme.channel_group.model_copy(update={
        "channels": tuple(reversed(second.scheme.channel_group.channels))})
    assert reverse.fingerprint == second.scheme.channel_group.fingerprint
    changed = second.model_copy(update={
        "scheme": second.scheme.model_copy(update={"channel_group": reverse})})
    reordered = compare_results([first, changed], quality_targets=targets, run_date=run_date)
    assert {row.candidate_id for row in reordered.rankable} == set(control.IMPEDANCE_MULTIPLES)
    translated = _shift_receivers(second, 0.1, all_points=True)
    assert translated.scheme.receiver_set.fingerprint != first.scheme.receiver_set.fingerprint
    same_table = compare_results([first, translated], quality_targets=targets, run_date=run_date)
    assert {row.candidate_id for row in same_table.rankable} == set(control.IMPEDANCE_MULTIPLES)


def _shift_receivers(result: SchemeResult, delta: float, *, all_points: bool) -> SchemeResult:
    points = tuple(point.model_copy(update={"position_m": (
        point.position_m[0] + delta if all_points or point.role.value == "surrounding"
        else point.position_m[0], *point.position_m[1:])})
        for point in result.scheme.receiver_set.points)
    receivers = result.scheme.receiver_set.model_copy(update={"points": points})
    return result.model_copy(update={"scheme": result.scheme.model_copy(
        update={"receiver_set": receivers})})


def _leaves(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        return {path: leaf for key, item in value.items()
                for path, leaf in _leaves(item, f"{prefix}.{key}").items()}
    if isinstance(value, (list, tuple)):
        if not value:
            return {f"{prefix}[]": ()}
        return {path: leaf for index, item in enumerate(value)
                for path, leaf in _leaves(item, f"{prefix}[{index}]").items()}
    return {prefix: value}


def _named_strings(value: object, key: str) -> set[str]:
    if isinstance(value, dict):
        direct = {item for name, item in value.items()
                  if name == key and isinstance(item, str)}
        return direct | set().union(*(_named_strings(item, key) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return set().union(*(_named_strings(item, key) for item in value))
    return set()


def test_pipeline_real_capability_changes_only_validation_leaves(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory, worker_id: str,
) -> None:
    expected = shared_control_result(tmp_path_factory, worker_id, "wall-1").candidate
    for module, name, fake in control.STAND_INS:
        monkeypatch.setattr(module, name, fake)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
    table = load_capabilities(config_path("capabilities.toml"))
    result = pipeline.run_scheme(_scheme("wall-1"), capabilities=table,
        directivity=DIRECTIVITY, quality_targets_path=control.TARGETS,
        engine_commit="control", run_date=date(2026, 9, 27))
    left = _leaves(expected.model_dump(mode="json"))
    right = _leaves(result.candidate.model_dump(mode="json"))
    differences = {path for path in left.keys() | right.keys()
                   if left.get(path) != right.get(path)}
    allowed = (
        ".payload.bands[].model_validation_status",
        ".payload.channels[].payload.model_validation_frequency_range_hz[]",
        ".payload.channels[].payload.model_validation_status",
        ".payload.point_sources[].channels[].payload.model_validation_frequency_range_hz[]",
        ".payload.point_sources[].channels[].payload.model_validation_status",
    )
    canonical = {re.sub(r"\[\d+\]", "[]", path) for path in differences}
    assert all(any(path.endswith(pattern) for pattern in allowed) for path in canonical), sorted(canonical)
    assert all(any(path.endswith(pattern) for path in canonical) for pattern in allowed)
    status_paths = [path for path in differences if path.endswith("model_validation_status")]
    assert all(left[path] == "unchecked" and right[path] == "experimental"
               for path in status_paths)
    range_values = {right[path] for path in differences
                    if "model_validation_frequency_range_hz[" in path and path in right}
    assert range_values == {20.0, 11166.0}
    assert all(pair.report.capability.status == "experimental" for pair in result.pairs)


def _analytic_scheme() -> Scheme:
    original = _scheme("wall-1")
    assert isinstance(original, Scheme)
    return original.model_copy(update={
        "source_model": "product_default",
        "speakers": {"spk-a": original.speakers["left"],
                     "spk-b": original.speakers["right"]},
        "channel_group": ChannelGroup(
            channels=(ChannelDefinition(role="left", speaker_id="spk-a"),
                      ChannelDefinition(role="right", speaker_id="spk-b")),
            comparisons=(ChannelComparison(left_role="left", right_role="right"),),
            feature_match_tolerance_hz=original.channel_group.feature_match_tolerance_hz,
        ),
    })


def _hand_pair(scheme: Scheme, role: str, receiver_id: str,
               table: CapabilityTable) -> tuple[ReportOutput, ReflectionInput]:
    speaker_id = next(item.speaker_id for item in scheme.channel_group.channels
                      if item.role == role)
    receiver = next(item for item in scheme.receiver_set.points
                    if item.receiver_id == receiver_id)
    model = default_source_model(Point(*scheme.receiver_set.primary.position_m), DIRECTIVITY)
    document = scheme.scene.model_dump(mode="json", exclude_unset=True) | {
        "source_m": scheme.speakers[speaker_id].__dict__,
        "receiver_m": dict(zip(("x", "y", "z"), receiver.position_m, strict=True)),
        "source_model": model.model_dump(mode="json"),
    }
    inputs = report_io.load_input_document(document, table, DIRECTIVITY)
    solved = report_io.solver_inputs(inputs)
    raw = three_lane_report.solve_three_lane_report(**solved._asdict())
    output = output_from_report(raw, inputs=inputs, with_points=True, path_table_inputs=solved)
    lane = raw.geometric_lane
    return output, ReflectionInput(
        role=role, receiver_id=receiver_id, report=output,
        screen=build_reflection_screen(inputs, lane.frequencies_hz),
        window=build_reflection_window(inputs, frequencies_hz=lane.frequencies_hz,
                                       scattering_coefficient=lane.scattering,
                                       window_s=control.WINDOW_S),
        third_octave_decay=build_third_octave_decay(raw, inputs),
        report_id=f"report-{speaker_id}-{receiver_id}", engine_commit="control",
        speaker_id=speaker_id,
    )


def _curve(report: ReportOutput) -> tuple[tuple[float, ...], tuple[float, ...]]:
    assert report.points is not None
    return (tuple(item.frequency_hz for item in report.points),
            tuple(item.total_energy for item in report.points))


def _hand_points(scheme: Scheme, reports: dict[tuple[str, str], ReportOutput],
                 timbres: dict[tuple[str, str], CategoryEvaluation],
                 listening: CategoryEvaluation) -> tuple[ChannelPointInput, ...]:
    return tuple(ChannelPointInput(
        receiver_id=point.receiver_id,
        receiver_set_fingerprint=scheme.receiver_set.fingerprint,
        timbre_settings_fingerprint=timbres["left", point.receiver_id].settings_fingerprint,
        listening_area_settings_fingerprint=listening.settings_fingerprint,
        channel_group_fingerprint=scheme.channel_group.fingerprint,
        responses=tuple(ChannelResponse(
            role=channel.role,
            timbre_evaluation=timbres[channel.role, point.receiver_id],
            frequencies_hz=_curve(reports[channel.role, point.receiver_id])[0],
            total_energy=_curve(reports[channel.role, point.receiver_id])[1],
            direct_distance_m=math.dist(
                reports[channel.role, point.receiver_id].scene.source_m.as_tuple(),
                reports[channel.role, point.receiver_id].scene.receiver_m.as_tuple()),
        ) for channel in scheme.channel_group.channels),
    ) for point in scheme.receiver_set.points)


def _hand_listening(scheme: Scheme, reports: dict[tuple[str, str], ReportOutput],
                    timbres: dict[tuple[str, str], CategoryEvaluation],
                    scene: str, settings: str) -> CategoryEvaluation:
    singles = {channel.role: evaluate_listening_area(
        scheme.receiver_set, tuple(ReceiverPointResult(
            receiver_id=point.receiver_id,
            receiver_set_fingerprint=scheme.receiver_set.fingerprint,
            timbre_evaluation=timbres[channel.role, point.receiver_id],
            frequencies_hz=_curve(reports[channel.role, point.receiver_id])[0],
            total_energy=_curve(reports[channel.role, point.receiver_id])[1],
        ) for point in scheme.receiver_set.points),
        candidate_id=scheme.scheme_id, speaker_id=channel.speaker_id,
        timbre_settings_fingerprint=settings, scene_fingerprint=scene,
        feature_match_tolerance_hz=scheme.channel_group.feature_match_tolerance_hz,
        broadband_range_hz=(20.0, 8000.0),
    ) for channel in scheme.channel_group.channels}
    return evaluate_listening_area_channels(
        scheme.channel_group, scheme.receiver_set, singles,
        candidate_id=scheme.scheme_id, scene_fingerprint=scene,
        listening_area_settings_fingerprint=singles["left"].settings_fingerprint,
    )


def _hand_candidate(scheme: Scheme) -> CandidateEvaluation:
    table = load_capabilities(config_path("capabilities.toml"))
    data = {(channel.role, point.receiver_id): _hand_pair(
        scheme, channel.role, point.receiver_id, table)
        for channel in scheme.channel_group.channels for point in scheme.receiver_set.points}
    reports = {key: value[0] for key, value in data.items()}
    timbres = {(role, receiver): evaluate_timbre(timbre_input_from_report(
        report, candidate_id=scheme.scheme_id,
        speaker_id=next(item.speaker_id for item in scheme.channel_group.channels if item.role == role),
        receiver_id=receiver, source_reference="三路接合報表共同能量基準",
        provenance=InputProvenance(
            report_id=f"report-{next(item.speaker_id for item in scheme.channel_group.channels if item.role == role)}-{receiver}",
            engine_commit="control",
            speaker_id=next(item.speaker_id for item in scheme.channel_group.channels if item.role == role),
            receiver_id=receiver),
    ), purpose=scheme.purpose, quality_targets_path=control.TARGETS)
        for (role, receiver), report in reports.items()}
    primary = scheme.receiver_set.primary.receiver_id
    scene = timbres["left", primary].scene_fingerprint
    settings = timbres["left", primary].settings_fingerprint
    listening = _hand_listening(scheme, reports, timbres, scene, settings)
    reflections = evaluate_reflections(
        scheme.channel_group, tuple(item[1] for item in data.values()),
        primary_receiver_id=primary, candidate_id=scheme.scheme_id,
        purpose=scheme.purpose, quality_targets_path=control.TARGETS)
    evaluations = (
        evaluate_timbre_channels(scheme.channel_group, primary,
            {channel.role: timbres[channel.role, primary] for channel in scheme.channel_group.channels},
            candidate_id=scheme.scheme_id, scene_fingerprint=scene,
            timbre_settings_fingerprint=settings),
        listening, reflections,
        evaluate_channel_matching(scheme.receiver_set,
            _hand_points(scheme, reports, timbres, listening),
            candidate_id=scheme.scheme_id, scene_fingerprint=scene,
            timbre_settings_fingerprint=settings,
            listening_area_settings_fingerprint=listening.settings_fingerprint,
            channel_group=scheme.channel_group, purpose=scheme.purpose,
            quality_targets_path=control.TARGETS, reflections=reflections,
            sound_speed_m_s=scheme.scene.sound_speed_m_s),
        evaluate_reverberation(reports["left", primary], candidate_id=scheme.scheme_id,
            provenance=InputProvenance(
                report_id="report-spk-a-main", engine_commit="control",
                speaker_id="spk-a", receiver_id=primary), logarithm_base=2.0),
    )
    return CandidateEvaluation(schema_version=CONTRACT_SCHEMA_VERSION,
        candidate_id=scheme.scheme_id, scene_fingerprint=scene, evaluations=evaluations)


def test_pipeline_analytic_source_uses_speaker_ids_and_all_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scheme = _analytic_scheme()
    for module, name, fake in control.STAND_INS:
        monkeypatch.setattr(module, name, fake)
    expected = _hand_candidate(scheme)

    def _forbidden(**kwargs: object) -> tuple[float, ...]:
        raise AssertionError("管線走了單對 FEM")

    monkeypatch.setattr(three_lane_report, "_solve_fem_energy", _forbidden)
    monkeypatch.setattr(three_lane_report, "_solve_fem_energies", _many_fem)
    monkeypatch.setattr(pipeline, "report_capability", lambda table: three_lane_report._unchecked_capability())
    table = load_capabilities(config_path("capabilities.toml"))
    result = pipeline.run_scheme(scheme, capabilities=table, directivity=DIRECTIVITY,
        quality_targets_path=control.TARGETS, engine_commit="control",
        run_date=date(2026, 9, 27))
    assert result.candidate == expected
    listening = next(item for item in result.candidate.evaluations
                     if item.category is QualityCategory.LISTENING_AREA_STABILITY)
    assert isinstance(listening.payload, ListeningAreaChannelsPayload)
    assert {(item.role, item.speaker_id) for item in listening.payload.channels} == {
        ("left", "spk-a"), ("right", "spk-b")}
    assert {pair.speaker_id for pair in result.pairs} == {"spk-a", "spk-b"}
    assert {pair.role for pair in result.pairs} == {"left", "right"}
    candidate_document = result.candidate.model_dump(mode="json")
    assert {name for name in _named_strings(candidate_document, "speaker_id")
            if name.startswith("spk-")} == {"spk-a", "spk-b"}
    assert {name for name in _named_strings(candidate_document, "role")
            if name in {"left", "right"}} == {"left", "right"}
    assert all(item.state.value == "measured" and not item.reason_codes
               for item in result.candidate.evaluations)
