"""把一份具體方案接到既有批次物理與評分入口。"""
from __future__ import annotations

import time
from datetime import date
from pathlib import Path

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.geometry.shoebox import Point
from aosr.physics import report_io, three_lane_report
from aosr.physics.reflection_screen import build_reflection_screen
from aosr.physics.reflection_window import build_reflection_window
from aosr.physics.report_output import output_from_report, report_capability
from aosr.physics.report_source import default_source_model
from aosr.physics.third_octave_decay import build_third_octave_decay
from aosr.scoring.contract import CONTRACT_SCHEMA_VERSION, CandidateEvaluation
from aosr.reporting.result import (
    RESULT_SCHEMA_VERSION, PairResult, SchemeResult, Timings, _evaluate_parts,
    read_registry_settings,
)
from aosr.reporting.scheme import Scheme, expected_pairs


def _listening_channel(scheme: Scheme) -> tuple[str, str]:
    """這一片只取聲道宣告順序的第一支；左右各算留給下一片。"""
    channel = scheme.channel_group.channels[0]
    return channel.role, channel.speaker_id


def _input_document(scheme: Scheme, source: Point, receiver: Point,
                    source_model: object) -> dict[str, object]:
    scene = scheme.scene.model_dump(mode="json", exclude_none=True)
    scene["source_m"] = {"x": source.x, "y": source.y, "z": source.z}
    scene["receiver_m"] = {"x": receiver.x, "y": receiver.y, "z": receiver.z}
    scene["source_model"] = source_model
    return scene


def _inputs(scheme: Scheme, capabilities: CapabilityTable,
            directivity: DirectivityDefaults) -> dict[tuple[str, str], tuple[dict[str, object], report_io.ReportInput]]:
    model = ({"kind": "omnidirectional"} if scheme.source_model == "omnidirectional"
             else default_source_model(Point(*scheme.receiver_set.primary.position_m),
                                       directivity).model_dump(mode="json"))
    documents = {}
    receivers = {point.receiver_id: Point(*point.position_m)
                 for point in scheme.receiver_set.points}
    for speaker_id, receiver_id, _ in expected_pairs(scheme):
        document = _input_document(scheme, scheme.speakers[speaker_id],
                                   receivers[receiver_id], model)
        inputs = report_io.load_input_document(document, capabilities, directivity)
        documents[speaker_id, receiver_id] = (document, inputs)
    fingerprints = {report_io.scene_fingerprint(item[1]) for item in documents.values()}
    if len(fingerprints) != 1:
        raise ValueError("方案各對報表的場景指紋不同")
    return documents


def _pair(scheme: Scheme, key: tuple[str, str], document: dict[str, object],
          inputs: report_io.ReportInput, raw: three_lane_report.ThreeLaneReport,
          window_s: float) -> PairResult:
    speaker_id, receiver_id = key
    role = next(role for speaker, receiver, role in expected_pairs(scheme)
                if (speaker, receiver) == key)
    lane = raw.geometric_lane
    return PairResult(
        role=role, speaker_id=speaker_id, receiver_id=receiver_id,
        report_id=f"report-{speaker_id}-{receiver_id}", input_document=document,
        report=output_from_report(raw, inputs=inputs, with_points=True,
                                  path_table_inputs=report_io.solver_inputs(inputs)),
        screen=build_reflection_screen(inputs, lane.frequencies_hz),
        window=build_reflection_window(inputs, frequencies_hz=lane.frequencies_hz,
                                       scattering_coefficient=lane.scattering,
                                       window_s=window_s),
        third_octave_decay=build_third_octave_decay(raw, inputs),
    )


def run_scheme(
    scheme: Scheme, *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
    quality_targets_path: Path, engine_commit: str, run_date: date,
) -> SchemeResult:
    """驗每一對、批次求解一次、組零件並評估一份候選。"""
    start = time.perf_counter()
    documents = _inputs(scheme, capabilities, directivity)
    first = next(iter(documents.values()))[1]
    solved = report_io.solver_inputs(first)
    registry = read_registry_settings(quality_targets_path, scheme.purpose)
    before_solve = time.perf_counter()
    raw = three_lane_report.solve_three_lane_reports(
        source_model=solved.source_model, room=solved.room,
        sources=scheme.speakers,
        receivers={point.receiver_id: Point(*point.position_m)
                   for point in scheme.receiver_set.points},
        sound_speed_m_s=solved.sound_speed_m_s,
        density_kg_m3=solved.density_kg_m3,
        impedance_by_wall=solved.impedance_by_wall,
        scattering_by_wall=solved.scattering_by_wall,
        reflection_order_k=solved.reflection_order_k,
        low_frequency_axis=solved.low_frequency_axis,
        capability=report_capability(capabilities),
    )
    before_output = time.perf_counter()
    pairs = tuple(_pair(scheme, key, *documents[key], raw[key], registry.window_s)
                  for key in documents)
    del raw
    before_evaluate = time.perf_counter()
    role, speaker_id = _listening_channel(scheme)
    temporary = SchemeResult(
        schema_version=RESULT_SCHEMA_VERSION,
        scheme=scheme, engine_commit=engine_commit, run_date=run_date,
        listening_area_channel_role=role, listening_area_speaker_id=speaker_id,
        timings=Timings(solve_s=before_output - before_solve,
                        output_s=before_evaluate - before_output, evaluate_s=0.0,
                        total_s=0.0),
        pairs=pairs,
        candidate=CandidateEvaluation(schema_version=CONTRACT_SCHEMA_VERSION,
                                      candidate_id=scheme.scheme_id,
                                      scene_fingerprint=report_io.scene_fingerprint(first),
                                      evaluations=()),
    )
    candidate = _evaluate_parts(temporary, quality_targets_path, registry)
    end = time.perf_counter()
    return SchemeResult.model_validate(temporary.model_copy(update={
        "candidate": candidate,
        "timings": Timings(solve_s=before_output - before_solve,
                           output_s=before_evaluate - before_output,
                           evaluate_s=end - before_evaluate,
                           total_s=end - start),
    }))
