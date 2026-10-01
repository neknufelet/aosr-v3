"""驗方案、批次求解與組裝物理零件；品質登記簿與評估由呼叫端處理。"""
from __future__ import annotations

import time

from pydantic import BaseModel, ConfigDict, Field

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.geometry.shoebox import Point
from aosr.physics import report_io, three_lane_report
from aosr.physics.reflection_screen import ReflectionScreen, build_reflection_screen
from aosr.physics.report_io import ReportInput, ReportOutput
from aosr.physics.report_output import output_from_report, report_capability
from aosr.physics.third_octave_decay import ThirdOctaveDecay, build_third_octave_decay
from aosr.reporting.scheme import Scheme, expected_pairs
from aosr.reporting.validation import checked_inputs


FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class PhysicsPair(BaseModel):
    """一支喇叭到一個座位的物理零件，尚未依品質設定補算反射窗。"""

    model_config = FROZEN
    role: str
    speaker_id: str
    receiver_id: str
    report_id: str
    input_document: dict[str, object]
    report: ReportOutput
    screen: ReflectionScreen
    third_octave_decay: ThirdOctaveDecay


class SchemePhysics(BaseModel):
    """依輸入宣告順序保存物理零件、同一份已驗輸入與求解／輸出牆鐘秒數。"""

    model_config = FROZEN
    pairs: tuple[PhysicsPair, ...]
    scene_fingerprint: str
    solve_s: float = Field(ge=0.0)
    output_s: float = Field(ge=0.0)
    inputs_by_pair: dict[tuple[str, str], ReportInput]


def _pair(scheme: Scheme, key: tuple[str, str], document: dict[str, object],
          inputs: ReportInput, raw: three_lane_report.ThreeLaneReport) -> PhysicsPair:
    speaker_id, receiver_id = key
    role = next(role for speaker, receiver, role in expected_pairs(scheme)
                if (speaker, receiver) == key)
    lane = raw.geometric_lane
    return PhysicsPair(
        role=role, speaker_id=speaker_id, receiver_id=receiver_id,
        report_id=f"report-{speaker_id}-{receiver_id}", input_document=document,
        report=output_from_report(raw, inputs=inputs, with_points=True,
                                  path_table_inputs=report_io.solver_inputs(inputs)),
        screen=build_reflection_screen(inputs, lane.frequencies_hz),
        third_octave_decay=build_third_octave_decay(raw, inputs),
    )


def solve_scheme_physics(
    scheme: Scheme | object, *, capabilities: CapabilityTable, directivity: DirectivityDefaults,
) -> tuple[Scheme, SchemePhysics]:
    """驗每一對、批次求解一次並組物理零件；回傳已驗方案與原輸入供呼叫端建窗。"""
    scheme, documents = checked_inputs(scheme, capabilities=capabilities,
                                       directivity=directivity)
    return scheme, solve_checked_physics(scheme, documents, capabilities=capabilities)


def solve_checked_physics(
    scheme: Scheme,
    documents: dict[tuple[str, str], tuple[dict[str, object], ReportInput]],
    *, capabilities: CapabilityTable,
) -> SchemePhysics:
    """已經過 ``checked_inputs`` 的方案與逐對輸入：批次求解一次並組物理零件。

    主管線先驗每一對、再讀品質登記簿、最後才求解（錯誤先後跟拆分前一樣），所以驗與解分兩支。
    """
    first = next(iter(documents.values()))[1]
    solved = report_io.solver_inputs(first)
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
    pairs = tuple(_pair(scheme, key, *documents[key], raw[key]) for key in documents)
    del raw
    after_output = time.perf_counter()
    return SchemePhysics(
        pairs=pairs, scene_fingerprint=report_io.scene_fingerprint(first),
        solve_s=before_output - before_solve, output_s=after_output - before_output,
        inputs_by_pair={key: inputs for key, (_document, inputs) in documents.items()},
    )
