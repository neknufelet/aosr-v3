"""方案結果、可重評的零件與 JSON 存讀。"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Literal, NamedTuple, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.quality_targets import QualityPurpose, SettingEntry, load_quality_targets
from aosr.geometry.shoebox import Point
from aosr.physics.reflection_screen import ReflectionScreen
from aosr.physics.reflection_window import ReflectionWindow
from aosr.physics.report_io import ReportOutput, load_input_document
from aosr.physics.third_octave_decay import ThirdOctaveDecay
from aosr.scoring.channel_matching import ChannelPointInput, ChannelResponse, evaluate_channel_matching
from aosr.scoring.contract import (CONTRACT_SCHEMA_VERSION, CandidateEvaluation,
                                   CategoryEvaluation, InputProvenance, QualityCategory)
from aosr.scoring.listening_area import ReceiverPointResult, evaluate_listening_area
from aosr.scoring.reflections import ReflectionInput, evaluate_reflections
from aosr.scoring.reverberation import evaluate_reverberation
from aosr.scoring.timbre import evaluate_timbre, timbre_input_from_report
from aosr.scoring.timbre_channels import evaluate_timbre_channels
from aosr.reporting.scheme import Scheme, expected_pairs


RESULT_SCHEMA_VERSION: Literal["aosr.scheme_result.v1"] = "aosr.scheme_result.v1"
SOURCE_REFERENCE = "三路接合報表共同能量基準"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Timings(BaseModel):
    """四段牆鐘秒數，不參與任何評估身分。"""

    model_config = FROZEN
    solve_s: float = Field(ge=0.0)
    output_s: float = Field(ge=0.0)
    evaluate_s: float = Field(ge=0.0)
    total_s: float = Field(ge=0.0)


class PairResult(BaseModel):
    """一支喇叭到一個座位的可重評零件。"""

    model_config = FROZEN
    role: str
    speaker_id: str
    receiver_id: str
    report_id: str
    input_document: dict[str, object]
    report: ReportOutput
    screen: ReflectionScreen
    window: ReflectionWindow
    third_octave_decay: ThirdOctaveDecay

    def reflection_input(self, engine_commit: str) -> ReflectionInput:
        return ReflectionInput(
            role=self.role, receiver_id=self.receiver_id, report=self.report,
            screen=self.screen, window=self.window,
            third_octave_decay=self.third_octave_decay, report_id=self.report_id,
            engine_commit=engine_commit, speaker_id=self.speaker_id,
        )

    def provenance(self, engine_commit: str) -> InputProvenance:
        return InputProvenance(report_id=self.report_id, engine_commit=engine_commit,
                               speaker_id=self.speaker_id, receiver_id=self.receiver_id)


class SchemeResult(BaseModel):
    """一份已跑完且可存讀、重評的方案。"""

    model_config = FROZEN
    schema_version: Literal["aosr.scheme_result.v1"]
    scheme: Scheme
    engine_commit: str = Field(min_length=1)
    run_date: date
    listening_area_channel_role: str
    listening_area_speaker_id: str
    timings: Timings
    pairs: tuple[PairResult, ...]
    candidate: CandidateEvaluation

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.candidate.candidate_id != self.scheme.scheme_id:
            raise ValueError("candidate_id 與 scheme_id 不同")
        expected = {(speaker, receiver): role
                    for speaker, receiver, role in expected_pairs(self.scheme)}
        actual = {(pair.speaker_id, pair.receiver_id): pair.role for pair in self.pairs}
        if len(self.pairs) != len(expected) or actual != expected:
            raise ValueError("pairs 必須剛好是聲道組喇叭與座位組的笛卡兒積，角色一致")
        first = self.scheme.channel_group.channels[0]
        if self.listening_area_channel_role != first.role:
            raise ValueError("listening_area_channel_role 必須是首位聲道角色")
        if self.listening_area_speaker_id != first.speaker_id:
            raise ValueError("listening_area_speaker_id 必須是首位聲道喇叭")
        self._check_listening_payload()
        self._check_documents()
        return self

    def _check_listening_payload(self) -> None:
        for evaluation in self.candidate.evaluations:
            if evaluation.category is not QualityCategory.LISTENING_AREA_STABILITY:
                continue
            payload = evaluation.payload
            payload_speaker = getattr(payload, "speaker_id", None)
            provenance_speaker = evaluation.provenance.speaker_id
            if (payload_speaker is not None and payload_speaker != self.listening_area_speaker_id
                    or provenance_speaker != self.listening_area_speaker_id):
                raise ValueError("聆聽區 payload 或 provenance 的 speaker_id 與選定喇叭不同")

    def _check_documents(self) -> None:
        from aosr.reporting.pipeline import _input_document

        receivers = {point.receiver_id: point.position_m
                     for point in self.scheme.receiver_set.points}
        for pair in self.pairs:
            source_model = pair.input_document.get("source_model")
            kind = source_model.get("kind") if isinstance(source_model, dict) else None
            expected_kind = ("omnidirectional" if self.scheme.source_model == "omnidirectional"
                             else "analytic_axisymmetric_two_parameter_v1")
            if kind != expected_kind:
                raise ValueError("input_document 的 source_model.kind 與方案不同")
            source = self.scheme.speakers[pair.speaker_id]
            receiver = Point(*receivers[pair.receiver_id])
            expected = _input_document(self.scheme, source, receiver, source_model)
            if pair.input_document != expected:
                raise ValueError("input_document 的場景或喇叭座位座標與方案不同")


class RegistrySettings(NamedTuple):
    window_s: float
    broadband_range_hz: tuple[float, float]
    logarithm_base: float


def _setting(purpose: QualityPurpose, key: str, unit: str) -> float | tuple[float, ...]:
    found = purpose.entry(key)
    if not isinstance(found, SettingEntry) or found.unit != unit:
        raise ValueError(f"{key} 必須是 {unit} 量法設定")
    return found.value


def read_registry_settings(path: Path, purpose_name: str) -> RegistrySettings:
    """同一次載入核反射窗、寬頻與殘響對數底的值及單位。"""
    purpose = load_quality_targets(path).purpose(purpose_name)
    window = _setting(purpose, "reflections_and_echo.window_upper_ms", "ms")
    broadband = _setting(purpose, "channel_matching.broadband_range_hz", "Hz")
    base = _setting(purpose, "reverberation.adjacent_t20_logarithm_base", "1")
    if isinstance(window, tuple) or isinstance(base, tuple):
        raise ValueError("反射窗與殘響對數底必須是單值")
    if not isinstance(broadband, tuple) or len(broadband) != 2:
        raise ValueError("寬頻範圍必須有兩端點")
    return RegistrySettings(float(window) / 1000.0,
                            (float(broadband[0]), float(broadband[1])), float(base))


def _curve(report: ReportOutput) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if report.points is None:
        raise ValueError("存下來的報表缺逐點資料")
    return (tuple(point.frequency_hz for point in report.points),
            tuple(point.total_energy for point in report.points))


def _distance(report: ReportOutput) -> float:
    return math.dist(report.scene.source_m.as_tuple(), report.scene.receiver_m.as_tuple())


def _channel_points(
    result: SchemeResult, timbres: dict[tuple[str, str], CategoryEvaluation],
    listening: CategoryEvaluation,
) -> tuple[ChannelPointInput, ...]:
    pairs = {(pair.role, pair.receiver_id): pair for pair in result.pairs}
    group = result.scheme.channel_group
    first = group.channels[0].role
    return tuple(ChannelPointInput(
        receiver_id=point.receiver_id,
        receiver_set_fingerprint=result.scheme.receiver_set.fingerprint,
        timbre_settings_fingerprint=timbres[first, point.receiver_id].settings_fingerprint,
        listening_area_settings_fingerprint=listening.settings_fingerprint,
        channel_group_fingerprint=group.fingerprint,
        responses=tuple(ChannelResponse(
            role=channel.role,
            timbre_evaluation=timbres[channel.role, point.receiver_id],
            frequencies_hz=_curve(pairs[channel.role, point.receiver_id].report)[0],
            total_energy=_curve(pairs[channel.role, point.receiver_id].report)[1],
            direct_distance_m=_distance(pairs[channel.role, point.receiver_id].report),
        ) for channel in group.channels),
    ) for point in result.scheme.receiver_set.points)


def _evaluate_parts(result: SchemeResult, quality_targets_path: Path,
                    settings_values: RegistrySettings) -> CandidateEvaluation:
    scheme = result.scheme
    pairs = {(pair.role, pair.receiver_id): pair for pair in result.pairs}
    timbres = {(role, receiver): evaluate_timbre(timbre_input_from_report(
        pair.report, candidate_id=scheme.scheme_id, speaker_id=pair.speaker_id,
        receiver_id=receiver, source_reference=SOURCE_REFERENCE,
        provenance=pair.provenance(result.engine_commit),
    ), purpose=scheme.purpose, quality_targets_path=quality_targets_path)
        for (role, receiver), pair in pairs.items()}
    first = result.listening_area_channel_role
    primary = scheme.receiver_set.primary.receiver_id
    scene = timbres[first, primary].scene_fingerprint
    settings = timbres[first, primary].settings_fingerprint
    # 聆聽區的寬頻量與聲道匹配呼叫同一支寬頻能量計算，兩者共用登記簿這個鍵。
    listening = evaluate_listening_area(
        scheme.receiver_set,
        tuple(ReceiverPointResult(
            receiver_id=point.receiver_id,
            receiver_set_fingerprint=scheme.receiver_set.fingerprint,
            timbre_evaluation=timbres[first, point.receiver_id],
            frequencies_hz=_curve(pairs[first, point.receiver_id].report)[0],
            total_energy=_curve(pairs[first, point.receiver_id].report)[1],
        ) for point in scheme.receiver_set.points),
        candidate_id=scheme.scheme_id, speaker_id=result.listening_area_speaker_id,
        timbre_settings_fingerprint=settings, scene_fingerprint=scene,
        feature_match_tolerance_hz=scheme.channel_group.feature_match_tolerance_hz,
        broadband_range_hz=settings_values.broadband_range_hz,
    )
    reflections = evaluate_reflections(
        scheme.channel_group,
        tuple(pair.reflection_input(result.engine_commit) for pair in result.pairs),
        primary_receiver_id=primary, candidate_id=scheme.scheme_id,
        purpose=scheme.purpose, quality_targets_path=quality_targets_path,
    )
    evaluations = (
        evaluate_timbre_channels(
            scheme.channel_group, primary,
            {channel.role: timbres[channel.role, primary]
             for channel in scheme.channel_group.channels},
            candidate_id=scheme.scheme_id, scene_fingerprint=scene,
            timbre_settings_fingerprint=settings,
        ),
        listening, reflections,
        evaluate_channel_matching(
            scheme.receiver_set, _channel_points(result, timbres, listening),
            candidate_id=scheme.scheme_id, scene_fingerprint=scene,
            timbre_settings_fingerprint=settings,
            listening_area_settings_fingerprint=listening.settings_fingerprint,
            channel_group=scheme.channel_group, purpose=scheme.purpose,
            quality_targets_path=quality_targets_path, reflections=reflections,
            sound_speed_m_s=scheme.scene.sound_speed_m_s,
        ),
        evaluate_reverberation(
            pairs[first, primary].report, candidate_id=scheme.scheme_id,
            provenance=pairs[first, primary].provenance(result.engine_commit),
            logarithm_base=settings_values.logarithm_base,
        ),
    )
    return CandidateEvaluation(schema_version=CONTRACT_SCHEMA_VERSION,
                               candidate_id=scheme.scheme_id,
                               scene_fingerprint=scene, evaluations=evaluations)


def reevaluate(result: SchemeResult, *, quality_targets_path: Path) -> CandidateEvaluation:
    """只讀保存的報表與反射零件，重跑五類評估。"""
    settings_values = read_registry_settings(quality_targets_path, result.scheme.purpose)
    return _evaluate_parts(result, quality_targets_path, settings_values)


def save_result(result: SchemeResult, path: Path) -> None:
    """寫出完整 JSON，目的地由呼叫端決定。"""
    path.write_text(result.model_dump_json(), encoding="utf-8")


def load_result(path: Path, *, capabilities: CapabilityTable,
                directivity: DirectivityDefaults) -> SchemeResult:
    """讀入 JSON 並重新走報表輸入驗證。"""
    with path.open(encoding="utf-8") as handle:
        result = SchemeResult.model_validate(json.load(handle))
    for pair in result.pairs:
        load_input_document(pair.input_document, capabilities, directivity)
    return result
