"""方案結果、可重評的零件與 JSON 存讀。"""
from __future__ import annotations

import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.geometry.shoebox import Point
from aosr.physics.reflection_screen import ReflectionScreen
from aosr.physics.reflection_window import ReflectionWindow
from aosr.physics.report_io import ReportOutput
from aosr.physics.third_octave_decay import ThirdOctaveDecay
from aosr.scoring.contract import (
    CandidateEvaluation, InputProvenance, ListeningAreaChannelsPayload, QualityCategory,
)
from aosr.scoring.reflections import ReflectionInput
from aosr.reporting.scheme import Scheme, expected_pairs, pair_input_document


RESULT_SCHEMA_VERSION: Literal["aosr.scheme_result.v3"] = "aosr.scheme_result.v3"
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
    schema_version: Literal["aosr.scheme_result.v3"]
    scheme: Scheme
    engine_commit: str = Field(min_length=1)
    calculation_fingerprint: str = Field(pattern=r"^calc-v1:[0-9a-f]{64}$")
    run_date: date
    # 存檔時用的品質登記簿指紋（`QualityTargets.fingerprint`，驗證後內容的正規化雜湊，
    # 跟排名表頭的登記簿指紋同一把尺；註解、換行不算）。讀回時靠它分清
    # 「登記簿內容換了」與「檔案內容被改過」。
    quality_targets_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
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
        self._check_listening_payload()
        self._check_documents()
        return self

    def _check_listening_payload(self) -> None:
        for evaluation in self.candidate.evaluations:
            if evaluation.category is not QualityCategory.LISTENING_AREA_STABILITY:
                continue
            payload = evaluation.payload
            if not isinstance(payload, ListeningAreaChannelsPayload):
                continue
            expected = {item.role: item.speaker_id for item in self.scheme.channel_group.channels}
            actual = {item.role: item.speaker_id for item in payload.channels}
            if actual != expected or payload.channel_group_fingerprint != self.scheme.channel_group.fingerprint:
                raise ValueError("聆聽區聲道清單與方案聲道組不同")
            primary = self.scheme.receiver_set.primary.receiver_id
            if (payload.receiver_set_fingerprint != self.scheme.receiver_set.fingerprint
                    or payload.primary_receiver_id != primary):
                raise ValueError("聆聽區彙總座位組與方案不同")
            if (evaluation.provenance.speaker_id
                    != f"channel-group:{self.scheme.channel_group.fingerprint}"
                    or evaluation.provenance.receiver_id != primary):
                raise ValueError("聆聽區彙總 provenance 出身與方案不同")
            for channel in payload.channels:
                if (channel.payload.speaker_id != channel.speaker_id
                        or channel.provenance.speaker_id != channel.speaker_id):
                    raise ValueError("聆聽區單支 payload 或 provenance 的喇叭代號不同")
                if (channel.payload.receiver_set_fingerprint != payload.receiver_set_fingerprint
                        or channel.provenance.receiver_id != primary
                        or channel.payload.candidate_id != self.scheme.scheme_id
                        or channel.payload.settings_fingerprint
                        != payload.listening_area_settings_fingerprint):
                    raise ValueError("聆聽區單支身分與彙總或方案不同")

    def _check_documents(self) -> None:
        receivers = {point.receiver_id: point.position_m
                     for point in self.scheme.receiver_set.points}
        for pair in self.pairs:
            if pair.report_id != f"report-{pair.speaker_id}-{pair.receiver_id}":
                raise ValueError("候選包跟存下來的零件對不上：report_id 與喇叭座位代號不同")
            for name in ("source_m", "receiver_m"):
                report_point = getattr(pair.report.scene, name).as_tuple()
                input_point = pair.input_document.get(name)
                expected_point = (tuple(input_point.get(axis) for axis in ("x", "y", "z"))
                                  if isinstance(input_point, dict) else None)
                if report_point != expected_point:
                    raise ValueError(f"候選包跟存下來的零件對不上：report.scene.{name} 與 input_document 不同")
            source_model = pair.input_document.get("source_model")
            kind = source_model.get("kind") if isinstance(source_model, dict) else None
            expected_kind = ("omnidirectional" if self.scheme.source_model == "omnidirectional"
                             else "analytic_axisymmetric_two_parameter_v1")
            if kind != expected_kind:
                raise ValueError("input_document 的 source_model.kind 與方案不同")
            source = self.scheme.speakers[pair.speaker_id]
            receiver = Point(*receivers[pair.receiver_id])
            expected = pair_input_document(self.scheme, source, receiver, source_model)
            if pair.input_document != expected:
                raise ValueError("input_document 的場景或喇叭座位座標與方案不同")


def save_result(result: SchemeResult, path: Path) -> None:
    """同目錄寫完才替換目的地；失敗清除暫存檔。"""
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(result.model_dump_json())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)  # noqa: PTH105  # expires=2026-12-08 reason=同目錄原子替換需明用 os.replace
    finally:
        temporary.unlink(missing_ok=True)
