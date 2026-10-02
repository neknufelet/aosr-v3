"""方案結果、可重評的零件與 JSON 存讀。"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aosr.geometry.shoebox import Point
from aosr.physics.reflection_window import ReflectionWindow
from aosr.scoring.contract import (
    CONTRACT_SCHEMA_VERSION, CandidateEvaluation, InputProvenance,
    ListeningAreaChannelsPayload, QualityCategory,
)
from aosr.scoring.reflections import ReflectionInput
from aosr.reporting.physics_stage import PhysicsPair
from aosr.reporting.scheme import Scheme, expected_pairs, pair_input_document


RESULT_SCHEMA_VERSION: Literal["aosr.scheme_result.v4"] = "aosr.scheme_result.v4"
FROZEN = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class Timings(BaseModel):
    """四段牆鐘秒數，不參與任何評估身分。"""

    model_config = FROZEN
    solve_s: float = Field(ge=0.0)
    output_s: float = Field(ge=0.0)
    evaluate_s: float = Field(ge=0.0)
    total_s: float = Field(ge=0.0)


class PurposeSettings(BaseModel):
    """存檔時該用途的完整評分設定；內容與摘要必須互相對得上。"""

    model_config = FROZEN
    purpose: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    content: dict[str, object]

    @model_validator(mode="after")
    def _digest_matches(self) -> Self:
        canonical = json.dumps(self.content, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if self.fingerprint != digest:
            raise ValueError("purpose_settings.fingerprint 與 content 不同")
        return self


class ResultOrigin(BaseModel):
    """結果的產生入口；一般執行與搜尋候選保留不同的出處。

    run 是一般執行，search_baseline 是搜尋原方案，search_candidate 是搜尋候選。
    """

    model_config = FROZEN
    kind: Literal["run", "search_baseline", "search_candidate"]
    search_id: str | None = None
    trial_number: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _fields_match_kind(self) -> Self:
        if self.kind == "run":
            if self.search_id is not None or self.trial_number is not None:
                raise ValueError("run 不可帶 search_id 或 trial_number")
        elif self.kind == "search_baseline":
            if self.search_id is None or not self.search_id.strip() or self.trial_number is not None:
                raise ValueError("search_baseline 必須帶非空 search_id，不可帶 trial_number")
        elif (self.search_id is None or not self.search_id.strip()
              or self.trial_number is None):
            raise ValueError("search_candidate 必須帶非空 search_id 與 trial_number")
        return self


class PairResult(PhysicsPair):
    """一支喇叭到一個座位的可重評零件；窗由當次評分設定建立。"""

    def reflection_input(self, engine_commit: str, window: ReflectionWindow) -> ReflectionInput:
        return ReflectionInput(
            role=self.role, receiver_id=self.receiver_id, report=self.report,
            screen=self.screen, window=window,
            third_octave_decay=self.third_octave_decay, report_id=self.report_id,
            engine_commit=engine_commit, speaker_id=self.speaker_id,
        )

    def provenance(self, engine_commit: str) -> InputProvenance:
        return InputProvenance(report_id=self.report_id, engine_commit=engine_commit,
                               speaker_id=self.speaker_id, receiver_id=self.receiver_id)


class SchemeResult(BaseModel):
    """一份已跑完且可存讀、重評的方案。"""

    model_config = FROZEN
    schema_version: Literal["aosr.scheme_result.v4"]
    scheme: Scheme
    engine_commit: str = Field(min_length=1)
    program_fingerprint: str = Field(pattern=r"^calc-v1:[0-9a-f]{64}$")
    physics_identity: str = Field(pattern=r"^phys-v1:[0-9a-f]{64}$")
    purpose_settings: PurposeSettings
    origin: ResultOrigin
    scope: Literal["stage_two_subset"]
    run_date: date
    # 存檔時用的品質登記簿指紋（`QualityTargets.fingerprint`，驗證後內容的正規化雜湊，
    # 跟排名表頭的登記簿指紋同一把尺；註解、換行不算）。讀回時保留作資訊，
    # 分級以物理身分、重新量出的候選包與該用途快照判斷。
    quality_targets_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    timings: Timings
    pairs: tuple[PairResult, ...]
    candidate: CandidateEvaluation

    @classmethod
    def validate_saved_parts(cls, document: object) -> tuple[Self, dict[str, object]]:
        """存檔候選包只核外層代號；空包讓零件與所有其他欄位先經原本的驗證。

        空包不帶聆聽區 payload（候選內容），完整候選檢查留給重新量出的那一份。
        存檔包的評分契約改版不能阻擋零件讀回，但非物件或候選代號損壞仍拒收。
        """
        if not isinstance(document, dict):
            raise ValueError("結果檔不是物件")
        stored = document.get("candidate")
        if not isinstance(stored, dict):
            raise ValueError("存下的 candidate 不是物件")
        candidate_id = stored.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("存下的 candidate_id 無效")
        empty = CandidateEvaluation(schema_version=CONTRACT_SCHEMA_VERSION,
                                    candidate_id=candidate_id,
                                    scene_fingerprint="0" * 64, evaluations=())
        return cls.model_validate({**document, "candidate": empty}), stored

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.purpose_settings.purpose != self.scheme.purpose:
            raise ValueError("purpose_settings.purpose 與 scheme.purpose 不同")
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
