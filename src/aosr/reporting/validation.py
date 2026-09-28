"""方案與逐對報表輸入共用的無求解驗證入口。"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.geometry.shoebox import Point
from aosr.physics import report_io
from aosr.physics.report_source import default_source_model
from aosr.reporting.scheme import Scheme, expected_pairs, pair_input_document


@dataclass(frozen=True)
class SchemeProblem:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}：{self.message}"


class SchemeValidationError(ValueError):
    def __init__(self, problems: tuple[SchemeProblem, ...]) -> None:
        self.problems = problems
        super().__init__("；".join(str(problem) for problem in problems))


def validated_scheme(document: object) -> Scheme:
    if isinstance(document, Scheme):
        return document
    try:
        return Scheme.model_validate(document)
    except ValidationError as exc:
        problems = []
        for error in exc.errors():
            message = " ".join(str(error["msg"]).removeprefix("Value error, ").split())
            if error.get("input") is None and error["type"] != "missing":
                message = "必填"
            path = ".".join(map(str, error["loc"]))
            if not path:
                path = "scheme_id" if message.startswith("scheme_id") else "scheme"
            problems.append(SchemeProblem(path, message))
        raise SchemeValidationError(tuple(problems)) from exc


def checked_inputs(document: object, *, capabilities: CapabilityTable,
                   directivity: DirectivityDefaults,
                   ) -> tuple[Scheme, dict[tuple[str, str],
                                           tuple[dict[str, object], report_io.ReportInput]]]:
    """驗方案和每一對輸入；失敗時同一種欄位路徑訊息。"""
    scheme = validated_scheme(document)
    model = ({"kind": "omnidirectional"} if scheme.source_model == "omnidirectional"
             else default_source_model(Point(*scheme.receiver_set.primary.position_m),
                                       directivity).model_dump(mode="json"))
    receivers = {point.receiver_id: Point(*point.position_m)
                 for point in scheme.receiver_set.points}
    documents: dict[tuple[str, str], tuple[dict[str, object], report_io.ReportInput]] = {}
    problems: list[SchemeProblem] = []
    for speaker_id, receiver_id, _ in expected_pairs(scheme):
        pair = pair_input_document(scheme, scheme.speakers[speaker_id],
                                   receivers[receiver_id], model)
        try:
            inputs = report_io.load_input_document(pair, capabilities, directivity)
        except report_io.ReportInputError as exc:
            for field, message, pair_specific in exc.issues:
                path = (f"pairs.{speaker_id}.{receiver_id}.{field}" if pair_specific
                        else f"scene.{field}" if field else "scene")
                problem = SchemeProblem(path, message)
                if pair_specific or problem not in problems:
                    problems.append(problem)
        else:
            documents[speaker_id, receiver_id] = (pair, inputs)
    if problems:
        raise SchemeValidationError(tuple(problems))
    fingerprints = {report_io.scene_fingerprint(item[1]) for item in documents.values()}
    if len(fingerprints) != 1:
        raise SchemeValidationError((SchemeProblem("scene", "方案各對報表的場景指紋不同"),))
    return scheme, documents


def validate_scheme(document_or_scheme: object, *, capabilities: CapabilityTable,
                    directivity: DirectivityDefaults) -> tuple[SchemeProblem, ...]:
    """跟執行入口同路驗證，但不求解。"""
    try:
        checked_inputs(document_or_scheme, capabilities=capabilities,
                       directivity=directivity)
    except SchemeValidationError as exc:
        return exc.problems
    return ()
