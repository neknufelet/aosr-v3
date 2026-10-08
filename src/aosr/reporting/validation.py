"""方案與逐對報表輸入共用的無求解驗證入口。"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from aosr.config.capabilities import CapabilityTable
from aosr.config.directivity_defaults import DirectivityDefaults
from aosr.config.precision_contracts import default_precision_contracts_path, furniture_contact_rel
from aosr.geometry.furniture import FurnitureKind, contact_margin_m
from aosr.geometry.shoebox import Point
from aosr.physics import report_io
from aosr.physics.report_source import default_source_model
from aosr.reporting.furniture_layout import direct_blockers, furniture_boxes
from aosr.reporting.scheme import ListenerPlacement, Scheme, expected_pairs, pair_input_document


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


def furniture_problems(scheme: Scheme) -> tuple[SchemeProblem, ...]:
    """方案與搜尋共用的擺放問題族，含喇叭高度；沒有家具仍須核落地高度。"""
    contact_rel = furniture_contact_rel(default_precision_contracts_path())
    heights = _speaker_height_problems(scheme, contact_rel=contact_rel)
    try:
        layout = furniture_boxes(scheme, contact_rel=contact_rel)
        blockers = direct_blockers(scheme, layout)
    except ValueError as exc:
        return heights + (SchemeProblem("furniture", str(exc)),)
    return heights + tuple(SchemeProblem(
        f"pairs.{speaker}.{receiver}", f"不符合擺位要求：喇叭 {speaker} 到座位 {receiver} 的直達路徑被家具 {'、'.join(ids)} 擋住")
        for (speaker, receiver), ids in blockers.items() if ids)


def _speaker_height_problems(scheme: Scheme, *, contact_rel: float) -> tuple[SchemeProblem, ...]:
    """施工單第 3 條的公式與主對話定稿訊息；數字 repr 原樣印，座標不覆寫。"""
    setup = scheme.speaker_setup
    if setup is None or setup.mount == "stand":
        return ()
    center = setup.cabinet.acoustic_center_above_bottom_m
    if setup.mount == "desk":
        table = next(item for item in scheme.furniture or ()
                     if item.kind in (FurnitureKind.COFFEE_TABLE, FurnitureKind.DESK))
        placement = table.placement
        if not isinstance(placement, ListenerPlacement):
            raise ValueError("桌面必須跟著主位")
        top = placement.bottom_height_m + table.height_m
        expected = top + center
        reason = f"桌面頂 {top!r} m＋聲學中心離箱底 {center!r} m＝{expected!r} m"
    else:
        expected = center
        reason = f"落地喇叭聲學中心離地 {center!r} m"
    room = scheme.scene.room_m
    margin = contact_margin_m((room.Lx, room.Ly, room.Lz), contact_rel=contact_rel)
    return tuple(SchemeProblem(f"speakers.{speaker_id}.z",
                 f"喇叭 {speaker_id} 的高度 {point.z!r} m 跟擺法推出值不同：{reason}")
                 for speaker_id, point in scheme.speakers.items() if abs(point.z - expected) > margin)


def _checked_furniture(scheme: Scheme) -> None:
    """先查喇叭高度、家具擺放與直達。"""
    problems = furniture_problems(scheme)
    if problems:
        raise SchemeValidationError(problems)


def checked_inputs(document: object, *, capabilities: CapabilityTable,
                   directivity: DirectivityDefaults,
                   ) -> tuple[Scheme, dict[tuple[str, str],
                                           tuple[dict[str, object], report_io.ReportInput]]]:
    """驗方案和每一對輸入；失敗時同一種欄位路徑訊息。"""
    scheme = validated_scheme(document)
    if scheme.furniture is not None or scheme.speaker_setup is not None:
        _checked_furniture(scheme)
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
