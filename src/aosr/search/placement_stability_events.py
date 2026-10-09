"""讀結果路徑表，列家具反射出入事件與基準點離邊界的距離；不重算。"""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel

from aosr.config.quality_targets import QualityPurpose
from aosr.geometry.furniture import FaceDirection, FurnitureBox, Vec3
from aosr.reporting.furniture_layout import furniture_boxes
from aosr.reporting.result import SchemeResult
from aosr.reporting.scheme import Scheme
from aosr.scoring.direction_zones import zone_limits


class PathAngles(BaseModel):
    elevation_deg: float


class FurniturePath(BaseModel):
    """已讀回路徑列的所需欄位；不載入物理層，不改結果文件契約。"""

    wall_sequence: tuple[str, ...]
    furniture_id: str | None = None
    furniture_face: FaceDirection | None = None
    reflection_point_m: Vec3 | None = None
    direction_angles: PathAngles


class PathTable(BaseModel):
    rows: tuple[FurniturePath, ...]


class PathReport(BaseModel):
    path_table: PathTable | None = None


class PathPair(BaseModel):
    speaker_id: str
    receiver_id: str
    report: PathReport


class PathResult(BaseModel):
    """完整結果文件的唯讀投影；忽略算術不需要的評分包，不求解也不重評。"""

    scheme: Scheme
    pairs: tuple[PathPair, ...]


@dataclass(frozen=True)
class FurnitureEvent:
    speaker_id: str
    receiver_id: str
    furniture_id: str
    face: str
    change: Literal["appeared", "disappeared"]
    is_primary: bool


@dataclass(frozen=True)
class BoundaryDistance:
    speaker_id: str
    receiver_id: str
    furniture_id: str
    face: str
    edge_distance_m: float
    vertical_boundary_distance_deg: float


def _view(result: SchemeResult | dict[str, object]) -> PathResult:
    document = result.model_dump(include={"scheme", "pairs"}) if isinstance(result, SchemeResult) else result
    return PathResult.model_validate(document)


def _rows(result: PathResult) -> dict[tuple[str, str], tuple[FurniturePath, ...]]:
    rows: dict[tuple[str, str], tuple[FurniturePath, ...]] = {}
    for pair in result.pairs:
        key = pair.speaker_id, pair.receiver_id
        if key in rows or pair.report.path_table is None:
            raise ValueError("每對路徑表必須存在且不能重複")
        rows[key] = pair.report.path_table.rows
    return rows


def _identities(rows: tuple[FurniturePath, ...]) -> set[tuple[str, str]]:
    return {(row.furniture_id, row.furniture_face.value) for row in rows
            if row.furniture_id is not None and row.furniture_face is not None}


def furniture_events(baseline: SchemeResult | dict[str, object],
                     shifted: SchemeResult | dict[str, object]) -> tuple[FurnitureEvent, ...]:
    before, after = _view(baseline), _view(shifted)
    original, moved = _rows(before), _rows(after)
    primary = before.scheme.receiver_set.primary.receiver_id
    if original.keys() != moved.keys() or primary != after.scheme.receiver_set.primary.receiver_id:
        raise ValueError("比較必須保留同一組喇叭座位對與主位")
    events: list[FurnitureEvent] = []
    for speaker, receiver in sorted(original):
        old = _identities(original[speaker, receiver])
        new = _identities(moved[speaker, receiver])
        changes: tuple[tuple[Literal["appeared", "disappeared"], set[tuple[str, str]]], ...] = (
            ("appeared", new - old), ("disappeared", old - new))
        for change, identities in changes:
            for identifier, face in sorted(identities):
                events.append(FurnitureEvent(speaker, receiver, identifier, face, change, receiver == primary))
    return tuple(events)


def _edge_distance(row: FurniturePath, box: FurnitureBox) -> float:
    if row.furniture_face is None or row.reflection_point_m is None:
        raise ValueError("家具列缺面或反射點")
    axes = FaceDirection(row.furniture_face).edge_axes
    return min(min(abs(row.reflection_point_m[axis] - box.minimum_m[axis]),
                   abs(box.maximum_m[axis] - row.reflection_point_m[axis])) for axis in axes)


def baseline_boundary_distances(result: SchemeResult | dict[str, object], *,
                                purpose: QualityPurpose, contact_rel: float) -> tuple[BoundaryDistance, ...]:
    view = _view(result)
    rows = _rows(view)
    boxes = {item.furniture_id: item.box for item in furniture_boxes(view.scheme, contact_rel=contact_rel).furniture}
    limit = zone_limits(purpose)[0].vertical_min_abs_elevation_deg
    primary = view.scheme.receiver_set.primary.receiver_id
    distances = []
    for (speaker, receiver), paths in sorted(rows.items()):
        if receiver != primary:
            continue
        for row in paths:
            if row.furniture_id is None or row.furniture_face is None:
                continue
            distances.append(BoundaryDistance(speaker, receiver, row.furniture_id, row.furniture_face.value,
                _edge_distance(row, boxes[row.furniture_id]), abs(abs(row.direction_angles.elevation_deg) - limit)))
    return tuple(distances)
